# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Tests for the Ascend language_extra primitives (dispatch layer).

Covers: pack/unpack (pure bit ops), ld/st + ld_vector/st_vector,
local atomic_add (including its old-value return) and cross-rank remote
atomic_add through aclshmem symmetric memory.
"""
import pytest
import torch
import torch_npu  # noqa: F401  (registers the npu backend)
import triton
import triton.language as tl
from triton.language.extra.cann.extension import sub_vec_id

from triton_dist.language import make_vector
from triton_dist.language.extra import libshmem_device
from triton_dist.language.extra.language_extra import (
    atomic_add,
    ld,
    ld_vector,
    pack,
    st,
    st_vector,
    unpack,
)

G_ASH_SIZE = 1024 * 1024 * 1024
G_IP_PORT = "tcp://127.0.0.1:8666"


# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------
@triton.jit
def _pack_unpack_kernel(in_ptr, out_ptr):
    """Pack two int32 into one int64, unpack back, store round-trip result."""
    if sub_vec_id() == 0:
        lo = tl.load(in_ptr)
        hi = tl.load(in_ptr + 1)
        packed = pack(make_vector([lo, hi]), tl.int64)
        vals = unpack(packed, tl.int32)
        tl.store(out_ptr, vals[0])
        tl.store(out_ptr + 1, vals[1])


@triton.jit
def _ld_st_kernel(in_ptr, out_ptr):
    """Scalar ld/st plus a 4-element ld_vector/st_vector round trip."""
    if sub_vec_id() == 0:
        v0 = ld(in_ptr, "gpu", "relaxed")
        st(out_ptr, v0 + 1, "gpu", "relaxed")
        vec = ld_vector(in_ptr + 4, 4, "gpu", "relaxed")
        st_vector(out_ptr + 4, vec, "gpu", "relaxed")


@triton.jit
def _atomic_add_local_kernel(counter_ptr, old_ptr):
    """Local atomic_add; record the returned old value."""
    if sub_vec_id() == 0:
        old = atomic_add(counter_ptr, 5, "gpu", "relaxed")
        st(old_ptr, old, "gpu", "relaxed")


@triton.jit
def _atomic_add_remote_kernel(counter_ptr, my_rank: tl.constexpr, world_size: tl.constexpr):
    """Each PE: +1 on its own counter, then +(rank+1) on the next PE's counter."""
    libshmem_device.barrier_all_vec()

    if sub_vec_id() == 0:
        atomic_add(counter_ptr, 1, "gpu", "relaxed")
        next_pe = (my_rank + 1) % world_size
        remote = libshmem_device.remote_ptr(counter_ptr, next_pe)
        atomic_add(remote, my_rank + 1, "gpu", "relaxed")

    libshmem_device.barrier_all_vec()


# ---------------------------------------------------------------------------
# single-device tests (no aclshmem required)
# ---------------------------------------------------------------------------
@pytest.mark.dist
def test_pack_unpack():
    src = torch.tensor([0x12345678, -1], dtype=torch.int32).npu()
    out = torch.zeros(2, dtype=torch.int32).npu()

    _pack_unpack_kernel[(1, 1, 1)](src, out)

    assert torch.equal(out.cpu(), src.cpu()), f"pack/unpack round trip mismatch: {out.cpu()} != {src.cpu()}"


@pytest.mark.dist
def test_ld_st():
    src = torch.arange(8, dtype=torch.int32).npu()
    out = torch.zeros(8, dtype=torch.int32).npu()

    _ld_st_kernel[(1, 1, 1)](src, out)

    expected = torch.zeros(8, dtype=torch.int32)
    expected[0] = src.cpu()[0] + 1  # scalar ld/st path
    expected[4:8] = src.cpu()[4:8]  # ld_vector/st_vector path
    result = out.cpu()
    assert torch.equal(result, expected), f"ld/st mismatch: got {result}, expected {expected}"


@pytest.mark.dist
def test_atomic_add_local():
    counter = torch.tensor([7], dtype=torch.int32).npu()
    old = torch.zeros(1, dtype=torch.int32).npu()

    _atomic_add_local_kernel[(1, 1, 1)](counter, old)

    assert counter.cpu().item() == 12, f"counter should be 7 + 5, got {counter.cpu().item()}"
    assert old.cpu().item() == 7, f"atomic_add should return the old value 7, got {old.cpu().item()}"


# ---------------------------------------------------------------------------
# distributed test (2 ranks, aclshmem symmetric memory)
# ---------------------------------------------------------------------------
def _init_aclshmem(rank, world_size):
    import shmem as ash

    ret = ash.set_conf_store_tls(False, "")
    if ret != 0:
        raise ValueError("[ERROR] set_conf_store_tls failed")
    attributes = ash.InitAttr()
    attributes.my_rank = rank
    attributes.n_ranks = world_size
    attributes.local_mem_size = G_ASH_SIZE
    attributes.ip_port = G_IP_PORT
    attributes.option_attr.data_op_engine_type = ash.OpEngineType.MTE
    ret = ash.aclshmem_init(attributes)
    if ret != 0:
        raise ValueError("[ERROR] aclshmem_init failed")
    return ash


def _run_atomic_add_remote(rank, world_size):
    import torch.distributed as dist

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    ash = _init_aclshmem(rank, world_size)

    counter = ash.aclshmem_create_tensor([1], dtype=torch.int32, device_id=rank)
    try:
        counter.zero_()
        dist.barrier()
        _atomic_add_remote_kernel[(1, 1, 1)](counter, rank, world_size)
        dist.barrier()

        prev_rank = (rank - 1) % world_size
        expected = 1 + (prev_rank + 1)  # local +1, remote +(prev_rank+1) from prev PE
        actual = counter.cpu().item()
        assert actual == expected, f"Rank {rank}: expected counter {expected}, got {actual}"
    finally:
        ash.aclshmem_free_tensor(counter)
        _ = ash.aclshmem_finalize()


@pytest.mark.dist
def test_atomic_add_remote(dist_test):
    dist_test(_run_atomic_add_remote, world_size=2)
