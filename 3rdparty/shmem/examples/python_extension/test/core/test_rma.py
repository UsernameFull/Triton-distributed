# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
import ctypes
import torch
import torch.distributed as dist
import acl
import shmem as ash
import shmem.core as core


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024
g_signal_size = 4
g_value = 1
g_sig_value = 2

# ACL memory copy direction constants (acl Python module does not export named enums)
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3


def _read_int32_from_device(addr):
    """Read a single int32 value from device memory at the given address."""
    host_val = ctypes.c_int32(0)
    ret = acl.rt.memcpy(ctypes.addressof(host_val), 4, addr, 4,
                        ACL_MEMCPY_DEVICE_TO_HOST)
    if ret != 0:
        raise RuntimeError(f"[ERROR] acl.rt.memcpy failed, ret={ret}")
    return host_val.value


def run_put_signal_test():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    next_pe = (pe + 1) % world_size
    prev_pe = (pe - 1 + world_size) % world_size
    ret = ash.set_conf_store_tls(False, "")

    # 0. disabel TLS
    if ret != 0:
        raise ValueError("[ERROR] disable tls failed.")

    # 1. get unique id
    uid_size = 512
    tensor = torch.zeros(uid_size, dtype=torch.uint8)
    if pe == 0:
        unique_id = core.get_unique_id()
        if unique_id is None:
            raise ValueError('[ERROR] get unique id failed')
        tensor = torch.tensor(list(unique_id), dtype=torch.uint8)
    dist.broadcast(tensor, src=0)
    if pe != 0:
        unique_id = bytes(tensor.tolist())

    # 2. init with unique id
    core.init(rank=pe, nranks=world_size, mem_size=g_ash_size, uid=unique_id, initializer_method='uid')

    # 3. malloc buffer
    send_aclshmem_buffer = core.buffer(g_malloc_size)
    if (send_aclshmem_buffer.addr is None) or (send_aclshmem_buffer.length != g_malloc_size):
        raise ValueError('[ERROR] create send buffer failed')
    acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)

    recv_aclshmem_buffer = core.buffer(g_malloc_size)
    if (recv_aclshmem_buffer.addr is None) or (recv_aclshmem_buffer.length != g_malloc_size):
        raise ValueError('[ERROR] create recv buffer failed')
    acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)

    signal_aclshmem_buffer = core.buffer(g_signal_size)
    if (signal_aclshmem_buffer.addr is None) or (signal_aclshmem_buffer.length != g_signal_size):
        raise ValueError('[ERROR] create signal buffer failed')
    acl.rt.memset(signal_aclshmem_buffer.addr, g_signal_size, 0, g_signal_size)

    # 4. write known value to local send buffer (scalar put to own PE)
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * pe, pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    send_val = _read_int32_from_device(send_aclshmem_buffer.addr)
    assert send_val == g_value * pe, \
        f"[FAIL] step 4: expected send buffer {g_value * pe}, got {send_val}"

    # 5. put_signal: copy local send buffer to next PE's recv buffer, set signal on next PE
    core.put_signal(recv_aclshmem_buffer, send_aclshmem_buffer, signal_aclshmem_buffer, g_sig_value,
                    core.direct.SignalOp.SIGNAL_SET, next_pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    dist.barrier()
    signal_val = _read_int32_from_device(signal_aclshmem_buffer.addr)
    assert signal_val == g_sig_value, \
        f"[FAIL] step 5: expected signal {g_sig_value}, got {signal_val} on {pe=}"
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * prev_pe, \
        f"[FAIL] step 5: expected recv buffer {g_value * prev_pe} from PE {prev_pe}, got {recv_val}"

    # 6. signal_op + signal_wait: atomic signal op on own PE, then wait for it
    stream, _ = acl.rt.create_stream()
    acl.rt.memset(signal_aclshmem_buffer.addr, g_signal_size, 0, g_signal_size)
    core.signal_op(signal_aclshmem_buffer, g_sig_value, core.direct.SignalOp.SIGNAL_SET, pe, stream=stream)
    core.signal_wait(signal_aclshmem_buffer, g_sig_value, core.direct.ComparisonType.CMP_EQ, stream=stream)
    acl.rt.synchronize_stream(stream)
    signal_val = _read_int32_from_device(signal_aclshmem_buffer.addr)
    assert signal_val == g_sig_value, \
        f"[FAIL] step 6: expected signal {g_sig_value} after signal_op, got {signal_val}"
    acl.rt.destroy_stream(stream)

    # 7. put: non-blocking put to next PE on a stream
    stream, _ = acl.rt.create_stream()
    acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)
    acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * pe, pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    core.put(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream)
    acl.rt.synchronize_stream(stream)
    acl.rt.destroy_stream(stream)
    dist.barrier()
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * prev_pe, \
        f"[FAIL] step 7: expected recv buffer {g_value * prev_pe} from PE {prev_pe}, got {recv_val}"

    # 8. get: non-blocking get from next PE on a stream
    stream, _ = acl.rt.create_stream()
    acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)
    acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * next_pe, next_pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    core.get(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream)
    acl.rt.synchronize_stream(stream)
    acl.rt.destroy_stream(stream)
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * next_pe, \
        f"[FAIL] step 8: expected recv buffer {g_value * next_pe} from PE {next_pe}, got {recv_val}"

    # 9. quiet: ensure all outstanding RMA operations complete before verifying
    acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)
    acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * pe, pe)
    core.quiet(stream=0)
    torch.npu.synchronize()
    core.put(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream=0)
    core.quiet(stream=0)
    torch.npu.synchronize()
    dist.barrier()
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * prev_pe, \
        f"[FAIL] step 9: expected recv buffer {g_value * prev_pe} from PE {prev_pe}, got {recv_val}"

    # 10. quiet_on_stream: explicit stream put + quiet_on_stream + synchronize
    stream, _ = acl.rt.create_stream()
    acl.rt.memset(send_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)
    acl.rt.memset(recv_aclshmem_buffer.addr, g_malloc_size, 0, g_malloc_size)
    ash._pyshmem.aclshmem_int32_p(send_aclshmem_buffer.addr, g_value * pe, pe)
    torch.npu.synchronize()
    core.put(recv_aclshmem_buffer, send_aclshmem_buffer, next_pe, stream)
    core.quiet(stream)
    acl.rt.synchronize_stream(stream)
    acl.rt.destroy_stream(stream)
    dist.barrier()
    recv_val = _read_int32_from_device(recv_aclshmem_buffer.addr)
    assert recv_val == g_value * prev_pe, \
        f"[FAIL] step 10: expected recv buffer {g_value * prev_pe} from PE {prev_pe}, got {recv_val}"

    # 11. free and finialize
    core.free(send_aclshmem_buffer)
    core.free(recv_aclshmem_buffer)
    core.free(signal_aclshmem_buffer)
    core.finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)

    dist.init_process_group(backend="gloo", init_method="env://")
    run_put_signal_test()
    print("test_rma running success!")
