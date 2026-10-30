"""Two-node Triton-distributed put/get correctness smoke test (one NPU per node)."""

import argparse
from datetime import timedelta
import ipaddress
import os
import socket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shmem-port", type=int, default=8765,
                        help="SHMEM bootstrap port on MASTER_ADDR, separate from torchrun")
    parser.add_argument("--elements", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180,
                        help="HCCL timeout in seconds; use an outer timeout for device hangs")
    args = parser.parse_args()
    if not 1 <= args.shmem_port <= 65535:
        parser.error("--shmem-port must be between 1 and 65535")
    if not 1 <= args.elements <= 1048576 or args.iterations < 1 or args.timeout < 1:
        parser.error("elements must be 1..1048576; iterations and timeout must be positive")
    if (int(os.environ.get("WORLD_SIZE", "0")) != 2 or
            int(os.environ.get("LOCAL_WORLD_SIZE", "0")) != 1):
        parser.error("launch with torchrun --nnodes=2 --nproc-per-node=1")
    address = os.environ.get("MASTER_ADDR", "")
    try:
        master_ip = ipaddress.IPv4Address(address)
    except ipaddress.AddressValueError:
        parser.error("MASTER_ADDR must be node 0's reachable IPv4 address")
    if master_ip.is_loopback or master_ip.is_unspecified:
        parser.error("MASTER_ADDR must be reachable from both nodes, not a loopback/unspecified address")
    if args.shmem_port == int(os.environ.get("MASTER_PORT", "0")):
        parser.error("SHMEM and torchrun must use different ports")
    return args


def main(args: argparse.Namespace) -> None:
    # Keep --help and argument validation available without NPU libraries.
    import shmem as ash
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401: registers torch.npu
    import triton
    import triton.language as tl
    from triton.runtime import driver
    from triton_dist.language.extra import libshmem_device

    @triton.jit
    def put_kernel(src, dst, nbytes: tl.constexpr, peer: tl.constexpr):
        libshmem_device.putmem(dst, src, nbytes, peer)

    @triton.jit
    def get_kernel(src, dst, nbytes: tl.constexpr, peer: tl.constexpr):
        libshmem_device.getmem(dst, src, nbytes, peer)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl", timeout=timedelta(seconds=args.timeout))
    try:
        rank = dist.get_rank()
        peer = 1 - rank
        target = driver.active.get_current_target()
        if target.backend not in ("ascend", "npu"):
            raise RuntimeError(f"Expected Ascend Triton backend, got {target}")
        endpoint = f"tcp://{os.environ['MASTER_ADDR']}:{args.shmem_port}"
        print(f"START host={socket.gethostname()} rank={rank} local_rank={local_rank} "
              f"target={target} shmem={endpoint} engine=ROCE", flush=True)
        if ash.set_conf_store_tls(False, "") != 0:
            raise RuntimeError("set_conf_store_tls failed")
        attr = ash.InitAttr()
        attr.my_rank = rank
        attr.n_ranks = 2
        attr.local_mem_size = 1024 * 1024 * 1024
        attr.ip_port = endpoint
        attr.option_attr.data_op_engine_type = ash.OpEngineType.ROCE
        ret = ash.aclshmem_init(attr)
        if ret != 0:
            raise RuntimeError(f"aclshmem_init(ROCE) returned {ret}; check SHMEM RDMA build and RoCE network")

        src = ash.aclshmem_create_tensor([args.elements], dtype=torch.int32, device_id=local_rank)
        dst = ash.aclshmem_create_tensor([args.elements], dtype=torch.int32, device_id=local_rank)
        nbytes = args.elements * src.element_size()
        base = torch.arange(args.elements, dtype=torch.int32)

        for iteration in range(args.iterations):
            # Distinct payloads expose both stale data and wrong-rank transfers.
            src.copy_(base + rank * 10000000 + iteration)
            expected = base + peer * 10000000 + iteration
            for operation, kernel in (("putmem", put_kernel), ("getmem", get_kernel)):
                dst.fill_(-1)
                torch.npu.synchronize()
                dist.barrier()
                print(f"RUN rank={rank} iteration={iteration} op={operation}", flush=True)
                kernel[(1, 1, 1)](src, dst, nbytes, peer)
                torch.npu.synchronize()
                # Blocking put completion on both senders precedes destination reads.
                dist.barrier()
                received = dst.cpu()
                local_ok = torch.equal(received, expected)
                verdict = torch.tensor([int(local_ok)], dtype=torch.int32, device=src.device)
                dist.all_reduce(verdict, op=dist.ReduceOp.MIN)
                if verdict.item() != 1:
                    detail = "peer rank reported incorrect data"
                    if not local_ok:
                        index = (received != expected).nonzero()[0].item()
                        detail = f"index={index}, got={received[index].item()}, expected={expected[index].item()}"
                    raise AssertionError(f"rank={rank} iteration={iteration} {operation}: {detail}")
                print(f"PASS rank={rank} iteration={iteration} op={operation} bytes={nbytes}", flush=True)

        # Only enter collective SHMEM cleanup after every data check passes.
        dist.barrier()
        ash.aclshmem_free_tensor(dst)
        ash.aclshmem_free_tensor(src)
        ret = ash.aclshmem_finalize()
        if ret not in (None, 0):
            raise RuntimeError(f"aclshmem_finalize returned {ret}")
        dist.barrier()
        print(f"PASS: Triton-distributed two-node put/get rank={rank}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main(parse_args())
