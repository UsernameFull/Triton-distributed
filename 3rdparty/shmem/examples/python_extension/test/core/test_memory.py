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
import torch
import torch.distributed as dist
import shmem as ash
import shmem.core as core


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024


def run_memory_test():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    next = (pe + 1) % world_size
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
    aclshmem_buffer = core.buffer(g_malloc_size)
    print(f'pe[{pe}]: aclshmem_ptr: {aclshmem_buffer.addr} with length {type(aclshmem_buffer.length)}')
    if (aclshmem_buffer.addr is None) or (aclshmem_buffer.length != g_malloc_size):
        raise ValueError('[ERROR] create buffer failed')

    # 4. get next pe buffer
    next_aclshmem_buffer = core.get_peer_buffer(aclshmem_buffer, next)
    if (next_aclshmem_buffer.addr is None) or (next_aclshmem_buffer.length != g_malloc_size):
        raise ValueError('[ERROR] get peer buffer failed')

    # 5. free buffer
    core.free(aclshmem_buffer)

    # 6. finialize
    core.finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)

    dist.init_process_group(backend="gloo", init_method="env://")
    run_memory_test()
    print("test_memory running success!")