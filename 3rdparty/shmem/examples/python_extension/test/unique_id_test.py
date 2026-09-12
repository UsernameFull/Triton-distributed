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
import torch_npu
import shmem as ash


g_ash_size = 1024 * 1024 * 1024
g_malloc_size = 8 * 1024 * 1024


def run_init_with_unique_id_tests():
    pe = dist.get_rank()
    world_size = dist.get_world_size()
    ret = ash.set_conf_store_tls(False, "")

    # 0. disabel TLS
    if ret != 0:
        raise ValueError("[ERROR] disable tls failed.")

    # 1. get unique id
    uid_size = 512
    tensor = torch.zeros(uid_size, dtype=torch.uint8)
    if pe == 0:
        unique_id = ash.aclshmem_get_unique_id()
        if unique_id is None:
            raise ValueError('[ERROR] get unique id failed')
        tensor = torch.tensor(list(unique_id), dtype=torch.uint8)
    dist.broadcast(tensor, src=0)
    if pe != 0:
        unique_id = bytes(tensor.tolist())
    # 2. init with unique id
    ret = ash.aclshmem_init_using_unique_id(pe, world_size, g_ash_size, unique_id)
    if ret != 0:
        raise ValueError('[ERROR] aclshmem_init failed')

    # test malloc
    aclshmem_ptr = ash.aclshmem_malloc(g_malloc_size)
    print(f'pe[{pe}]: aclshmem_ptr: {aclshmem_ptr} with type {type(aclshmem_ptr)}')
    if aclshmem_ptr is None:
        raise ValueError('[ERROR] aclshmem_malloc failed')

    # test pe
    my_pe, pe_count = ash.my_pe(), ash.pe_count()
    print(f'pe[{pe}]: my_pe:{my_pe} and pe_count:{pe_count}')
    if not (my_pe == pe and pe_count == world_size):
        raise ValueError('[ERROR] pe/world failed')

    # test free
    _ = ash.aclshmem_free(aclshmem_ptr)

    # test finialize
    _ = ash.aclshmem_finalize()


if __name__ == "__main__":
    local_pe = int(os.environ.get("LOCAL_RANK", "0"))
    torch.npu.set_device(local_pe)

    dist.init_process_group(backend="gloo", init_method="env://")
    run_init_with_unique_id_tests()
    print("test.py running success!")