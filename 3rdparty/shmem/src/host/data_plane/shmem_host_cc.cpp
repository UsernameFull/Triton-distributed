/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "host/data_plane/shmem_host_cc.h"

#include "acl/acl.h"

#include "shmemi_host_common.h"
#include "gm2gm/shmemi_device_cc_kernel.h"

namespace {
int aclshmemi_barrier_on_stream(aclshmem_team_t team, aclrtStream stream)
{
    if (team == -1) { // not in team
        return ACLSHMEM_SUCCESS;
    }
    // ACLSHMEM_CHECK_RET(aclshmemx_init_status(), "aclshmemx_init_status failed.");
    aclshmemx_init_status();
    ACLSHMEM_CHECK_RET(aclshmemi_call_barrier_on_stream_kernel(team, stream),
        "aclshmemi_call_barrier_on_stream_kernel failed.");
    return ACLSHMEM_SUCCESS;
}
} // namespace

void aclshmem_barrier(aclshmem_team_t team)
{
    aclshmemi_barrier_on_stream(team, g_state_host.default_stream);
    aclrtSynchronizeStream(g_state_host.default_stream);
}

void aclshmem_barrier_all()
{
    aclshmemi_barrier_on_stream(ACLSHMEM_TEAM_WORLD, g_state_host.default_stream);
    aclrtSynchronizeStream(g_state_host.default_stream);
}

void aclshmemx_barrier_on_stream(aclshmem_team_t team, aclrtStream stream)
{
    aclshmemi_barrier_on_stream(team, stream);
}

void aclshmemx_barrier_all_on_stream(aclrtStream stream)
{
    aclshmemi_barrier_on_stream(ACLSHMEM_TEAM_WORLD, stream);
}

void aclshmem_sync(aclshmem_team_t team)
{
    aclshmem_barrier(team);
}

void aclshmem_sync_all()
{
    aclshmem_sync(ACLSHMEM_TEAM_WORLD);
}
