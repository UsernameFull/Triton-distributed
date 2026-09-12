/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <iostream>

#include "acl/acl.h"
#include "shmemi_host_common.h"
#include "gm2gm/shmemi_device_cc_kernel.h"
#include "gm2gm/shmemi_device_p2p_sync_kernel.h"

extern "C" int rtGetC2cCtrlAddr(uint64_t *config, uint32_t *len);

static uint64_t ffts_config;

int32_t aclshmemi_sync_init()
{
    uint32_t len;
#ifdef FFTS_CONFIG
    return rtGetC2cCtrlAddr(&ffts_config, &len);
#else
    len = 0;
    ffts_config = 0;
    return 0;
#endif
}

uint64_t util_get_ffts_config()
{
    return ffts_config;
}

void aclshmemx_handle_wait(aclshmem_handle_t handle, aclrtStream stream)
{
    aclshmemi_handle_wait_on_stream(handle, stream);
}

void aclshmemx_signal_wait_until_on_stream(int32_t *sig_addr, int cmp, int32_t cmp_value, aclrtStream stream)
{
    if (stream == nullptr) {
        stream = g_state_host.default_stream;
    }
    call_aclshmemi_signal_wait_until_on_stream_kernel(sig_addr, cmp, cmp_value, stream);                            
}

ACLSHMEM_HOST_API void aclshmem_signal_wait_until(int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    if (cmp < 0 || cmp > 5) {
        SHM_LOG_ERROR("compare op is invalid.");
        return;
    }
    aclshmem_do_signal_wait_until(sig_addr, cmp, cmp_val, g_state_host.default_stream,
        g_state_host.default_block_num);

    aclrtSynchronizeStream(g_state_host.default_stream);
}

// The same macro names are used in public headers for declarations.
// Undefine them here before redefining for implementation generation.
#undef ACLSHMEM_WAIT_UNTIL
#undef ACLSHMEM_WAIT
#undef ACLSHMEM_WAIT_UNTIL_ALL
#undef ACLSHMEM_WAIT_UNTIL_ANY
#undef ACLSHMEM_WAIT_UNTIL_SOME
#undef ACLSHMEM_WAIT_UNTIL_ALL_VECTOR
#undef ACLSHMEM_WAIT_UNTIL_ANY_VECTOR
#undef ACLSHMEM_WAIT_UNTIL_SOME_VECTOR
#undef ACLSHMEM_TEST
#undef ACLSHMEM_TEST_ANY
#undef ACLSHMEM_TEST_SOME
#undef ACLSHMEM_TEST_ALL_VECTOR
#undef ACLSHMEM_TEST_ANY_VECTOR
#undef ACLSHMEM_TEST_SOME_VECTOR

#define ACLSHMEM_WAIT_UNTIL(NAME, TYPE)                                                                                \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until(TYPE *ivar, int cmp, TYPE cmp_value)                           \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_wait_until(ivar, cmp, cmp_value, g_state_host.default_stream,                             \
            g_state_host.default_block_num);                                                                           \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL);

#define ACLSHMEM_WAIT(NAME, TYPE)                                                                                      \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait(TYPE *ivar, TYPE cmp_value)                                          \
    {                                                                                                                  \
        aclshmem_##NAME##_do_wait(ivar, cmp_value, g_state_host.default_stream,                                        \
            g_state_host.default_block_num);                                                                           \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT);

#define ACLSHMEM_WAIT_UNTIL_ALL(NAME, TYPE)                                                                            \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_all(TYPE *ivars, size_t nelems, const int *status,             \
        int cmp, TYPE cmp_value)                                                                                       \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_wait_until_all(ivars, nelems, status, cmp, cmp_value,                                     \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ALL);

#define ACLSHMEM_WAIT_UNTIL_ANY(NAME, TYPE)                                                                            \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_any(TYPE *ivars, size_t nelems, const int *status,             \
        int cmp, TYPE cmp_value, size_t *res_out)                                                                      \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = SIZE_MAX;                                                                                       \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_wait_until_any(ivars, nelems, status, cmp, cmp_value, res_out,                            \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ANY);

#define ACLSHMEM_WAIT_UNTIL_SOME(NAME, TYPE)                                                                           \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_some(TYPE *ivars, size_t nelems, size_t *indices,              \
        const int *status, int cmp, TYPE cmp_value, size_t *res_out)                                                   \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = 0;                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_wait_until_some(ivars, nelems, indices, status, cmp, cmp_value, res_out,                  \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_SOME);

#define ACLSHMEM_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_all_vector(TYPE *ivars, size_t nelems, const int *status,      \
        int cmp, TYPE *cmp_values)                                                                                     \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_wait_until_all_vector(ivars, nelems, status, cmp, cmp_values,                             \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ALL_VECTOR);

#define ACLSHMEM_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_any_vector(TYPE *ivars, size_t nelems, const int *status,      \
        int cmp, TYPE *cmp_values, size_t *res_out)                                                                    \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = SIZE_MAX;                                                                                       \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_wait_until_any_vector(ivars, nelems, status, cmp, cmp_values, res_out,                    \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_ANY_VECTOR);

#define ACLSHMEM_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                                                    \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t *indices,       \
        const int *status, int cmp, TYPE *cmp_values, size_t *res_out)                                                 \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = 0;                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_wait_until_some_vector(ivars, nelems, indices, status, cmp, cmp_values, res_out,          \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_WAIT_UNTIL_SOME_VECTOR);

#define ACLSHMEM_TEST(NAME, TYPE)                                                                                      \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test(TYPE *ivars, int cmp, TYPE cmp_value, int *res_out)                  \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = 0;                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_test(ivars, cmp, cmp_value, res_out,                                                      \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST);

#define ACLSHMEM_TEST_ANY(NAME, TYPE)                                                                                  \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_any(                                                                 \
        TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value, size_t *res_out                        \
    )                                                                                                                  \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = SIZE_MAX;                                                                                       \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_test_any(ivars, nelems, status, cmp, cmp_value, res_out,                                  \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ANY);

#define ACLSHMEM_TEST_SOME(NAME, TYPE)                                                                                 \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_some(TYPE *ivars, size_t nelems, size_t *indices, const int *status, \
        int cmp, TYPE cmp_value, size_t *res_out)                                                                      \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = 0;                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_test_some(ivars, nelems, indices, status, cmp, cmp_value, res_out,                        \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_SOME);

#define ACLSHMEM_TEST_ALL_VECTOR(NAME, TYPE)                                                                           \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp,   \
        TYPE *cmp_values, int *res_out)                                                                                \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = 0;                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_test_all_vector(ivars, nelems, status, cmp, cmp_values, res_out,                          \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ALL_VECTOR);

#define ACLSHMEM_TEST_ANY_VECTOR(NAME, TYPE)                                                                           \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp,   \
        TYPE *cmp_values, size_t *res_out)                                                                             \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = SIZE_MAX;                                                                                       \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_test_any_vector(ivars, nelems, status, cmp, cmp_values, res_out,                          \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_ANY_VECTOR);

#define ACLSHMEM_TEST_SOME_VECTOR(NAME, TYPE)                                                                          \
    ACLSHMEM_HOST_API void aclshmem_##NAME##_test_some_vector(TYPE *ivars, size_t nelems, size_t *indices,             \
        const int *status, int cmp, TYPE *cmp_values, size_t *res_out)                                                 \
    {                                                                                                                  \
        if (cmp < 0 || cmp > 5) {                                                                                      \
            SHM_LOG_ERROR("compare op is invalid.");                                                                   \
            *res_out = 0;                                                                                              \
            return;                                                                                                    \
        }                                                                                                              \
        aclshmem_##NAME##_do_test_some_vector(ivars, nelems, indices, status, cmp, cmp_values, res_out,                \
            g_state_host.default_stream, g_state_host.default_block_num);                                              \
                                                                                                                       \
        aclrtSynchronizeStream(g_state_host.default_stream);                                                           \
    }

ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_TEST_SOME_VECTOR);
