/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acl/acl.h"
#include "kernel_operator.h"

#include "shmem.h"
#include "host_device/shmem_common_types.h"

// kernels
ACLSHMEM_GLOBAL void aclshmem_signal_wait_until_kernel(__gm__ int32_t *sig_addr, int cmp, int32_t cmp_val)
{
    aclshmem_signal_wait_until(sig_addr, cmp, cmp_val);
}

#define TEST_ACLSHMEMI_WAIT_UNTIL(NAME, TYPE)                                                                          \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_wait_until_kernel(__gm__ TYPE *ivar, int cmp, TYPE cmp_value)               \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until(ivar, cmp, cmp_value);                                                            \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_WAIT_UNTIL);

#define TEST_ACLSHMEMI_WAIT(NAME, TYPE)                                                                                \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_wait_kernel(__gm__ TYPE *ivar, TYPE cmp_value)                              \
    {                                                                                                                  \
        aclshmem_##NAME##_wait(ivar, cmp_value);                                                                       \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_WAIT);

#define TEST_ACLSHMEMI_WAIT_UNTIL_ALL(NAME, TYPE)                                                                      \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_wait_until_all_kernel(__gm__ TYPE *ivars, size_t nelems,                    \
        __gm__ const int *status, int cmp, TYPE cmp_value)                                                             \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_all(ivars, nelems, status, cmp, cmp_value);                                       \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_WAIT_UNTIL_ALL);

#define TEST_ACLSHMEMI_WAIT_UNTIL_ANY(NAME, TYPE)                                                                      \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_wait_until_any_kernel(__gm__ TYPE *ivars, size_t nelems,                    \
        __gm__ const int *status, int cmp, TYPE cmp_value, __gm__ size_t *res_out)                                     \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_wait_until_any(ivars, nelems, status, cmp, cmp_value);                            \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_WAIT_UNTIL_ANY);

#define TEST_ACLSHMEMI_WAIT_UNTIL_SOME(NAME, TYPE)                                                                     \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_wait_until_some_kernel(__gm__ TYPE *ivars, size_t nelems,                   \
        __gm__ size_t *indices, __gm__ const int *status, int cmp, TYPE cmp_value, __gm__ size_t *res_out)             \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_wait_until_some(ivars, nelems, indices, status, cmp, cmp_value);                  \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_WAIT_UNTIL_SOME);

#define TEST_ACLSHMEMI_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)                                                               \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_wait_until_all_vector_kernel(__gm__ TYPE *ivars, size_t nelems,             \
        __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values)                                                    \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_all_vector(ivars, nelems, status, cmp, cmp_values);                               \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_WAIT_UNTIL_ALL_VECTOR);

#define TEST_ACLSHMEMI_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                                                               \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_wait_until_any_vector_kernel(__gm__ TYPE *ivars, size_t nelems,             \
        __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values, __gm__ size_t *res_out)                            \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_wait_until_any_vector(ivars, nelems, status, cmp, cmp_values);                    \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_WAIT_UNTIL_ANY_VECTOR);

#define TEST_ACLSHMEMI_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                                              \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_wait_until_some_vector_kernel(__gm__ TYPE *ivars, size_t nelems,            \
        __gm__ size_t *indices, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values, __gm__ size_t *res_out)    \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_wait_until_some_vector(ivars, nelems, indices, status, cmp, cmp_values);          \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_WAIT_UNTIL_SOME_VECTOR);

#define TEST_ACLSHMEMI_TEST(NAME, TYPE)                                                                                \
    ACLSHMEM_GLOBAL void aclshmemi_##NAME##_test_kernel(__gm__ TYPE *ivars, int cmp, TYPE cmp_value,                   \
        __gm__ int *res_out)                                                                                           \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_test(ivars, cmp, cmp_value);                                                      \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_TEST);

#define TEST_ACLSHMEMI_TEST_ANY(NAME, TYPE)                                                                            \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_test_any_kernel(__gm__ TYPE *ivars, size_t nelems,                          \
        __gm__ const int *status, int cmp, TYPE cmp_value, __gm__ size_t *res_out)                                     \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_test_any(ivars, nelems, status, cmp, cmp_value);                                  \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_TEST_ANY);

#define TEST_ACLSHMEMI_TEST_SOME(NAME, TYPE)                                                                           \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_test_some_kernel(__gm__ TYPE *ivars, size_t nelems, __gm__ size_t *indices, \
        __gm__ const int *status, int cmp, TYPE cmp_value, __gm__ size_t *res_out)                                     \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_test_some(ivars, nelems, indices, status, cmp, cmp_value);                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_TEST_SOME);

#define TEST_ACLSHMEMI_TEST_ALL_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_test_all_vector_kernel(__gm__ TYPE *ivars, size_t nelems,                   \
        __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values, __gm__ int *res_out)                               \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_test_all_vector(ivars, nelems, status, cmp, cmp_values);                          \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_TEST_ALL_VECTOR);

#define TEST_ACLSHMEMI_TEST_ANY_VECTOR(NAME, TYPE)                                                                     \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_test_any_vector_kernel(__gm__ TYPE *ivars, size_t nelems,                   \
        __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values, __gm__ size_t *res_out)                            \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_test_any_vector(ivars, nelems, status, cmp, cmp_values);                          \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_TEST_ANY_VECTOR);

#define TEST_ACLSHMEMI_TEST_SOME_VECTOR(NAME, TYPE)                                                                    \
    ACLSHMEM_GLOBAL void aclshmem_##NAME##_test_some_vector_kernel(__gm__ TYPE *ivars, size_t nelems,                  \
        __gm__ size_t *indices, __gm__ const int *status, int cmp, __gm__ TYPE *cmp_values, __gm__ size_t *res_out)    \
    {                                                                                                                  \
        *res_out = aclshmem_##NAME##_test_some_vector(ivars, nelems, indices, status, cmp, cmp_values);                \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(TEST_ACLSHMEMI_TEST_SOME_VECTOR);

// interface
void aclshmem_do_signal_wait_until(int32_t *sig_addr, int cmp, int32_t cmp_val, aclrtStream stream = nullptr,
    size_t block_dim = 1)
{
    aclshmem_signal_wait_until_kernel<<<block_dim, nullptr, stream>>>(sig_addr, cmp, cmp_val);
}

#define ACLSHMEM_DO_WAIT_UNTIL(NAME, TYPE)                                                                             \
    void aclshmem_##NAME##_do_wait_until(TYPE *ivar, int cmp, TYPE cmp_value, aclrtStream stream = nullptr,            \
        size_t block_dim = 1)                                                                                          \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_kernel<<<block_dim, nullptr, stream>>>(ivar, cmp, cmp_value);                     \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL);

#define ACLSHMEM_DO_WAIT(NAME, TYPE)                                                                                   \
    void aclshmem_##NAME##_do_wait(TYPE *ivar, TYPE cmp_value, aclrtStream stream = nullptr, size_t block_dim = 1)     \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_kernel<<<block_dim, nullptr, stream>>>(ivar, cmp_value);                                \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT);

#define ACLSHMEM_DO_WAIT_UNTIL_ALL(NAME, TYPE)                                                                         \
    void aclshmem_##NAME##_do_wait_until_all(TYPE *ivars, size_t nelems, const int *status,                            \
        int cmp, TYPE cmp_value, aclrtStream stream = nullptr, size_t block_dim = 1)                                   \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_all_kernel<<<block_dim, nullptr, stream>>>(                                       \
            ivars, nelems, status, cmp, cmp_value                                                                      \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_ALL);

#define ACLSHMEM_DO_WAIT_UNTIL_ANY(NAME, TYPE)                                                                         \
    void aclshmem_##NAME##_do_wait_until_any(TYPE *ivars, size_t nelems, const int *status,                            \
        int cmp, TYPE cmp_value, size_t *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                  \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_any_kernel<<<block_dim, nullptr, stream>>>(                                       \
            ivars, nelems, status, cmp, cmp_value, res_out                                                             \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_ANY);

#define ACLSHMEM_DO_WAIT_UNTIL_SOME(NAME, TYPE)                                                                        \
    void aclshmem_##NAME##_do_wait_until_some(TYPE *ivars, size_t nelems, size_t *indices, const int *status,          \
        int cmp, TYPE cmp_value, size_t *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                  \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_some_kernel<<<block_dim, nullptr, stream>>>(                                      \
            ivars, nelems, indices, status, cmp, cmp_value, res_out                                                    \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_SOME);

#define ACLSHMEM_DO_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)                                                                  \
    void aclshmem_##NAME##_do_wait_until_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp,            \
        TYPE *cmp_values, aclrtStream stream = nullptr, size_t block_dim = 1)                                          \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_all_vector_kernel<<<block_dim, nullptr, stream>>>(                                \
            ivars, nelems, status, cmp, cmp_values                                                                     \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_ALL_VECTOR);

#define ACLSHMEM_DO_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)                                                                  \
    void aclshmem_##NAME##_do_wait_until_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp,            \
        TYPE *cmp_values, size_t *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                         \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_any_vector_kernel<<<block_dim, nullptr, stream>>>(                                \
            ivars, nelems, status, cmp, cmp_values, res_out                                                            \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_ANY_VECTOR);

#define ACLSHMEM_DO_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                                                 \
    void aclshmem_##NAME##_do_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status,   \
        int cmp, TYPE *cmp_values, size_t *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                \
    {                                                                                                                  \
        aclshmem_##NAME##_wait_until_some_vector_kernel<<<block_dim, nullptr, stream>>>(ivars, nelems, indices,        \
            status, cmp, cmp_values, res_out);                                                                         \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_WAIT_UNTIL_SOME_VECTOR);

#define ACLSHMEM_DO_TEST(NAME, TYPE)                                                                                   \
    void aclshmem_##NAME##_do_test(TYPE *ivars, int cmp, TYPE cmp_value, int *res_out, aclrtStream stream = nullptr,   \
        size_t block_dim = 1)                                                                                          \
    {                                                                                                                  \
        aclshmemi_##NAME##_test_kernel<<<block_dim, nullptr, stream>>>(ivars, cmp, cmp_value, res_out);                \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST);

#define ACLSHMEM_DO_TEST_ANY(NAME, TYPE)                                                                               \
    void aclshmem_##NAME##_do_test_any(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE cmp_value,         \
        size_t *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                                           \
    {                                                                                                                  \
        aclshmem_##NAME##_test_any_kernel<<<block_dim, nullptr, stream>>>(                                             \
            ivars, nelems, status, cmp, cmp_value, res_out                                                             \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_ANY);

#define ACLSHMEM_DO_TEST_SOME(NAME, TYPE)                                                                              \
    void aclshmem_##NAME##_do_test_some(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp,       \
        TYPE cmp_value, size_t *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                           \
    {                                                                                                                  \
        aclshmem_##NAME##_test_some_kernel<<<block_dim, nullptr, stream>>>(                                            \
            ivars, nelems, indices, status, cmp, cmp_value, res_out                                                    \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_SOME);

#define ACLSHMEM_DO_TEST_ALL_VECTOR(NAME, TYPE)                                                                        \
    void aclshmem_##NAME##_do_test_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp,                  \
        TYPE *cmp_values, int *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                            \
    {                                                                                                                  \
        aclshmem_##NAME##_test_all_vector_kernel<<<block_dim, nullptr, stream>>>(                                      \
            ivars, nelems, status, cmp, cmp_values, res_out                                                            \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_ALL_VECTOR);

#define ACLSHMEM_DO_TEST_ANY_VECTOR(NAME, TYPE)                                                                        \
    void aclshmem_##NAME##_do_test_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp,                  \
        TYPE *cmp_values, size_t *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                         \
    {                                                                                                                  \
        aclshmem_##NAME##_test_any_vector_kernel<<<block_dim, nullptr, stream>>>(                                      \
            ivars, nelems, status, cmp, cmp_values, res_out                                                            \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_ANY_VECTOR);

#define ACLSHMEM_DO_TEST_SOME_VECTOR(NAME, TYPE)                                                                       \
    void aclshmem_##NAME##_do_test_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status,         \
        int cmp, TYPE *cmp_values, size_t *res_out, aclrtStream stream = nullptr, size_t block_dim = 1)                \
    {                                                                                                                  \
        aclshmem_##NAME##_test_some_vector_kernel<<<block_dim, nullptr, stream>>>(                                     \
            ivars, nelems, indices, status, cmp, cmp_values, res_out                                                   \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_FUNC(ACLSHMEM_DO_TEST_SOME_VECTOR);