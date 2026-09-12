/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "shmem.h"

#define ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(FUNC)                                                                          \
    FUNC(float, float, 10.0f);                                                                                         \
    FUNC(int8, int8_t, 10);                                                                                            \
    FUNC(int16, int16_t, 10);                                                                                          \
    FUNC(int32, int32_t, 10);                                                                                          \
    FUNC(int64, int64_t, 10);                                                                                          \
    FUNC(uint8, uint8_t, 10);                                                                                          \
    FUNC(uint16, uint16_t, 10);                                                                                        \
    FUNC(uint32, uint32_t, 10);                                                                                        \
    FUNC(uint64, uint64_t, 10);                                                                                        \
    FUNC(char, char, 'b')

extern "C" ACLSHMEM_GLOBAL void p2p_chain(uint64_t config, GM_ADDR addr, int rank_id, int rank_size) {
    util_set_ffts_config(config);
    auto sig_addr = (__gm__ int32_t *)addr;
    int32_t val = *sig_addr;
    int next = (rank_id + 1) % rank_size;

    aclshmem_barrier_all();

#if defined(__DAV_C220_VEC__) || defined(__DAV_C310_VEC__)
    if (rank_id == 0) {
        aclshmemx_signal_op(sig_addr, 1, ACLSHMEM_SIGNAL_SET, next);
        aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_EQ, 1);
    } else {
        aclshmem_signal_wait_until(sig_addr, ACLSHMEM_CMP_EQ, 1);
        aclshmemx_signal_op(sig_addr, 1, ACLSHMEM_SIGNAL_SET, next);
    }
#endif

    aclshmem_barrier_all();
}

void p2p_chain_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)
{
    p2p_chain<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size);
}

#define P2P_CHAIN_WAIT_UNTIL(NAME, TYPE, CMP_VAL)                                                                      \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until(uint64_t config, GM_ADDR addr,                       \
        int rank_id, int rank_size)                                                                                    \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_EQ, CMP_VAL);                                                  \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_NE, (CMP_VAL) - 1);                                            \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);                                            \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_GE, CMP_VAL);                                                  \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);                                            \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);                                            \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_LE, CMP_VAL);                                                  \
        aclshmem_##NAME##_wait_until(ivar, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);                                            \
                                                                                                                       \
        aclshmem_##NAME##_wait(ivar, (CMP_VAL) - 1);                                                                   \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL);

#define P2P_CHAIN_DO_WAIT_UNTIL(NAME, TYPE, CMP_VAL)                                                                   \
    void p2p_chain_do_##NAME##_wait_until(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)    \
    {                                                                                                                  \
        p2p_chain_##NAME##_wait_until<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size);                       \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL);

#define P2P_CHAIN_WAIT_UNTIL_ALL(NAME, TYPE, CMP_VAL)                                                                  \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until_all(uint64_t config, GM_ADDR addr,                   \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status)                                           \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_EQ, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_NE, (CMP_VAL) - 1);                        \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);                        \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_GE, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);                        \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);                        \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_LE, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_all(ivar, nelems, status, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);                        \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL_ALL);

#define P2P_CHAIN_DO_WAIT_UNTIL_ALL(NAME, TYPE, CMP_VAL)                                                               \
    void p2p_chain_do_##NAME##_wait_until_all(void *stream, uint64_t config, uint8_t *addr, int rank_id,               \
        int rank_size, size_t nelems, const int *status)                                                               \
    {                                                                                                                  \
        p2p_chain_##NAME##_wait_until_all<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, nelems, status);   \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_ALL);

#define P2P_CHAIN_WAIT_UNTIL_ANY(NAME, TYPE, CMP_VAL)                                                                  \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until_any(uint64_t config, GM_ADDR addr,                   \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status)                                           \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until_any(ivar, nelems, status, ACLSHMEM_CMP_EQ, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_any(ivar, nelems, status, ACLSHMEM_CMP_NE, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_any(ivar, nelems, status, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);                        \
        aclshmem_##NAME##_wait_until_any(ivar, nelems, status, ACLSHMEM_CMP_GE, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_any(ivar, nelems, status, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);                        \
        aclshmem_##NAME##_wait_until_any(ivar, nelems, status, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);                        \
        aclshmem_##NAME##_wait_until_any(ivar, nelems, status, ACLSHMEM_CMP_LE, CMP_VAL);                              \
        aclshmem_##NAME##_wait_until_any(ivar, nelems, status, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);                        \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL_ANY);

#define P2P_CHAIN_DO_WAIT_UNTIL_ANY(NAME, TYPE, CMP_VAL)                                                               \
    void p2p_chain_do_##NAME##_wait_until_any(void *stream, uint64_t config, uint8_t *addr, int rank_id,               \
        int rank_size, size_t nelems, const int *status)                                                               \
    {                                                                                                                  \
        p2p_chain_##NAME##_wait_until_any<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, nelems, status);   \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_ANY);

#define P2P_CHAIN_WAIT_UNTIL_SOME(NAME, TYPE, CMP_VAL)                                                                 \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until_some(uint64_t config, GM_ADDR addr,                  \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ size_t *indices)                   \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until_some(ivar, nelems, indices, status, ACLSHMEM_CMP_EQ, CMP_VAL);                    \
        aclshmem_##NAME##_wait_until_some(ivar, nelems, indices, status, ACLSHMEM_CMP_NE, CMP_VAL);                    \
        aclshmem_##NAME##_wait_until_some(ivar, nelems, indices, status, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);              \
        aclshmem_##NAME##_wait_until_some(ivar, nelems, indices, status, ACLSHMEM_CMP_GE, CMP_VAL);                    \
        aclshmem_##NAME##_wait_until_some(ivar, nelems, indices, status, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);              \
        aclshmem_##NAME##_wait_until_some(ivar, nelems, indices, status, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);              \
        aclshmem_##NAME##_wait_until_some(ivar, nelems, indices, status, ACLSHMEM_CMP_LE, CMP_VAL);                    \
        aclshmem_##NAME##_wait_until_some(ivar, nelems, indices, status, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);              \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL_SOME);

#define P2P_CHAIN_DO_WAIT_UNTIL_SOME(NAME, TYPE, CMP_VAL)                                                              \
    void p2p_chain_do_##NAME##_wait_until_some(void *stream, uint64_t config, uint8_t *addr, int rank_id,              \
        int rank_size, size_t nelems, const int *status, size_t *indices)                                              \
    {                                                                                                                  \
        p2p_chain_##NAME##_wait_until_some<<<1, nullptr, stream>>>(                                                    \
            config, addr, rank_id, rank_size, nelems, status, indices                                                  \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_SOME);

#define P2P_CHAIN_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE, CMP_VAL)                                                           \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until_all_vector(uint64_t config, GM_ADDR addr,            \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ TYPE *cmp_values)                  \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until_all_vector(ivar, nelems, status, ACLSHMEM_CMP_EQ, cmp_values);                    \
        aclshmem_##NAME##_wait_until_all_vector(ivar, nelems, status, ACLSHMEM_CMP_NE, cmp_values + nelems);           \
        aclshmem_##NAME##_wait_until_all_vector(ivar, nelems, status, ACLSHMEM_CMP_GT, cmp_values + nelems);           \
        aclshmem_##NAME##_wait_until_all_vector(ivar, nelems, status, ACLSHMEM_CMP_GE, cmp_values);                    \
        aclshmem_##NAME##_wait_until_all_vector(ivar, nelems, status, ACLSHMEM_CMP_GE, cmp_values + nelems);           \
        aclshmem_##NAME##_wait_until_all_vector(ivar, nelems, status, ACLSHMEM_CMP_LT, cmp_values + 2 * nelems);       \
        aclshmem_##NAME##_wait_until_all_vector(ivar, nelems, status, ACLSHMEM_CMP_LE, cmp_values);                    \
        aclshmem_##NAME##_wait_until_all_vector(ivar, nelems, status, ACLSHMEM_CMP_LE, cmp_values + 2 * nelems);       \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL_ALL_VECTOR);

#define P2P_CHAIN_DO_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE, CMP_VAL)                                                        \
    void p2p_chain_do_##NAME##_wait_until_all_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,        \
        int rank_size, size_t nelems, const int *status, TYPE *cmp_values)                                             \
    {                                                                                                                  \
        p2p_chain_##NAME##_wait_until_all_vector<<<1, nullptr, stream>>>(                                              \
            config, addr, rank_id, rank_size, nelems, status, cmp_values                                               \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_ALL_VECTOR);

#define P2P_CHAIN_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE, CMP_VAL)                                                           \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until_any_vector(uint64_t config, GM_ADDR addr,            \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ TYPE *cmp_values)                  \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until_any_vector(ivar, nelems, status, ACLSHMEM_CMP_EQ, cmp_values);                    \
        aclshmem_##NAME##_wait_until_any_vector(ivar, nelems, status, ACLSHMEM_CMP_NE, cmp_values);                    \
        aclshmem_##NAME##_wait_until_any_vector(ivar, nelems, status, ACLSHMEM_CMP_GT, cmp_values + nelems);           \
        aclshmem_##NAME##_wait_until_any_vector(ivar, nelems, status, ACLSHMEM_CMP_GE, cmp_values);                    \
        aclshmem_##NAME##_wait_until_any_vector(ivar, nelems, status, ACLSHMEM_CMP_GE, cmp_values + nelems);           \
        aclshmem_##NAME##_wait_until_any_vector(ivar, nelems, status, ACLSHMEM_CMP_LT, cmp_values + 2 * nelems);       \
        aclshmem_##NAME##_wait_until_any_vector(ivar, nelems, status, ACLSHMEM_CMP_LE, cmp_values);                    \
        aclshmem_##NAME##_wait_until_any_vector(ivar, nelems, status, ACLSHMEM_CMP_LE, cmp_values + 2 * nelems);       \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL_ANY_VECTOR);

#define P2P_CHAIN_DO_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE, CMP_VAL)                                                        \
    void p2p_chain_do_##NAME##_wait_until_any_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,        \
        int rank_size, size_t nelems, const int *status, TYPE *cmp_values)                                             \
    {                                                                                                                  \
        p2p_chain_##NAME##_wait_until_any_vector<<<1, nullptr, stream>>>(                                              \
            config, addr, rank_id, rank_size, nelems, status, cmp_values                                               \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_ANY_VECTOR);

#define P2P_CHAIN_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE, CMP_VAL)                                                          \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_wait_until_some_vector(uint64_t config, GM_ADDR addr,           \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ size_t *indices,                   \
        __gm__ TYPE *cmp_values)                                                                                       \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        aclshmem_##NAME##_wait_until_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_EQ, cmp_values);          \
        aclshmem_##NAME##_wait_until_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_NE, cmp_values);          \
        aclshmem_##NAME##_wait_until_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_GT, cmp_values + nelems); \
        aclshmem_##NAME##_wait_until_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_GE, cmp_values);          \
        aclshmem_##NAME##_wait_until_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_GE, cmp_values + nelems); \
        aclshmem_##NAME##_wait_until_some_vector(                                                                      \
            ivar, nelems, indices, status, ACLSHMEM_CMP_LT, cmp_values + 2 * nelems                                    \
        );                                                                                                             \
        aclshmem_##NAME##_wait_until_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_LE, cmp_values);          \
        aclshmem_##NAME##_wait_until_some_vector(                                                                      \
            ivar, nelems, indices, status, ACLSHMEM_CMP_LE, cmp_values + 2 * nelems                                    \
        );                                                                                                             \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_WAIT_UNTIL_SOME_VECTOR);

#define P2P_CHAIN_DO_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE, CMP_VAL)                                                       \
    void p2p_chain_do_##NAME##_wait_until_some_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,       \
        int rank_size, size_t nelems, const int *status, size_t *indices, TYPE *cmp_values)                            \
    {                                                                                                                  \
        p2p_chain_##NAME##_wait_until_some_vector<<<1, nullptr, stream>>>(                                             \
            config, addr, rank_id, rank_size, nelems, status, indices, cmp_values                                      \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_SOME_VECTOR);

#define P2P_CHAIN_TEST(NAME, TYPE, CMP_VAL)                                                                            \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_test(uint64_t config, GM_ADDR addr,                             \
        int rank_id, int rank_size, __gm__ int *res_addr)                                                              \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        res_addr[0] = aclshmem_##NAME##_test(ivar, ACLSHMEM_CMP_EQ, CMP_VAL);                                          \
        res_addr[1] = aclshmem_##NAME##_test(ivar, ACLSHMEM_CMP_NE, (CMP_VAL) - 1);                                    \
        res_addr[2] = aclshmem_##NAME##_test(ivar, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);                                    \
        res_addr[3] = aclshmem_##NAME##_test(ivar, ACLSHMEM_CMP_GE, CMP_VAL);                                          \
        res_addr[4] = aclshmem_##NAME##_test(ivar, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);                                    \
        res_addr[5] = aclshmem_##NAME##_test(ivar, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);                                    \
        res_addr[6] = aclshmem_##NAME##_test(ivar, ACLSHMEM_CMP_LE, CMP_VAL);                                          \
        res_addr[7] = aclshmem_##NAME##_test(ivar, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);                                    \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_TEST);

#define P2P_CHAIN_DO_TEST(NAME, TYPE, CMP_VAL)                                                                         \
    void p2p_chain_do_##NAME##_test(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size,          \
        int *res_addr)                                                                                                 \
    {                                                                                                                  \
        p2p_chain_##NAME##_test<<<1, nullptr, stream>>>(config, addr, rank_id, rank_size, res_addr);                   \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST);

#define P2P_CHAIN_TEST_ANY(NAME, TYPE, CMP_VAL)                                                                        \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_test_any(uint64_t config, GM_ADDR addr,                         \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ size_t *res_addr)                  \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        res_addr[0] = aclshmem_##NAME##_test_any(ivar, nelems, status, ACLSHMEM_CMP_EQ, CMP_VAL);                      \
        res_addr[1] = aclshmem_##NAME##_test_any(ivar, nelems, status, ACLSHMEM_CMP_NE, CMP_VAL);                      \
        res_addr[2] = aclshmem_##NAME##_test_any(ivar, nelems, status, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);                \
        res_addr[3] = aclshmem_##NAME##_test_any(ivar, nelems, status, ACLSHMEM_CMP_GE, CMP_VAL);                      \
        res_addr[4] = aclshmem_##NAME##_test_any(ivar, nelems, status, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);                \
        res_addr[5] = aclshmem_##NAME##_test_any(ivar, nelems, status, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);                \
        res_addr[6] = aclshmem_##NAME##_test_any(ivar, nelems, status, ACLSHMEM_CMP_LE, CMP_VAL);                      \
        res_addr[7] = aclshmem_##NAME##_test_any(ivar, nelems, status, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);                \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_TEST_ANY);

#define P2P_CHAIN_DO_TEST_ANY(NAME, TYPE, CMP_VAL)                                                                     \
    void p2p_chain_do_##NAME##_test_any(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size,      \
        size_t nelems, const int *status, size_t *res_addr)                                                            \
    {                                                                                                                  \
        p2p_chain_##NAME##_test_any<<<1, nullptr, stream>>>(                                                           \
            config, addr, rank_id, rank_size, nelems, status, res_addr                                                 \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_ANY);

#define P2P_CHAIN_TEST_SOME(NAME, TYPE, CMP_VAL)                                                                       \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_test_some(uint64_t config, GM_ADDR addr,                        \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ size_t *indices,                   \
        __gm__ size_t *res_addr)                                                                                       \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        res_addr[0] = aclshmem_##NAME##_test_some(ivar, nelems, indices, status, ACLSHMEM_CMP_EQ, CMP_VAL);            \
        res_addr[1] = aclshmem_##NAME##_test_some(ivar, nelems, indices, status, ACLSHMEM_CMP_NE, CMP_VAL);            \
        res_addr[2] = aclshmem_##NAME##_test_some(ivar, nelems, indices, status, ACLSHMEM_CMP_GT, (CMP_VAL) - 1);      \
        res_addr[3] = aclshmem_##NAME##_test_some(ivar, nelems, indices, status, ACLSHMEM_CMP_GE, CMP_VAL);            \
        res_addr[4] = aclshmem_##NAME##_test_some(ivar, nelems, indices, status, ACLSHMEM_CMP_GE, (CMP_VAL) - 1);      \
        res_addr[5] = aclshmem_##NAME##_test_some(ivar, nelems, indices, status, ACLSHMEM_CMP_LT, (CMP_VAL) + 1);      \
        res_addr[6] = aclshmem_##NAME##_test_some(ivar, nelems, indices, status, ACLSHMEM_CMP_LE, CMP_VAL);            \
        res_addr[7] = aclshmem_##NAME##_test_some(ivar, nelems, indices, status, ACLSHMEM_CMP_LE, (CMP_VAL) + 1);      \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_TEST_SOME);

#define P2P_CHAIN_DO_TEST_SOME(NAME, TYPE, CMP_VAL)                                                                    \
    void p2p_chain_do_##NAME##_test_some(void *stream, uint64_t config, uint8_t *addr, int rank_id,                    \
        int rank_size, size_t nelems, const int *status, size_t *indices, size_t *res_addr)                            \
    {                                                                                                                  \
        p2p_chain_##NAME##_test_some<<<1, nullptr, stream>>>(                                                          \
            config, addr, rank_id, rank_size, nelems, status, indices, res_addr                                        \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_SOME);

#define P2P_CHAIN_TEST_ALL_VECTOR(NAME, TYPE, CMP_VAL)                                                                 \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_test_all_vector(uint64_t config, GM_ADDR addr,                  \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ TYPE *cmp_values,                  \
        __gm__ int *res_addr)                                                                                          \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        res_addr[0] = aclshmem_##NAME##_test_all_vector(ivar, nelems, status, ACLSHMEM_CMP_EQ, cmp_values);            \
        res_addr[1] = aclshmem_##NAME##_test_all_vector(ivar, nelems, status, ACLSHMEM_CMP_NE, cmp_values + nelems);   \
        res_addr[2] = aclshmem_##NAME##_test_all_vector(ivar, nelems, status, ACLSHMEM_CMP_GT, cmp_values + nelems);   \
        res_addr[3] = aclshmem_##NAME##_test_all_vector(ivar, nelems, status, ACLSHMEM_CMP_GE, cmp_values);            \
        res_addr[4] = aclshmem_##NAME##_test_all_vector(ivar, nelems, status, ACLSHMEM_CMP_GE, cmp_values + nelems);   \
        res_addr[5] = aclshmem_##NAME##_test_all_vector(                                                               \
            ivar, nelems, status, ACLSHMEM_CMP_LT, cmp_values + 2 * nelems                                             \
        );                                                                                                             \
        res_addr[6] = aclshmem_##NAME##_test_all_vector(ivar, nelems, status, ACLSHMEM_CMP_LE, cmp_values);            \
        res_addr[7] = aclshmem_##NAME##_test_all_vector(                                                               \
            ivar, nelems, status, ACLSHMEM_CMP_LE, cmp_values + 2 * nelems                                             \
        );                                                                                                             \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_TEST_ALL_VECTOR);

#define P2P_CHAIN_DO_TEST_ALL_VECTOR(NAME, TYPE, CMP_VAL)                                                              \
    void p2p_chain_do_##NAME##_test_all_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,              \
        int rank_size, size_t nelems, const int *status, TYPE *cmp_values, int *res_addr)                              \
    {                                                                                                                  \
        p2p_chain_##NAME##_test_all_vector<<<1, nullptr, stream>>>(                                                    \
            config, addr, rank_id, rank_size, nelems, status, cmp_values, res_addr                                     \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_ALL_VECTOR);

#define P2P_CHAIN_TEST_ANY_VECTOR(NAME, TYPE, CMP_VAL)                                                                 \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_test_any_vector(uint64_t config, GM_ADDR addr,                  \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ TYPE *cmp_values,                  \
        __gm__ size_t *res_addr)                                                                                       \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        res_addr[0] = aclshmem_##NAME##_test_any_vector(ivar, nelems, status, ACLSHMEM_CMP_EQ, cmp_values);            \
        res_addr[1] = aclshmem_##NAME##_test_any_vector(ivar, nelems, status, ACLSHMEM_CMP_NE, cmp_values);            \
        res_addr[2] = aclshmem_##NAME##_test_any_vector(ivar, nelems, status, ACLSHMEM_CMP_GT, cmp_values + nelems);   \
        res_addr[3] = aclshmem_##NAME##_test_any_vector(ivar, nelems, status, ACLSHMEM_CMP_GE, cmp_values);            \
        res_addr[4] = aclshmem_##NAME##_test_any_vector(ivar, nelems, status, ACLSHMEM_CMP_GE, cmp_values + nelems);   \
        res_addr[5] = aclshmem_##NAME##_test_any_vector(                                                               \
            ivar, nelems, status, ACLSHMEM_CMP_LT, cmp_values + 2 * nelems                                             \
        );                                                                                                             \
        res_addr[6] = aclshmem_##NAME##_test_any_vector(ivar, nelems, status, ACLSHMEM_CMP_LE, cmp_values);            \
        res_addr[7] = aclshmem_##NAME##_test_any_vector(                                                               \
            ivar, nelems, status, ACLSHMEM_CMP_LE, cmp_values + 2 * nelems                                             \
        );                                                                                                             \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_TEST_ANY_VECTOR);

#define P2P_CHAIN_DO_TEST_ANY_VECTOR(NAME, TYPE, CMP_VAL)                                                              \
    void p2p_chain_do_##NAME##_test_any_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,              \
        int rank_size, size_t nelems, const int *status, TYPE *cmp_values, size_t *res_addr)                           \
    {                                                                                                                  \
        p2p_chain_##NAME##_test_any_vector<<<1, nullptr, stream>>>(                                                    \
            config, addr, rank_id, rank_size, nelems, status, cmp_values, res_addr                                    \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_ANY_VECTOR);

#define P2P_CHAIN_TEST_SOME_VECTOR(NAME, TYPE, CMP_VAL)                                                                \
    extern "C" ACLSHMEM_GLOBAL void p2p_chain_##NAME##_test_some_vector(uint64_t config, GM_ADDR addr,                 \
        int rank_id, int rank_size, size_t nelems, __gm__ const int *status, __gm__ size_t *indices,                   \
        __gm__ TYPE *cmp_values, __gm__ size_t *res_addr)                                                              \
    {                                                                                                                  \
        util_set_ffts_config(config);                                                                                  \
        int next = (rank_id + 1) % rank_size;                                                                          \
        auto *remote_addr = aclshmem_ptr(addr, next);                                                                  \
        auto ivar = (__gm__ TYPE *)addr;                                                                               \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
                                                                                                                       \
        res_addr[0] = aclshmem_##NAME##_test_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_EQ, cmp_values);  \
        res_addr[1] = aclshmem_##NAME##_test_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_NE, cmp_values);  \
        res_addr[2] = aclshmem_##NAME##_test_some_vector(                                                              \
            ivar, nelems, indices, status, ACLSHMEM_CMP_GT, cmp_values + nelems                                        \
        );                                                                                                             \
        res_addr[3] = aclshmem_##NAME##_test_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_GE, cmp_values);  \
        res_addr[4] = aclshmem_##NAME##_test_some_vector(                                                              \
            ivar, nelems, indices, status, ACLSHMEM_CMP_GE, cmp_values + nelems                                        \
        );                                                                                                             \
        res_addr[5] = aclshmem_##NAME##_test_some_vector(                                                              \
            ivar, nelems, indices, status, ACLSHMEM_CMP_LT, cmp_values + 2 * nelems                                    \
        );                                                                                                             \
        res_addr[6] = aclshmem_##NAME##_test_some_vector(ivar, nelems, indices, status, ACLSHMEM_CMP_LE, cmp_values);  \
        res_addr[7] = aclshmem_##NAME##_test_some_vector(                                                              \
            ivar, nelems, indices, status, ACLSHMEM_CMP_LE, cmp_values + 2 * nelems                                    \
        );                                                                                                             \
                                                                                                                       \
        aclshmem_barrier_all();                                                                                        \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_TEST_SOME_VECTOR);

#define P2P_CHAIN_DO_TEST_SOME_VECTOR(NAME, TYPE, CMP_VAL)                                                             \
    void p2p_chain_do_##NAME##_test_some_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,             \
        int rank_size, size_t nelems, const int *status, size_t *indices, TYPE *cmp_values, size_t *res_addr)          \
    {                                                                                                                  \
        p2p_chain_##NAME##_test_some_vector<<<1, nullptr, stream>>>(                                                   \
            config, addr, rank_id, rank_size, nelems, status, indices, cmp_values, res_addr                           \
        );                                                                                                             \
    }
ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_SOME_VECTOR);