/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef P2P_KERNEL_H
#define P2P_KERNEL_H

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

void p2p_chain_do(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size);

#define P2P_CHAIN_DO_WAIT_UNTIL(NAME, TYPE, CMP_VAL)                                                                   \
    void p2p_chain_do_##NAME##_wait_until(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size)

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL);

#define P2P_CHAIN_DO_WAIT_UNTIL_ALL(NAME, TYPE, CMP_VAL)                                                               \
    void p2p_chain_do_##NAME##_wait_until_all(void *stream, uint64_t config, uint8_t *addr, int rank_id,               \
        int rank_size, size_t nelems, const int *status)                                                               \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_ALL);

#define P2P_CHAIN_DO_WAIT_UNTIL_ANY(NAME, TYPE, CMP_VAL)                                                               \
    void p2p_chain_do_##NAME##_wait_until_any(void *stream, uint64_t config, uint8_t *addr, int rank_id,               \
        int rank_size, size_t nelems, const int *status)                                                               \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_ANY);

#define P2P_CHAIN_DO_WAIT_UNTIL_SOME(NAME, TYPE, CMP_VAL)                                                              \
    void p2p_chain_do_##NAME##_wait_until_some(void *stream, uint64_t config, uint8_t *addr, int rank_id,              \
        int rank_size, size_t nelems, const int *status, size_t *indices)                                              \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_SOME);

#define P2P_CHAIN_DO_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE, CMP_VAL)                                                        \
    void p2p_chain_do_##NAME##_wait_until_all_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,        \
        int rank_size, size_t nelems, const int *status, TYPE *cmp_values)                                             \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_ALL_VECTOR);

#define P2P_CHAIN_DO_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE, CMP_VAL)                                                        \
    void p2p_chain_do_##NAME##_wait_until_any_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,        \
        int rank_size, size_t nelems, const int *status, TYPE *cmp_values)                                             \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_ANY_VECTOR);

#define P2P_CHAIN_DO_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE, CMP_VAL)                                                       \
    void p2p_chain_do_##NAME##_wait_until_some_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,       \
        int rank_size, size_t nelems, const int *status, size_t *indices, TYPE *cmp_values)                            \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_WAIT_UNTIL_SOME_VECTOR);

#define P2P_CHAIN_DO_TEST(NAME, TYPE, CMP_VAL)                                                                         \
    void p2p_chain_do_##NAME##_test(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size,          \
        int *res_addr)                                                                                                 \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST);

#define P2P_CHAIN_DO_TEST_ANY(NAME, TYPE, CMP_VAL)                                                                     \
    void p2p_chain_do_##NAME##_test_any(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size,      \
        size_t nelems, const int *status, size_t *res_addr)                                                            \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_ANY);

#define P2P_CHAIN_DO_TEST_SOME(NAME, TYPE, CMP_VAL)                                                                    \
    void p2p_chain_do_##NAME##_test_some(void *stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size,     \
        size_t nelems, const int *status, size_t *indices, size_t *res_addr)                                           \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_SOME);

#define P2P_CHAIN_DO_TEST_ALL_VECTOR(NAME, TYPE, CMP_VAL)                                                              \
    void p2p_chain_do_##NAME##_test_all_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,              \
        int rank_size, size_t nelems, const int *status, TYPE *cmp_values, int *res_addr)                              \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_ALL_VECTOR);

#define P2P_CHAIN_DO_TEST_ANY_VECTOR(NAME, TYPE, CMP_VAL)                                                              \
    void p2p_chain_do_##NAME##_test_any_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,              \
        int rank_size, size_t nelems, const int *status, TYPE *cmp_values, size_t *res_addr)                           \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_ANY_VECTOR);

#define P2P_CHAIN_DO_TEST_SOME_VECTOR(NAME, TYPE, CMP_VAL)                                                             \
    void p2p_chain_do_##NAME##_test_some_vector(void *stream, uint64_t config, uint8_t *addr, int rank_id,             \
        int rank_size, size_t nelems, const int *status, size_t *indices, TYPE *cmp_values, size_t *res_addr)          \

ACLSHMEM_P2P_SYNC_TYPE_VAL_FUNC(P2P_CHAIN_DO_TEST_SOME_VECTOR);


#endif