/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UNITTEST_H
#define UNITTEST_H

extern int test_global_ranks;
extern int test_gnpu_num;
extern char test_global_ipport[ACLSHMEM_MAX_IP_PORT_LEN];
extern int test_first_rank;
extern int test_first_npu;

int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port,
                       aclshmemx_init_attr_t *attributes);
void test_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
int32_t test_rdma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
int32_t test_udma_init(int rank_id, int n_ranks, uint64_t local_mem_size, aclrtStream *st);
void test_cross_init(int pe_id, int n_pes, uint64_t local_mem_size, aclrtStream *st);
void test_finalize(aclrtStream stream, int device_id);
void test_mutil_task(std::function<void(int, int, uint64_t)> func, uint64_t local_mem_size, int process_count);

#endif // UNITTEST_H