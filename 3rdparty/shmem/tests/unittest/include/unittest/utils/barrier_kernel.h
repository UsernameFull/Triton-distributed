/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef BARRIER_KERNEL_H
#define BARRIER_KERNEL_H

void increase_do(void* stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size);
void increase_vec_do(void* stream, uint64_t config, uint8_t *addr, int rank_id, int rank_size);
void increase_do_odd_team(void* stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, aclshmem_team_t team_id);
void increase_vec_do_odd_team(void* stream, uint64_t config, uint8_t *addr, int rank_id,
    int rank_size, aclshmem_team_t team_id);

#endif // BARRIER_KERNEL_H