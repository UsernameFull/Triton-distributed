/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DISPATCH_KERNEL_H
#define DISPATCH_KERNEL_H

#include <cstdint>

// `expert_per_pe` is capped because the kernel keeps fixed local-expert
// bookkeeping arrays on the AI core stack.
constexpr int32_t DISPATCH_MAX_LOCAL_EXPERT_NUM = 1024;

template <class T>
void dispatch_demo(uint32_t block_dim, void *stream, uint64_t fftsAddr, uint8_t *x, int32_t *expert_ids,
                   uint8_t *expand_x, int32_t *assist_info_for_combine, int32_t *ep_recv_count,
                   int32_t *expert_token_nums, uint8_t *shmem_window, int bs, int h, int k, int moe_expert_num,
                   int magic, int perf_mode = 0, int full_frame_id = 0, int comm_frame_id = 1,
                   int warmup_count = 0, int loop_count = 1);

#endif // DISPATCH_KERNEL_H
