/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_HOST_DEVICE_CONSTANT_H
#define SHMEMI_HOST_DEVICE_CONSTANT_H

namespace shm {

// Qp constant
constexpr uint32_t UDMA_CQ_DEPTH_DEFAULT = 16384;
constexpr uint32_t UDMA_SQ_DEPTH_DEFAULT = 4096;
constexpr uint32_t UDMA_RQ_DEPTH_DEFAULT = 256;
constexpr uint32_t UDMA_MAX_SQE_BB_NUM = 4;
constexpr uint32_t UDMA_SQ_BASKBLK_CNT = UDMA_SQ_DEPTH_DEFAULT * UDMA_MAX_SQE_BB_NUM;

} // namespace shm

#endif