/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBM_CORE_DL_RT_API_H
#define MF_HYBM_CORE_DL_RT_API_H

#include <mutex>
#include "dl_rt_def.h"
#include "shmemi_functions.h"
#include "host/shmem_host_def.h"

namespace shm {
using rtStreamGetSqidFunc = int32_t (*)(const void *, uint32_t *);
using rtStreamGetCqidFunc = int32_t (*)(const void *, uint32_t *, uint32_t *);

class DlRtApi {
public:
    static Result LoadLibrary();
    static void CleanupLibrary();

    static inline Result RtStreamGetSqid(const void *stm, uint32_t *sqId)
    {
        if (pRtStreamGetSqid == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtStreamGetSqid(stm, sqId);
    }

    static inline Result RtStreamGetCqid(const void *stm, uint32_t *cqId, uint32_t *logicCqId)
    {
        if (pRtStreamGetCqid == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtStreamGetCqid(stm, cqId, logicCqId);
    }

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void *rtHandle;
    static const char *gRtLibName;

    static rtStreamGetSqidFunc pRtStreamGetSqid;
    static rtStreamGetCqidFunc pRtStreamGetCqid;
};
} // namespace shm

#endif  // MF_HYBM_CORE_DL_RT_API_H
