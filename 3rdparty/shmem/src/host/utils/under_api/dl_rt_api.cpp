/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dl_rt_api.h"

#include <dlfcn.h>
#include "shmemi_file_util.h"

namespace shm {
bool DlRtApi::gLoaded = false;
std::mutex DlRtApi::gMutex;
void *DlRtApi::rtHandle;
const char *DlRtApi::gRtLibName = "libruntime.so";

rtStreamGetSqidFunc DlRtApi::pRtStreamGetSqid = nullptr;
rtStreamGetCqidFunc DlRtApi::pRtStreamGetCqid = nullptr;

Result DlRtApi::LoadLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return ACLSHMEM_SUCCESS;
    }

    /* dlopen library */
    rtHandle = dlopen(gRtLibName, RTLD_NOW);
    if (rtHandle == nullptr) {
        SHM_LOG_ERROR("Failed to open library error: " << dlerror());
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    /* load sym */
    DL_LOAD_SYM(pRtStreamGetSqid, rtStreamGetSqidFunc, rtHandle, "rtStreamGetSqid");
    DL_LOAD_SYM(pRtStreamGetCqid, rtStreamGetCqidFunc, rtHandle, "rtStreamGetCqid");

    gLoaded = true;
    return ACLSHMEM_SUCCESS;
}

void DlRtApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    pRtStreamGetSqid = nullptr;
    pRtStreamGetCqid = nullptr;

    if (rtHandle != nullptr) {
        dlclose(rtHandle);
        rtHandle = nullptr;
    }

    gLoaded = false;
}
} // namespace shm
