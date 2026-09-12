/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dl_opapi_api.h"

#include <dlfcn.h>
#include "shmemi_file_util.h"

namespace shm {
bool DlOpapiApi::gLoaded = false;
std::mutex DlOpapiApi::gMutex;
void *DlOpapiApi::opapiHandle;
const char *DlOpapiApi::gOpapiLibName = "libopapi.so";

aclnnShmemSdmaStarsQueryGetWorkspaceSizeFunc DlOpapiApi::pAclnnShmemSdmaStarsQueryGetWorkspaceSize = nullptr;
aclnnShmemSdmaStarsQueryFunc DlOpapiApi::pAclnnShmemSdmaStarsQuery = nullptr;

Result DlOpapiApi::LoadLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return ACLSHMEM_SUCCESS;
    }

    /* dlopen library */
    opapiHandle = dlopen(gOpapiLibName, RTLD_NOW);
    if (opapiHandle == nullptr) {
        SHM_LOG_ERROR("Failed to open library error: " << dlerror());
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    /* load sym */
    DL_LOAD_SYM(pAclnnShmemSdmaStarsQueryGetWorkspaceSize, aclnnShmemSdmaStarsQueryGetWorkspaceSizeFunc, opapiHandle,
        "aclnnShmemSdmaStarsQueryGetWorkspaceSize");
    DL_LOAD_SYM(pAclnnShmemSdmaStarsQuery, aclnnShmemSdmaStarsQueryFunc, opapiHandle, "aclnnShmemSdmaStarsQuery");

    gLoaded = true;
    return ACLSHMEM_SUCCESS;
}

void DlOpapiApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    pAclnnShmemSdmaStarsQueryGetWorkspaceSize = nullptr;
    pAclnnShmemSdmaStarsQuery = nullptr;

    if (opapiHandle != nullptr) {
        dlclose(opapiHandle);
        opapiHandle = nullptr;
    }

    gLoaded = false;
}
} // namespace shm
