/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <dlfcn.h>
#include "dl_acl_api.h"
#include "shmemi_file_util.h"

namespace shm {
bool DlAclApi::gLoaded = false;
std::mutex DlAclApi::gMutex;
void *DlAclApi::rtHandle;
const char *DlAclApi::gAscendAclLibName = "libascendcl.so";

aclrtGetDeviceFunc DlAclApi::pAclrtGetDevice = nullptr;
aclrtSetDeviceFunc DlAclApi::pAclrtSetDevice = nullptr;
aclrtDeviceEnablePeerAccessFunc DlAclApi::pAclrtDeviceEnablePeerAccess = nullptr;
aclrtCreateStreamFunc DlAclApi::pAclrtCreateStream = nullptr;
aclrtDestroyStreamFunc DlAclApi::pAclrtDestroyStream = nullptr;
aclrtSynchronizeStreamFunc DlAclApi::pAclrtSynchronizeStream = nullptr;
aclrtMallocFunc DlAclApi::pAclrtMalloc = nullptr;
aclrtFreeFunc DlAclApi::pAclrtFree = nullptr;
aclrtMemcpyFunc DlAclApi::pAclrtMemcpy = nullptr;
aclrtMemcpyAsyncFunc DlAclApi::pAclrtMemcpyAsync = nullptr;
aclrtMemsetFunc DlAclApi::pAclrtMemset = nullptr;
rtDeviceGetBareTgidFunc DlAclApi::pRtDeviceGetBareTgid = nullptr;
rtGetDeviceInfoFunc DlAclApi::pRtGetDeviceInfo = nullptr;
rtSetIpcMemorySuperPodPidFunc DlAclApi::pRtSetIpcMemorySuperPodPid = nullptr;
rtIpcDestroyMemoryNameFunc DlAclApi::pRtIpcDestroyMemoryName = nullptr;
rtIpcSetMemoryNameFunc DlAclApi::pRtIpcSetMemoryName = nullptr;
rtIpcOpenMemoryFunc DlAclApi::pRtIpcOpenMemory = nullptr;
rtIpcCloseMemoryFunc DlAclApi::pRtIpcCloseMemory = nullptr;
aclrtGetSocNameFunc DlAclApi::pAclrtGetSocName = nullptr;
rtGetLogicDevIdByUserDevIdFunc DlAclApi::pRtGetLogicDevIdByUserDevId = nullptr;
aclrtGetPhyDevIdByLogicDevIdFunc DlAclApi::pAclrtGetPhyDevIdByLogicDevId = nullptr;

Result DlAclApi::LoadLibrary(const std::string &libDirPath)
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return ACLSHMEM_SUCCESS;
    }

    std::string realPath;
    if (!shm::utils::FileUtil::LibraryRealPath(libDirPath, std::string(gAscendAclLibName), realPath)) {
        SHM_LOG_ERROR(libDirPath << " get lib path failed.");
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    /* dlopen library */
    rtHandle = dlopen(realPath.c_str(), RTLD_NOW);
    if (rtHandle == nullptr) {
        SHM_LOG_ERROR("Failed to open library error: " << dlerror());
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    /* load sym */
    DL_LOAD_SYM(pAclrtGetDevice, aclrtGetDeviceFunc, rtHandle, "aclrtGetDevice");
    DL_LOAD_SYM(pAclrtSetDevice, aclrtSetDeviceFunc, rtHandle, "aclrtSetDevice");
    DL_LOAD_SYM(pAclrtDeviceEnablePeerAccess, aclrtDeviceEnablePeerAccessFunc, rtHandle, "aclrtDeviceEnablePeerAccess");
    DL_LOAD_SYM(pAclrtCreateStream, aclrtCreateStreamFunc, rtHandle, "aclrtCreateStream");
    DL_LOAD_SYM(pAclrtDestroyStream, aclrtDestroyStreamFunc, rtHandle, "aclrtDestroyStream");
    DL_LOAD_SYM(pAclrtSynchronizeStream, aclrtSynchronizeStreamFunc, rtHandle, "aclrtSynchronizeStream");
    DL_LOAD_SYM(pAclrtMalloc, aclrtMallocFunc, rtHandle, "aclrtMalloc");
    DL_LOAD_SYM(pAclrtFree, aclrtFreeFunc, rtHandle, "aclrtFree");
    DL_LOAD_SYM(pAclrtMemcpy, aclrtMemcpyFunc, rtHandle, "aclrtMemcpy");
    DL_LOAD_SYM(pAclrtMemcpyAsync, aclrtMemcpyAsyncFunc, rtHandle, "aclrtMemcpyAsync");
    DL_LOAD_SYM(pAclrtMemset, aclrtMemsetFunc, rtHandle, "aclrtMemset");
    DL_LOAD_SYM(pRtDeviceGetBareTgid, rtDeviceGetBareTgidFunc, rtHandle, "rtDeviceGetBareTgid");
    DL_LOAD_SYM(pRtGetDeviceInfo, rtGetDeviceInfoFunc, rtHandle, "rtGetDeviceInfo");
    DL_LOAD_SYM(pRtSetIpcMemorySuperPodPid, rtSetIpcMemorySuperPodPidFunc, rtHandle, "rtSetIpcMemorySuperPodPid");
    DL_LOAD_SYM(pRtIpcSetMemoryName, rtIpcSetMemoryNameFunc, rtHandle, "rtIpcSetMemoryName");
    DL_LOAD_SYM(pRtIpcDestroyMemoryName, rtIpcDestroyMemoryNameFunc, rtHandle, "rtIpcDestroyMemoryName");
    DL_LOAD_SYM(pRtIpcOpenMemory, rtIpcOpenMemoryFunc, rtHandle, "rtIpcOpenMemory");
    DL_LOAD_SYM(pRtIpcCloseMemory, rtIpcCloseMemoryFunc, rtHandle, "rtIpcCloseMemory");
    DL_LOAD_SYM(pAclrtGetSocName, aclrtGetSocNameFunc, rtHandle, "aclrtGetSocName");
    DL_LOAD_SYM(pRtGetLogicDevIdByUserDevId, rtGetLogicDevIdByUserDevIdFunc, rtHandle, "rtGetLogicDevIdByUserDevId");
    DL_LOAD_SYM(
        pAclrtGetPhyDevIdByLogicDevId, aclrtGetPhyDevIdByLogicDevIdFunc, rtHandle,
        "aclrtGetPhyDevIdByLogicDevId");

    gLoaded = true;
    SHM_LOG_INFO("Load " << realPath << " success.");
    return ACLSHMEM_SUCCESS;
}

void DlAclApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    pAclrtGetDevice = nullptr;
    pAclrtSetDevice = nullptr;
    pAclrtDeviceEnablePeerAccess = nullptr;
    pAclrtCreateStream = nullptr;
    pAclrtDestroyStream = nullptr;
    pAclrtSynchronizeStream = nullptr;
    pAclrtMalloc = nullptr;
    pAclrtFree = nullptr;
    pAclrtMemcpy = nullptr;
    pAclrtMemcpyAsync = nullptr;
    pAclrtMemset = nullptr;
    pRtDeviceGetBareTgid = nullptr;
    pRtGetDeviceInfo = nullptr;
    pRtSetIpcMemorySuperPodPid = nullptr;
    pRtIpcDestroyMemoryName = nullptr;
    pRtIpcSetMemoryName = nullptr;
    pAclrtGetSocName = nullptr;
    pRtGetLogicDevIdByUserDevId = nullptr;
    pAclrtGetPhyDevIdByLogicDevId = nullptr;

    if (rtHandle != nullptr) {
        dlclose(rtHandle);
        rtHandle = nullptr;
    }

    gLoaded = false;
}
}
