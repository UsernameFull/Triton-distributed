/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBM_CORE_DL_ACL_API_H
#define MF_HYBM_CORE_DL_ACL_API_H

#include <string>
#include <mutex>
#include "shmemi_functions.h"
#include "host/shmem_host_def.h"

namespace shm {

using aclrtSetDeviceFunc = int32_t (*)(int32_t);
using aclrtGetDeviceFunc = int32_t (*)(int32_t *);
using aclrtDeviceEnablePeerAccessFunc = int32_t (*)(int32_t, uint32_t);
using aclrtCreateStreamFunc = int (*)(void **);
using aclrtDestroyStreamFunc = int (*)(void *);
using aclrtSynchronizeStreamFunc = int (*)(void *);
using aclrtMallocFunc = int32_t (*)(void **, size_t, uint32_t);
using aclrtFreeFunc = int (*)(void *);
using aclrtMemcpyFunc = int32_t (*)(void *, size_t, const void *, size_t, uint32_t);
using aclrtMemcpyAsyncFunc = int32_t (*)(void *, size_t, const void *, size_t, uint32_t, void *);
using aclrtMemsetFunc = int32_t (*)(void *, size_t, int32_t, size_t);
using rtDeviceGetBareTgidFunc = int32_t (*)(uint32_t *);
using rtGetDeviceInfoFunc = int32_t (*)(uint32_t, int32_t, int32_t, int64_t *val);
using rtIpcSetMemoryNameFunc = int32_t (*)(const void *, uint64_t, char *, uint32_t);
using rtSetIpcMemorySuperPodPidFunc = int32_t (*)(const char *, uint32_t, int32_t *, int32_t);
using rtIpcDestroyMemoryNameFunc = int32_t (*)(const char *);
using rtIpcOpenMemoryFunc = int32_t (*)(void **, const char *);
using rtIpcCloseMemoryFunc = int32_t (*)(const void *);
using aclrtGetSocNameFunc = const char *(*)();
using rtGetLogicDevIdByUserDevIdFunc = int32_t (*)(const int32_t, int32_t *const);
using aclrtGetPhyDevIdByLogicDevIdFunc = int32_t (*)(const int32_t, int32_t *const);

class DlAclApi {
public:
    static Result LoadLibrary(const std::string &libDirPath);
    static void CleanupLibrary();

    static inline Result AclrtSetDevice(int32_t deviceId)
    {
        if (pAclrtSetDevice == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtSetDevice(deviceId);
    }

    static inline Result AclrtGetDevice(int32_t *deviceId)
    {
        if (pAclrtGetDevice == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtGetDevice(deviceId);
    }

    static inline Result AclrtDeviceEnablePeerAccess(int32_t peerDeviceId, uint32_t flags)
    {
        if (pAclrtDeviceEnablePeerAccess == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtDeviceEnablePeerAccess(peerDeviceId, flags);
    }

    static inline Result AclrtCreateStream(void **stream)
    {
        if (pAclrtCreateStream == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtCreateStream(stream);
    }

    static inline Result AclrtDestroyStream(void *stream)
    {
        if (pAclrtDestroyStream == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtDestroyStream(stream);
    }

    static inline Result AclrtSynchronizeStream(void *stream)
    {
        if (pAclrtSynchronizeStream == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtSynchronizeStream(stream);
    }

    static inline Result AclrtMalloc(void **ptr, size_t count, uint32_t type)
    {
        if (pAclrtMalloc == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtMalloc(ptr, count, type);
    }

    static inline Result AclrtFree(void *ptr)
    {
        if (pAclrtFree == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        auto ret = pAclrtFree(ptr);
        return ret;
    }

    static inline Result AclrtMemcpy(void *dst, size_t destMax, const void *src, size_t count, uint32_t kind)
    {
        if (pAclrtMemcpy == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtMemcpy(dst, destMax, src, count, kind);
    }

    static inline Result AclrtMemcpyAsync(void *dst, size_t destMax, const void *src, size_t count, uint32_t kind,
                                          void *stream)
    {
        if (pAclrtMemcpyAsync == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtMemcpyAsync(dst, destMax, src, count, kind, stream);
    }

    static inline Result AclrtMemset(void *dst, size_t destMax, int32_t value, size_t count)
    {
        if (pAclrtMemset == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtMemset(dst, destMax, value, count);
    }

    static inline Result RtDeviceGetBareTgid(uint32_t *pid)
    {
        if (pRtDeviceGetBareTgid == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtDeviceGetBareTgid(pid);
    }

    static inline Result RtGetDeviceInfo(uint32_t deviceId, int32_t moduleType, int32_t infoType, int64_t *val)
    {
        if (pRtGetDeviceInfo == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtGetDeviceInfo(deviceId, moduleType, infoType, val);
    }

    static inline Result RtSetIpcMemorySuperPodPid(const char *name, uint32_t sdid, int32_t pid[], int32_t num)
    {
        if (pRtSetIpcMemorySuperPodPid == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtSetIpcMemorySuperPodPid(name, sdid, pid, num);
    }

    static inline Result RtIpcSetMemoryName(const void *ptr, uint64_t byteCount, char *name, uint32_t len)
    {
        if (pRtIpcSetMemoryName == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtIpcSetMemoryName(ptr, byteCount, name, len);
    }

    static inline Result RtIpcDestroyMemoryName(const char *name)
    {
        if (pRtIpcDestroyMemoryName == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtIpcDestroyMemoryName(name);
    }

    static inline Result RtIpcOpenMemory(void **ptr, const char *name)
    {
        if (pRtIpcOpenMemory == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtIpcOpenMemory(ptr, name);
    }

    static inline Result RtIpcCloseMemory(const void *ptr)
    {
        if (pRtIpcCloseMemory == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pRtIpcCloseMemory(ptr);
    }

    static inline const char *AclrtGetSocName()
    {
        return pAclrtGetSocName();
    }

    static inline Result RtGetLogicDevIdByUserDevId(const int32_t userDevId, int32_t * const logicDevId)
    {
        return pRtGetLogicDevIdByUserDevId(userDevId, logicDevId);
    }

    static inline Result AclrtGetPhyDevIdByLogicDevId(const int32_t logicDevId, int32_t * const phyDevId)
    {
        if (pAclrtGetPhyDevIdByLogicDevId == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclrtGetPhyDevIdByLogicDevId(logicDevId, phyDevId);
    }

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void *rtHandle;
    static const char *gAscendAclLibName;

    static aclrtSetDeviceFunc pAclrtSetDevice;
    static aclrtGetDeviceFunc pAclrtGetDevice;
    static aclrtDeviceEnablePeerAccessFunc pAclrtDeviceEnablePeerAccess;
    static aclrtCreateStreamFunc pAclrtCreateStream;
    static aclrtDestroyStreamFunc pAclrtDestroyStream;
    static aclrtSynchronizeStreamFunc pAclrtSynchronizeStream;
    static aclrtMallocFunc pAclrtMalloc;
    static aclrtFreeFunc pAclrtFree;
    static aclrtMemcpyFunc pAclrtMemcpy;
    static aclrtMemcpyAsyncFunc pAclrtMemcpyAsync;
    static aclrtMemsetFunc pAclrtMemset;
    static rtDeviceGetBareTgidFunc pRtDeviceGetBareTgid;
    static rtGetDeviceInfoFunc pRtGetDeviceInfo;
    static rtSetIpcMemorySuperPodPidFunc pRtSetIpcMemorySuperPodPid;
    static rtIpcSetMemoryNameFunc pRtIpcSetMemoryName;
    static rtIpcDestroyMemoryNameFunc pRtIpcDestroyMemoryName;
    static rtIpcOpenMemoryFunc pRtIpcOpenMemory;
    static rtIpcCloseMemoryFunc pRtIpcCloseMemory;
    static aclrtGetSocNameFunc pAclrtGetSocName;
    static rtGetLogicDevIdByUserDevIdFunc pRtGetLogicDevIdByUserDevId;
    static aclrtGetPhyDevIdByLogicDevIdFunc pAclrtGetPhyDevIdByLogicDevId;
};
}

#endif  // MF_HYBM_CORE_DL_ACL_API_H
