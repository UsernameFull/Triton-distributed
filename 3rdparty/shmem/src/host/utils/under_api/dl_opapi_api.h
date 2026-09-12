/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBM_CORE_DL_OPAPI_API_H
#define MF_HYBM_CORE_DL_OPAPI_API_H

#include <mutex>
#include "aclnn/aclnn_base.h"
#include "shmemi_functions.h"
#include "host/shmem_host_def.h"

namespace shm {
using aclnnShmemSdmaStarsQueryGetWorkspaceSizeFunc =
    aclnnStatus (*)(const aclTensor *, aclTensor *, uint64_t *, aclOpExecutor **);
using aclnnShmemSdmaStarsQueryFunc = aclnnStatus (*)(void *, uint64_t, aclOpExecutor *, aclrtStream);

class DlOpapiApi {
public:
    static Result LoadLibrary();
    static void CleanupLibrary();

    static inline Result AclnnShmemSdmaStarsQueryGetWorkspaceSize(const aclTensor *input, aclTensor *out,
                                                                  uint64_t *workspaceSize, aclOpExecutor **executor)
    {
        if (pAclnnShmemSdmaStarsQueryGetWorkspaceSize == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclnnShmemSdmaStarsQueryGetWorkspaceSize(input, out, workspaceSize, executor);
    }

    static inline Result AclnnShmemSdmaStarsQuery(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                                  aclrtStream stream)
    {
        if (pAclnnShmemSdmaStarsQuery == nullptr) {
            return ACLSHMEM_UNDER_API_UNLOAD;
        }
        return pAclnnShmemSdmaStarsQuery(workspace, workspaceSize, executor, stream);
    }

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void *opapiHandle;
    static const char *gOpapiLibName;

    static aclnnShmemSdmaStarsQueryGetWorkspaceSizeFunc pAclnnShmemSdmaStarsQueryGetWorkspaceSize;
    static aclnnShmemSdmaStarsQueryFunc pAclnnShmemSdmaStarsQuery;
};
} // namespace shm

#endif  // MF_HYBM_CORE_DL_OPAPI_API_H
