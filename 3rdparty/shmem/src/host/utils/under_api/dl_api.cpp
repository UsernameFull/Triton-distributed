/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dl_api.h"
#include "dl_acl_api.h"
#include "dl_hal_api.h"
#include "dl_hccp_api.h"
#include "dl_hccp_v2_api.h"
#include "dl_rt_api.h"
#include "dl_opapi_api.h"
#include "utils/shmemi_logger.h"

namespace shm {

Result DlApi::LoadLibrary(const std::string &libDirPath)
{
    auto result = DlAclApi::LoadLibrary(libDirPath);
    if (result != ACLSHMEM_SUCCESS) {
        return result;
    }
    AscendSocType socType = GetAscendSocType();
    result = DlHalApi::LoadLibrary(socType);
    if (result != ACLSHMEM_SUCCESS) {
        DlAclApi::CleanupLibrary();
        return result;
    }

    return ACLSHMEM_SUCCESS;
}

void DlApi::CleanupLibrary()
{
    DlHccpApi::CleanupLibrary();
    DlAclApi::CleanupLibrary();
    DlHalApi::CleanupLibrary();
}

Result DlApi::LoadExtendLibrary(DlApiExtendLibraryType libraryType)
{
    if (libraryType == DL_EXT_LIB_DEVICE_RDMA) {
        return DlHccpApi::LoadLibrary();
    }
    if (libraryType == DL_EXT_LIB_DEVICE_SDMA) {
        auto result = DlRtApi::LoadLibrary();
        if (result != ACLSHMEM_SUCCESS) {
            return result;
        }
        result = DlOpapiApi::LoadLibrary();
        if (result != ACLSHMEM_SUCCESS) {
            DlRtApi::CleanupLibrary();
            return result;
        }
    }
    if (libraryType == DL_EXT_LIB_DEVICE_UDMA) {
        return DlHccpV2Api::LoadLibrary();
    }

    return ACLSHMEM_SUCCESS;
}

AscendSocType DlApi::GetAscendSocType()
{
    static AscendSocType cachedSocType = [&]() -> AscendSocType {
        auto name = DlAclApi::AclrtGetSocName();
        if (name == nullptr) {
            SHM_LOG_ERROR("AclrtGetSocName() failed.");
            return ASCEND_UNKNOWN;
        }
        SHM_LOG_DEBUG("success get soc name: " << name);
        std::string socName{name};
        if (socName.find("Ascend910B") != std::string::npos) {
            return AscendSocType::ASCEND_910B;
        } else if (socName.find("Ascend910_93") != std::string::npos) {
            return AscendSocType::ASCEND_910C;
        } else if (socName.find("Ascend950") != std::string::npos) {
            return AscendSocType::ASCEND_950;
        }

        return ASCEND_UNKNOWN;
    }();

    return cachedSocType;
}
}