/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "dl_acl_api.h"
#include "dl_comm_def.h"
#include "shmemi_logger.h"
#include "device_chip_info.h"

namespace shm {
namespace transport {
namespace device {
int DeviceChipInfo::Init() noexcept
{
    int64_t infoValue = 0U;
    auto ret = DlAclApi::RtGetDeviceInfo(deviceId_, 0, INFO_TYPE_PHY_CHIP_ID, &infoValue);
    if (ret != 0) {
        SHM_LOG_ERROR("RtGetDeviceInfo(INFO_TYPE_PHY_CHIP_ID) failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    chipId_ = static_cast<uint32_t>(infoValue);

    ret = DlAclApi::RtGetDeviceInfo(deviceId_, 0, INFO_TYPE_PHY_DIE_ID, &infoValue);
    if (ret != 0) {
        SHM_LOG_ERROR("RtGetDeviceInfo(INFO_TYPE_PHY_DIE_ID) failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    dieId_ = static_cast<uint32_t>(infoValue);

    chipDieOffset_ = 0x10000000000UL; // RT_ASCEND910B1_DIE_ADDR_OFFSET;
    chipOffset_ = 0x80000000000UL; // RT_ASCEND910B1_CHIP_ADDR_OFFSET;
    chipBaseAddr_ = 0UL;
    ret = DlAclApi::RtGetDeviceInfo(deviceId_, 0, INFO_TYPE_ADDR_MODE, &infoValue);
    if (ret != 0) {
        SHM_LOG_ERROR("RtGetDeviceInfo(INFO_TYPE_ADDR_MODE) failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    addrFlatDevice_ = (infoValue != 0L);
    if (addrFlatDevice_) {
        chipBaseAddr_ = 0x200000000000ULL; // RT_CHIP_BASE_ADDR;
        chipOffset_ = 0x20000000000UL; // RT_ASCEND910B1_HCCS_CHIP_ADDR_OFFSET;
    }

    return 0;
}
}
}

}