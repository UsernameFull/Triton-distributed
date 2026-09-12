/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBM_CORE_DEVICE_CHIP_INFO_H
#define MF_HYBM_CORE_DEVICE_CHIP_INFO_H

#include <cstdint>

namespace shm {
namespace transport {
namespace device {
class DeviceChipInfo {
public:
    explicit DeviceChipInfo(uint32_t devId) noexcept: deviceId_{devId} {}
    int Init() noexcept;

public:
    inline uint32_t GetChipId() const noexcept
    {
        return chipId_;
    }

    inline uint32_t GetDieId() const noexcept
    {
        return dieId_;
    }

    inline uint64_t GetChipAddr() const noexcept
    {
        return chipBaseAddr_;
    }

    inline uint64_t GetChipOffset() const noexcept
    {
        return chipOffset_;
    }

    inline uint64_t GetDieOffset() const noexcept
    {
        return chipDieOffset_;
    }

private:
    const uint32_t deviceId_;
    uint32_t chipId_{0};
    uint32_t dieId_{0};
    bool addrFlatDevice_{false};
    uint64_t chipBaseAddr_{0};
    uint64_t chipOffset_{0};
    uint64_t chipDieOffset_{0};
};
}
}
}

#endif  // MF_HYBM_CORE_DEVICE_CHIP_INFO_H
