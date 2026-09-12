/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_QP_MANAGER_H
#define MF_HYBRID_DEVICE_QP_MANAGER_H

#include <netinet/in.h>
#include <cstdint>
#include <memory>
#include <unordered_map>
#include "hybm_def.h"
#include "device_rdma_common.h"

namespace shm {
namespace transport {
namespace device {

class DeviceQpManager {
public:
    DeviceQpManager(uint32_t deviceId, uint32_t rankId, uint32_t rankCount, mf_sockaddr devNet,
                    hybm_role_type role) noexcept;
    virtual ~DeviceQpManager() = default;

    virtual int SetRemoteRankInfo(const std::unordered_map<uint32_t, ConnectRankInfo> &ranks) noexcept = 0;
    virtual int SetLocalMemories(const MemoryRegionMap &mrs) noexcept = 0;
    virtual int Startup(void *rdma) noexcept = 0;
    virtual void Shutdown() noexcept = 0;
    virtual int WaitingConnectionReady() noexcept;
    virtual const void *GetQpInfoAddress() const noexcept;
    virtual void *GetQpHandleWithRankId(uint32_t rankId) const noexcept = 0;

protected:
    void *CreateLocalSocket() noexcept;
    int CreateServerSocket() noexcept;
    void DestroyServerSocket() noexcept;

protected:
    const uint32_t deviceId_;
    const uint32_t rankId_;
    const uint32_t rankCount_;
    const hybm_role_type rankRole_;
    mf_sockaddr deviceAddress_;
    void *serverSocketHandle_{nullptr};
};
}
}
}

#endif  // MF_HYBRID_DEVICE_QP_MANAGER_H
