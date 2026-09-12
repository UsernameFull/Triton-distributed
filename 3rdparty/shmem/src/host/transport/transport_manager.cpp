/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "transport_manager.h"

#include "shmemi_logger.h"
#include "transport_def.h"
#include "device_rdma_transport_manager.h"
#include "device_sdma_transport_manager.h"
#if defined(ACLSHMEM_UDMA_SUPPORT)
#include "device_udma_transport_manager.h"
#endif

using namespace shm;
using namespace shm::transport;

std::shared_ptr<TransportManager> TransportManager::Create(TransportType type)
{
    switch (type) {
        case TT_HCCP:
            return std::make_shared<device::RdmaTransportManager>();
        case TT_SDMA:
            return std::make_shared<device::SdmaTransportManager>();
#if defined(ACLSHMEM_UDMA_SUPPORT)
        case TT_UDMA:
            return std::make_shared<device::UdmaTransportManager>();
#endif
        default:
            SHM_LOG_ERROR("Invalid trans type: " << type);
            return nullptr;
    }
}

const void *TransportManager::GetQpInfo() const
{
    SHM_LOG_DEBUG("Not Implement GetQpInfo()");
    return nullptr;
}

Result TransportManager::ConnectWithOptions(const HybmTransPrepareOptions &options)
{
    SHM_LOG_DEBUG("ConnectWithOptions now connected=" << connected_);
    if (!connected_) {
        auto ret = Prepare(options);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("prepare connection failed: " << ret);
            return ret;
        }

        ret = Connect();
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("connect failed: " << ret);
            return ret;
        }

        connected_ = true;
        return ACLSHMEM_SUCCESS;
    }

    return UpdateRankOptions(options);
}
