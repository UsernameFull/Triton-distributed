/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shmemi_logger.h"
#include "dl_hccp_api.h"
#include "device_qp_manager.h"

namespace shm {
namespace transport {
namespace device {
DeviceQpManager::DeviceQpManager(uint32_t deviceId, uint32_t rankId, uint32_t rankCount, mf_sockaddr devNet,
                                 hybm_role_type role) noexcept
    : deviceId_{deviceId},
      rankId_{rankId},
      rankCount_{rankCount},
      deviceAddress_{devNet},
      rankRole_{role}
{
}

int DeviceQpManager::WaitingConnectionReady() noexcept
{
    return ACLSHMEM_SUCCESS;
}

const void *DeviceQpManager::GetQpInfoAddress() const noexcept
{
    return nullptr;
}

void *DeviceQpManager::CreateLocalSocket() noexcept
{
    void *socketHandle = nullptr;
    HccpRdev rdev;
    rdev.phyId = deviceId_;
    rdev.family = (deviceAddress_.type == IpV4) ? AF_INET : AF_INET6;
    if (deviceAddress_.type == IpV4) {
        rdev.localIp.addr = deviceAddress_.ip.ipv4.sin_addr;
    } else if (deviceAddress_.type == IpV6) {
        rdev.localIp.addr6 = deviceAddress_.ip.ipv6.sin6_addr;
    }
    auto ret = DlHccpApi::RaSocketInit(HccpNetworkMode::NETWORK_OFFLINE, rdev, socketHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("initialize socket handle failed: " << ret);
        return nullptr;
    }

    return socketHandle;
}

int DeviceQpManager::CreateServerSocket() noexcept
{
    if (serverSocketHandle_ != nullptr) {
        return ACLSHMEM_SUCCESS;
    }

    auto socketHandle = CreateLocalSocket();
    if (socketHandle == nullptr) {
        SHM_LOG_ERROR(rankId_ << " create local socket handle failed.");
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    HccpSocketListenInfo listenInfo{};
    listenInfo.handle = socketHandle;
    listenInfo.port = (deviceAddress_.type == IpV4) ? deviceAddress_.ip.ipv4.sin_port
        : deviceAddress_.ip.ipv6.sin6_port;
    bool successListen = false;
    uint16_t maxPort = std::numeric_limits<uint16_t>::max();
    while (listenInfo.port <= maxPort) {
        auto ret = DlHccpApi::RaSocketListenStart(&listenInfo, 1);
        if (ret == 0) {
            if (deviceAddress_.type == IpV4) {
                deviceAddress_.ip.ipv4.sin_port = listenInfo.port;
            } else if (deviceAddress_.type == IpV6) {
                deviceAddress_.ip.ipv6.sin6_port = listenInfo.port;
            }
            successListen = true;
            break;
        }
        if (listenInfo.port == maxPort) {
            break;
        }
        listenInfo.port++;
    }
    if (!successListen) {
        SHM_LOG_ERROR(rankId_ << " start to listen server socket failed.");
        DlHccpApi::RaSocketDeinit(socketHandle);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    SHM_LOG_INFO(rankId_ << " start to listen on port: " << listenInfo.port << " success.");
    serverSocketHandle_ = socketHandle;
    return ACLSHMEM_SUCCESS;
}

void DeviceQpManager::DestroyServerSocket() noexcept
{
    if (serverSocketHandle_ == nullptr) {
        return;
    }

    HccpSocketListenInfo listenInfo{};
    listenInfo.handle = serverSocketHandle_;
    listenInfo.port = (deviceAddress_.type == IpV4) ? deviceAddress_.ip.ipv4.sin_port
        : deviceAddress_.ip.ipv6.sin6_port;
    auto ret = DlHccpApi::RaSocketListenStop(&listenInfo, 1);
    if (ret != 0) {
        SHM_LOG_INFO("stop to listen on port: " << listenInfo.port << " return: " << ret);
    }
}
}
}

}