/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "mem_entity_def.h"
#include "shmemi_logger.h"
#include "dl_acl_api.h"
#include "dl_hccp_api.h"
#include "device_rdma_common.h"
#include "device_rdma_helper.h"
#include "fixed_ranks_qp_manager.h"
#include "dynamic_ranks_qp_manager.h"
#include "device_rdma_transport_manager.h"

namespace shm {
namespace transport {
namespace device {
bool RdmaTransportManager::tsdOpened_ = false;
bool RdmaTransportManager::deviceIpRetired_ = false;
bool RdmaTransportManager::raInitialized_ = false;
void* RdmaTransportManager::storedRdmaHandle_ = nullptr;

RdmaTransportManager::~RdmaTransportManager()
{
    ClearAllRegisterMRs();
    tsdOpened_ = false;
    raInitialized_ = false;
    deviceIpRetired_ = false;
    storedRdmaHandle_ = nullptr;
}

void RdmaTransportManager::InitializeDeviceAddress(mf_sockaddr &deviceAddr)
{
    if (deviceIp_.type == IpV4) {
        deviceAddr.ip.ipv4.sin_family = AF_INET;
        deviceAddr.ip.ipv4.sin_addr = deviceIp_.ip.ipv4;
        deviceAddr.ip.ipv4.sin_port = devicePort_;
        deviceAddr.type = IpV4;
    } else if (deviceIp_.type == IpV6) {
        deviceAddr.ip.ipv6.sin6_family = AF_INET6;
        deviceAddr.ip.ipv6.sin6_addr = deviceIp_.ip.ipv6;
        deviceAddr.ip.ipv6.sin6_port = devicePort_;
        deviceAddr.type = IpV6;
    }
}

Result RdmaTransportManager::OpenDevice(const TransportOptions &options)
{
    int32_t userId = -1;
    int32_t logicId = -1;

    SHM_LOG_DEBUG(rankId_ << " begin to open device with " << options);
    auto ret = DlAclApi::AclrtGetDevice(&userId);
    SHM_ASSERT_LOG_AND_RETURN(ret == 0 && userId >= 0,
                             "AclrtGetDevice() return=" << ret << ", output deviceId=" << userId,
                             ACLSHMEM_DL_FUNC_FAILED);

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(userId, &logicId);
    SHM_ASSERT_LOG_AND_RETURN(ret == 0 && logicId >= 0,
                             "RtGetLogicDevIdByUserDevId() return=" << ret << ", output deviceId=" << logicId,
                             ACLSHMEM_DL_FUNC_FAILED);

    deviceId_ = static_cast<uint32_t>(logicId);
    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    role_ = options.role;
    ret = ParseDeviceNic(options.nic, devicePort_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " parse input nic(" << options.nic << ") failed!");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.type == IpV4) {
        deviceIp_.type = IpV4;
    } else if (options.type == IpV6) {
        deviceIp_.type = IpV6;
    }

    if (!PrepareOpenDevice(userId, deviceId_, rankCount_, deviceIp_, rdmaHandle_)) {
        SHM_LOG_ERROR(deviceId_ << " PrepareOpenDevice failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    nicInfo_ = GenerateDeviceNic(deviceIp_, devicePort_);

    mf_sockaddr deviceAddr;
    InitializeDeviceAddress(deviceAddr);
    if (role_ == HYBM_ROLE_PEER) {
        qpManager_ = std::make_shared<FixedRanksQpManager>(userId, deviceId_, rankId_, rankCount_, deviceAddr);
    } else {
        qpManager_ = std::make_shared<DynamicRanksQpManager>(userId, deviceId_, rankId_, rankCount_, deviceAddr,
                                                             role_ == HYBM_ROLE_RECEIVER);
    }

    deviceChipInfo_ = std::make_shared<DeviceChipInfo>(userId);
    ret = deviceChipInfo_->Init();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " device info init failed: " << ret);
        return ret;
    }
    SHM_LOG_INFO(rankId_ << " open device with " << options << " success.");
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::CloseDevice()
{
    if (qpManager_ != nullptr) {
        qpManager_->Shutdown();
        qpManager_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion &mr)
{
    void *mrHandle = nullptr;
    HccpMrInfo info{};
    info.addr = (void *)(ptrdiff_t)mr.addr;
    info.size = mr.size;
    info.access = mr.access;
    auto ret = DlHccpApi::RaRegisterMR(rdmaHandle_, &info, mrHandle);
    if (ret != 0) {
        SHM_LOG_ERROR(rankId_ << " register MR=" << mr << " failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    RegMemResult result{mr.addr, mr.size, mrHandle, info.lkey, info.rkey};
    SHM_LOG_DEBUG(rankId_ << " register MR result=" << result);

    registerMRS_.emplace(mr.addr, result);
    ret = qpManager_->SetLocalMemories(registerMRS_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " qp manager set mr failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    auto pos = registerMRS_.find(addr);
    if (pos == registerMRS_.end()) {
        SHM_LOG_ERROR(rankId_ << " input address not register!");
        return ACLSHMEM_INVALID_PARAM;
    }

    auto ret = DlHccpApi::RaDeregisterMR(rdmaHandle_, pos->second.mrHandle);
    if (ret != 0) {
        SHM_LOG_ERROR(rankId_ << " Unregister MR addr failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    registerMRS_.erase(pos);
    ret = qpManager_->SetLocalMemories(registerMRS_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " qp manager set mr failed: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::QueryMemoryKey(uint64_t addr, TransportMemoryKey &key)
{
    RegMemKeyUnion keyUnion{};
    auto pos = registerMRS_.lower_bound(addr);
    if (pos == registerMRS_.end() || pos->first + pos->second.size <= addr) {
        SHM_LOG_ERROR(rankId_ << " input address not register");
        return ACLSHMEM_INVALID_PARAM;
    }

    keyUnion.deviceKey = pos->second;

    key = keyUnion.commonKey;
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::ParseMemoryKey(const TransportMemoryKey &key, uint64_t &addr, uint64_t &size)
{
    RegMemKeyUnion keyUnion{};
    keyUnion.commonKey = key;
    if (keyUnion.deviceKey.type != TT_HCCP) {
        SHM_LOG_ERROR(rankId_ << " parse memory key type invalid: " << keyUnion.deviceKey.type);
        return ACLSHMEM_INNER_ERROR;
    }

    addr = keyUnion.deviceKey.address;
    size = keyUnion.deviceKey.size;
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::Prepare(const HybmTransPrepareOptions &options)
{
    SHM_LOG_DEBUG(rankId_ << " RdmaTransportManager Prepare with : " << options);
    int ret;
    if ((ret = CheckPrepareOptions(options)) != 0) {
        return ret;
    }

    mf_sockaddr deviceNetwork;
    std::unordered_map<uint32_t, ConnectRankInfo> rankInfo;
    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        ret = ParseDeviceNic(it->second.nic, deviceNetwork);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(rankId_ << " parse networks[" << it->first << "]=" << it->second.nic << " failed: " << ret);
            return ACLSHMEM_INVALID_PARAM;
        }

        rankInfo.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.memKeys});
    }
    SHM_LOG_DEBUG(rankId_ << " SetRemoteRankInfo rankInfo.size=" << rankInfo.size());

    ret = qpManager_->SetRemoteRankInfo(rankInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " qp manager set remote rank info failed: " << ret);
        return ret;
    }

    ret = qpManager_->Startup(rdmaHandle_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " qp manager startup failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::Connect()
{
    auto ret = AsyncConnect();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " AsyncConnect() failed: " << ret);
        return ret;
    }

    ret = WaitForConnected(-1L);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " WaitForConnected(-1) failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::AsyncConnect()
{
    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::WaitForConnected(int64_t timeoutNs)
{
    if (qpManager_ == nullptr) {
        SHM_LOG_ERROR(rankId_ << " server side not listen!");
        return ACLSHMEM_INNER_ERROR;
    }

    auto ret = qpManager_->WaitingConnectionReady();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " wait for server side connected on device failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result RdmaTransportManager::UpdateRankOptions(const HybmTransPrepareOptions &options)
{
    SHM_LOG_DEBUG(rankId_ << " RdmaTransportManager Prepare with : " << options);
    if (qpManager_ == nullptr) {
        SHM_LOG_ERROR(rankId_ << " qp manager not created");
        return ACLSHMEM_INNER_ERROR;
    }

    mf_sockaddr deviceNetwork;
    std::unordered_map<uint32_t, ConnectRankInfo> ranksInfo;
    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        auto ret = ParseDeviceNic(it->second.nic, deviceNetwork);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(rankId_ << " update rank network(" << it->second.nic << ") invalid.");
            return ACLSHMEM_INVALID_PARAM;
        }
        SHM_LOG_INFO(rankId_ << " UpdateRankOptions update rank: " << it->first);
        ranksInfo.emplace(it->first, ConnectRankInfo{it->second.role, deviceNetwork, it->second.memKeys});
    }
    SHM_LOG_DEBUG(rankId_ << " UpdateRankOptions ranksInfo.size=" << ranksInfo.size());

    auto ret = qpManager_->SetRemoteRankInfo(ranksInfo);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " update rank options failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

const std::string &RdmaTransportManager::GetNic() const
{
    return nicInfo_;
}

const void *RdmaTransportManager::GetQpInfo() const
{
    if (qpManager_ == nullptr) {
        SHM_LOG_ERROR(rankId_ << " GetQpInfo(): connection manager not created.");
        return nullptr;
    }
    return qpManager_->GetQpInfoAddress();
}

bool RdmaTransportManager::PrepareOpenDevice(uint32_t userId, uint32_t device, uint32_t rankCount,
                                             net_addr_t &deviceIp, void *&rdmaHandle)
{
    // If can get rdmaHandle, maybe the device has been opened, can try get rdmaHandle directly.
    if (DlHccpApi::RaRdevGetHandle(device, rdmaHandle) == 0) {
        if (rdmaHandle != nullptr) {
            if (!RetireDeviceIp(device, deviceIp)) {
                SHM_LOG_ERROR(device << " RetireDeviceIp failed.");
                return false;
            }
            SHM_LOG_DEBUG(device << " Had prepared device and get rdmaHandle success.");
            return true;
        }
        SHM_LOG_INFO(device << " Had prepared device, but rdmaHandle is null, need init again.");
    }
    if (!OpenTsd(userId, rankCount)) {
        SHM_LOG_ERROR(device << " open tsd failed.");
        return false;
    }

    if (!RaInit(device)) {
        SHM_LOG_ERROR(device << " RaInit failed.");
        return false;
    }

    if (!RetireDeviceIp(device, deviceIp)) {
        SHM_LOG_ERROR(device << " RetireDeviceIp failed.");
        return false;
    }

    if (!RaRdevInit(device, deviceIp, rdmaHandle)) {
        SHM_LOG_ERROR(device << " RaRdevInit failed.");
        return false;
    }
    return true;
}

bool RdmaTransportManager::OpenTsd(uint32_t deviceId, uint32_t rankCount)
{
    if (tsdOpened_) {
        SHM_LOG_INFO(deviceId << " tsd already opened.");
        return true;
    }

    auto res = DlHccpApi::TsdOpen(deviceId, rankCount);
    if (res != 0) {
        SHM_LOG_ERROR("TsdOpen for (deviceId=" << deviceId << ", rankCount=" << rankCount << ") failed: " << res);
        return false;
    }

    SHM_LOG_DEBUG("open tsd for device id: " << deviceId << ", rank count: " << rankCount << " success.");
    tsdOpened_ = true;
    return true;
}

bool RdmaTransportManager::RaInit(uint32_t deviceId)
{
    if (raInitialized_) {
        SHM_LOG_INFO(deviceId << " ra already initialized.");
        return true;
    }
    const std::chrono::seconds WAIT_TIME(3);
    HccpRaInitConfig initConfig{};
    initConfig.phyId = deviceId;
    initConfig.nicPosition = NETWORK_OFFLINE;
    initConfig.hdcType = 6;  // HDC_SERVICE_TYPE_RDMA = 6
    SHM_LOG_DEBUG(deviceId << " RaInit=" << initConfig);
    std::this_thread::sleep_for(WAIT_TIME); // avoid hccl init conflict
    auto ret = DlHccpApi::RaInit(initConfig);
    if (ret != 0) {
        SHM_LOG_WARN(deviceId << " Hccp Init RA failed: " << ret);
        // maybe hccl have already initialized ra, wait 3s then return true.
        std::this_thread::sleep_for(WAIT_TIME);
        raInitialized_ = true;
        return true;
    }

    SHM_LOG_DEBUG(deviceId << " ra init for device id: " << deviceId << " success.");
    raInitialized_ = true;
    return true;
}

bool RdmaTransportManager::HandleRetiredDeviceIp(uint32_t deviceId, net_addr_t &deviceIp, net_addr_t &retiredIp)
{
    if (deviceIpRetired_ && deviceIp.type == IpV4) {
        SHM_LOG_INFO(deviceId << " device ip already retired : " << inet_ntoa(retiredIp.ip.ipv4));
        deviceIp = retiredIp;
        return true;
    } else if (deviceIpRetired_ && deviceIp.type == IpV6) {
        char ipv6Str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &retiredIp.ip.ipv6, ipv6Str, INET6_ADDRSTRLEN);
        SHM_LOG_INFO(deviceId << " device ip already retired : " << ipv6Str);
        deviceIp = retiredIp;
        return true;
    }
    return false;
}

bool RdmaTransportManager::RetireDeviceIp(uint32_t deviceId, net_addr_t &deviceIp)
{
    net_addr_t retiredIp{};

    auto isRetire = HandleRetiredDeviceIp(deviceId, deviceIp, retiredIp);
    if (isRetire) {
        return true;
    }

    uint32_t count = 0;
    std::vector<HccpInterfaceInfo> infos;

    HccpRaGetIfAttr config;
    config.phyId = deviceId;
    config.nicPosition = NETWORK_OFFLINE;
    config.isAll = true;

    auto ret = DlHccpApi::RaGetIfNum(config, count);
    if (ret != 0 || count == 0) {
        SHM_LOG_ERROR(deviceId << " get interface count failed: " << ret << ", count: " << count);
        return false;
    }

    infos.resize(count);
    ret = DlHccpApi::RaGetIfAddrs(config, infos.data(), count);
    if (ret != 0) {
        SHM_LOG_ERROR(deviceId << " get interface information failed: " << ret);
        return false;
    }

    for (auto &info : infos) {
        if (info.family == AF_INET) {
            deviceIp.ip.ipv4 = retiredIp.ip.ipv4 = info.ifaddr.ip.addr;
            deviceIp.type = IpV4;
            deviceIpRetired_ = true;
            SHM_LOG_DEBUG(deviceId << " retire device ip success : " << inet_ntoa(deviceIp.ip.ipv4));
            return true;
        }
        if (info.family == AF_INET6) {
            deviceIp.ip.ipv6 = retiredIp.ip.ipv6 = info.ifaddr.ip.addr6;
            deviceIp.type = IpV6;
            deviceIpRetired_ = true;
            char ipv6Str[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &deviceIp.ip.ipv6, ipv6Str, INET6_ADDRSTRLEN);
            SHM_LOG_DEBUG(deviceId << " retire device ip success : " << ipv6Str);
            return true;
        }
    }

    SHM_LOG_ERROR(deviceId << " not found network device of AF_INET or AF_INET6 on NPU.");
    return false;
}

bool RdmaTransportManager::RaRdevInit(uint32_t deviceId, net_addr_t deviceIp, void *&rdmaHandle)
{
    if (storedRdmaHandle_ != nullptr) {
        SHM_LOG_INFO(deviceId << " ra rdev already initialized.");
        rdmaHandle = storedRdmaHandle_;
        return true;
    }

    HccpRdevInitInfo info{};
    HccpRdev rdev{};

    info.mode = NETWORK_OFFLINE;
    info.notifyType = NOTIFY;
    info.enabled2mbLite = true;
    rdev.phyId = deviceId;
    rdev.family = (deviceIp.type == IpV4) ? AF_INET : AF_INET6;
    if (deviceIp.type == IpV4) {
        rdev.localIp.addr = deviceIp.ip.ipv4;
    } else if (deviceIp.type == IpV6) {
        rdev.localIp.addr6 = deviceIp.ip.ipv6;
    }
    SHM_LOG_DEBUG(deviceId << " RaRdevInitV2, info=" << info << "rdev=" << rdev);
    auto ret = DlHccpApi::RaRdevInitV2(info, rdev, rdmaHandle);
    if (ret != 0) {
        SHM_LOG_ERROR(deviceId << " Hccp Init RDev failed: " << ret);
        return false;
    }

    storedRdmaHandle_ = rdmaHandle;
    SHM_LOG_INFO(deviceId << " initialize RDev success.");
    return true;
}

void RdmaTransportManager::ClearAllRegisterMRs()
{
    for (auto it = registerMRS_.begin(); it != registerMRS_.end(); ++it) {
        auto ret = DlHccpApi::RaDeregisterMR(rdmaHandle_, it->second.mrHandle);
        if (ret != 0) {
            SHM_LOG_WARN(rankId_ << " Unregister:" << (void *)(ptrdiff_t)it->first << " : " << it->second << " failed: " << ret);
        }
    }
    registerMRS_.clear();
}

int RdmaTransportManager::CheckPrepareOptions(const shm::transport::HybmTransPrepareOptions &options)
{
    if (role_ != HYBM_ROLE_PEER) {
        SHM_LOG_INFO(rankId_ << " transport role: " << role_ << " check options passed.");
        return ACLSHMEM_SUCCESS;
    }

    if (options.options.size() > rankCount_) {
        SHM_LOG_ERROR(rankId_ << " options size():" << options.options.size() << " larger than rank count: " << rankCount_);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.options.find(rankId_) == options.options.end()) {
        SHM_LOG_ERROR(rankId_ << " options not contains self rankId: " << rankId_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        if (it->first >= rankCount_) {
            SHM_LOG_ERROR(rankId_ << " input options of nics contains rankId:" << it->first << ", rank count: " << rankCount_);
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    return ACLSHMEM_SUCCESS;
}
}
}
}
