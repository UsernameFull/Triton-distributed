/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

#include "dl_acl_api.h"
#include "dl_hccp_v2_api.h"
#include "mem_entity_def.h"
#include "shmemi_host_common.h"
#include "transport/topo/topo_reader.h"
#include "device_udma_transport_manager.h"

namespace shm {
namespace transport {
namespace device {

namespace {

struct ExchangedMrInfo {
    uint32_t eidIndex{0};
    uint32_t valid{0};
    RegMemResultInfo mr{};
};

constexpr int32_t INVALID_EID_INDEX = -1;

} // namespace

bool UdmaTransportManager::tsdOpened_ = false;
bool UdmaTransportManager::raInitialized_ = false;
std::map<uint32_t, void*> UdmaTransportManager::storedCtxHandleMap_;
int UdmaTransportManager::subPid_ = 0;

UdmaTransportManager::UdmaTransportManager() noexcept {}

UdmaTransportManager::~UdmaTransportManager() noexcept
{
    CleanupResources();
    tsdOpened_ = false;
    raInitialized_ = false;
    storedCtxHandleMap_.clear();
    subPid_ = 0;
}

Result UdmaTransportManager::OpenDevice(const TransportOptions& options)
{
    int32_t userId = -1;
    int32_t logicId = -1;

    SHM_LOG_DEBUG(options.rankId << " begin to open device with " << options);
    auto ret = DlAclApi::AclrtGetDevice(&userId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && userId >= 0, "AclrtGetDevice() return=" << ret << ", output deviceId=" << userId,
        ACLSHMEM_DL_FUNC_FAILED);

    ret = DlAclApi::RtGetLogicDevIdByUserDevId(userId, &logicId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && logicId >= 0, "RtGetLogicDevIdByUserDevId() return=" << ret << ", output deviceId=" << logicId,
        ACLSHMEM_DL_FUNC_FAILED);

    deviceId_ = static_cast<uint32_t>(logicId);
    rankId_ = options.rankId;
    rankCount_ = options.rankCount;
    role_ = options.role;

    if (!PrepareOpenDevice(deviceId_, rankCount_)) {
        SHM_LOG_ERROR("PrepareOpenDevice failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    deviceJettyManager_ = new DeviceJettyManager(deviceId_, rankId_, rankCount_, eidCount_);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::CloseDevice()
{
    if (deviceJettyManager_ != nullptr) {
        deviceJettyManager_->Shutdown();
        delete deviceJettyManager_;
        deviceJettyManager_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::RegisterMemoryRegion(const TransportMemoryRegion& mr)
{
    std::map<uint32_t, RegMemResultInfo> regResultMap;
    for (const auto& ctxEntry : storedCtxHandleMap_) {
        const uint32_t eidIndex = ctxEntry.first;
        void* ctxHandle = ctxEntry.second;
        auto handleIter = tokenIdHandleMap_.find(eidIndex);
        if (handleIter == tokenIdHandleMap_.end()) {
            SHM_LOG_ERROR("Failed to find a tokenIdHandle for EID index " << eidIndex);
            return ACLSHMEM_INNER_ERROR;
        }
        void* tokenIdHandle = handleIter->second;

        struct MrRegInfoT mrInfo = {};
        mrInfo.in.mem.addr = mr.addr;
        mrInfo.in.mem.size = mr.size;
        mrInfo.in.ub.tokenValue = mr.tokenValue;
        mrInfo.in.ub.tokenIdHandle = tokenIdHandle;
        mrInfo.in.ub.flags.bs.access = MEM_SEG_ACCESS_DEFAULT;
        mrInfo.in.ub.flags.bs.cacheable = mr.cacheable;
        mrInfo.in.ub.flags.bs.tokenIdValid = mr.tokenIdValid;
        mrInfo.in.ub.flags.bs.nonPin = 0;
        mrInfo.in.ub.flags.bs.userIova = 0;
        mrInfo.in.ub.flags.bs.tokenPolicy = URMA_TOKEN_PLAIN_TEXT;

        void* lmemHandle = nullptr;
        auto ret = shm::DlHccpV2Api::RaCtxLmemRegister(ctxHandle, &mrInfo, &lmemHandle);
        if (ret != 0) {
            SHM_LOG_ERROR("Failed to register the memory region for EID index " << eidIndex << ", ret = " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        lmemHandleMap_[eidIndex] = lmemHandle;

        RegMemResultInfo regResult{
            mr.addr,
            mr.size,
            lmemHandle,
            mrInfo.out.key,
            mrInfo.out.ub.tokenId,
            mr.tokenValue,
            mrInfo.out.ub.targetSegHandle,
            tokenIdHandle,
            mr.cacheable,
            mr.access};
        regResultMap[eidIndex] = regResult;
        ACLSHMEMUBmemInfo localMemInfo{};
        localMemInfo.token_value_valid = mr.tokenIdValid;
        localMemInfo.rmt_jetty_type = 1;
        localMemInfo.target_hint = 0;
        localMemInfo.tpn = 0;
        localMemInfo.tid = mrInfo.out.ub.tokenId >> 8;
        localMemInfo.rmt_token_value = mr.tokenValue;
        localMemInfo.len = mr.size;
        localMemInfo.addr = mr.addr;
        localMemInfoMap_[eidIndex] = localMemInfo;
    }

    registerMRS_[mr.addr] = regResultMap;
    localMrMap_ = regResultMap;

    auto ret = deviceJettyManager_->SetLocalMemInfos(localMemInfoMap_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Failed to set the local memory information in the jetty manager, ret = " << ret);
        return ret;
    }

    ret = deviceJettyManager_->SetEids(hccpEidMap_);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Failed to set the swap EID in the jetty manager, ret = " << ret);
        return ret;
    }

    SHM_LOG_INFO("Register MR success.");
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::UnregisterMemoryRegion(uint64_t addr)
{
    auto pos = registerMRS_.find(addr);
    if (pos == registerMRS_.end()) {
        SHM_LOG_ERROR("Input address is not registered, address = " << addr);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (const auto& mrEntry : pos->second) {
        uint32_t eidIndex = mrEntry.first;
        void* lmemHandle = mrEntry.second.lmemHandle;
        auto ctxIt = storedCtxHandleMap_.find(eidIndex);
        if (ctxIt == storedCtxHandleMap_.end() || lmemHandle == nullptr) {
            continue;
        }
        auto ret = shm::DlHccpV2Api::RaCtxLmemUnregister(ctxIt->second, lmemHandle);
        if (ret != 0) {
            SHM_LOG_ERROR("Failed to unregister the memory region for EID index " << eidIndex << ", ret = " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
    }
    registerMRS_.erase(pos);
    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::Prepare(const HybmTransPrepareOptions& options)
{
    SHM_LOG_DEBUG("UdmaTransportManager Prepare with options: " << options);
    int ret;
    if ((ret = CheckPrepareOptions(options)) != 0) {
        return ret;
    }

    deviceJettyManager_->SetCtxHandles(storedCtxHandleMap_);
    deviceJettyManager_->SetTokenIdHandles(tokenIdHandleMap_);
    deviceJettyManager_->SetLocalMemInfos(localMemInfoMap_);
    deviceJettyManager_->SetEids(hccpEidMap_);
    deviceJettyManager_->SetPeerRoutes(peerEidIndexMap_, peerRemoteEidIndexMap_);
    ret = deviceJettyManager_->Startup();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Jetty manager startup failed, ret = " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

Result UdmaTransportManager::Connect()
{
    auto ret = AsyncConnect();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Async connect failed, ret = " << ret);
        return ret;
    }
    SHM_LOG_INFO("Async connect success");

    return ACLSHMEM_SUCCESS;
}

const void* UdmaTransportManager::GetQpInfo() const { return deviceJettyManager_->GetJettyInfoAddress(); }

Result UdmaTransportManager::AsyncConnect()
{
    uint32_t localMrCount = static_cast<uint32_t>(localMrMap_.size());
    std::vector<uint32_t> allMrRegInfoCounts(rankCount_);
    g_boot_handle.allgather(&localMrCount, allMrRegInfoCounts.data(), sizeof(uint32_t), &g_boot_handle);

    const auto maxCountIt = std::max_element(allMrRegInfoCounts.begin(), allMrRegInfoCounts.end());
    const uint32_t maxMrCount = (maxCountIt == allMrRegInfoCounts.end()) ? 0 : *maxCountIt;
    if (maxMrCount == 0) {
        SHM_LOG_ERROR("No local MR information was exchanged.");
        return ACLSHMEM_INNER_ERROR;
    }

    std::vector<ExchangedMrInfo> localMrByEid(maxMrCount);
    uint32_t packedIndex = 0;
    for (const auto& mrEntry : localMrMap_) {
        localMrByEid[packedIndex].eidIndex = mrEntry.first;
        localMrByEid[packedIndex].valid = 1;
        localMrByEid[packedIndex].mr = mrEntry.second;
        ++packedIndex;
    }

    std::vector<ExchangedMrInfo> allMrRegInfo(rankCount_ * maxMrCount);
    g_boot_handle.allgather(
        localMrByEid.data(), allMrRegInfo.data(), static_cast<uint64_t>(sizeof(ExchangedMrInfo) * maxMrCount),
        &g_boot_handle);
    memoryHandleList_.resize(rankCount_);
    for (uint32_t peer = 0; peer < rankCount_; ++peer) {
        if (peer == rankId_) {
            continue;
        }
        auto ctxIt = ctxHandleMap_.find(peer);
        void* ctxHandle = (ctxIt != ctxHandleMap_.end()) ? ctxIt->second : nullptr;
        if (ctxHandle == nullptr) {
            SHM_LOG_ERROR("Failed to find ctxHandle for peer " << peer);
            return ACLSHMEM_INNER_ERROR;
        }

        auto remoteEidIt = peerRemoteEidIndexMap_.find(peer);
        if (remoteEidIt == peerRemoteEidIndexMap_.end()) {
            SHM_LOG_ERROR("Failed to find remote EID route for peer " << peer);
            return ACLSHMEM_INNER_ERROR;
        }
        uint32_t remoteEidIndex = remoteEidIt->second;

        const ExchangedMrInfo* remoteMrInfo = nullptr;
        const uint32_t peerMrCount = allMrRegInfoCounts[peer];
        for (uint32_t idx = 0; idx < peerMrCount; ++idx) {
            const ExchangedMrInfo& candidate = allMrRegInfo[peer * maxMrCount + idx];
            if (candidate.valid != 0 && candidate.eidIndex == remoteEidIndex) {
                remoteMrInfo = &candidate;
                break;
            }
        }
        if (remoteMrInfo == nullptr) {
            SHM_LOG_ERROR("No remote MR was exchanged for peer " << peer << " on remote EID index " << remoteEidIndex);
            return ACLSHMEM_INNER_ERROR;
        }

        const RegMemResultInfo& mr = remoteMrInfo->mr;

        struct MrImportInfoT mrImportInfo = {};
        mrImportInfo.in.key = mr.key;
        mrImportInfo.in.ub.tokenValue = mr.tokenValue;
        mrImportInfo.in.ub.flags.bs.cacheable = mr.cacheable;
        mrImportInfo.in.ub.flags.bs.access = mr.access;
        void* rmemHandle = nullptr;
        auto ret = shm::DlHccpV2Api::RaCtxRmemImport(ctxHandle, &mrImportInfo, &rmemHandle);
        if (ret != 0) {
            SHM_LOG_ERROR("Failed to import remote memory for peer " << peer << ", ret = " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
        memoryHandleList_[peer] = rmemHandle;
    }
    SHM_LOG_INFO("Import remote MR success.");
    return ACLSHMEM_SUCCESS;
}

bool UdmaTransportManager::PrepareOpenDevice(uint32_t deviceId, uint32_t rankCount)
{
    // Parse topo to build portEidMap
    RootInfo rootInfo;
    TopoInfo topoInfo;
    uint32_t localId = 0;
    uint32_t eidCount = 0;
    int32_t phyId = -1;

    auto ret = DlAclApi::AclrtGetPhyDevIdByLogicDevId(static_cast<int32_t>(deviceId), &phyId);
    SHM_ASSERT_LOG_AND_RETURN(
        ret == 0 && phyId >= 0,
        "AclrtGetPhyDevIdByLogicDevId() return=" << ret << ", input deviceId=" << deviceId
                                                   << ", output phyId="
                                               << phyId,
        false);

    if (!TopoReader::ParseRootInfo(phyId, rootInfo)) {
        SHM_LOG_ERROR("Failed to parse the rootinfo file.");
        return false;
    }
    phyId_ = static_cast<uint32_t>(phyId);
    SHM_LOG_INFO("Resolved phy id from current device mapping, deviceId=" << deviceId << ", phyId=" << phyId);
    if (!TopoReader::ParseTopoInfo(rootInfo.topo_file_path, topoInfo)) {
        SHM_LOG_ERROR("Failed to parse the topology file at path " << rootInfo.topo_file_path);
        return false;
    }

    if (!OpenTsd(deviceId, rankCount)) {
        SHM_LOG_ERROR("Open tsd failed.");
        return false;
    }

    rootInfo_ = rootInfo;
    if (!RaInit(phyId_)) {
        SHM_LOG_ERROR("RaInit failed.");
        return false;
    }

    if (!TopoReader::GetLocalId(rootInfo, phyId_, localId)) {
        SHM_LOG_ERROR("Failed to find localId for phyId: " << phyId_);
        return false;
    }
    if (!TopoReader::GetEidCount(rootInfo, eidCount)) {
        SHM_LOG_ERROR("Failed to find eid count from rootinfo.");
        return false;
    }

    std::vector<uint32_t> eidCountList(rankCount);
    g_boot_handle.allgather(&eidCount, eidCountList.data(), sizeof(uint32_t), &g_boot_handle);
    const auto maxEidCountIt = std::max_element(eidCountList.begin(), eidCountList.end());
    eidCount_ = (maxEidCountIt == eidCountList.end()) ? 0 : *maxEidCountIt;
    if (eidCount_ == 0) {
        SHM_LOG_ERROR("Invalid eidSlotCount resolved from rootinfo rank_addr_list.");
        return false;
    }

    // allgather rankId -> localId
    std::vector<uint32_t> localIdList(rankCount);
    g_boot_handle.allgather(&localId, localIdList.data(), sizeof(uint32_t), &g_boot_handle);

    std::vector<int32_t> localRouteByPeer(rankCount, INVALID_EID_INDEX);

    for (uint32_t peer = 0; peer < rankCount; ++peer) {
        if (peer == rankId_) {
            continue;
        }
        uint32_t eidIndex = 0;
        std::array<uint8_t, 16> eidRaw{};
        uint32_t peerLocalId = localIdList[peer];
        if (!TopoReader::GetLocalEidRouteForPeer(rootInfo, topoInfo, localId, peerLocalId, eidIndex, eidRaw)) {
            SHM_LOG_ERROR(
                "Failed to resolve the local EID route for peer rank " << peer << ". The localId was " << localId
                                                                       << " and the peer localId was " << peerLocalId);
            return false;
        }
        peerEidIndexMap_[peer] = eidIndex;
        localRouteByPeer[peer] = static_cast<int32_t>(eidIndex);

        void* ctxHandle = nullptr;
        if (!RaCtxInit(phyId_, eidIndex, eidRaw, ctxHandle)) {
            SHM_LOG_ERROR("RaCtxInit failed for peer " << peer << " with EID index " << eidIndex);
            return false;
        }
        ctxHandleMap_[peer] = ctxHandle;
    }

    std::vector<int32_t> allRouteByPeer(rankCount * rankCount, INVALID_EID_INDEX);
    g_boot_handle.allgather(
        localRouteByPeer.data(), allRouteByPeer.data(), sizeof(int32_t) * rankCount, &g_boot_handle);

    for (uint32_t peer = 0; peer < rankCount; ++peer) {
        if (peer == rankId_) {
            continue;
        }
        int32_t remoteRoute = allRouteByPeer[peer * rankCount + rankId_];
        if (remoteRoute < 0 || static_cast<uint32_t>(remoteRoute) >= eidCount_) {
            SHM_LOG_ERROR(
                "Invalid remote EID route for peer rank "
                << peer << ", remoteRoute = " << remoteRoute << ", local rank = " << rankId_ << ", localId = "
                << localId << ", peerLocalId = " << localIdList[peer] << ", eidSlotCount = " << eidCount_);
            return false;
        }
        peerRemoteEidIndexMap_[peer] = static_cast<uint32_t>(remoteRoute);
    }

    return true;
}

bool UdmaTransportManager::OpenTsd(uint32_t deviceId, uint32_t rankCount)
{
    if (tsdOpened_) {
        SHM_LOG_INFO("Tsd already opened.");
        return true;
    }
    struct ProcOpenArgs args = {};
    args.procType = TSD_SUB_PROC_HCCP;
    args.filePath = nullptr;
    char paramInfo0[20] = "--hdcType=18";
    ProcExtParam extParam;
    extParam.paramInfo = paramInfo0;
    extParam.paramLen = sizeof("--hdcType=18");
    args.extParamList = &extParam;
    args.extParamCnt = 1UL;
    args.pathLen = 0UL;
    int subPid = 0;
    args.subPid = &subPid;
    auto ret = shm::DlHccpV2Api::TsdProcessOpen(deviceId, &args);
    if (ret != 0) {
        SHM_LOG_ERROR(
            "TsdProcessOpen failed, deviceId = " << deviceId << ", rankCount = " << rankCount << ", ret = " << ret);
        return false;
    }
    subPid_ = subPid;
    SHM_LOG_DEBUG("Open tsd for device id: " << deviceId << ", rank count: " << rankCount << " success.");
    tsdOpened_ = true;
    return true;
}

bool UdmaTransportManager::RaInit(uint32_t deviceId)
{
    if (raInitialized_) {
        SHM_LOG_INFO("Ra already initialized.");
        return true;
    }

    const std::chrono::seconds WAIT_TIME(3);
    RaInitConfig initConfig{};
    initConfig.phyId = deviceId;
    initConfig.nicPosition = NETWORK_OFFLINE;
    initConfig.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
    initConfig.enableHdcAsync = 1;

    SHM_LOG_DEBUG("RaInit = " << initConfig);
    std::this_thread::sleep_for(WAIT_TIME); // avoid hccl init conflict
    auto ret = shm::DlHccpV2Api::RaInit(&initConfig);
    if (ret != 0) {
        SHM_LOG_WARN("Hccp Init RA failed, ret = " << ret << ", device id: " << deviceId);
        // maybe hccl have already initialized ra, wait 3s then return true.
        std::this_thread::sleep_for(WAIT_TIME);
        raInitialized_ = true;
        return true;
    }

    SHM_LOG_DEBUG("RaInit for device id: " << deviceId << " success.");
    raInitialized_ = true;
    return true;
}

bool UdmaTransportManager::RaGetDevEidInfoNum(uint32_t deviceId, unsigned int& num)
{
    struct RaInfo info = {NETWORK_PEER_ONLINE, 0};
    info.phyId = deviceId;
    info.mode = NETWORK_OFFLINE;
    SHM_LOG_DEBUG("RaInfo=" << info);
    auto ret = shm::DlHccpV2Api::RaGetDevEidInfoNum(info, &num);
    if (ret != 0) {
        SHM_LOG_ERROR("RaGetDevEidInfoNum failed, ret = " << ret);
        return false;
    }
    SHM_LOG_DEBUG("RaGetDevEidInfoNum = " << num);
    return true;
}

bool UdmaTransportManager::RaGetDevEidInfoList(
    uint32_t deviceId, unsigned int eidNum, const std::array<uint8_t, 16>& targetEidRaw, struct CtxInitAttr* attr)
{
    struct RaInfo info = {NETWORK_PEER_ONLINE, 0};
    info.phyId = deviceId;
    info.mode = NETWORK_OFFLINE;
    SHM_LOG_DEBUG("RaInfo=" << info);
    unsigned int infoListNum = eidNum;
    std::vector<DevEidInfo> infoList(eidNum);
    int ret = shm::DlHccpV2Api::RaGetDevEidInfoList(info, infoList.data(), &infoListNum);
    if (ret != 0 || infoListNum != eidNum) {
        SHM_LOG_ERROR(
            "Get eid information failed, ret = " << ret << ", expected eidNum = " << eidNum
                                                 << ", actual eidNum = " << infoListNum);
        return false;
    }
    SHM_LOG_INFO("Get eid information success.");

    for (unsigned int index = 0; index < infoListNum; ++index) {
        if (std::memcmp(infoList[index].eid.raw, targetEidRaw.data(), targetEidRaw.size()) != 0) {
            continue;
        }
        attr->ub.eid = infoList[index].eid;
        attr->ub.eidIndex = infoList[index].eidIndex;
        SHM_LOG_INFO("Matched eid raw for route eidIndex " << attr->ub.eidIndex);
        return true;
    }

    SHM_LOG_ERROR("Failed to match target eid raw in device eid info list.");
    return false;
}

bool UdmaTransportManager::RaCtxTokenId(void* ctxHandle, void*& tokenIdHandle)
{
    struct HccpTokenId tokenId = {0};
    auto ret = shm::DlHccpV2Api::RaCtxTokenIdAlloc(ctxHandle, &tokenId, &tokenIdHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("RaCtxTokenIdAlloc failed, ret = " << ret);
        return false;
    }
    SHM_LOG_DEBUG("RaCtxTokenIdAlloc success, tokenId = " << tokenId.tokenId);
    return true;
}

bool UdmaTransportManager::RaCtxInit(
    uint32_t deviceId, uint32_t eidIndex, const std::array<uint8_t, 16>& targetEidRaw, void*& ctxHandle)
{
    auto it = storedCtxHandleMap_.find(eidIndex);
    if (it != storedCtxHandleMap_.end()) {
        SHM_LOG_INFO("The ctxHandle has already been initialized for EID index " << eidIndex);
        ctxHandle = it->second;
        return true;
    }

    unsigned int eidNum = 0;
    if (!RaGetDevEidInfoNum(deviceId, eidNum)) {
        return false;
    }
    struct CtxInitAttr attr = {};
    attr.phyId = deviceId;
    struct CtxInitCfg cfg = {};
    cfg.mode = NETWORK_OFFLINE;
    if (!RaGetDevEidInfoList(deviceId, eidNum, targetEidRaw, &attr)) {
        return false;
    }
    hccpEidMap_[eidIndex] = attr.ub.eid;
    SHM_LOG_DEBUG("CtxInitAttr = " << attr);
    auto ret = shm::DlHccpV2Api::RaCtxInit(&cfg, &attr, &ctxHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("RaCtxInit failed, ret = " << ret);
        return false;
    }
    void* tokenIdHandle = nullptr;
    if (!RaCtxTokenId(ctxHandle, tokenIdHandle)) {
        return false;
    }
    tokenIdHandleMap_[eidIndex] = tokenIdHandle;
    storedCtxHandleMap_[eidIndex] = ctxHandle;
    SHM_LOG_INFO("RaCtxInit succeeded for EID index " << eidIndex);
    return true;
}

Result UdmaTransportManager::CheckPrepareOptions(const HybmTransPrepareOptions& options)
{
    if (role_ != HYBM_ROLE_PEER) {
        SHM_LOG_INFO("Transport role: " << role_ << " check options passed.");
        return ACLSHMEM_SUCCESS;
    }

    if (options.options.size() > rankCount_) {
        SHM_LOG_ERROR("Options size() is " << options.options.size() << " larger than rank count: " << rankCount_);
        return ACLSHMEM_INVALID_PARAM;
    }

    if (options.options.find(rankId_) == options.options.end()) {
        SHM_LOG_ERROR("Options do not contain self rankId: " << rankId_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = options.options.begin(); it != options.options.end(); ++it) {
        if (it->first >= rankCount_) {
            SHM_LOG_ERROR("RankId: " << it->first << " is out of range [0, " << rankCount_ << ")");
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    return ACLSHMEM_SUCCESS;
}

void UdmaTransportManager::CleanupResources()
{
    // Unimport remote handle
    if (memoryHandleList_.size() > 0) {
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            void* ctxHandle = ctxHandleMap_.count(peer) ? ctxHandleMap_.at(peer) : nullptr;
            if (ctxHandle == nullptr || memoryHandleList_[peer] == nullptr) {
                continue;
            }
            auto ret = shm::DlHccpV2Api::RaCtxRmemUnimport(ctxHandle, memoryHandleList_[peer]);
            if (ret != 0) {
                SHM_LOG_WARN("Rmem unimport failed, rankId = " << peer << ", ret = " << ret);
            }
            memoryHandleList_[peer] = nullptr;
        }
        SHM_LOG_INFO("Unimport remote handle success.");
    }

    // Unregister lmem on each EID
    for (const auto& mrEntry : registerMRS_) {
        for (const auto& eidMrEntry : mrEntry.second) {
            uint32_t eidIndex = eidMrEntry.first;
            void* lmemHandle = eidMrEntry.second.lmemHandle;
            auto ctxIt = storedCtxHandleMap_.find(eidIndex);
            if (ctxIt == storedCtxHandleMap_.end() || lmemHandle == nullptr) {
                continue;
            }
            auto ret = shm::DlHccpV2Api::RaCtxLmemUnregister(ctxIt->second, lmemHandle);
            if (ret != 0) {
                SHM_LOG_WARN("Failed to unregister the memory region for EID index " << eidIndex << ", ret = " << ret);
            }
        }
    }
    registerMRS_.clear();
    lmemHandleMap_.clear();
    localMrMap_.clear();
    localMemInfoMap_.clear();
    SHM_LOG_INFO("Unregister memory success.");

    // Free tokenIdHandle
    for (auto& eidTokenEntry : tokenIdHandleMap_) {
        uint32_t eidIndex = eidTokenEntry.first;
        void* tokenIdHandle = eidTokenEntry.second;
        auto ctxIt = storedCtxHandleMap_.find(eidIndex);
        if (ctxIt == storedCtxHandleMap_.end() || tokenIdHandle == nullptr) {
            continue;
        }
        auto ret = shm::DlHccpV2Api::RaCtxTokenIdFree(ctxIt->second, tokenIdHandle);
        if (ret != 0) {
            SHM_LOG_WARN("RaCtxTokenIdFree failed for EID index " << eidIndex << ", ret = " << ret);
        }
    }
    tokenIdHandleMap_.clear();
    SHM_LOG_INFO("RaCtxTokenIdFree success.");

    // Deinit ctx on each EID
    for (auto& ctxEntry : storedCtxHandleMap_) {
        uint32_t eidIndex = ctxEntry.first;
        void* ctxHandle = ctxEntry.second;
        if (ctxHandle == nullptr) {
            continue;
        }
        auto ret = shm::DlHccpV2Api::RaCtxDeinit(ctxHandle);
        if (ret != 0) {
            SHM_LOG_WARN("RaCtxDeinit failed for EID index " << eidIndex << ", ret = " << ret);
        }
    }
    SHM_LOG_INFO("RaCtxDeinit success.");

    if (raInitialized_) {
        RaInitConfig deinitConfig{};
        deinitConfig.phyId = phyId_;
        deinitConfig.nicPosition = NETWORK_OFFLINE;
        deinitConfig.hdcType = HDC_SERVICE_TYPE_RDMA_V2;
        deinitConfig.enableHdcAsync = 1;
        auto ret = shm::DlHccpV2Api::RaDeinit(&deinitConfig);
        if (ret != 0) {
            SHM_LOG_WARN("RaDeinit failed, ret = " << ret << ", phy id: " << phyId_);
        }
        SHM_LOG_INFO("RaDeinit success.");
    }

    if (tsdOpened_ && subPid_ > 0) {
        auto ret = shm::DlHccpV2Api::TsdProcessClose(deviceId_, subPid_);
        if (ret != 0) {
            SHM_LOG_WARN(
                "TsdProcessClose failed, device id: " << deviceId_ << ", subPid: " << subPid_ << ", ret = " << ret);
        }
        SHM_LOG_INFO("TsdProcessClose success.");
    }

    storedCtxHandleMap_.clear();
    ctxHandleMap_.clear();
    peerEidIndexMap_.clear();
    peerRemoteEidIndexMap_.clear();
    hccpEidMap_.clear();
    memoryHandleList_.clear();
    lmemHandleMap_.clear();
}

} // namespace device
} // namespace transport
} // namespace shm
