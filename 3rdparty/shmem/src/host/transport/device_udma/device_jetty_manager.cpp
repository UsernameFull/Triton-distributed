/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <acl/acl.h>
#include <cmath>
#include <cstring>
#include <set>
#include <vector>

#include "host/shmem_host_def.h"
#include "../host_device/shmemi_host_device_constant.h"
#include "runtime/mem.h"
#include "shmemi_host_common.h"
#include "device_jetty_manager.h"

constexpr uint8_t RNR_RETRY_COUNT_DEFAULT = 7;

namespace shm {
namespace transport {
namespace device {
DeviceJettyManager::DeviceJettyManager(
    uint32_t deviceId, uint32_t rankId, uint32_t rankCount, uint32_t eidSlotCount) noexcept
    : deviceId_{deviceId}, rankId_{rankId}, rankCount_{rankCount}, eidCount_{eidSlotCount}
{}

DeviceJettyManager::~DeviceJettyManager() noexcept { Shutdown(); }

Result DeviceJettyManager::SetCtxHandles(const std::map<uint32_t, void*>& ctxHandleMap) noexcept
{
    ctxHandleMap_ = ctxHandleMap;
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::SetLocalMemInfos(const std::map<uint32_t, ACLSHMEMUBmemInfo>& localMemInfoMap) noexcept
{
    localMemInfoMap_ = localMemInfoMap;
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::SetEids(const std::map<uint32_t, HccpEid>& hccpEidMap) noexcept
{
    localHccpEidMap_ = hccpEidMap;
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::SetTokenIdHandles(const std::map<uint32_t, void*>& tokenIdHandleMap) noexcept
{
    tokenIdHandleMap_ = tokenIdHandleMap;
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::SetPeerRoutes(
    const std::map<uint32_t, uint32_t>& peerLocalEidMap, const std::map<uint32_t, uint32_t>& peerRemoteEidMap) noexcept
{
    peerLocalEidMap_ = peerLocalEidMap;
    peerRemoteEidMap_ = peerRemoteEidMap;
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::Shutdown() noexcept
{
    int ret = 0;
    for (auto& stateEntry : jettyStateMap_) {
        auto& state = stateEntry.second;
        if (transportMode_ != TransportModeT::CONN_RM && state.qpHandle != nullptr) {
            ret = DlHccpV2Api::RaCtxQpUnbind(state.qpHandle);
            if (ret != 0) {
                SHM_LOG_WARN("Qp unbind failed, eidIndex = " << state.eidIndex << ", ret = " << ret);
            }
        }

        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_ || state.remoteQpHandleList.empty() || state.remoteQpHandleList[peer] == nullptr) {
                continue;
            }
            ret = DlHccpV2Api::RaCtxQpUnimport(state.ctxHandle, state.remoteQpHandleList[peer]);
            if (ret != 0) {
                SHM_LOG_WARN(
                    "Qp unimport failed, eidIndex: " << state.eidIndex << ", rankId: " << peer << ", ret: " << ret);
            }
            state.remoteQpHandleList[peer] = nullptr;
        }

        if (state.qpHandle != nullptr) {
            ret = DlHccpV2Api::RaCtxQpDestroy(state.qpHandle);
            if (ret != 0) {
                SHM_LOG_WARN("Qp destroy failed, eidIndex = " << state.eidIndex << ", ret = " << ret);
            }
            state.qpHandle = nullptr;
        }

        if (state.cqHandle != nullptr) {
            ret = DlHccpV2Api::RaCtxCqDestroy(state.ctxHandle, state.cqHandle);
            if (ret != 0) {
                SHM_LOG_WARN("Cq destroy failed, eidIndex = " << state.eidIndex << ", ret = " << ret);
            }
            state.cqHandle = nullptr;
        }

        if (state.chanHandle != nullptr) {
            ret = DlHccpV2Api::RaCtxChanDestroy(state.ctxHandle, state.chanHandle);
            if (ret != 0) {
                SHM_LOG_WARN("Channel destroy failed, eidIndex = " << state.eidIndex << ", ret = " << ret);
            }
            state.chanHandle = nullptr;
        }

        if (state.cqPiAddr != nullptr) {
            aclrtFree(state.cqPiAddr);
            state.cqPiAddr = nullptr;
        }
        if (state.cqCiAddr != nullptr) {
            aclrtFree(state.cqCiAddr);
            state.cqCiAddr = nullptr;
        }
        if (state.sqPiAddr != nullptr) {
            aclrtFree(state.sqPiAddr);
            state.sqPiAddr = nullptr;
        }
        if (state.sqCiAddr != nullptr) {
            aclrtFree(state.sqCiAddr);
            state.sqCiAddr = nullptr;
        }
        if (state.wqeCntAddr != nullptr) {
            aclrtFree(state.wqeCntAddr);
            state.wqeCntAddr = nullptr;
        }
        if (state.amoAddr != nullptr) {
            aclrtFree(state.amoAddr);
            state.amoAddr = nullptr;
        }
    }
    jettyStateMap_.clear();

    if (udmaInfo_ != nullptr) {
        aclrtFree(udmaInfo_);
        udmaInfo_ = nullptr;
    }
    if (hccpEidDevice_ != nullptr) {
        aclrtFree(hccpEidDevice_);
        hccpEidDevice_ = nullptr;
    }
    return ACLSHMEM_SUCCESS;
}

bool DeviceJettyManager::ReserveUdmaInfoSpace() noexcept
{
    if (udmaInfo_ != nullptr) {
        return true;
    }

    constexpr uint32_t qpNum = 1;
    auto wqSize = sizeof(ACLSHMEMUDMAWQCtx) * qpNum;
    auto cqSize = sizeof(ACLSHMEMUDMACqCtx) * qpNum;
    auto oneQpSize = 2U * (wqSize + cqSize) + sizeof(ACLSHMEMUBmemInfo) * qpNum; // 2 means (sq + rq) (scq + rcq)
    udmaInfoSize_ = sizeof(ACLSHMEMAIVUDMAInfo) + oneQpSize * rankCount_;

    SHM_VALIDATE_RETURN(
        aclrtMalloc(&udmaInfo_, udmaInfoSize_, ACL_MEM_MALLOC_HUGE_FIRST) == 0,
        "Allocate device size: " << udmaInfoSize_ << " for udmaInfo failed", false);
    SHM_VALIDATE_RETURN(
        aclrtMalloc(&hccpEidDevice_, rankCount_ * eidCount_ * sizeof(HccpEid), ACL_MEM_MALLOC_HUGE_FIRST) == 0,
        "Allocate device size for eid table failed", false);
    return true;
}

std::vector<uint32_t> DeviceJettyManager::CollectUsedLocalEids() const noexcept
{
    std::set<uint32_t> eidSet;
    for (const auto& routeEntry : peerLocalEidMap_) {
        if (routeEntry.first == rankId_) {
            continue;
        }
        eidSet.insert(routeEntry.second);
    }
    return std::vector<uint32_t>(eidSet.begin(), eidSet.end());
}

uint32_t DeviceJettyManager::GetFallbackLocalEid() const noexcept
{
    if (!peerLocalEidMap_.empty()) {
        uint32_t fallbackEid = peerLocalEidMap_.begin()->second;
        SHM_LOG_INFO("Select fallback local EID from peer route map: " << fallbackEid);
        return fallbackEid;
    }
    if (!ctxHandleMap_.empty()) {
        uint32_t fallbackEid = ctxHandleMap_.begin()->first;
        SHM_LOG_INFO("Select fallback local EID from ctx handle map: " << fallbackEid);
        return fallbackEid;
    }
    SHM_LOG_WARN("Select fallback local EID defaulting to 0 because no peer route or ctx handle is available.");
    return 0;
}

HccpEid DeviceJettyManager::ToImportedEid(const HccpEid& hccpEid) const noexcept
{
    // after import jetty, eid should be __builtin_bswap64
    HccpEid swapped{};
    uint64_t eidL = 0;
    uint64_t eidH = 0;
    std::memcpy(&eidL, hccpEid.raw, sizeof(uint64_t));
    std::memcpy(&eidH, hccpEid.raw + sizeof(uint64_t), sizeof(uint64_t));
    eidL = __builtin_bswap64(eidL);
    eidH = __builtin_bswap64(eidH);
    std::memcpy(swapped.raw, &eidH, sizeof(uint64_t));
    std::memcpy(swapped.raw + sizeof(uint64_t), &eidL, sizeof(uint64_t));
    return swapped;
}

Result DeviceJettyManager::JFCCreate(PerEidJettyState& state) noexcept
{
    ChanInfoT chanInfo = {0};
    chanInfo.in.dataPlaneFlag.bs.poolCqCstm = 1; // default 0:hccp poll cq; 1: caller poll cq
    int ret = DlHccpV2Api::RaCtxChanCreate(state.ctxHandle, &chanInfo, &state.chanHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("Create udma channel failed: " << ret << ", eidIndex = " << state.eidIndex);
        return ACLSHMEM_INNER_ERROR;
    }

    state.cqInfo.in.chanHandle = state.chanHandle;
    state.cqInfo.in.depth = shm::UDMA_CQ_DEPTH_DEFAULT; // optional, normal mode default 16384
    state.cqInfo.in.ub.userCtx = 0;                     // optional, default 0
    state.cqInfo.in.ub.mode = JFC_MODE_USER_CTL_NORMAL; // corresponding with jetty mode : JETTY_MODE_USER_CTL_NORMAL
    state.cqInfo.in.ub.ceqn = 0;                        // optional, default 0
    state.cqInfo.in.ub.flag.bs.lockFree = 0;            // optional, default 0
    state.cqInfo.in.ub.flag.bs.jfcInline = 0;           // optional, default 0
    ret = DlHccpV2Api::RaCtxCqCreate(state.ctxHandle, &state.cqInfo, &state.cqHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("Create udma jfc create failed, ret = " << ret << ", eidIndex = " << state.eidIndex);
        return ACLSHMEM_INNER_ERROR;
    }
    state.cqVa = state.cqInfo.out.va;
    // save & allgather cq related info
    state.localCq.cqn = 0;
    state.localCq.bufAddr = state.cqInfo.out.bufAddr;
    state.localCq.cqeShiftSize = log2(state.cqInfo.out.cqeSize); // cqeSize = 64 = 2^6, cqeShiftSize此处取6
    state.localCq.depth = state.cqInfo.in.depth;
    aclrtMalloc(&state.cqPiAddr, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(state.cqPiAddr, sizeof(uint32_t), 0, sizeof(uint32_t));
    state.localCq.headAddr = reinterpret_cast<uintptr_t>(state.cqPiAddr);
    aclrtMalloc(&state.cqCiAddr, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(state.cqCiAddr, sizeof(uint32_t), 0, sizeof(uint32_t));
    state.localCq.tailAddr = reinterpret_cast<uintptr_t>(state.cqCiAddr);
    state.localCq.dbMode = ACLSHMEMUDMADBMode::SW_DB;
    state.localCq.dbAddr = state.cqInfo.out.swdbAddr;

    SHM_LOG_INFO("Cq create success, eidIndex = " << state.eidIndex);
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::JettyCreate(PerEidJettyState& state) noexcept
{
    QpCreateAttr qpCreateAttr = {0};
    qpCreateAttr.scqHandle = state.cqHandle;
    qpCreateAttr.rcqHandle = state.cqHandle;
    qpCreateAttr.srqHandle = state.cqHandle;
    qpCreateAttr.sqDepth = shm::UDMA_SQ_DEPTH_DEFAULT; // optional, default 4096
    qpCreateAttr.rqDepth = shm::UDMA_RQ_DEPTH_DEFAULT; // optional, default 256
    qpCreateAttr.transportMode = transportMode_;

    qpCreateAttr.ub.mode = JettyMode::JETTY_MODE_USER_CTL_NORMAL;
    qpCreateAttr.ub.jettyId = 0;       // 0 means not specified
    qpCreateAttr.ub.flag.value = 1;    // URMA_SHARE_JFR
    qpCreateAttr.ub.jfsFlag.value = 2; // 0b10
    /* default as 0, lock protected */
    /*  1: error suspend */
    qpCreateAttr.ub.tokenValue = TOKEN_VALUE;
    qpCreateAttr.ub.priority = 0;
    qpCreateAttr.ub.rnrRetry = RNR_RETRY_COUNT_DEFAULT;
    qpCreateAttr.ub.errTimeout = 0;

    qpCreateAttr.ub.extMode.piType = 0; // optional, default 0 op mode
    qpCreateAttr.ub.extMode.cstmFlag.bs.sqCstm =
        0; // optional, USER_CTL_NORMAL default is 0, sqbuff no need, others default 1
    qpCreateAttr.ub.extMode.sqebbNum = shm::UDMA_SQ_DEPTH_DEFAULT;
    qpCreateAttr.ub.tokenIdHandle = state.tokenIdHandle;

    int ret = DlHccpV2Api::RaCtxQpCreate(state.ctxHandle, &qpCreateAttr, &state.qpCreateInfo_, &state.qpHandle);
    if (ret != 0) {
        SHM_LOG_ERROR("Qp create failed, ret = " << ret << ", eidIndex = " << state.eidIndex);
        return ACLSHMEM_INNER_ERROR;
    }
    // save & allgather wq related info
    state.localWq.wqn = 0;
    state.localWq.bufAddr = state.qpCreateInfo_.ub.sqBuffVa;
    state.localWq.wqeShiftSize = log2(state.qpCreateInfo_.ub.wqebbSize); // wqeSize = 64 = 2^6, wqeShiftSize此处取6
    state.localWq.depth = shm::UDMA_SQ_BASKBLK_CNT;
    aclrtMalloc(&state.sqPiAddr, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(state.sqPiAddr, sizeof(uint32_t), 0, sizeof(uint32_t));
    state.localWq.headAddr = reinterpret_cast<uintptr_t>(state.sqPiAddr);
    aclrtMalloc(&state.sqCiAddr, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(state.sqCiAddr, sizeof(uint32_t), 0, sizeof(uint32_t));
    state.localWq.tailAddr = reinterpret_cast<uintptr_t>(state.sqCiAddr);
    state.localWq.dbMode = ACLSHMEMUDMADBMode::SW_DB;
    state.localWq.dbAddr = state.qpCreateInfo_.ub.dbAddr;
    state.localWq.sl = 0;
    aclrtMalloc(&state.wqeCntAddr, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(state.wqeCntAddr, sizeof(uint32_t), 0, sizeof(uint32_t));
    state.localWq.wqeCntAddr = reinterpret_cast<uintptr_t>(state.wqeCntAddr);
    aclrtMalloc(&state.amoAddr, sizeof(uint64_t), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemset(state.amoAddr, sizeof(uint64_t), 0, sizeof(uint64_t));
    state.localWq.amoAddr = reinterpret_cast<uintptr_t>(state.amoAddr);

    SHM_LOG_INFO("Qp create success, eidIndex = " << state.eidIndex);
    return ACLSHMEM_SUCCESS;
}

bool DeviceJettyManager::BuildLocalQpPublishByEid(
    std::vector<QpImportInfoT>& qpImportByEid, std::vector<QpKeyT>& qpKeyByEid) const noexcept
{
    qpImportByEid.assign(eidCount_, QpImportInfoT{});
    qpKeyByEid.assign(eidCount_, QpKeyT{});
    for (const auto& stateEntry : jettyStateMap_) {
        const auto& state = stateEntry.second;
        if (state.eidIndex >= eidCount_) {
            SHM_LOG_ERROR("EID index out of range when publishing qp info: " << state.eidIndex);
            return false;
        }
        qpImportByEid[state.eidIndex].in.ub.mode = JettyImportMode::JETTY_IMPORT_MODE_NORMAL;
        qpImportByEid[state.eidIndex].in.ub.tokenValue = TOKEN_VALUE; // same as qpCreateattr.ub.tokenValue
        qpImportByEid[state.eidIndex].in.ub.policy = JettyGrpPolicy::JETTY_GRP_POLICY_RR;
        qpImportByEid[state.eidIndex].in.ub.type = TargetType::TARGET_TYPE_JETTY;
        qpImportByEid[state.eidIndex].in.ub.flag.bs.tokenPolicy = TokenPolicy::TOKEN_POLICY_PLAIN_TEXT;
        qpImportByEid[state.eidIndex].in.ub.tpType = 1; // mode ctp
        qpKeyByEid[state.eidIndex] = state.qpCreateInfo_.key;
    }
    return true;
}

Result DeviceJettyManager::JettyImport() noexcept
{
    std::vector<QpImportInfoT> localQpImportByEid;
    std::vector<QpKeyT> localQpKeyByEid;
    SHM_VALIDATE_RETURN(
        BuildLocalQpPublishByEid(localQpImportByEid, localQpKeyByEid), "Build local qp publish info failed.",
        ACLSHMEM_INNER_ERROR);

    std::vector<QpImportInfoT> allQpImportByEid(rankCount_ * eidCount_);
    std::vector<QpKeyT> allQpKeyByEid(rankCount_ * eidCount_);
    g_boot_handle.allgather(
        localQpImportByEid.data(), allQpImportByEid.data(), sizeof(QpImportInfoT) * eidCount_, &g_boot_handle);
    g_boot_handle.allgather(localQpKeyByEid.data(), allQpKeyByEid.data(), sizeof(QpKeyT) * eidCount_, &g_boot_handle);

    for (auto& stateEntry : jettyStateMap_) {
        auto& state = stateEntry.second;
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            auto localRouteIt = peerLocalEidMap_.find(peer);
            if (localRouteIt == peerLocalEidMap_.end() || localRouteIt->second != state.eidIndex) {
                continue;
            }
            auto remoteRouteIt = peerRemoteEidMap_.find(peer);
            if (remoteRouteIt == peerRemoteEidMap_.end()) {
                SHM_LOG_ERROR("Missing remote route for peer " << peer);
                return ACLSHMEM_INNER_ERROR;
            }
            uint32_t remoteEid = remoteRouteIt->second;
            if (remoteEid >= eidCount_) {
                SHM_LOG_ERROR("Remote EID index out of range for peer " << peer << ": " << remoteEid);
                return ACLSHMEM_INNER_ERROR;
            }

            QpImportInfoT qpImportInfo = allQpImportByEid[peer * eidCount_ + remoteEid];
            qpImportInfo.in.key = allQpKeyByEid[peer * eidCount_ + remoteEid];
            int ret = DlHccpV2Api::RaCtxQpImport(state.ctxHandle, &qpImportInfo, &state.remoteQpHandleList[peer]);
            if (ret != 0) {
                SHM_LOG_ERROR(
                    "Qp import failed, eidIndex: " << state.eidIndex << " rankId: " << peer
                                                   << " remoteEid: " << remoteEid << " ret: " << ret);
                return ACLSHMEM_INNER_ERROR;
            }
            state.tpnList[peer] = qpImportInfo.out.ub.tpn;
        }
    }

    SHM_LOG_INFO("Qp import success");
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::JettyBind() noexcept
{
    if (transportMode_ == TransportModeT::CONN_RM) {
        return ACLSHMEM_SUCCESS; // no need to bind in RM mode
    }
    for (auto& stateEntry : jettyStateMap_) {
        auto& state = stateEntry.second;
        for (uint32_t peer = 0; peer < rankCount_; ++peer) {
            if (peer == rankId_) {
                continue;
            }
            auto localRouteIt = peerLocalEidMap_.find(peer);
            if (localRouteIt == peerLocalEidMap_.end() || localRouteIt->second != state.eidIndex) {
                continue;
            }
            int ret = DlHccpV2Api::RaCtxQpBind(state.qpHandle, state.remoteQpHandleList[peer]);
            if (ret != 0) {
                SHM_LOG_ERROR("Qp bind failed, eidIndex: " << state.eidIndex << " rankId: " << peer << " ret: " << ret);
                return ACLSHMEM_INNER_ERROR;
            }
        }
    }
    SHM_LOG_INFO("Qp bind success.");
    return ACLSHMEM_SUCCESS;
}

Result DeviceJettyManager::Startup() noexcept
{
    if (!ReserveUdmaInfoSpace()) {
        SHM_LOG_ERROR("Reserve UDMA info space failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    for (uint32_t eidIndex : CollectUsedLocalEids()) {
        auto ctxIt = ctxHandleMap_.find(eidIndex);
        auto tokenIt = tokenIdHandleMap_.find(eidIndex);
        if (ctxIt == ctxHandleMap_.end() || tokenIt == tokenIdHandleMap_.end()) {
            SHM_LOG_ERROR("Missing ctxHandle or tokenIdHandle for EID index " << eidIndex);
            return ACLSHMEM_INNER_ERROR;
        }

        auto& state = jettyStateMap_[eidIndex];
        state.eidIndex = eidIndex;
        state.ctxHandle = ctxIt->second;
        state.tokenIdHandle = tokenIt->second;
        state.remoteQpHandleList.assign(rankCount_, nullptr);
        state.tpnList.assign(rankCount_, 0);

        SHM_VALIDATE_RETURN(JFCCreate(state) == 0, "Create JFC failed.", ACLSHMEM_INNER_ERROR);
        SHM_VALIDATE_RETURN(JettyCreate(state) == 0, "Create Jetty failed.", ACLSHMEM_INNER_ERROR);
    }

    if (jettyStateMap_.empty()) {
        SHM_LOG_ERROR("No jetty state was created. Check peer EID route initialization before startup.");
        return ACLSHMEM_INNER_ERROR;
    }

    SHM_VALIDATE_RETURN(JettyImport() == 0, "Jetty import failed.", ACLSHMEM_INNER_ERROR);
    SHM_VALIDATE_RETURN(JettyBind() == 0, "Jetty bind failed.", ACLSHMEM_INNER_ERROR);
    SHM_VALIDATE_RETURN(FillUdmaInfo() == ACLSHMEM_SUCCESS, "Fill udma info failed.", ACLSHMEM_INNER_ERROR);
    return ACLSHMEM_SUCCESS;
}

void* DeviceJettyManager::GetJettyInfoAddress() noexcept { return udmaInfo_; }

uint64_t DeviceJettyManager::GetJFCInfoAddress() const noexcept
{
    if (jettyStateMap_.empty()) {
        SHM_LOG_WARN("GetJFCInfoAddress returns 0 because jettyStateMap_ is empty.");
        return 0;
    }
    return jettyStateMap_.begin()->second.cqVa;
}

void DeviceJettyManager::FillUdmaWq(ACLSHMEMUDMAWQCtx& srcWq, ACLSHMEMUDMAWQCtx& dstWq) const
{
    dstWq.wqn = srcWq.wqn;
    dstWq.bufAddr = srcWq.bufAddr;
    dstWq.wqeShiftSize = srcWq.wqeShiftSize;
    dstWq.depth = srcWq.depth;
    dstWq.headAddr = srcWq.headAddr;
    dstWq.tailAddr = srcWq.tailAddr;
    dstWq.dbMode = srcWq.dbMode;
    dstWq.dbAddr = srcWq.dbAddr;
    dstWq.sl = srcWq.sl;
    dstWq.wqeCntAddr = srcWq.wqeCntAddr;
    dstWq.amoAddr = srcWq.amoAddr;
}

void DeviceJettyManager::FillUdmaCq(ACLSHMEMUDMACqCtx& srcCq, ACLSHMEMUDMACqCtx& dstCq) const
{
    dstCq.cqn = srcCq.cqn;
    dstCq.bufAddr = srcCq.bufAddr;
    dstCq.cqeShiftSize = srcCq.cqeShiftSize;
    dstCq.depth = srcCq.depth;
    dstCq.headAddr = srcCq.headAddr;
    dstCq.tailAddr = srcCq.tailAddr;
    dstCq.dbMode = srcCq.dbMode;
    dstCq.dbAddr = srcCq.dbAddr;
}

void DeviceJettyManager::FillUdmaMem(ACLSHMEMUBmemInfo& srcMem, ACLSHMEMUBmemInfo& dstMem) const
{
    dstMem.token_value_valid = srcMem.token_value_valid;
    dstMem.rmt_jetty_type = srcMem.rmt_jetty_type;
    dstMem.target_hint = srcMem.target_hint;
    dstMem.tpn = srcMem.tpn;
    dstMem.tid = srcMem.tid;
    dstMem.rmt_token_value = srcMem.rmt_token_value;
    dstMem.len = srcMem.len;
    dstMem.addr = srcMem.addr;
}

void DeviceJettyManager::PrintHostInfo(ACLSHMEMAIVUDMAInfo& hostInfo) const
{
    SHM_LOG_DEBUG("=======================rank [" << rankId_ << "] host info====================");
    auto tempWQCtx = ((ACLSHMEMUDMAWQCtx*)hostInfo.sqPtr)[rankId_];
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.wqn: " << tempWQCtx.wqn);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.bufAddr: " << tempWQCtx.bufAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.wqeShiftSize: " << tempWQCtx.wqeShiftSize);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.depth: " << tempWQCtx.depth);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.headAddr: " << tempWQCtx.headAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.tailAddr: " << tempWQCtx.tailAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.dbMode: " << static_cast<int>(tempWQCtx.dbMode));
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.dbAddr: " << tempWQCtx.dbAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.sl: " << tempWQCtx.sl);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] WQCtx.wqeCntAddr: " << tempWQCtx.wqeCntAddr);

    auto tempCQCtx = ((ACLSHMEMUDMACqCtx*)hostInfo.scqPtr)[rankId_];
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.cqn: " << tempCQCtx.cqn);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.bufAddr: " << tempCQCtx.bufAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.cqeShiftSize: " << tempCQCtx.cqeShiftSize);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.depth: " << tempCQCtx.depth);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.headAddr: " << tempCQCtx.headAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.tailAddr: " << tempCQCtx.tailAddr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.dbMode: " << static_cast<int>(tempCQCtx.dbMode));
    SHM_LOG_DEBUG("rank[" << rankId_ << "] CQCtx.dbAddr: " << tempCQCtx.dbAddr);

    auto tempMemInfo = ((ACLSHMEMUBmemInfo*)hostInfo.memPtr)[rankId_];
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.token_value_valid: " << tempMemInfo.token_value_valid);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.rmt_jetty_type: " << tempMemInfo.rmt_jetty_type);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.target_hint: " << tempMemInfo.target_hint);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.tpn: " << tempMemInfo.tpn);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.tid: " << tempMemInfo.tid);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.rmt_token_value: " << tempMemInfo.rmt_token_value);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.len: " << tempMemInfo.len);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.addr: " << tempMemInfo.addr);
    SHM_LOG_DEBUG("rank[" << rankId_ << "] MemInfo.eidAddr: " << tempMemInfo.eidAddr);

    // eidAddr points to device memory, so only log the pointer here.
}

Result DeviceJettyManager::FillUdmaInfo() noexcept
{
    std::vector<ACLSHMEMUBmemInfo> localMemByEid(eidCount_);
    for (const auto& memEntry : localMemInfoMap_) {
        if (memEntry.first >= eidCount_) {
            SHM_LOG_ERROR("Local mem EID index out of range: " << memEntry.first);
            return ACLSHMEM_INNER_ERROR;
        }
        localMemByEid[memEntry.first] = memEntry.second;
    }

    std::vector<HccpEid> localEidByEid(eidCount_);
    for (const auto& eidEntry : localHccpEidMap_) {
        if (eidEntry.first >= eidCount_) {
            SHM_LOG_ERROR("Local HCCP EID index out of range: " << eidEntry.first);
            return ACLSHMEM_INNER_ERROR;
        }
        localEidByEid[eidEntry.first] = ToImportedEid(eidEntry.second);
    }

    std::vector<ACLSHMEMUBmemInfo> allMemByEid(rankCount_ * eidCount_);
    std::vector<HccpEid> allEidByEid(rankCount_ * eidCount_);
    g_boot_handle.allgather(
        localMemByEid.data(), allMemByEid.data(), sizeof(ACLSHMEMUBmemInfo) * eidCount_, &g_boot_handle);
    g_boot_handle.allgather(localEidByEid.data(), allEidByEid.data(), sizeof(HccpEid) * eidCount_, &g_boot_handle);
    g_boot_handle.barrier(&g_boot_handle);

    auto ret = aclrtMemcpy(
        hccpEidDevice_, rankCount_ * eidCount_ * sizeof(HccpEid), allEidByEid.data(),
        rankCount_ * eidCount_ * sizeof(HccpEid), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("Copy eid info to device failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    // construct udma info in host
    constexpr uint32_t qpNum = 1;
    std::vector<uint8_t> udmaInfoBuffer(udmaInfoSize_, 0);
    auto copyInfo = reinterpret_cast<ACLSHMEMAIVUDMAInfo*>(udmaInfoBuffer.data());
    copyInfo->qpNum = qpNum;
    copyInfo->sqPtr = (uint64_t)(copyInfo + 1);
    copyInfo->rqPtr = (uint64_t)((ACLSHMEMUDMAWQCtx*)copyInfo->sqPtr + rankCount_ * qpNum);
    copyInfo->scqPtr = (uint64_t)((ACLSHMEMUDMAWQCtx*)copyInfo->rqPtr + rankCount_ * qpNum);
    copyInfo->rcqPtr = (uint64_t)((ACLSHMEMUDMACqCtx*)copyInfo->scqPtr + rankCount_ * qpNum);
    copyInfo->memPtr = (uint64_t)((ACLSHMEMUDMACqCtx*)copyInfo->rcqPtr + rankCount_ * qpNum);

    uint32_t fallbackLocalEid = GetFallbackLocalEid();
    const auto fallbackStateIt = jettyStateMap_.find(fallbackLocalEid);
    for (uint32_t rank = 0; rank < rankCount_; ++rank) {
        uint32_t localEid = fallbackLocalEid;
        uint32_t remoteEid = fallbackLocalEid;
        if (rank != rankId_) {
            auto localRouteIt = peerLocalEidMap_.find(rank);
            auto remoteRouteIt = peerRemoteEidMap_.find(rank);
            if (localRouteIt == peerLocalEidMap_.end() || remoteRouteIt == peerRemoteEidMap_.end()) {
                SHM_LOG_ERROR("Missing route for peer rank " << rank);
                return ACLSHMEM_INNER_ERROR;
            }
            localEid = localRouteIt->second;
            remoteEid = remoteRouteIt->second;
        }

        auto stateIt = jettyStateMap_.find(localEid);
        if (stateIt == jettyStateMap_.end()) {
            if (fallbackStateIt == jettyStateMap_.end()) {
                SHM_LOG_ERROR("Missing local jetty state for EID index " << localEid);
                return ACLSHMEM_INNER_ERROR;
            }
            stateIt = fallbackStateIt;
        }
        auto& state = stateIt->second;

        FillUdmaWq(state.localWq, ((ACLSHMEMUDMAWQCtx*)copyInfo->sqPtr)[rank]);
        FillUdmaWq(state.localWq, ((ACLSHMEMUDMAWQCtx*)copyInfo->rqPtr)[rank]);
        FillUdmaCq(state.localCq, ((ACLSHMEMUDMACqCtx*)copyInfo->scqPtr)[rank]);
        FillUdmaCq(state.localCq, ((ACLSHMEMUDMACqCtx*)copyInfo->rcqPtr)[rank]);

        ACLSHMEMUBmemInfo memInfo{};
        if (rank == rankId_) {
            auto localMemIt = localMemInfoMap_.find(localEid);
            if (localMemIt != localMemInfoMap_.end()) {
                memInfo = localMemIt->second;
            }
        } else {
            memInfo = allMemByEid[rank * eidCount_ + remoteEid];
            memInfo.tpn = state.tpnList[rank];
        }
        FillUdmaMem(memInfo, ((ACLSHMEMUBmemInfo*)copyInfo->memPtr)[rank]);
        ((ACLSHMEMUBmemInfo*)copyInfo->memPtr)[rank].eidAddr =
            (uint64_t)((HccpEid*)hccpEidDevice_ + rank * eidCount_ + remoteEid);
    }
    PrintHostInfo(*copyInfo);
    // link position in device
    copyInfo->sqPtr = (uint64_t)((ACLSHMEMAIVUDMAInfo*)udmaInfo_ + 1);
    copyInfo->rqPtr = (uint64_t)((ACLSHMEMUDMAWQCtx*)copyInfo->sqPtr + rankCount_ * qpNum);
    copyInfo->scqPtr = (uint64_t)((ACLSHMEMUDMAWQCtx*)copyInfo->rqPtr + rankCount_ * qpNum);
    copyInfo->rcqPtr = (uint64_t)((ACLSHMEMUDMACqCtx*)copyInfo->scqPtr + rankCount_ * qpNum);
    copyInfo->memPtr = (uint64_t)((ACLSHMEMUDMACqCtx*)copyInfo->rcqPtr + rankCount_ * qpNum);
    ret = aclrtMemcpy(udmaInfo_, udmaInfoSize_, copyInfo, udmaInfoSize_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("Copy udma info to device failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("Copy udma info success");

    return ACLSHMEM_SUCCESS;
}

} // namespace device
} // namespace transport
} // namespace shm
