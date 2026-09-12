/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_JETTY_MANAGER_H
#define MF_HYBRID_DEVICE_JETTY_MANAGER_H

#include <netinet/in.h>
#include <cstdint>
#include <map>
#include <memory>
#include <vector>

#include "hybm_def.h"
#include "dl_hccp_v2_api.h"
#include "dl_hccp_v2_def.h"
#include "device_udma_def.h"

namespace shm {
namespace transport {
namespace device {

struct PerEidJettyState {
    uint32_t eidIndex{0};
    void* ctxHandle{nullptr};
    void* tokenIdHandle{nullptr};
    void* chanHandle{nullptr};
    uint64_t cqVa{0};
    CqInfoT cqInfo = {0};
    void* cqHandle{nullptr};
    void* qpHandle{nullptr};
    QpCreateInfo qpCreateInfo_;
    std::vector<void*> remoteQpHandleList;
    std::vector<uint32_t> tpnList;
    void* cqPiAddr{nullptr};
    void* cqCiAddr{nullptr};
    void* sqPiAddr{nullptr};
    void* sqCiAddr{nullptr};
    void *wqeCntAddr{nullptr};
    void *amoAddr{nullptr};
    ACLSHMEMUDMAWQCtx localWq{};
    ACLSHMEMUDMACqCtx localCq{};
};

class DeviceJettyManager {
public:
    DeviceJettyManager(uint32_t deviceId, uint32_t rankId, uint32_t rankCount, uint32_t eidSlotCount) noexcept;
    ~DeviceJettyManager() noexcept;

    Result SetCtxHandles(const std::map<uint32_t, void*>& ctxHandleMap) noexcept;
    Result SetLocalMemInfos(const std::map<uint32_t, ACLSHMEMUBmemInfo>& localMemInfoMap) noexcept;
    Result SetEids(const std::map<uint32_t, HccpEid>& hccpEidMap) noexcept;
    Result SetTokenIdHandles(const std::map<uint32_t, void*>& tokenIdHandleMap) noexcept;
    Result SetPeerRoutes(
        const std::map<uint32_t, uint32_t>& peerLocalEidMap,
        const std::map<uint32_t, uint32_t>& peerRemoteEidMap) noexcept;
    Result Startup() noexcept;
    Result Shutdown() noexcept;
    void* GetJettyInfoAddress() noexcept;
    uint64_t GetJFCInfoAddress() const noexcept;

private:
    Result JFCCreate(PerEidJettyState& state) noexcept;
    Result JettyCreate(PerEidJettyState& state) noexcept;
    Result JettyImport() noexcept;
    Result JettyBind() noexcept;
    bool ReserveUdmaInfoSpace() noexcept;
    std::vector<uint32_t> CollectUsedLocalEids() const noexcept;
    bool BuildLocalQpPublishByEid(
        std::vector<QpImportInfoT>& qpImportByEid, std::vector<QpKeyT>& qpKeyByEid) const noexcept;
    uint32_t GetFallbackLocalEid() const noexcept;
    HccpEid ToImportedEid(const HccpEid& hccpEid) const noexcept;

    void FillUdmaWq(ACLSHMEMUDMAWQCtx& srcWq, ACLSHMEMUDMAWQCtx& dstWq) const;
    void FillUdmaCq(ACLSHMEMUDMACqCtx& srcCq, ACLSHMEMUDMACqCtx& dstCq) const;
    void FillUdmaMem(ACLSHMEMUBmemInfo& srcMem, ACLSHMEMUBmemInfo& dstMem) const;
    Result FillUdmaInfo() noexcept;
    void PrintHostInfo(ACLSHMEMAIVUDMAInfo& hostInfo) const;

    const uint32_t deviceId_;
    const uint32_t rankId_;
    const uint32_t rankCount_;
    const uint32_t eidCount_;
    std::map<uint32_t, void*> ctxHandleMap_;                // eidIndex -> ctxHandle
    std::map<uint32_t, ACLSHMEMUBmemInfo> localMemInfoMap_; // eidIndex -> local UDMA mem info
    std::map<uint32_t, HccpEid> localHccpEidMap_;           // eidIndex -> local HCCP EID
    std::map<uint32_t, void*> tokenIdHandleMap_;            // eidIndex -> tokenIdHandle
    std::map<uint32_t, uint32_t> peerLocalEidMap_;          // peerRankId -> local eidIndex
    std::map<uint32_t, uint32_t> peerRemoteEidMap_;         // peerRankId -> remote eidIndex
    std::map<uint32_t, PerEidJettyState> jettyStateMap_;    // eidIndex -> per-EID jetty state
    TransportModeT transportMode_ = TransportModeT::CONN_RM;

    // device
    void* udmaInfo_{nullptr};
    void* hccpEidDevice_{nullptr};

    // host
    uint32_t udmaInfoSize_{0};
};
} // namespace device
} // namespace transport
} // namespace shm

#endif // MF_HYBRID_DEVICE_JETTY_MANAGER_H
