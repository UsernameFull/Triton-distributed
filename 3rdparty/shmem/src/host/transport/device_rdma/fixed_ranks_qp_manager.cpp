/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <algorithm>

#include <chrono>
#include "shmemi_logger.h"
#include "dl_acl_api.h"
#include "dl_hccp_api.h"
#include "fixed_ranks_qp_manager.h"

namespace shm {
namespace transport {
namespace device {
static constexpr uint32_t SEND_CQ_DEPTH = 8192;
static constexpr uint32_t RECV_CQ_DEPTH = 128;
static constexpr uint32_t MAX_SEND_WR = 8192;
static constexpr uint32_t MAX_RECV_WR = 128;
static constexpr uint32_t QP_MODE = 2;
FixedRanksQpManager::FixedRanksQpManager(uint32_t userDeviceId, uint32_t deviceId, uint32_t rankId, uint32_t rankCount,
                                         mf_sockaddr devNet) noexcept
    : DeviceQpManager(deviceId, rankId, rankCount, devNet, HYBM_ROLE_PEER)
{
    userDeviceId_ = userDeviceId;
}

FixedRanksQpManager::~FixedRanksQpManager() noexcept
{
}

int FixedRanksQpManager::SetRemoteRankInfo(const std::unordered_map<uint32_t, ConnectRankInfo> &ranks) noexcept
{
    if (started_.load()) {
        SHM_LOG_ERROR(rankId_ << " fixed ranks not support update ranks info after startup");
        return ACLSHMEM_INNER_ERROR;
    }

    currentRanksInfo_ = ranks;
    return ACLSHMEM_SUCCESS;
}

int FixedRanksQpManager::SetLocalMemories(const MemoryRegionMap &mrs) noexcept
{
    if (started_.load()) {
        SHM_LOG_INFO(rankId_ << " fixed ranks not support update register MRs after startup");
        return ACLSHMEM_SUCCESS;
    }

    currentLocalMrs_ = mrs;
    return ACLSHMEM_SUCCESS;
}

int FixedRanksQpManager::Startup(void *rdma) noexcept
{
    if (rdma == nullptr) {
        SHM_LOG_ERROR(rankId_ << " input rdma is null");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (started_.load()) {
        SHM_LOG_ERROR(rankId_ << " already started.");
        return ACLSHMEM_INNER_ERROR;
    }

    rdmaHandle_ = rdma;
    if (!ReserveQpInfoSpace()) {
        SHM_LOG_ERROR(rankId_ << " reserve qp info space failed.");
        return ACLSHMEM_INNER_ERROR;
    }

    if (currentRanksInfo_.size() != rankCount_) {
        SHM_LOG_ERROR(rankId_ << " set rank count = " << currentRanksInfo_.size() << ", but rank_size = " << rankCount_);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (auto it = currentRanksInfo_.begin(); it != currentRanksInfo_.end(); ++it) {
        if (it->first >= rankCount_) {
            SHM_LOG_ERROR(rankId_ << " input options of nics contains rankId:" << it->first << ", rank count: " << rankCount_);
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    auto ret = StartServerSide();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " start server side failed: " << ret);
        return ret;
    }

    ret = StartClientSide();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " start client side failed: " << ret);
        return ret;
    }

    started_.store(true);
    return ACLSHMEM_SUCCESS;
}

void FixedRanksQpManager::Shutdown() noexcept
{
    CloseServices();
    if (qpInfo_ != nullptr) {
        auto ret = DlAclApi::AclrtFree(qpInfo_);
        if (ret != 0) {
            SHM_LOG_WARN(rankId_ << " free qpInfo device memory failed: " << ret);
        }
        qpInfo_ = nullptr;
    }
}

int FixedRanksQpManager::WaitingConnectionReady() noexcept
{
    if (serverConnectThread_ != nullptr) {
        serverConnectThread_->join();
        serverConnectThread_ = nullptr;
    }

    if (clientConnectThread_ != nullptr) {
        clientConnectThread_->join();
        clientConnectThread_ = nullptr;
    }

    if (serverConnectResult == ACLSHMEM_SUCCESS && clientConnectResult == ACLSHMEM_SUCCESS) {
        SHM_LOG_INFO(rankId_ << " client & server connections ready.");
        return ACLSHMEM_SUCCESS;
    }

    SHM_LOG_ERROR(rankId_ << " background connection thread not started.");
    return ACLSHMEM_INNER_ERROR;
}

const void *FixedRanksQpManager::GetQpInfoAddress() const noexcept
{
    return qpInfo_;
}

void *FixedRanksQpManager::GetQpHandleWithRankId(uint32_t rankId) const noexcept
{
    auto connections = rankId < rankId_ ? &clientConnections_ : &serverConnections_;
    auto pos = connections->find(rankId);
    if (pos == connections->end()) {
        return nullptr;
    }

    return pos->second.qpHandles[CONN_QP_STARS];
}

bool FixedRanksQpManager::ReserveQpInfoSpace() noexcept
{
    if (qpInfo_ != nullptr) {
        return true;
    }

    void *ptr = nullptr;
    auto oneQpSize = 2U * (sizeof(AiQpRMAWQ) + sizeof(AiQpRMACQ)) + sizeof(RdmaMemRegionInfo);
    qpInfoSize_ = sizeof(AiQpRMAQueueInfo) + oneQpSize * rankCount_;
    auto ret = DlAclApi::AclrtMalloc(&ptr, qpInfoSize_, 0);
    if (ret != 0) {
        SHM_LOG_ERROR(rankId_ << " allocate device size: " << qpInfoSize_ << ", failed: " << ret);
        return false;
    }

    qpInfo_ = (AiQpRMAQueueInfo *)ptr;
    return true;
}

int FixedRanksQpManager::StartServerSide() noexcept
{
    if (rankId_ + 1U == rankCount_) {
        serverConnectResult = 0;
        return ACLSHMEM_SUCCESS;
    }

    auto ret = CreateServerSocket();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR(rankId_ << " create server socket failed: " << ret);
        return ret;
    }

    ret = GenerateWhiteList();
    if (ret != 0) {
        SHM_LOG_ERROR(rankId_ << " generate white list failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    serverConnectThread_ = std::make_shared<std::thread>([this]() {
        DlAclApi::AclrtSetDevice(userDeviceId_);
        auto ret = WaitConnectionsReady(serverConnections_);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(rankId_ << " wait connection ready failed: " << ret);
            serverConnectResult = ret;
            return;
        }
        ret = CreateQpWaitingReady(serverConnections_, CONN_QP_AI_CORE, CONN_SERVER_SIDE);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(rankId_ << " wait connection AI qp ready failed: " << ret);
            serverConnectResult = ret;
            return;
        }

        serverConnectResult = ACLSHMEM_SUCCESS;
    });

    return ACLSHMEM_SUCCESS;
}

void FixedRanksQpManager::InitClientConnectThread()
{
    clientConnectThread_ = std::make_shared<std::thread>([this]() {
        DlAclApi::AclrtSetDevice(userDeviceId_);
        auto ret = WaitConnectionsReady(clientConnections_);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(rankId_ << " client wait connections failed: " << ret);
            clientConnectResult = ret;
            CloseClientConnections();
            return;
        }

        ret = CreateQpWaitingReady(clientConnections_, CONN_QP_AI_CORE, CONN_CLIENT_SIDE);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR(rankId_ << " client create qp for AI CORE failed: " << ret);
            clientConnectResult = ret;
            CloseClientConnections();
            return;
        }

        clientConnectResult = ACLSHMEM_SUCCESS;
    });
}

int FixedRanksQpManager::StartClientSide() noexcept
{
    if (rankId_ == 0U) {
        SHM_LOG_INFO(rankId_ << " need not connect to others.");
        clientConnectResult = ACLSHMEM_SUCCESS;
        return ACLSHMEM_SUCCESS;
    }

    std::vector<HccpSocketConnectInfo> connectInfos;
    for (auto it = currentRanksInfo_.begin(); it != currentRanksInfo_.end(); ++it) {
        if (it->first >= rankId_) {
            continue;  // client connect to small ranks.
        }

        auto socketHandle = CreateLocalSocket();
        if (socketHandle == nullptr) {
            SHM_LOG_ERROR(rankId_ << " create local socket handle failed");
            CloseClientConnections();
            return ACLSHMEM_DL_FUNC_FAILED;
        }

        net_addr_t addr;
        if (it->second.network.type == IpV4) {
            addr.type = IpV4;
            addr.ip.ipv4 = it->second.network.ip.ipv4.sin_addr;
        } else if (it->second.network.type == IpV6) {
            addr.type = IpV6;
            addr.ip.ipv6 = it->second.network.ip.ipv6.sin6_addr;
        }

        clientConnections_.emplace(it->first, ConnectionChannel{addr, socketHandle});
        HccpSocketConnectInfo connectInfo;
        connectInfo.handle = socketHandle;
        if (it->second.network.type == IpV4) {
            connectInfo.remoteIp.addr = it->second.network.ip.ipv4.sin_addr;
        } else if (it->second.network.type == IpV6) {
            connectInfo.remoteIp.addr6 = it->second.network.ip.ipv6.sin6_addr;
        }
        connectInfo.port = (it->second.network.type == IpV4) ? it->second.network.ip.ipv4.sin_port
            : it->second.network.ip.ipv6.sin6_port;
        bzero(connectInfo.tag, sizeof(connectInfo.tag));
        SHM_LOG_DEBUG(rankId_ << " add connecting server " << connectInfo);
        connectInfos.emplace_back(connectInfo);
    }

    for (size_t i = 0; i < connectInfos.size(); i += RA_MAX_BATCH_NUM) {
        size_t currentBatchSize = (connectInfos.size() - i) >= RA_MAX_BATCH_NUM ? RA_MAX_BATCH_NUM : (connectInfos.size() - i);
        auto batchStart = connectInfos.begin() + i;
        auto batchEnd = batchStart + currentBatchSize;
        std::vector<HccpSocketConnectInfo> currentBatch(batchStart, batchEnd);

        auto ret = DlHccpApi::RaSocketBatchConnect(currentBatch.data(), currentBatch.size());
        if (ret != 0) {
            SHM_LOG_ERROR(rankId_ << " connect to all servers failed: " << ret << ", servers count = " << connectInfos.size());
            CloseClientConnections();
            return ACLSHMEM_DL_FUNC_FAILED;
        }
    }

    InitClientConnectThread();
    return ACLSHMEM_SUCCESS;
}

int FixedRanksQpManager::GenerateWhiteList() noexcept
{
    std::vector<HccpSocketWhiteListInfo> whitelist;
    for (auto it = currentRanksInfo_.begin(); it != currentRanksInfo_.end(); ++it) {
        if (it->first <= rankId_) {
            continue;  // small id as server, large id as client
        }
        HccpSocketWhiteListInfo info{};
        net_addr_t addr;
        if (it->second.network.type == IpV4) {
            addr.type = IpV4;
            addr.ip.ipv4 = it->second.network.ip.ipv4.sin_addr;
            info.remoteIp.addr = it->second.network.ip.ipv4.sin_addr;
        } else if (it->second.network.type == IpV6) {
            addr.type = IpV6;
            addr.ip.ipv6 = it->second.network.ip.ipv6.sin6_addr;
            info.remoteIp.addr6 = it->second.network.ip.ipv6.sin6_addr;
        }
        info.connLimit = rankCount_;
        bzero(info.tag, sizeof(info.tag));
        whitelist.emplace_back(info);
        serverConnections_.emplace(it->first, ConnectionChannel{addr, serverSocketHandle_});
    }

    if (whitelist.empty()) {
        return ACLSHMEM_SUCCESS;
    }

    for (size_t i = 0; i < whitelist.size(); i += RA_MAX_BATCH_NUM) {
        size_t currentBatchSize = (whitelist.size() - i) >= RA_MAX_BATCH_NUM ? RA_MAX_BATCH_NUM : (whitelist.size() - i);
        auto batchStart = whitelist.begin() + i;
        auto batchEnd = batchStart + currentBatchSize;
        std::vector<HccpSocketWhiteListInfo> currentBatch(batchStart, batchEnd);
        auto ret = DlHccpApi::RaSocketWhiteListAdd(serverSocketHandle_, currentBatch.data(), currentBatch.size());
        if (ret != 0) {
            SHM_LOG_ERROR(rankId_ << " socket handle add white list failed: " << ret);
            return ACLSHMEM_INNER_ERROR;
        }
    }

    return ACLSHMEM_SUCCESS;
}

int FixedRanksQpManager::WaitConnectionsReady(std::unordered_map<uint32_t, ConnectionChannel> &connections) noexcept
{
    IpType type{};
    uint32_t cnt = 0;
    auto start = std::chrono::steady_clock::now();
    constexpr auto TIMEOUT_MINUTES = 2;
    auto timeout = start + std::chrono::minutes(TIMEOUT_MINUTES);
    SHM_LOG_DEBUG(rankId_ << " begin wait connections, size=" << connections.size());

    std::vector<HccpSocketInfo> socketInfos;
    std::unordered_map<net_addr_t, uint32_t> addr2index;
    for (auto it = connections.begin(); it != connections.end(); ++it) {
        if (it->second.socketFd != nullptr) {
            continue;
        }

        HccpSocketInfo info{};
        info.handle = it->second.socketHandle;
        info.fd = nullptr;
        if (it->second.remoteIp.type == IpV4) {
            info.remoteIp.addr = it->second.remoteIp.ip.ipv4;
            type = IpV4;
        } else if (it->second.remoteIp.type == IpV6) {
            info.remoteIp.addr6 = it->second.remoteIp.ip.ipv6;
            type = IpV6;
        }
        info.status = 0;
        bzero(info.tag, sizeof(info.tag));
        socketInfos.push_back(info);
        addr2index.emplace(it->second.remoteIp, it->first);
    }

    if (socketInfos.empty()) {
        SHM_LOG_WARN(rankId_ << " socketInfos is empty.");
        return ACLSHMEM_SUCCESS;
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    auto role = (&connections == &clientConnections_) ? 1 : 0;
    std::string roleStr = (role == 1) ? "client" : "server";

    do {
        auto now = std::chrono::steady_clock::now();
        if (now >= timeout) {
            std::string pendingRanks;
            for (const auto& info : socketInfos) {
                auto it = addr2index.find(type == IpV4 ? net_addr_t::from_ipv4(info.remoteIp.addr) 
                                                        : net_addr_t::from_ipv6(info.remoteIp.addr6));
                if (it != addr2index.end()) {
                    pendingRanks += " rank[" + std::to_string(it->second) + "]";
                }
            }
            SHM_LOG_ERROR(rankId_ << " " << roleStr << " wait connections timeout after " << TIMEOUT_MINUTES 
                << " minutes. Pending connections:" << pendingRanks);
            return ACLSHMEM_TIMEOUT_ERROR;
        }

        uint32_t getSize = socketInfos.size() < RA_MAX_BATCH_NUM ? socketInfos.size() : RA_MAX_BATCH_NUM;
        auto ret = DlHccpApi::RaGetSockets(role, socketInfos.data(), getSize, cnt);
        if (ret != 0) {
            SHM_LOG_ERROR(rankId_ << " role(" << role << ") side get sockets failed: " << ret);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100L));
        if (cnt == 0) {
            continue;
        }
        bool progress = false;
        for (auto i = 0U; i < getSize; i++) {
            if (socketInfos[i].status != 1) {
                continue;
            }
            progress = true;
            net_addr_t addr;
            char ipStr[INET6_ADDRSTRLEN];
            char* result {};
            if (type == IpV4) {
                addr = net_addr_t::from_ipv4(socketInfos[i].remoteIp.addr);
                result = inet_ntoa(socketInfos[i].remoteIp.addr);
            } else if (type == IpV6) {
                addr = net_addr_t::from_ipv6(socketInfos[i].remoteIp.addr6);
                inet_ntop(AF_INET6, &socketInfos[i].remoteIp.addr6, ipStr, INET6_ADDRSTRLEN);
                result = ipStr;
            }
            auto socketInfoPos = addr2index.find(addr);
            if (socketInfoPos == addr2index.end()) {
                SHM_LOG_ERROR(rankId_ << " socket ip(" << result << ") should not exist.");
                return ACLSHMEM_DL_FUNC_FAILED;
            }

            auto rankId = socketInfoPos->second;
            auto pos = connections.find(rankId);
            if (pos == connections.end()) {
                SHM_LOG_ERROR(rankId_ << " -> " << rankId << ":socket ip(" << result << ") should not exist.");
                return ACLSHMEM_DL_FUNC_FAILED;
            }

            if (pos->second.socketFd != nullptr) {
                SHM_LOG_ERROR(rankId_ << " -> " << rankId << ":get socket ip(" << result << ") already get socket fd.");
                return ACLSHMEM_DL_FUNC_FAILED;
            }

            if (pos->second.socketHandle != socketInfos[i].handle) {
                SHM_LOG_ERROR(rankId_ << " -> " << rankId << ":get socket ip(" << result << ") socket handle not match.");
                return ACLSHMEM_DL_FUNC_FAILED;
            }

            pos->second.socketFd = socketInfos[i].fd;
            SHM_LOG_INFO(rankId_ << " connect to (" << rankId << ") ready.");
        }
        if (progress) {
            start = std::chrono::steady_clock::now();
            timeout = start + std::chrono::minutes(TIMEOUT_MINUTES);
        }
        socketInfos.erase(
            std::remove_if(socketInfos.begin(), socketInfos.end(),
                [](const auto& info) {
                    return info.status == 1;
                }),
            socketInfos.end()
        );
    } while (socketInfos.size() > 0);
    return ACLSHMEM_SUCCESS;
}

int FixedRanksQpManager::CreateQpWaitingReady(std::unordered_map<uint32_t, ConnectionChannel> &connections,
                                              ConnQpType qpType, ConnSide side) noexcept
{
    const int accessLevel = 7;
    std::string sideName = (side == CONN_CLIENT_SIDE) ? "client" : "server";
    SHM_LOG_DEBUG(rankId_ << ":" << sideName << " begin create qp and register mr");
    for (auto it = connections.begin(); it != connections.end(); ++it) {
        auto ret = CreateOneQp(qpType, it->second);
        if (ret != 0) {
            SHM_LOG_ERROR(rankId_ << ":" << sideName << " create QP type:" << qpType <<
                " to " << it->first << " failed: " << ret);
            return ACLSHMEM_DL_FUNC_FAILED;
        }

        SHM_LOG_DEBUG(rankId_ << ":" << sideName << " create qp success, qptype=" << qpType);

        for (auto pos = currentLocalMrs_.begin(); pos != currentLocalMrs_.end(); ++pos) {
            HccpMrInfo info{};
            info.addr = (void *)(ptrdiff_t)pos->second.address;
            info.size = pos->second.size;
            info.access = accessLevel;
            ret = DlHccpApi::RaMrReg(it->second.qpHandles[qpType], info);
            if (ret != 0) {
                SHM_LOG_ERROR(rankId_ << ":" << sideName << " register MR failed: " << ret);
                return ACLSHMEM_DL_FUNC_FAILED;
            }
            SHM_LOG_DEBUG(rankId_ << ":" << sideName << " register mr size:" << pos->second.size << " success");
        }

        ret = DlHccpApi::RaQpConnectAsync(it->second.qpHandles[qpType], it->second.socketFd);
        if (ret != 0) {
            SHM_LOG_ERROR(rankId_ << ":" << sideName << " connect AI QP to " << it->first << " failed: " << ret);
            return ACLSHMEM_DL_FUNC_FAILED;
        }
    }

    SHM_LOG_DEBUG(rankId_ << ":" << sideName << " begin wait create qp ready.");
    auto start = std::chrono::steady_clock::now();
    auto timeout = start + std::chrono::minutes(1);
    while (std::chrono::steady_clock::now() < timeout) {
        int connectingCount = 0;
        for (auto it = connections.begin(); it != connections.end(); ++it) {
            int status = 0;
            auto ret = DlHccpApi::RaGetQpStatus(it->second.qpHandles[qpType], status);
            if (ret != 0) {
                SHM_LOG_ERROR(rankId_ << ":" << sideName << " get AI QP status to " << it->first << " failed: " << ret);
                return ACLSHMEM_DL_FUNC_FAILED;
            }

            // rs_qp_status: 0-disconnect, 1-connected, 2-timeout, 3-connecting, 4-fd_close, 5-pause
            SHM_LOG_DEBUG(rankId_ << ":" << sideName << " get create qp status to " << it->first << "=" << status);
            if (status != 1) {
                connectingCount++;
            }
        }
        if (connectingCount == 0) {
            return FillQpInfo(qpType);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    SHM_LOG_ERROR(rankId_ << ":" << sideName << " wait create qp ready timeout");
    return ACLSHMEM_INNER_TIMEOUT;
}

int FixedRanksQpManager::CreateOneQp(ConnQpType qpType, ConnectionChannel &channel) noexcept
{
    int ret;
    if (qpType == CONN_QP_AI_CORE) {
        HccpQpExtAttrs attr{};
        attr.qpMode = NETWORK_OFFLINE;
        attr.version = 1;
        attr.cqAttr.sendCqDepth = SEND_CQ_DEPTH;
        attr.cqAttr.recvDqDepth = RECV_CQ_DEPTH;
        attr.qp_attr.cap.max_recv_sge = 1;
        attr.qp_attr.cap.max_recv_wr = MAX_RECV_WR;
        attr.qp_attr.cap.max_recv_sge = 1;
        attr.qp_attr.qp_type = IBV_QPT_RC;
        attr.qp_attr.cap.max_send_wr = MAX_SEND_WR;
        attr.data_plane_flag.bs.cq_cstm = 1;
        ret = DlHccpApi::RaQpAiCreate(rdmaHandle_, attr, channel.aiQpInfo, channel.qpHandles[qpType]);
    } else {
        ret = DlHccpApi::RaQpCreate(rdmaHandle_, 0, QP_MODE, channel.qpHandles[qpType]);
    }
    return ret;
}

void FixedRanksQpManager::FillQpPreSettingCopyInfo(AiQpRMAQueueInfo *&copyInfo)
{
    copyInfo->count = 1;
    copyInfo->sq = (AiQpRMAWQ *)(void *)(copyInfo + 1);
    copyInfo->rq = (AiQpRMAWQ *)(void *)(copyInfo->sq + rankCount_);
    copyInfo->scq = (AiQpRMACQ *)(void *)(copyInfo->rq + rankCount_);
    copyInfo->rcq = (AiQpRMACQ *)(void *)(copyInfo->scq + rankCount_);
    copyInfo->mr = (RdmaMemRegionInfo *)(void *)(copyInfo->rcq + rankCount_);
}

void FixedRanksQpManager::FillQpPostSettingCopyInfo(AiQpRMAQueueInfo *&copyInfo)
{
    auto pointer = (ptrdiff_t)(void *)(qpInfo_);
    pointer += sizeof(AiQpRMAQueueInfo);
    copyInfo->sq = (AiQpRMAWQ *)(void *)(pointer);

    pointer += static_cast<ptrdiff_t>(sizeof(AiQpRMAWQ) * rankCount_);
    copyInfo->rq = (AiQpRMAWQ *)(void *)(pointer);

    pointer += static_cast<ptrdiff_t>(sizeof(AiQpRMAWQ) * rankCount_);
    copyInfo->scq = (AiQpRMACQ *)(void *)(pointer);

    pointer += static_cast<ptrdiff_t>(sizeof(AiQpRMACQ) * rankCount_);
    copyInfo->rcq = (AiQpRMACQ *)(void *)(pointer);

    pointer += static_cast<ptrdiff_t>(sizeof(AiQpRMACQ) * rankCount_);
    copyInfo->mr = (RdmaMemRegionInfo *)(void *)pointer;
}

int FixedRanksQpManager::FillQpInfo(ConnQpType qpType) noexcept
{
    if (qpType != CONN_QP_AI_CORE) {
        return ACLSHMEM_SUCCESS;
    }

    const uint32_t slevel = 4;
    std::vector<uint8_t> qpInfoBuffer(qpInfoSize_);
    auto copyInfo = (AiQpRMAQueueInfo *)(void *)qpInfoBuffer.data();
    FillQpPreSettingCopyInfo(copyInfo);
    for (auto it = currentRanksInfo_.begin(); it != currentRanksInfo_.end(); ++it) {
        auto &map = it->second.memoryMap;
        if (map.empty()) {
            continue;
        }
        copyInfo->mr[it->first].size = map.begin()->second.size;
        copyInfo->mr[it->first].addr = map.begin()->second.address;
        copyInfo->mr[it->first].lkey = map.begin()->second.lkey;
        copyInfo->mr[it->first].rkey = map.begin()->second.rkey;
        if (it->first == rankId_) {
            continue;
        }

        std::unordered_map<uint32_t, ConnectionChannel> *connections;
        if (it->first < rankId_) {
            connections = &clientConnections_;
        } else {
            connections = &serverConnections_;
        }

        auto pos = connections->find(it->first);
        if (pos == connections->end()) {
            SHM_LOG_ERROR(rankId_ << " missing for remote: " << it->first);
            return ACLSHMEM_INNER_ERROR;
        }

        CopyAiWQInfo(copyInfo->sq[it->first], pos->second.aiQpInfo.data_plane_info.sq, DBMode::HW_DB, slevel);
        CopyAiWQInfo(copyInfo->rq[it->first], pos->second.aiQpInfo.data_plane_info.rq, DBMode::SW_DB, slevel);
        CopyAiCQInfo(copyInfo->scq[it->first], pos->second.aiQpInfo.data_plane_info.scq, DBMode::HW_DB);
        CopyAiCQInfo(copyInfo->rcq[it->first], pos->second.aiQpInfo.data_plane_info.rcq, DBMode::SW_DB);
    }

    FillQpPostSettingCopyInfo(copyInfo);

    auto ret = DlAclApi::AclrtMemcpy(qpInfo_, qpInfoSize_, copyInfo, qpInfoSize_, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR(rankId_ << " copy qp info to device failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    SHM_LOG_INFO(rankId_ << " copy qp info success");

    return ACLSHMEM_SUCCESS;
}

void FixedRanksQpManager::CopyAiWQInfo(struct AiQpRMAWQ &dest, const struct ai_data_plane_wq &src, DBMode dbMode,
                                       uint32_t sl) noexcept
{
    dest.wqn = src.wqn;
    dest.bufAddr = src.buf_addr;
    dest.wqeSize = src.wqebb_size;
    dest.depth = src.depth;
    dest.headAddr = src.head_addr;
    dest.tailAddr = src.tail_addr;
    dest.dbMode = dbMode;
    if (dbMode == DBMode::SW_DB) {
        dest.dbAddr = src.swdb_addr;
    } else if (dbMode == DBMode::HW_DB) {
        dest.dbAddr = src.db_reg;
    }
    dest.sl = sl;
}

void FixedRanksQpManager::CopyAiCQInfo(struct AiQpRMACQ &dest, const ai_data_plane_cq &source, DBMode dbMode) noexcept
{
    dest.cqn = source.cqn;
    dest.bufAddr = source.buf_addr;
    dest.cqeSize = source.cqe_size;
    dest.depth = source.depth;
    dest.headAddr = source.head_addr;
    dest.tailAddr = source.tail_addr;
    dest.dbMode = dbMode;
    if (dbMode == DBMode::SW_DB) {
        dest.dbAddr = source.swdb_addr;
    } else if (dbMode == DBMode::HW_DB) {
        dest.dbAddr = source.db_reg;
    }
}

void FixedRanksQpManager::CloseServices() noexcept
{
    if (serverConnectThread_ != nullptr) {
        serverConnectThread_->join();
        serverConnectThread_ = nullptr;
    }

    if (clientConnectThread_ != nullptr) {
        clientConnectThread_->join();
        clientConnectThread_ = nullptr;
    }

    CloseServerConnections();
    SHM_LOG_INFO(rankId_ << " server connections closed.");

    CloseClientConnections();
    SHM_LOG_INFO(rankId_ << " client connections closed.");
}

void FixedRanksQpManager::CloseClientConnections() noexcept
{
    CloseConnections(clientConnections_);
}

void FixedRanksQpManager::CloseServerConnections() noexcept
{
    DestroyServerSocket();
    CloseConnections(serverConnections_);
}

void FixedRanksQpManager::CloseConnections(std::unordered_map<uint32_t, ConnectionChannel> &connections) noexcept
{
    std::vector<HccpSocketCloseInfo> socketCloseInfos;
    for (auto it = connections.begin(); it != connections.end(); ++it) {
        if (it->second.qpHandles[CONN_QP_AI_CORE] != nullptr) {
            auto ret = DlHccpApi::RaQpDestroy(it->second.qpHandles[CONN_QP_AI_CORE]);
            if (ret != 0) {
                SHM_LOG_WARN(rankId_ << " destroy AI QP to server: " << it->first << " failed: " << ret);
            }
            it->second.qpHandles[CONN_QP_AI_CORE] = nullptr;
        }

        if (it->second.socketFd != nullptr) {
            HccpSocketCloseInfo info;
            info.handle = it->second.socketHandle;
            info.fd = it->second.socketFd;
            info.linger = 0;
            socketCloseInfos.push_back(info);
            it->second.socketFd = nullptr;
        }
    }

    if (!socketCloseInfos.empty()) {
        auto ret = DlHccpApi::RaSocketBatchClose(socketCloseInfos.data(), socketCloseInfos.size());
        if (ret != 0) {
            SHM_LOG_INFO(rankId_ << " close sockets return: " << ret);
        }
    }

    if (&connections == &serverConnections_) {
        if (serverSocketHandle_ != nullptr) {
            auto ret = DlHccpApi::RaSocketDeinit(serverSocketHandle_);
            if (ret != 0) {
                SHM_LOG_INFO(rankId_ << " server deinit socket return: " << ret);
            }
            serverSocketHandle_ = nullptr;
            SHM_LOG_INFO(rankId_ << " as server close connection done, size is " << 1);
        }
    } else {
        for (auto it = connections.begin(); it != connections.end(); ++it) {
            auto ret = DlHccpApi::RaSocketDeinit(it->second.socketHandle);
            if (ret != 0) {
                SHM_LOG_INFO(rankId_ << " deinit socket to server: " << it->first << " return: " << ret);
            }
        }
        SHM_LOG_INFO(rankId_ << " as client close connection done, size is " << connections.size());
    }

    connections.clear();
}
}
}
}