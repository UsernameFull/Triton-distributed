/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef STORE_TCP_CONFIG_SERVER_H
#define STORE_TCP_CONFIG_SERVER_H

#include <list>
#include <mutex>
#include <chrono>
#include <mutex>
#include <condition_variable>
#include <unordered_map>
#include <unordered_set>
#include <system_error>

#include "acc_tcp_server.h"
#include "store_message_packer.h"
#include "store_obj_ref.h"
#include "store_utils.h"

namespace shm {
namespace store {
class StoreWaitContext {
public:
    StoreWaitContext(int64_t tmMs, std::string key, const shm::acc::AccTcpRequestContext &reqCtx) noexcept
        : id_{idGen_.fetch_add(1UL)},
          timeoutMs_{tmMs},
          key_{std::move(key)},
          reqCtx_{reqCtx}
    {
    }

    uint64_t Id() const noexcept
    {
        return id_;
    }

    int64_t TimeoutMs() const noexcept
    {
        return timeoutMs_;
    }

    const std::string &Key() const noexcept
    {
        return key_;
    }

    const shm::acc::AccTcpRequestContext &ReqCtx() const noexcept
    {
        return reqCtx_;
    }

    shm::acc::AccTcpRequestContext &ReqCtx() noexcept
    {
        return reqCtx_;
    }

private:
    const uint64_t id_;
    const int64_t timeoutMs_;
    const std::string key_;
    shm::acc::AccTcpRequestContext reqCtx_;
    static std::atomic<uint64_t> idGen_;
};

class AccStoreServer : public SmReferable {
public:
    AccStoreServer(std::string ip, uint16_t port, int32_t sockFd = -1) noexcept;
    ~AccStoreServer() override = default;

    Result Startup(const AcclinkTlsOption &tlsOption) noexcept;
    void Shutdown(bool afterFork = false) noexcept;

private:
    Result ReceiveMessageHandler(const shm::acc::AccTcpRequestContext &context) noexcept;
    Result LinkConnectedHandler(const shm::acc::AccConnReq &req, const shm::acc::AccTcpLinkComplexPtr &link) noexcept;
    Result LinkBrokenHandler(const shm::acc::AccTcpLinkComplexPtr &link) noexcept;

    /* business handler */
    Result SetHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept;
    Result GetHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept;
    Result AddHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept;
    Result RemoveHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept;
    Result AppendHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept;
    Result CasHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept;

    std::list<shm::acc::AccTcpRequestContext> GetOutWaitersInLock(const std::unordered_set<uint64_t> &ids) noexcept;
    void WakeupWaiters(const std::list<shm::acc::AccTcpRequestContext> &waiters,
                       const std::vector<uint8_t> &value) noexcept;
    void ReplyWithMessage(const shm::acc::AccTcpRequestContext &ctx, int16_t code, const std::string &message) noexcept;
    void ReplyWithMessage(const shm::acc::AccTcpRequestContext &ctx, int16_t code,
                          const std::vector<uint8_t> &message) noexcept;
    void TimerThreadTask() noexcept;
    Result AccServerStart(shm::acc::AccTcpServerPtr &accTcpServer, const AcclinkTlsOption &tlsOption) noexcept;

private:
    static constexpr uint32_t MAX_KEY_LEN_SERVER = 2048U;

    using MessageHandle = int32_t (AccStoreServer::*)(const shm::acc::AccTcpRequestContext &, SmemMessage &);
    const std::unordered_map<MessageType, MessageHandle> requestHandlers_;

    std::mutex storeMutex_;
    std::condition_variable storeCond_;
    std::unordered_map<std::string, std::vector<uint8_t>> kvStore_;
    std::unordered_map<uint64_t, StoreWaitContext> waitCtx_;
    std::unordered_map<std::string, std::unordered_set<uint64_t>> keyWaiters_;
    shm::acc::AccTcpServerPtr accTcpServer_;
    std::unordered_map<int64_t, std::unordered_set<uint64_t>> timedWaiters_;
    std::thread timerThread_;
    bool running_{false};

    const std::string listenIp_;
    const uint16_t listenPort_;
    int32_t sockFd_;
    std::mutex mutex_;
};
using AccStoreServerPtr = SmRef<AccStoreServer>;
}  // namespace store
}  // namespace shm

#endif  // STORE_TCP_CONFIG_SERVER_H
