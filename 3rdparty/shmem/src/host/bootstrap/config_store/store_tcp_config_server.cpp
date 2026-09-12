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
#include "host/shmem_host_def.h"
#include "shmemi_logger.h"
#include "store_message_packer.h"
#include "store_op.h"
#include "store_net_common.h"
#include "store_tcp_config_server.h"

namespace shm {
namespace store {
std::atomic<uint64_t> StoreWaitContext::idGen_{1UL};
AccStoreServer::AccStoreServer(std::string ip, uint16_t port, int32_t sockFd) noexcept
    : listenIp_{std::move(ip)},
      listenPort_{port},
      sockFd_{sockFd},
      requestHandlers_{
          {MessageType::SET, &AccStoreServer::SetHandler},       {MessageType::GET, &AccStoreServer::GetHandler},
          {MessageType::ADD, &AccStoreServer::AddHandler},       {MessageType::REMOVE, &AccStoreServer::RemoveHandler},
          {MessageType::APPEND, &AccStoreServer::AppendHandler}, {MessageType::CAS, &AccStoreServer::CasHandler}}
{
}

Result AccStoreServer::AccServerStart(shm::acc::AccTcpServerPtr &accTcpServer,
                                           const AcclinkTlsOption &tlsOption) noexcept
{
    shm::acc::AccTcpServerOptions options;
    options.listenIp = listenIp_;
    options.listenPort = listenPort_;
    options.enableListener = true;
    options.linkSendQueueSize = shm::acc::UNO_48;
    options.sockFd = sockFd_;

    shm::acc::AccTlsOption tlsOpt = ConvertTlsOption(tlsOption);
    Result result;
    if (tlsOpt.enableTls) {
        if (accTcpServer->LoadDynamicLib(tlsOption.packagePath) != 0) {
            SHM_LOG_ERROR("Load openssl failed");
            return ACLSHMEM_SMEM_ERROR;
        }
        if (tlsOption.decryptHandler_) {
            // pk must be encrypt, decrypt handler not null means pk password encyrpted
            accTcpServer->RegisterDecryptHandler(tlsOption.decryptHandler_);
        }
        result = accTcpServer->Start(options, tlsOpt);
    } else {
        result = accTcpServer->Start(options);
    }
    if (result == shm::acc::ACC_LINK_ADDRESS_IN_USE) {
        SHM_LOG_INFO("startup acc tcp server on port: " << listenPort_ << " already in use.");
        return SM_RESOURCE_IN_USE;
    }
    if (result != shm::acc::ACC_OK) {
        SHM_LOG_ERROR("startup acc tcp server on port: " << listenPort_ << " failed: " << result);
        return ACLSHMEM_SMEM_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

Result AccStoreServer::Startup(const AcclinkTlsOption &tlsOption) noexcept
{
    std::lock_guard<std::mutex> guard(mutex_);
    if (accTcpServer_ != nullptr) {
        SHM_LOG_INFO("tcp store server already startup");
        return ACLSHMEM_SUCCESS;
    }

    auto tmpAccTcpServer = shm::acc::AccTcpServer::Create();
    if (tmpAccTcpServer == nullptr) {
        SHM_LOG_ERROR("create acc tcp server failed");
        return SM_NEW_OBJECT_FAILED;
    }

    tmpAccTcpServer->RegisterNewRequestHandler(
        0, [this](const shm::acc::AccTcpRequestContext &context) { return ReceiveMessageHandler(context); });
    tmpAccTcpServer->RegisterNewLinkHandler(
        [this](const shm::acc::AccConnReq &req, const shm::acc::AccTcpLinkComplexPtr &link) {
            return LinkConnectedHandler(req, link);
        });
    tmpAccTcpServer->RegisterLinkBrokenHandler(
        [this](const shm::acc::AccTcpLinkComplexPtr &link) { return LinkBrokenHandler(link); });

    Result result = AccServerStart(tmpAccTcpServer, tlsOption);
    if (result != ACLSHMEM_SUCCESS) {
        return result;
    }

    accTcpServer_ = tmpAccTcpServer;

    std::unique_lock<std::mutex> lockGuard{storeMutex_};
    running_ = true;
    lockGuard.unlock();

    timerThread_ = std::thread{[this]() {
        TimerThreadTask();
    }};

    return ACLSHMEM_SUCCESS;
}

void AccStoreServer::Shutdown(bool afterFork) noexcept
{
    SHM_LOG_INFO("start to shutdown Acc Store Server");
    if (accTcpServer_ == nullptr) {
        return;
    }

    if (afterFork) {
        accTcpServer_->StopAfterFork();
        running_ = false;
        if (timerThread_.joinable()) {
            timerThread_.detach();
        }
    } else {
        accTcpServer_->Stop();
        std::unique_lock<std::mutex> lockGuard{storeMutex_};
        running_ = false;
        lockGuard.unlock();
        storeCond_.notify_one();

        if (timerThread_.joinable()) {
            try {
                timerThread_.join();
            } catch (const std::system_error& e) {
                SHM_LOG_ERROR("thread join failed: " << e.what());
            }
        }
    }
    sockFd_ = -1;
    accTcpServer_ = nullptr;
    SHM_LOG_INFO("finished shutdown Acc Store Server");
}

Result AccStoreServer::ReceiveMessageHandler(const shm::acc::AccTcpRequestContext &context) noexcept
{
    auto data = reinterpret_cast<const uint8_t *>(context.DataPtr());
    if (data == nullptr) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle get null request body");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "request no body");
        return SM_INVALID_PARAM;
    }

    SmemMessage requestMessage;
    auto size = SmemMessagePacker::Unpack(data, context.DataLen(), requestMessage);
    if (size < 0) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle invalid body");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request");
        return ACLSHMEM_SMEM_ERROR;
    }

    auto pos = requestHandlers_.find(requestMessage.mt);
    if (pos == requestHandlers_.end()) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle invalid message type: " << requestMessage.mt);
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request message type");
        return ACLSHMEM_SMEM_ERROR;
    }

    return (this->*(pos->second))(context, requestMessage);
}

Result AccStoreServer::LinkConnectedHandler(const shm::acc::AccConnReq &req,
                                            const shm::acc::AccTcpLinkComplexPtr &link) noexcept
{
    SHM_LOG_INFO("new link connected, linkId: " << link->Id() << ", rank: " << req.rankId);
    return ACLSHMEM_SUCCESS;
}

Result AccStoreServer::LinkBrokenHandler(const shm::acc::AccTcpLinkComplexPtr &link) noexcept
{
    SHM_LOG_INFO("link broken, linkId: " << link->Id());
    return ACLSHMEM_SUCCESS;
}

Result AccStoreServer::SetHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept
{
    if (request.keys.size() != 1 || request.values.size() != 1) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle invalid body");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request: key value should be one");
        return SM_INVALID_PARAM;
    }

    auto &key = request.keys[0];
    auto &value = request.values[0];
    if (key.length() > MAX_KEY_LEN_SERVER) {
        SHM_LOG_ERROR("key length too large, length: " << key.length());
        return StoreErrorCode::INVALID_KEY;
    }

    SHM_LOG_DEBUG("SET REQUEST(" << context.SeqNo() << ") for key(" << key << ") start.");
    std::list<shm::acc::AccTcpRequestContext> wakeupWaiters;
    std::vector<uint8_t> reqVal;
    std::unique_lock<std::mutex> lockGuard{storeMutex_};
    auto pos = kvStore_.find(key);
    if (pos == kvStore_.end()) {
        auto wPos = keyWaiters_.find(key);
        if (wPos != keyWaiters_.end()) {
            wakeupWaiters = GetOutWaitersInLock(wPos->second);
            reqVal = value;
            keyWaiters_.erase(wPos);
        }
        kvStore_.emplace(key, std::move(value));
    } else {
        pos->second = std::move(value);
    }
    lockGuard.unlock();

    ReplyWithMessage(context, StoreErrorCode::SUCCESS, "success");
    if (!wakeupWaiters.empty()) {
        WakeupWaiters(wakeupWaiters, reqVal);
    }

    return ACLSHMEM_SUCCESS;
}

Result AccStoreServer::GetHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept
{
    if (request.keys.size() != 1 || !request.values.empty()) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle invalid body");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request: key should be one and no values.");
        return SM_INVALID_PARAM;
    }

    auto &key = request.keys[0];
    if (key.length() > MAX_KEY_LEN_SERVER) {
        SHM_LOG_ERROR("key length too large, length: " << key.length());
        return StoreErrorCode::INVALID_KEY;
    }

    SHM_LOG_DEBUG("GET REQUEST(" << context.SeqNo() << ") for key(" << key << ") start.");
    SmemMessage responseMessage{request.mt};
    std::unique_lock<std::mutex> lockGuard{storeMutex_};
    auto pos = kvStore_.find(key);
    if (pos != kvStore_.end()) {
        responseMessage.values.push_back(pos->second);
        lockGuard.unlock();

        SHM_LOG_DEBUG("GET REQUEST(" << context.SeqNo() << ") for key(" << key << ") success.");
        auto response = SmemMessagePacker::Pack(responseMessage);
        ReplyWithMessage(context, StoreErrorCode::SUCCESS, response);
        return ACLSHMEM_SUCCESS;
    }

    if (request.userDef == 0) {
        lockGuard.unlock();

        SHM_LOG_DEBUG("GET REQUEST(" << context.SeqNo() << ") for key(" << key << ") not exist.");
        ReplyWithMessage(context, StoreErrorCode::NOT_EXIST, "<not exist>");
        return ACLSHMEM_SMEM_ERROR;
    }

    if (request.userDef > std::numeric_limits<int>::max()) {
        SHM_LOG_DEBUG("GET REQUEST(" << context.SeqNo() << ") for key(" << key << "): invalid timeout.");
        return ACLSHMEM_SMEM_ERROR;
    }
    SHM_LOG_DEBUG("GET REQUEST(" << context.SeqNo() << ") for key(" << key << ") waiting timeout=" << request.userDef);
    auto timeout = std::chrono::steady_clock::now() + std::chrono::milliseconds(request.userDef);
    auto timeoutMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeout.time_since_epoch()).count();
    SHM_LOG_DEBUG("GET REQUEST(" << context.SeqNo() << ") for key(" << key << ") waiting timeout=" << timeoutMs);
    StoreWaitContext waitContext{timeoutMs, key, context};
    auto pair = waitCtx_.emplace(waitContext.Id(), std::move(waitContext));
    auto wPos = keyWaiters_.find(key);
    if (wPos != keyWaiters_.end()) {
        wPos->second.emplace(pair.first->first);
    } else {
        keyWaiters_.emplace(key, std::unordered_set<uint64_t>{pair.first->first});
    }

    if (request.userDef > 0) {
        auto timerPos = timedWaiters_.find(timeoutMs);
        if (timerPos == timedWaiters_.end()) {
            timedWaiters_.emplace(timeoutMs, std::unordered_set<uint64_t>{pair.first->first});
        } else {
            timerPos->second.emplace(pair.first->first);
        }
    }
    lockGuard.unlock();

    return ACLSHMEM_SUCCESS;
}

Result AccStoreServer::AddHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept
{
    if (request.keys.size() != 1 || request.values.size() != 1) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle invalid body");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request: key value should be one.");
        return SM_INVALID_PARAM;
    }

    auto &key = request.keys[0];
    auto &value = request.values[0];
    SHM_VALIDATE_RETURN(key.length() <= MAX_KEY_LEN_SERVER, "key length too large, length: "
                       << key.length(), StoreErrorCode::INVALID_KEY);

    std::string valueStr{value.begin(), value.end()};
    SHM_LOG_DEBUG("ADD REQUEST(" << context.SeqNo() << ") for key(" << key << ") value(" << valueStr << ") start.");

    long valueNum;
    SHM_VALIDATE_RETURN(StrToLong(valueStr, valueNum), "convert string to long failed.", ACLSHMEM_SMEM_ERROR);

    if (valueStr != std::to_string(valueNum)) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") add for key(" << key << ") value is not a number");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request: value should be a number.");
        return ACLSHMEM_SMEM_ERROR;
    }

    auto responseValue = valueNum;
    std::list<shm::acc::AccTcpRequestContext> wakeupWaiters;
    std::vector<uint8_t> reqVal;
    std::unique_lock<std::mutex> lockGuard{storeMutex_};
    auto pos = kvStore_.find(key);
    if (pos == kvStore_.end()) {
        auto wPos = keyWaiters_.find(key);
        if (wPos != keyWaiters_.end()) {
            wakeupWaiters = GetOutWaitersInLock(wPos->second);
            reqVal = value;
            keyWaiters_.erase(wPos);
        }
        kvStore_.emplace(key, std::move(value));
    } else {
        std::string oldValueStr{pos->second.begin(), pos->second.end()};
        long storedValueNum = 0;
        SHM_VALIDATE_RETURN(StrToLong(oldValueStr, storedValueNum), "convert string to long failed.", ACLSHMEM_SMEM_ERROR);

        storedValueNum += valueNum;
        auto storedValueStr = std::to_string(storedValueNum);
        pos->second = std::vector<uint8_t>(storedValueStr.begin(), storedValueStr.end());
        responseValue = storedValueNum;
    }
    lockGuard.unlock();
    SHM_LOG_DEBUG("ADD REQUEST(" << context.SeqNo() << ") for key(" << key << ") value(" << responseValue << ") end.");
    ReplyWithMessage(context, StoreErrorCode::SUCCESS, std::to_string(responseValue));
    if (!wakeupWaiters.empty()) {
        WakeupWaiters(wakeupWaiters, reqVal);
    }
    return ACLSHMEM_SUCCESS;
}

Result AccStoreServer::RemoveHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept
{
    if (request.keys.size() != 1 || !request.values.empty()) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle invalid body");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request: key should be one and no values.");
        return SM_INVALID_PARAM;
    }

    auto &key = request.keys[0];
    if (key.length() > MAX_KEY_LEN_SERVER) {
        SHM_LOG_ERROR("key length too large, length: " << key.length());
        return StoreErrorCode::INVALID_KEY;
    }

    SHM_LOG_DEBUG("REMOVE REQUEST(" << context.SeqNo() << ") for key(" << key << ") start.");
    bool removed = false;
    std::unique_lock<std::mutex> lockGuard{storeMutex_};
    auto pos = kvStore_.find(key);
    if (pos != kvStore_.end()) {
        kvStore_.erase(pos);
        removed = true;
    }
    lockGuard.unlock();
    ReplyWithMessage(context, removed ? StoreErrorCode::SUCCESS : StoreErrorCode::NOT_EXIST,
                     removed ? "success" : "not exist");

    return ACLSHMEM_SUCCESS;
}

Result AccStoreServer::AppendHandler(const shm::acc::AccTcpRequestContext &context, SmemMessage &request) noexcept
{
    if (request.keys.size() != 1 || request.values.size() != 1) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle invalid body");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request: key & value should be one.");
        return SM_INVALID_PARAM;
    }

    auto &key = request.keys[0];
    auto &value = request.values[0];
    if (key.length() > MAX_KEY_LEN_SERVER) {
        SHM_LOG_ERROR("key length too large, length: " << key.length());
        return StoreErrorCode::INVALID_KEY;
    }

    SHM_LOG_DEBUG("APPEND REQUEST(" << context.SeqNo() << ") for key(" << key << ") start.");
    uint64_t newSize;
    std::list<shm::acc::AccTcpRequestContext> wakeupWaiters;
    std::vector<uint8_t> reqVal;
    std::unique_lock<std::mutex> lockGuard{storeMutex_};
    auto pos = kvStore_.find(key);
    if (pos != kvStore_.end()) {
        pos->second.insert(pos->second.end(), value.begin(), value.end());
        newSize = pos->second.size();
    } else {
        newSize = value.size();
        auto wPos = keyWaiters_.find(key);
        if (wPos != keyWaiters_.end()) {
            wakeupWaiters = GetOutWaitersInLock(wPos->second);
            reqVal = value;
            keyWaiters_.erase(wPos);
        }
        kvStore_.emplace(key, std::move(value));
    }
    lockGuard.unlock();
    ReplyWithMessage(context, StoreErrorCode::SUCCESS, std::to_string(newSize));
    if (!wakeupWaiters.empty()) {
        WakeupWaiters(wakeupWaiters, value);
    }

    return ACLSHMEM_SUCCESS;
}

Result AccStoreServer::CasHandler(const shm::acc::AccTcpRequestContext &context,
                                  shm::store::SmemMessage &request) noexcept
{
    const size_t EXPECTED_KEYS_SIZE = 1;
    const size_t EXPECTED_VALUES_SIZE = 2;
    if (request.keys.size() != EXPECTED_KEYS_SIZE || request.values.size() != EXPECTED_VALUES_SIZE) {
        SHM_LOG_ERROR("request(" << context.SeqNo() << ") handle invalid body");
        ReplyWithMessage(context, StoreErrorCode::INVALID_MESSAGE, "invalid request: count(key)=1 & count(value)=2");
        return SM_INVALID_PARAM;
    }

    auto &key = request.keys[0];
    auto &expected = request.values[0];
    auto &exchange = request.values[1];
    auto newValue = exchange;
    if (key.length() > MAX_KEY_LEN_SERVER) {
        SHM_LOG_ERROR("key length too large, length: " << key.length());
        return StoreErrorCode::INVALID_KEY;
    }

    std::vector<uint8_t> exists;
    SmemMessage responseMessage{request.mt};
    std::list<shm::acc::AccTcpRequestContext> wakeupWaiters;
    SHM_LOG_DEBUG("CAS REQUEST(" << context.SeqNo() << ") for key(" << key << ") start.");

    std::unique_lock<std::mutex> lockGuard{storeMutex_};
    auto pos = kvStore_.find(key);
    if (pos != kvStore_.end()) {
        if (expected == pos->second) {
            exists = std::move(pos->second);
            pos->second = std::move(exchange);
        } else {
            exists = pos->second;
        }
    } else {
        if (expected.empty()) {
            kvStore_.emplace(key, std::move(exchange));
            auto wPos = keyWaiters_.find(key);
            if (wPos != keyWaiters_.end()) {
                wakeupWaiters = GetOutWaitersInLock(wPos->second);
                keyWaiters_.erase(wPos);
            }
        }
    }
    lockGuard.unlock();
    SHM_LOG_DEBUG("CAS REQUEST(" << context.SeqNo() << ") for key(" << key << ") finished.");

    responseMessage.values.push_back(exists);
    auto response = SmemMessagePacker::Pack(responseMessage);
    ReplyWithMessage(context, StoreErrorCode::SUCCESS, response);
    if (!wakeupWaiters.empty()) {
        WakeupWaiters(wakeupWaiters, newValue);
    }
    return ACLSHMEM_SUCCESS;
}

std::list<shm::acc::AccTcpRequestContext> AccStoreServer::GetOutWaitersInLock(
    const std::unordered_set<uint64_t> &ids) noexcept
{
    std::list<shm::acc::AccTcpRequestContext> reqCtx;
    for (auto id : ids) {
        auto it = waitCtx_.find(id);
        if (it != waitCtx_.end()) {
            reqCtx.emplace_back(std::move(it->second.ReqCtx()));
            auto wit = timedWaiters_.find(it->second.TimeoutMs());
            if (wit != timedWaiters_.end()) {
                wit->second.erase(it->second.Id());
                if (wit->second.empty()) {
                    timedWaiters_.erase(wit);
                }
            }
            waitCtx_.erase(it);
        }
    }
    return std::move(reqCtx);
}

void AccStoreServer::WakeupWaiters(const std::list<shm::acc::AccTcpRequestContext> &waiters,
                                   const std::vector<uint8_t> &value) noexcept
{
    SmemMessage responseMessage{MessageType::GET};
    responseMessage.values.push_back(value);
    auto response = SmemMessagePacker::Pack(responseMessage);
    for (auto &context : waiters) {
        SHM_LOG_DEBUG("WAKEUP REQUEST(" << context.SeqNo() << ").");
        ReplyWithMessage(context, StoreErrorCode::SUCCESS, response);
    }
}

void AccStoreServer::ReplyWithMessage(const shm::acc::AccTcpRequestContext &ctx, int16_t code,
                                      const std::string &message) noexcept
{
    auto response = shm::acc::AccDataBuffer::Create(message.c_str(), message.size());
    if (response == nullptr) {
        SHM_LOG_ERROR("create response message failed");
        return;
    }

    ctx.Reply(code, response);
}

void AccStoreServer::ReplyWithMessage(const shm::acc::AccTcpRequestContext &ctx, int16_t code,
                                      const std::vector<uint8_t> &message) noexcept
{
    auto response = shm::acc::AccDataBuffer::Create(message.data(), message.size());
    if (response == nullptr) {
        SHM_LOG_ERROR("create response message failed");
        return;
    }

    ctx.Reply(code, response);
}

void AccStoreServer::TimerThreadTask() noexcept
{
    std::unordered_set<uint64_t> timeoutIds;
    std::unique_lock<std::mutex> lockerGuard{storeMutex_};
    while (running_) {
        auto now = std::chrono::steady_clock::now().time_since_epoch();
        auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now).count();
        while (!timedWaiters_.empty()) {
            auto it = timedWaiters_.begin();
            if (it->first > timestamp) {
                break;
            }
            timeoutIds.insert(it->second.begin(), it->second.end());
            timedWaiters_.erase(it);
        }

        auto timeoutContexts = GetOutWaitersInLock(timeoutIds);
        lockerGuard.unlock();

        timeoutIds.clear();
        for (auto &ctx : timeoutContexts) {
            SHM_LOG_DEBUG("reply timeout response for : " << ctx.SeqNo());
            ReplyWithMessage(ctx, StoreErrorCode::TIMEOUT, "<timeout>");
        }

        lockerGuard.lock();
        storeCond_.wait_for(lockerGuard, std::chrono::milliseconds(1), [this]() { return !running_; });
    }
}
}  // namespace store
}  // namespace shm