/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netdb.h>
#include <net/if.h>

#include "shmemi_host_def.h"
#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"
#include "utils/shmemi_host_types.h"
#include "init/bootstrap/shmemi_bootstrap.h"

#include "store_op.h"
#include "store_net_group_engine.h"
#include "store_factory.h"
#include "sotre_net.h"
#include "store_net_common.h"

#include "socket/uid_utils.h"
#include "config_store/store_net_utils.h"

constexpr int DEFAULT_ID = 0;

static std::mutex g_store_mutex;
// Count Used to guard Multi instance scenario
static int g_store_ref = 0;

struct ConfigStoreState {
    shm::store::StorePtr store_ = nullptr;
    shm::store::SmemGroupEnginePtr group_engine_ = nullptr;
};

using namespace shm::utils;

int config_store_get_unique_id(void* uid) {
    char pta_env_ip[MAX_IP];
    uint16_t pta_env_port;
    sa_family_t sockType;
    const char *ipPort = std::getenv("SHMEM_UID_SESSION_ID");
    const char *ipInfo = std::getenv("SHMEM_UID_SOCK_IFNAME");
    bool is_from_ifa = false;
    if (ipPort != nullptr) {
        if (aclshmemi_get_ip_from_env(pta_env_ip, pta_env_port, sockType, ipPort) != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("cant get pta master addr.");
            return ACLSHMEM_INVALID_PARAM;
        }
    } else {
        is_from_ifa = true;
        if (aclshmemi_get_ip_from_ifa(pta_env_ip, sockType, ipInfo) != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("cant get available ip port.");
            return ACLSHMEM_INVALID_PARAM;
        }
    }
    SHM_LOG_INFO("get master IP value:" << pta_env_ip);
    return aclshmemi_set_ip_info((aclshmemx_uniqueid_t *)uid, sockType, pta_env_ip, pta_env_port, is_from_ifa);
}

// Plugin pre-initialization entry function. 
int aclshmemi_bootstrap_plugin_pre_init(aclshmemi_bootstrap_handle_t* handle)
{
    if (!handle) {
        SHM_LOG_ERROR("bootstrap pre_init: invalid arguments (nullptr).");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (handle->pre_init_ops == nullptr) {
        SHM_LOG_DEBUG(" bootstrap plugin pre init start.");
        ACLSHMEM_CHECK_RET(ACLSHMEM_BOOTSTRAP_CALLOC(&handle->pre_init_ops, 1));
        handle->pre_init_ops->get_unique_id = config_store_get_unique_id;
        handle->pre_init_ops->get_unique_id_static_magic = nullptr;
        handle->pre_init_ops->cookie = nullptr;
        SHM_LOG_DEBUG(" bootstrap plugin pre init end.");
    } else {
        SHM_LOG_DEBUG(" pre_init_ops had already prepared.");
    }
    return ACLSHMEM_SUCCESS;
}

void config_store_bootstrap_global_exit(int status, aclshmemi_bootstrap_handle_t* handle)
{
    if (!handle) {
        SHM_LOG_ERROR("bootstrap global_exit: invalid arguments (nullptr).");
        return;
    }

    // Get Bootstrap state
    auto state = static_cast<ConfigStoreState*>(handle->bootstrap_state);
    if (!state) {
        SHM_LOG_ERROR("handle->state is NULL"); 
        return;
    }

    if (state->group_engine_ == nullptr) {
        SHM_LOG_ERROR("Group is NULL"); 
        return; 
    } 
    state->group_engine_->GroupBroadcastExit(status);
}

int32_t config_store_set_tls_info(bool enable, const char *tls_info, const uint32_t tls_info_len)
{
    return shm::store::StoreFactory::SetTlsInfo(enable, tls_info, tls_info_len);
}

int32_t config_store_bootstrap_set_tls_key(
    const char *tls_pk, const uint32_t tls_pk_len,
    const char *tls_pk_pw, const uint32_t tls_pk_pw_len, const aclshmem_decrypt_handler decrypt_handler)
{
    return shm::store::StoreFactory::SetTlsPkInfo(tls_pk, tls_pk_len, tls_pk_pw, tls_pk_pw_len, decrypt_handler);
}

static int config_store_bootstrap_finalize(aclshmemi_bootstrap_handle_t *handle) {
    std::lock_guard<std::mutex> lock(g_store_mutex);
    g_store_ref--;

    if (!handle) {
        return ACLSHMEM_SUCCESS;
    }

    if (handle->pre_init_ops) {
        ACLSHMEM_BOOTSTRAP_PTR_FREE(handle->pre_init_ops);
        handle->pre_init_ops = nullptr;
    }

    // Get Bootstrap state
    auto state = static_cast<ConfigStoreState*>(handle->bootstrap_state);
    if (!state) return ACLSHMEM_INNER_ERROR;

    // DestroyStore will destroy global resource, only do this when all stores are finalized.
    if (state->store_ != nullptr) {
        if (g_store_ref == 0) {
            shm::store::StoreFactory::DestroyStore();
            SHM_LOG_INFO("Global Store Destroyed. ");
        } else {
            SHM_LOG_INFO("Store ref count is " << g_store_ref << ", Skip Global DestroyStore.");
        }
    }

    state->group_engine_ = nullptr;
    state->store_ = nullptr;
    delete state;

    return ACLSHMEM_SUCCESS;
}

static int config_store_bootstrap_allgather(const void *in, void *out, int len, aclshmemi_bootstrap_handle_t *handle) {
    if (!in || !out || !handle) {
        SHM_LOG_ERROR("bootstrap allgather: invalid arguments (nullptr).");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    // Get Bootstrap state
    auto state = static_cast<ConfigStoreState*>(handle->bootstrap_state);
    if (!state) return ACLSHMEM_INNER_ERROR;    

    int rank = handle->mype;
    int nranks = handle->npes;

    auto ret = (shm::store::SmemGroupEnginePtr(state->group_engine_))->GroupAllGather((char *)in, len, (char *)out, len * nranks);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Group AllGather timeout or store failure");
        return ACLSHMEM_SMEM_ERROR;
    }
    return ACLSHMEM_SUCCESS;
}

static int config_store_bootstrap_barrier(aclshmemi_bootstrap_handle_t *handle) {
    if (!handle) {
        SHM_LOG_ERROR("bootstrap barrier: invalid arguments (nullptr).");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    // Get Bootstrap state
    auto state = static_cast<ConfigStoreState*>(handle->bootstrap_state);
    if (!state) return ACLSHMEM_INNER_ERROR; 

    int rank = handle->mype;
    int tag = 0;
    int nranks = handle->npes;

    if (nranks == 1) {
        SHM_LOG_DEBUG("Single rank, skip barrier");
        return ACLSHMEM_SUCCESS;
    }

    SHM_LOG_INFO("group_engine_bootstrap_barrier");
    SHM_LOG_DEBUG("Barrier start. rank: " << rank << " nranks: " << nranks <<" tag: "<< tag);

    if (state->group_engine_ == nullptr) {
        SHM_LOG_ERROR("Group is NULL");
        return ACLSHMEM_INNER_ERROR;
    }

    auto ret = (shm::store::SmemGroupEnginePtr(state->group_engine_))->GroupBarrier();
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("Group barrier timeout or store failure");
        return ACLSHMEM_SMEM_ERROR;
    }

    SHM_LOG_DEBUG("Barrier end. rank: " << rank << " nranks: " << nranks <<" tag: "<< tag);
    return ACLSHMEM_SUCCESS;
}

int32_t init_config_store(aclshmemi_bootstrap_handle_t* handle)
{
    if (!handle) {
        SHM_LOG_ERROR("bootstrap init_config_store: invalid arguments (nullptr).");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    // Get Bootstrap state
    auto state = static_cast<ConfigStoreState*>(handle->bootstrap_state);
    if (!state) return ACLSHMEM_INNER_ERROR;

    shm::store::StoreFactory::SetTlsInfo(false, nullptr, 0);
    int32_t sock_fd = handle->sockFd;
    shm::store::UrlExtraction option;
    std::string url(handle->ipport);
    SHM_ASSERT_RETURN(option.ExtractIpPortFromUrl(url) == ACLSHMEM_SUCCESS, ACLSHMEM_INVALID_PARAM);
    if (handle->mype == 0) {
        state->store_ = shm::store::StoreFactory::CreateStore(option.ip, option.port, true, 0, -1, sock_fd);
    }
    else {
        state->store_ = shm::store::StoreFactory::CreateStore(option.ip, option.port, false, handle->mype, handle->timeOut);
    }
    return ACLSHMEM_SUCCESS;
}

int32_t init_group_engine(aclshmemi_bootstrap_handle_t* handle)
{
    if (!handle) {
        SHM_LOG_ERROR("bootstrap init_group_engine: invalid arguments (nullptr).");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    // Get Bootstrap state
    auto state = static_cast<ConfigStoreState*>(handle->bootstrap_state);
    if (!state) return ACLSHMEM_INNER_ERROR;

    std::string prefix = "SHM_(" + std::to_string(DEFAULT_ID) + ")_";
    shm::store::StorePtr store_ptr = shm::store::StoreFactory::PrefixStore(state->store_, prefix);
    shm::store::SmemGroupOption opt = {
        (uint32_t)handle->npes, (uint32_t)handle->mype, handle->timeControlOut * 1000U, false, nullptr, nullptr};
    shm::store::SmemGroupEnginePtr group = shm::store::SmemNetGroupEngine::Create(store_ptr, opt);
    SHM_ASSERT_RETURN(group != nullptr, ACLSHMEM_SMEM_ERROR);
    state->group_engine_ = group;
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_bootstrap_plugin_init(void* comm, aclshmemi_bootstrap_handle_t* handle)
{
    std::lock_guard<std::mutex> lock(g_store_mutex);
    g_store_ref++;

    int status = ACLSHMEM_SUCCESS;
    if (handle == nullptr) {
        SHM_LOG_ERROR(" aclshmemi_bootstrap_plugin_init: invalid arguments (nullptr)");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    ConfigStoreState *state = new (std::nothrow) ConfigStoreState();
    if (state == nullptr) {
        SHM_LOG_ERROR("Failed to allocate memory for bootstrap state. ");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    handle->bootstrap_state = static_cast<void *>(state);

    /* If we don't get ip_port, now get it from uid. */
    std::string ipPort;
    if (handle->use_attr_ipport == false) {
        aclshmemi_bootstrap_uid_state_t* uid_args = (aclshmemi_bootstrap_uid_state_t*)comm;
        if (uid_args->addr.type == ADDR_IPv6) {
            char ipStr[INET6_ADDRSTRLEN] = {0};
            if (inet_ntop(AF_INET6, &(uid_args->addr.addr.addr6.sin6_addr), ipStr, sizeof(ipStr)) == nullptr) {
                SHM_LOG_ERROR("inet_ntop failed for IPv6");
                return ACLSHMEM_INNER_ERROR;
            }
            uint16_t port = ntohs(uid_args->addr.addr.addr6.sin6_port);
            ipPort = "tcp6://[" + std::string(ipStr) + "]:" + std::to_string(port);
        } else {
            char ipStr[INET_ADDRSTRLEN] = {0};
            if (inet_ntop(AF_INET, &(uid_args->addr.addr.addr4.sin_addr), ipStr, sizeof(ipStr)) == nullptr) {
                SHM_LOG_ERROR("inet_ntop failed for IPv4");
                return ACLSHMEM_INNER_ERROR;
            }
            uint16_t port = ntohs(uid_args->addr.addr.addr4.sin_port);
            ipPort = "tcp://" + std::string(ipStr) + ":" + std::to_string(port);
        }

        /* Set bootstrap Ip params. */
        handle->sockFd = uid_args->inner_sockFd;
        handle->timeOut = DEFAULT_TIMEOUT;
        handle->timeControlOut = DEFAULT_TIMEOUT;

        size_t ip_len = sizeof(handle->ipport);
        if (ipPort.length() > ip_len) {
            SHM_LOG_ERROR("Generated IP String Illegal!!");
            return ACLSHMEM_INNER_ERROR;
        }

        strncpy(handle->ipport, ipPort.c_str(), ip_len - 1);
        handle->ipport[ip_len - 1] = '\0';
        SHM_LOG_INFO("extract ip port:" << ipPort);
    }

    /* Set TLS Config */
    status = config_store_set_tls_info(handle->tls_enable, handle->tls_info, handle->tls_info_len);
    if (handle->tls_enable == true) {
        status = config_store_bootstrap_set_tls_key(
                    handle->tls_pk, handle->tls_pk_len, 
                    handle->tls_pk_pw, handle->tls_pk_pw_len, handle->decrypt_handler);
    }

    /* Init Config Store */
    status = init_config_store(handle);
    if (status != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("shmem init config store failed, error: " << status);
        return status;
    }

    /* Init Host Communication Group Engine */
    status = init_group_engine(handle);
    if (status != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("shmem init group engine failed, error: " << status);
        return status;
    }

    handle->allgather = config_store_bootstrap_allgather;
    handle->barrier = config_store_bootstrap_barrier;
    handle->finalize = config_store_bootstrap_finalize;
    handle->alltoall = nullptr;
    handle->global_exit = config_store_bootstrap_global_exit;

    SHM_LOG_INFO("pe " << handle->mype << ": bootstrap plugin initialized successfully");
    return ACLSHMEM_SUCCESS;
}