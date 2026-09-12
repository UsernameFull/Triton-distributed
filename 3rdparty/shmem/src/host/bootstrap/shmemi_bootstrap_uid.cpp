/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <fstream>
#include <ifaddrs.h>
#include <arpa/inet.h>
#include <net/if.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/resource.h>
#include <algorithm>
#include "socket/uid_socket.h"
#include "socket/uid_utils.h"

#define MAX_ATTEMPTS 500
#define MAX_IFCONFIG_LENGTH 23
#define MAX_IP 48
#define DEFAULT_IFNAME_LNEGTH 4
#define BOOTSTRAP_IN_PLACE (void*)0x1
#define SOCKET_MAGIC 0x243ab9f2fc4b9d6cULL

static const char* env_ip_port = nullptr;
static const char* env_ifname = nullptr;
static aclshmemi_bootstrap_uid_state_t aclshmemi_bootstrap_uid_state;
static struct bootstrap_netstate priv_info;
static int static_magic_count = 1;

bool is_ipv6_loopback(const struct in6_addr *addr6) {
    static const struct in6_addr loopback6 = IN6ADDR_LOOPBACK_INIT;
    return memcmp(addr6, &loopback6, sizeof(struct in6_addr)) == 0;
}

// fe80 check
bool is_ipv6_link_local(const struct in6_addr *addr6) {
    if (addr6 == nullptr) {
        return false;
    }

    const uint8_t* bytes = addr6->s6_addr;
    
    if (bytes[0] != 0xfe) {
        return false;
    }
    
    if ((bytes[1] & 0xc0) != 0x80) {
        return false;
    }
    
    SHM_LOG_DEBUG("It is IPv6 link-local address (fe80::/10).");
    return true;
}

bool is_ipv4_loopback(const struct in_addr *addr4) {
    return ((ntohl(addr4->s_addr) >> 24) & 0xFF) == IN_LOOPBACKNET;
}

bool is_ipv4_link_local(const struct in_addr *addr4) {
    if (addr4 == nullptr) {
        return false;
    }

    uint32_t ip_addr = ntohl(addr4->s_addr);
    
    uint8_t byte1 = (ip_addr >> 24) & 0xff;
    uint8_t byte2 = (ip_addr >> 16) & 0xff;
    if (byte1 != 169 || byte2 != 254) {
        return false;
    }
    SHM_LOG_DEBUG("It is IPv4 link-local address (169.254.x.x).");
    return true;
}

static bool is_loopback_addr(const sockaddr_t* addr) {
    if (addr == nullptr) {
        return false;
    }
    if (addr->type == ADDR_IPv4) {
        return is_ipv4_loopback(&addr->addr.addr4.sin_addr);
    } else if (addr->type == ADDR_IPv6) {
        return is_ipv6_loopback(&addr->addr.addr6.sin6_addr);
    } else {
        return false;
    }
}

static bool is_link_local_addr(const sockaddr_t* addr) {
    if (addr == nullptr) {
        return false;
    }
    if (addr->type == ADDR_IPv4) {
        return is_ipv4_link_local(&addr->addr.addr4.sin_addr);
    } else if (addr->type == ADDR_IPv6) {
        return is_ipv6_link_local(&addr->addr.addr6.sin6_addr);
    } else {
        return false;
    }
}

static bool aclshmemi_parse_port(const std::string &portStr, uint16_t &port)
{
    constexpr int maxPort = 65535;
    try {
        size_t parsedLen = 0;
        int portInt = std::stoi(portStr, &parsedLen);
        if (parsedLen != portStr.size()) {
            SHM_LOG_ERROR("Invalid port format: " << portStr);
            return false;
        }
        if (portInt < 0 || portInt > maxPort) {
            SHM_LOG_ERROR("Invalid port value: " << portInt << ", must be in range [0, " << maxPort << "]");
            return false;
        }
        port = static_cast<uint16_t>(portInt);
        return true;
    } catch (const std::exception &e) {
        SHM_LOG_ERROR("Invalid port format: " << e.what());
        return false;
    }
}

static int32_t aclshmemi_get_uid_magic(aclshmemi_bootstrap_uid_state_t *innerUId)
{
    std::ifstream urandom("/dev/urandom", std::ios::binary);
    if (!urandom) {
        SHM_LOG_ERROR("open random failed");
        return ACLSHMEM_INNER_ERROR;
    }
    
    urandom.read(reinterpret_cast<char *>(&innerUId->magic), sizeof(innerUId->magic));
    if (urandom.fail()) {
        SHM_LOG_ERROR("read random failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_DEBUG("init magic id to " << innerUId->magic);
    return ACLSHMEM_SUCCESS;
}


static int32_t aclshmemi_uid_parse_interface_with_type(const char *ipInfo, char *IP, sa_family_t &sockType, bool &flag)
{
    const char *delim = ":";
    const char *sep = strchr(ipInfo, delim[0]);
    if (sep != nullptr) {
        size_t leftLen = sep - ipInfo;
        if (leftLen >= MAX_IFCONFIG_LENGTH - 1 || leftLen == 0) {
            SHM_LOG_ERROR("Invalid interface prefix length: " << leftLen);
            return ACLSHMEM_INVALID_VALUE;
        }
        strncpy(IP, ipInfo, leftLen);
        IP[leftLen] = '\0';
        sockType = (strcmp(sep + 1, "inet6") != 0) ? AF_INET : AF_INET6;
        flag = true;
        SHM_LOG_INFO("Parse ipInfo success: ifaPrefix=" << IP << ", sockType=" << (sockType == AF_INET ? "IPv4" : "IPv6"));
    }
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_traverse_ifa(
    struct ifaddrs *ifaddr,
    sa_family_t &sockType,
    bool flag,
    const char **prefixes,
    bool exclude,
    aclshmemi_bootstrap_uid_state_t *uid_args,
    bool skipStateCheck = false,
    bool allow_local = false
) {
    for (struct ifaddrs *ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) continue;
        const char* ifname = ifa->ifa_name;
        if (!allow_local && (strstr(ifname, "lo") != nullptr || strstr(ifname, "docker") != nullptr)) {
            SHM_LOG_DEBUG("Skip interface: " << ifname << " (lo/docker, allow_local=false)");
            continue;
        }

        bool match = false;
        const char **p = prefixes;
        
        while (*p != nullptr) {
            if (**p == '\0') {
                p++;
                continue;
            }
            size_t prefix_len = strlen(*p);
            size_t ifname_len = strlen(ifa->ifa_name);
            if (ifname_len < prefix_len) {
                p++;
                continue;
            }
            if (strncmp(ifa->ifa_name, *p, prefix_len) == 0) {
                match = true;
                break;
            }
            p++;
        }
        if (exclude && match) continue;
        if (!exclude && !match) continue;

        if (!skipStateCheck && (!(ifa->ifa_flags & IFF_UP) || !(ifa->ifa_flags & IFF_RUNNING))) continue;

        if (flag) {
            if (ifa->ifa_addr->sa_family != sockType) {
                SHM_LOG_DEBUG("Protocol family not match (flag=true), interface: " << ifa->ifa_name << ", get: " << ifa->ifa_addr->sa_family << ", expect: "<< sockType);
                continue;
            }
        }
        bool is_invalid_addr = false;
        if (!allow_local) {
            if (ifa->ifa_addr->sa_family == AF_INET) {
                struct sockaddr_in *addr4 = (struct sockaddr_in *)ifa->ifa_addr;
                if (is_ipv4_link_local(&addr4->sin_addr)) {
                    SHM_LOG_INFO("Blocked ipv4 link local address.");
                    continue;
                }
                if (is_ipv4_loopback(&addr4->sin_addr)) {
                    is_invalid_addr = true;
                }
            } else if (ifa->ifa_addr->sa_family == AF_INET6) {
                struct sockaddr_in6 *addr6 = (struct sockaddr_in6 *)ifa->ifa_addr;
                if (is_ipv6_link_local(&addr6->sin6_addr)) {
                    SHM_LOG_INFO("Blocked ipv6 link local address.");
                    continue;
                }
                if (is_ipv6_loopback(&addr6->sin6_addr)) {
                    is_invalid_addr = true;
                }
            }
        }
        if (is_invalid_addr) {
            SHM_LOG_DEBUG("Skip invalid address (lo/fe80, allow_local=false) on interface: " << ifname);
            continue;
        }

        if (ifa->ifa_addr->sa_family == AF_INET && (sockType == AF_UNSPEC || sockType == AF_INET)) {
            std::fill(reinterpret_cast<char*>(&uid_args->addr.addr.addr4), 
                      reinterpret_cast<char*>(&uid_args->addr.addr.addr4) + sizeof(struct sockaddr_in), 
                      0);
            uid_args->addr.type = ADDR_IPv4;
            uid_args->addr.addr.addr4 = *(struct sockaddr_in *)ifa->ifa_addr;
            uid_args->addr.addr.addr4.sin_port = 0;
            sockType = AF_INET;
            SHM_LOG_INFO("Assign IPv4 from interface: " << ifa->ifa_name);
            return ACLSHMEM_SUCCESS;
        }

        if (ifa->ifa_addr->sa_family == AF_INET6 && (sockType == AF_UNSPEC || sockType == AF_INET6)) {
            std::fill(reinterpret_cast<char*>(&uid_args->addr.addr.addr6), 
                      reinterpret_cast<char*>(&uid_args->addr.addr.addr6) + sizeof(struct sockaddr_in6), 
                      0);
            uid_args->addr.type = ADDR_IPv6;
            uid_args->addr.addr.addr6 = *(struct sockaddr_in6 *)ifa->ifa_addr;
            uid_args->addr.addr.addr6.sin6_port = 0;
            uid_args->addr.addr.addr6.sin6_flowinfo = 0;

            sockType = AF_INET6;
            SHM_LOG_INFO("Assign IPv6 from interface: " << ifa->ifa_name <<" scope_id: " << uid_args->addr.addr.addr6.sin6_scope_id);
            return ACLSHMEM_SUCCESS;
        }
    }
    return ACLSHMEM_INVALID_PARAM;
}

int32_t aclshmemi_get_ip_from_env(aclshmemi_bootstrap_uid_state_t *uid_args, const char *ipPort) {
    if (uid_args == nullptr || ipPort == nullptr || strlen(ipPort) == 0) {
        SHM_LOG_ERROR("Invalid param: uid_args is null or ipPort is empty");
        return ACLSHMEM_INVALID_PARAM;
    }

    aclshmemi_get_uid_magic(uid_args);
    SHM_LOG_DEBUG("get env SHMEM_UID_SESSION_ID value: " << ipPort);
    std::string ipPortStr = ipPort;
    
    if (ipPort[0] == '[') {
        size_t bracketEnd = ipPortStr.find_last_of(']');
        if (bracketEnd == std::string::npos) {
            SHM_LOG_ERROR("Invalid IPv6 format: no closing ']'");
            return ACLSHMEM_INVALID_PARAM;
        }
        if (bracketEnd + 1 >= ipPortStr.size() || ipPortStr[bracketEnd + 1] != ':') {
            SHM_LOG_ERROR("Invalid IPv6 format: missing ':' after ']'");
            return ACLSHMEM_INVALID_PARAM;
        }

        std::string ipWithScope = ipPortStr.substr(1, bracketEnd - 1);
        size_t scopeSep = ipWithScope.find('%');
        std::string ipStr;
        std::string ifName;

        std::fill(reinterpret_cast<char*>(&uid_args->addr.addr.addr6), 
                  reinterpret_cast<char*>(&uid_args->addr.addr.addr6) + sizeof(struct sockaddr_in6), 
                  0);
        uid_args->addr.type = ADDR_IPv6;
        uid_args->addr.addr.addr6.sin6_family = AF_INET6;

        if (scopeSep != std::string::npos) {
            ipStr = ipWithScope.substr(0, scopeSep);
            ifName = ipWithScope.substr(scopeSep + 1);
            uid_args->addr.addr.addr6.sin6_scope_id = if_nametoindex(ifName.c_str());
            if (uid_args->addr.addr.addr6.sin6_scope_id == 0) {
                SHM_LOG_INFO("Interface " << ifName.c_str() << "not found, scope_id set to 0");
            }
        } else {
            ipStr = ipWithScope;
            uid_args->addr.addr.addr6.sin6_scope_id = 0;
        }

        std::string portStr = ipPortStr.substr(bracketEnd + 2);
        if (portStr.empty()) {
            SHM_LOG_ERROR("IPv6 port is empty");
            return ACLSHMEM_INVALID_PARAM;
        }
        
        uint16_t port = 0;
        if (!aclshmemi_parse_port(portStr, port)) {
            return ACLSHMEM_INVALID_PARAM;
        }
        uid_args->addr.addr.addr6.sin6_port = htons(port);
        uid_args->addr.addr.addr6.sin6_flowinfo = 0;

        if (inet_pton(AF_INET6, ipStr.c_str(), &uid_args->addr.addr.addr6.sin6_addr) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv6 failed: " << strerror(errno));
            return ACLSHMEM_NOT_INITED;
        }
    } else {
        size_t colonPos = ipPortStr.find_last_of(':');
        if (colonPos == std::string::npos || ipPortStr.length() - colonPos <= 1) {
            SHM_LOG_ERROR("Invalid IPv4 format: no colon separator");
            return ACLSHMEM_INVALID_PARAM;
        }

        std::string ipStr = ipPortStr.substr(0, colonPos);
        std::string portStr = ipPortStr.substr(colonPos + 1);

        std::fill(reinterpret_cast<char*>(&uid_args->addr.addr.addr4), 
                  reinterpret_cast<char*>(&uid_args->addr.addr.addr4) + sizeof(struct sockaddr_in), 
                  0);
        uid_args->addr.type = ADDR_IPv4;
        uid_args->addr.addr.addr4.sin_family = AF_INET;
        
        uint16_t port = 0;
        if (!aclshmemi_parse_port(portStr, port)) {
            return ACLSHMEM_INVALID_PARAM;
        }
        uid_args->addr.addr.addr4.sin_port = htons(port);

        if (inet_pton(AF_INET, ipStr.c_str(), &uid_args->addr.addr.addr4.sin_addr) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv4 failed: " << strerror(errno));
            return ACLSHMEM_NOT_INITED;
        }
    }

    SHM_LOG_INFO("Assign IP/Port from env success");
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_get_ip_from_ifa(aclshmemi_bootstrap_uid_state_t *uid_args, const char *ipInfo) {
    if (uid_args == nullptr) {
        SHM_LOG_ERROR("uid_args is nullptr");
        return ACLSHMEM_INVALID_PARAM;
    }

    struct ifaddrs *ifaddr = nullptr;
    char ifaPrefix[MAX_IFCONFIG_LENGTH] = {0};
    bool flag = false;
    sa_family_t sockType = AF_INET;
    bool foundValidIp = false;

    aclshmemi_get_uid_magic(uid_args);

    bool isIpInfoConfigured = (ipInfo != nullptr && strlen(ipInfo) > 0);
    if (isIpInfoConfigured) {
        int32_t ret = aclshmemi_uid_parse_interface_with_type(ipInfo, ifaPrefix, sockType, flag);
        if (ret != ACLSHMEM_SUCCESS) {
            SHM_LOG_ERROR("Parse ipInfo failed, ret: " << ret);
            return ret;
        }
    }
    bool allow_local = isIpInfoConfigured;

    if (getifaddrs(&ifaddr) == -1) {
        SHM_LOG_ERROR("getifaddrs failed: " << strerror(errno));
        return ACLSHMEM_INVALID_PARAM;
    }

    if (isIpInfoConfigured) {
        const char *specifiedPrefixes[] = {ifaPrefix, nullptr};
        SHM_LOG_INFO("Search interface with specified prefix: " << ifaPrefix);
        foundValidIp = (aclshmemi_traverse_ifa(ifaddr, sockType, flag, specifiedPrefixes, false, uid_args, false, allow_local) == ACLSHMEM_SUCCESS);
    } else {
        const char *ethPrefixes[] = {"eth", nullptr};
        const char *excludePrefixes[] = {"docker", "lo", nullptr};

        SHM_LOG_INFO("Step 1: Search interfaces match 'eth'");
        foundValidIp = (aclshmemi_traverse_ifa(ifaddr, sockType, flag, ethPrefixes, false, uid_args, false, allow_local) == ACLSHMEM_SUCCESS);

        if (!foundValidIp) {
            SHM_LOG_INFO("Step 3: Search interfaces exclude 'docker' and 'lo/fe80'");
            foundValidIp = (aclshmemi_traverse_ifa(ifaddr, sockType, flag, excludePrefixes, true, uid_args, false, allow_local) == ACLSHMEM_SUCCESS);
        }
    }

    if (ifaddr != nullptr) {
        freeifaddrs(ifaddr);
        ifaddr = nullptr;
    }
    if (!foundValidIp) {
        SHM_LOG_ERROR("No valid IP address found from any interface!");
        return ACLSHMEM_INVALID_PARAM;
    }

    SHM_LOG_INFO("Assign IP/Port from interface success");
    return ACLSHMEM_SUCCESS;
}

int32_t aclshmemi_set_ip_info(void *uid, sa_family_t &sockType, char *pta_env_ip, uint16_t pta_env_port,
                          bool is_from_ifa)
{
    // init default uid
    aclshmemi_bootstrap_uid_state_t *innerUID = (aclshmemi_bootstrap_uid_state_t *)(uid);
    SHM_LOG_INFO(" ENV IP: " << pta_env_ip << " ENV port: " << pta_env_port << " sockType: " << sockType);
    ACLSHMEM_CHECK_RET(aclshmemi_get_uid_magic(innerUID));

    // fill ip port as part of uid
    uint16_t port = 0;
    if (is_from_ifa) {
        SHM_LOG_DEBUG("Automatically obtain the value of port. port: " << port);
    } else {
        port = pta_env_port;
        SHM_LOG_DEBUG("Get the port from the environment variable. port: " << port);
    }

    if (sockType == AF_INET) {
        SHM_LOG_INFO("SockType is AF_INET.");
        innerUID->addr.addr.addr4.sin_family = AF_INET;
        if (inet_pton(AF_INET, pta_env_ip, &(innerUID->addr.addr.addr4.sin_addr)) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv4 failed");
            return ACLSHMEM_NOT_INITED;
        }
        innerUID->addr.addr.addr4.sin_port = htons(port);
        innerUID->addr.type = ADDR_IPv4;
    } else if (sockType == AF_INET6) {
        SHM_LOG_INFO("SockType is AF_INET6.");
        innerUID->addr.addr.addr6.sin6_family = AF_INET6;
        if (inet_pton(AF_INET6, pta_env_ip, &(innerUID->addr.addr.addr6.sin6_addr)) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv6 failed");
            return ACLSHMEM_NOT_INITED;
        }
        innerUID->addr.addr.addr6.sin6_port = htons(port);
        innerUID->addr.type = ADDR_IPv6;
    } else {
        SHM_LOG_ERROR("IP Type is not IPv4 or IPv6");
        return ACLSHMEM_INVALID_PARAM;
    }
    SHM_LOG_INFO("gen unique id success.");
    return ACLSHMEM_SUCCESS;
}

static int aclshmemi_bootstrap_uid_finalize(aclshmemi_bootstrap_handle_t *handle) {
    if (!handle) {
        return ACLSHMEM_SUCCESS;
    }

    if (handle->bootstrap_state) {
        uid_bootstrap_state* state = (uid_bootstrap_state*) handle->bootstrap_state;
        unexpected_conn_t* elem = state->unexpected_conns;
        while (elem != NULL) {
            unexpected_conn_t* next = elem->next;
            socket_close(&elem->sock); // 关闭socket句柄
            ACLSHMEM_BOOTSTRAP_PTR_FREE(elem);
            elem = next;
        }
        state->unexpected_conns = NULL;
        socket_close(&state->listen_sock);
        socket_close(&state->ring_send_sock);
        socket_close(&state->ring_recv_sock);

        ACLSHMEM_BOOTSTRAP_PTR_FREE(state->peer_addrs);
        state->peer_addrs = nullptr;
        ACLSHMEM_BOOTSTRAP_PTR_FREE(state);
        handle->bootstrap_state = nullptr;
    }

    if (handle->pre_init_ops) {
        ACLSHMEM_BOOTSTRAP_PTR_FREE(handle->pre_init_ops);
        handle->pre_init_ops = nullptr;
    }

    return ACLSHMEM_SUCCESS;
}


static int aclshmemi_bootstrap_uid_allgather(const void *in, void *out, int len, aclshmemi_bootstrap_handle_t *handle) {
    if (!in || !out || !handle || !handle->bootstrap_state) {
        SHM_LOG_ERROR("bootstrap allgather: invalid arguments.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    uid_bootstrap_state* state = (uid_bootstrap_state*) handle->bootstrap_state;
    int rank = state->rank;
    int nranks = state->nranks;
    char* send_buf = (char*)in;

    if (state->ring_send_sock.state != SOCKET_STATE_READY || 
        state->ring_recv_sock.state != SOCKET_STATE_READY) {
        SHM_LOG_ERROR("bootstrap allgather: rank " << rank << ": sockets not ready for allgather");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (in != BOOTSTRAP_IN_PLACE) {
        std::copy_n(send_buf, len, (char*)out + (rank % nranks) * len);
    }

    for (int i = 0; i < nranks - 1; i++) {
        size_t rslice = (rank - i - 1 + nranks) % nranks;
        size_t sslice = (rank - i + nranks) % nranks;

        ACLSHMEM_CHECK_RET(socket_send(&state->ring_send_sock, ((char*)out + sslice * len), len), "pe " << rank << ": barrier send failed");
        ACLSHMEM_CHECK_RET(socket_recv(&state->ring_recv_sock, ((char*)out + rslice * len), len), "pe " << rank << ": barrier recv failed");
    }
    return ACLSHMEM_SUCCESS;
}


static int aclshmemi_bootstrap_uid_barrier(aclshmemi_bootstrap_handle_t *handle) {
    SHM_LOG_INFO("aclshmemi_bootstrap_uid_barrier");
    if (!handle || !handle->bootstrap_state) {
        SHM_LOG_ERROR("bootstrap barrier: invalid arguments");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    uid_bootstrap_state* state = (uid_bootstrap_state*) handle->bootstrap_state;
    int rank = state->rank;
    int nranks = state->nranks;

    if (nranks == 1) {
        return ACLSHMEM_SUCCESS;
    }

    if (state->ring_send_sock.state != SOCKET_STATE_READY || 
        state->ring_recv_sock.state != SOCKET_STATE_READY) {
        SHM_LOG_ERROR("bootstrap barrier: rank " << rank << ": sockets not ready for barrier");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    char token = 0;
    if (rank == 0) {
        ACLSHMEM_CHECK_RET(socket_send(&state->ring_send_sock, &token, 1), "pe 0: barrier send failed");
        ACLSHMEM_CHECK_RET(socket_recv(&state->ring_recv_sock, &token, 1), "pe 0: barrier recv failed");
    } else {
        ACLSHMEM_CHECK_RET(socket_recv(&state->ring_recv_sock, &token, 1), "pe " << rank << ": barrier recv failed");
        ACLSHMEM_CHECK_RET(socket_send(&state->ring_send_sock, &token, 1), "pe " << rank << ": barrier send failed");
    }
    return ACLSHMEM_SUCCESS;
}
static int unexpected_dequeue(uid_bootstrap_state* state, int peer, int tag, socket_t* sock, int* found) {
    SHM_LOG_INFO("unexpected_dequeue start.");
    if (state == NULL || sock == NULL || found == NULL) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    unexpected_conn_t* elem = state->unexpected_conns;
    unexpected_conn_t* prev = NULL;
    *found = 0;
    while (elem != NULL) {
        if (elem->peer == peer && elem->tag == tag) {
            if (prev == NULL) {
                state->unexpected_conns = elem->next;
            } else {
                prev->next = elem->next;
            }

            std::copy_n(reinterpret_cast<const char*>(&elem->sock), 
                        sizeof(socket_t), 
                        reinterpret_cast<char*>(sock));
            ACLSHMEM_BOOTSTRAP_PTR_FREE(elem);
            *found = 1;
            return ACLSHMEM_SUCCESS;
        }

        prev = elem;
        elem = elem->next;
    }
    return ACLSHMEM_SUCCESS;
}

static int unexpected_enqueue(uid_bootstrap_state* state, int peer, int tag, socket_t* sock) {
    SHM_LOG_INFO("unexpected_enqueue start.");
    if (state == NULL || sock == NULL) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    unexpected_conn_t* new_conn = NULL;
    ACLSHMEM_BOOTSTRAP_CALLOC(&new_conn, 1);
    if (new_conn == NULL) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    new_conn->peer = peer;
    new_conn->tag = tag;
    std::copy_n(reinterpret_cast<const char*>(sock), 
                sizeof(socket_t), 
                reinterpret_cast<char*>(&new_conn->sock));
    new_conn->next = NULL;
    if (state->unexpected_conns == NULL) {
        state->unexpected_conns = new_conn;
    } else {
        new_conn->next = state->unexpected_conns;
        state->unexpected_conns = new_conn;
    }

    return ACLSHMEM_SUCCESS;
}
static int bootstrap_send(void* comm_state, int peer, int tag, void* data, int size) {
    if (comm_state == nullptr || data == nullptr || size < 0 || peer < 0) {
        SHM_LOG_ERROR("bootstrap_send: invalid arguments");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    uid_bootstrap_state* state = (uid_bootstrap_state*)comm_state;
    socket_t sock;
    ACLSHMEM_CHECK_RET(socket_init(&sock, SOCKET_TYPE_BOOTSTRAP, state->magic, &state->peer_addrs[peer]), "bootstrap_send: socket_init failed for peer " << peer);

    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_connect(&sock), "bootstrap_send: socket_connect failed for peer " << peer, sock);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_send(&sock, &state->rank, sizeof(int)), "bootstrap_send: send rank failed to peer " << peer, sock);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_send(&sock, &tag, sizeof(int)), "bootstrap_send: send tag " << tag << " failed to peer " << peer, sock);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_send(&sock, data, size), "bootstrap_send: send data (size=" << size << ") failed to peer " << peer, sock);
    if (sock.fd >= 0) {
        socket_close(&sock);
    }
    return ACLSHMEM_SUCCESS;
}


static int bootstrap_recv(void* comm_state, int peer, int tag, void* data, int size) {
    if (comm_state == NULL || data == NULL || size < 0 || peer < 0) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    uid_bootstrap_state* state = (uid_bootstrap_state*)comm_state;
    socket_t sock;
    int found = 0;
    int retry_count = 0;
    int ret = ACLSHMEM_SUCCESS;
    ACLSHMEM_CHECK_RET(unexpected_dequeue(state, peer, tag, &sock, &found));

    if (found == 1) {
        ret = socket_recv(&sock, data, size);
        socket_close(&sock);
        return (ret == ACLSHMEM_SUCCESS) ? ACLSHMEM_SUCCESS : ACLSHMEM_BOOTSTRAP_ERROR;
    }
    while (1) {
        socket_t new_sock;
        int new_peer = -1;
        int new_tag = -1;
        ACLSHMEM_CHECK_RET(socket_init(&new_sock, SOCKET_TYPE_BOOTSTRAP, SOCKET_MAGIC, NULL), "socket_init new_sock failed");
        ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_accept(&new_sock, &state->listen_sock), "socket_accept new_sock failed", new_sock);
        ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_recv(&new_sock, &new_peer, sizeof(int)), "socket_recv new_peer failed", new_sock);
        ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_recv(&new_sock, &new_tag, sizeof(int)), "socket_recv new_tag failed", new_sock);
        if (new_peer == peer && new_tag == tag) {
            ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_recv(&new_sock, data, size), "socket_recv failed", new_sock);
            return ACLSHMEM_SUCCESS;
        } else {
            ACLSHMEM_CHECK_RET_CLOSE_SOCK(unexpected_enqueue(state, new_peer, new_tag, &new_sock), "unexpected_enqueue failed", new_sock);
        }
    }
}

static int aclshmemi_bootstrap_uid_barrier_v2(aclshmemi_bootstrap_handle_t *handle) {
    SHM_LOG_INFO("aclshmemi_bootstrap_uid_barrier_v2");
    uid_bootstrap_state* state = (uid_bootstrap_state*)(handle->bootstrap_state);
    int rank = state->rank;
    int tag = 0;
    int nranks = state->nranks;

    if (nranks == 1) {
        SHM_LOG_DEBUG("Single rank, skip barrier");
        return ACLSHMEM_SUCCESS;
    }

    SHM_LOG_DEBUG("Barrier start. rank: " << rank << " nranks: " << nranks <<" tag: "<< tag);

    int data[1];
    for (int mask = 1; mask < nranks; mask <<= 1) {
        int src = (rank - mask + nranks) % nranks;
        int dst = (rank + mask) % nranks;
        tag++;

        ACLSHMEM_CHECK_RET(bootstrap_send(state, dst, tag, data, sizeof(data)), "pe " << rank << ": barrier send failed, dst: " << dst << "tag: " << tag);
        ACLSHMEM_CHECK_RET(bootstrap_recv(state, src, tag, data, sizeof(data)), "pe " << rank << ": barrier recv failed, src: " << src << "tag: " << tag);
    }

    SHM_LOG_DEBUG("Barrier end. rank: " << rank << " nranks: " << nranks <<" tag: "<< tag);
    return ACLSHMEM_SUCCESS;
}

static int aclshmemi_bootstrap_uid_alltoall(const void *sendbuf, void *recvbuf, int length,
                                  aclshmemi_bootstrap_handle_t *handle) {
    (void)sendbuf;
    (void)recvbuf;
    (void)length;
    (void)handle;
    return ACLSHMEM_NOT_SUPPORTED;
}

static void aclshmemi_bootstrap_uid_global_exit(int status, aclshmemi_bootstrap_handle_t *handle) {

}

static bool matchSubnet(struct ifaddrs local_if, sockaddr_t* remote) {
    int family;
    bool is_lo_interface = (strncmp(local_if.ifa_name, "lo", 2) == 0);
    if (remote->type == ADDR_IPv4) {
        family = AF_INET;
    } else if (remote->type == ADDR_IPv6) {
        family = AF_INET6;
    } else {
        return false;
    }
    
    SHM_LOG_DEBUG("local_if family: " << local_if.ifa_addr->sa_family << " remote family: " << family);
    if (family != local_if.ifa_addr->sa_family) {
        SHM_LOG_DEBUG(" matchSubnet family unmatch.");
        return false;
    }

    if (family == AF_INET) {
        struct sockaddr_in* local_addr = (struct sockaddr_in*)(local_if.ifa_addr);
        struct sockaddr_in* mask = (struct sockaddr_in*)(local_if.ifa_netmask);
        struct sockaddr_in* remote_addr = &remote->addr.addr4;

        uint32_t local_subnet = local_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
        uint32_t remote_subnet = remote_addr->sin_addr.s_addr & mask->sin_addr.s_addr;
        SHM_LOG_DEBUG("ipv4 matchSubnet result:" << (local_subnet == remote_subnet));
        return local_subnet == remote_subnet;
    } else if (family == AF_INET6) {
        struct sockaddr_in6* local_addr = (struct sockaddr_in6*)(local_if.ifa_addr);
        struct sockaddr_in6* mask = (struct sockaddr_in6*)(local_if.ifa_netmask);
        struct sockaddr_in6* remote_addr = &remote->addr.addr6;

        bool same = true;
        for (int c = 0; c < 16; c++) {
            uint8_t l = local_addr->sin6_addr.s6_addr[c] & mask->sin6_addr.s6_addr[c];
            uint8_t r = remote_addr->sin6_addr.s6_addr[c] & mask->sin6_addr.s6_addr[c];
            if (l != r) {
                same = false;
                break;
            }
        }
        if (is_lo_interface) {
            SHM_LOG_DEBUG("IPv6 on lo interface, skipping sin6_scope_id validation");
            SHM_LOG_DEBUG("ipv6 matchSubnet result:" << same);
            return same;
        }
        same &= (local_addr->sin6_scope_id == remote_addr->sin6_scope_id);
        SHM_LOG_DEBUG("ipv6 matchSubnet result:" << same << " local_addr->sin6_scope_id: " <<local_addr->sin6_scope_id << " remote_addr->sin6_scope_id: "<<  remote_addr->sin6_scope_id);
        return same;
    }
    return false;
}

static int find_interface_match_subnet(char* ifNames, sockaddr_t* localAddrs, sockaddr_t* remoteAddr) {
    int found = 0;
    struct ifaddrs *interfaces, *interface;
    if (getifaddrs(&interfaces) != 0) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    if (remoteAddr) {
        if (remoteAddr->type == ADDR_IPv4) {
            char ip_str[INET_ADDRSTRLEN];
            ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &remoteAddr->addr.addr4.sin_addr, ip_str, INET_ADDRSTRLEN) == nullptr, "convert remote ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
            uint16_t port = ntohs(remoteAddr->addr.addr4.sin_port);
            SHM_LOG_INFO(" Type: IPv4" << " IP: " << ip_str <<" Port: " << (port ? port : 0) << " (0 means not set)");
        } else if (remoteAddr->type == ADDR_IPv6) {
            char ip_str[INET6_ADDRSTRLEN];
            ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &remoteAddr->addr.addr6.sin6_addr, ip_str, INET6_ADDRSTRLEN) == nullptr, "convert remote ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
            uint16_t port = ntohs(remoteAddr->addr.addr6.sin6_port);
            SHM_LOG_INFO(" Type: IPv6" << " IP: " << ip_str <<" Port: " << (port ? port : 0) << " (0 means not set)");
        } else {
            SHM_LOG_ERROR(" remoteAddr: Unknown address type is not within IPv4 or IPv6.");
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }
    } else {
        SHM_LOG_ERROR(" remoteAddr is NULL.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    ACLSHMEM_CHECK_RET(is_link_local_addr(remoteAddr), "Remote address is link_local", ACLSHMEM_BOOTSTRAP_ERROR);
    bool remote_is_loopback = is_loopback_addr(remoteAddr);
    SHM_LOG_INFO("Remote address is loopback:" << remote_is_loopback);

    if (remote_is_loopback) {
        SHM_LOG_DEBUG("Remote address is loopback, check lo interface first");
        for (interface = interfaces; interface && !found; interface = interface->ifa_next) {
            if (interface->ifa_addr == NULL) continue;
            int family = interface->ifa_addr->sa_family;
            if (family != AF_INET && family != AF_INET6) continue;
            if (strcmp(interface->ifa_name, "lo") != 0) continue;

            if (matchSubnet(*interface, remoteAddr)) {
                if (family == AF_INET) {
                    localAddrs->type = ADDR_IPv4;
                    std::copy_n(reinterpret_cast<const char*>(interface->ifa_addr), 
                                sizeof(struct sockaddr_in), 
                                reinterpret_cast<char*>(&localAddrs->addr.addr4));
                } else {
                    localAddrs->type = ADDR_IPv6;
                    std::copy_n(reinterpret_cast<const char*>(interface->ifa_addr), 
                                sizeof(struct sockaddr_in6), 
                                reinterpret_cast<char*>(&localAddrs->addr.addr6));
                }
                strncpy(ifNames, interface->ifa_name, MAX_IF_NAME_SIZE);
                ifNames[MAX_IF_NAME_SIZE] = '\0';
                found = 1;
                break;
            }
        }
    }
    if (!found) {
        for (interface = interfaces; interface && !found; interface = interface->ifa_next) {
            if (interface->ifa_addr == NULL) continue;
            int family = interface->ifa_addr->sa_family;
            if (family != AF_INET && family != AF_INET6) continue;

            if (!remote_is_loopback && strcmp(interface->ifa_name, "lo") == 0) {
                continue;
            }

            if (matchSubnet(*interface, remoteAddr)) {
                if (family == AF_INET) {
                    localAddrs->type = ADDR_IPv4;
                    std::copy_n(reinterpret_cast<const char*>(interface->ifa_addr), 
                                sizeof(struct sockaddr_in), 
                                reinterpret_cast<char*>(&localAddrs->addr.addr4));
                } else {
                    localAddrs->type = ADDR_IPv6;
                    std::copy_n(reinterpret_cast<const char*>(interface->ifa_addr), 
                                sizeof(struct sockaddr_in6), 
                                reinterpret_cast<char*>(&localAddrs->addr.addr6));
                }
                strncpy(ifNames, interface->ifa_name, MAX_IF_NAME_SIZE);
                ifNames[MAX_IF_NAME_SIZE] = '\0';
                found = 1;
                break;
            }
        }
    }

    freeifaddrs(interfaces);
    return (found == 0) ? ACLSHMEM_BOOTSTRAP_ERROR : ACLSHMEM_SUCCESS;
}

static int bootstrap_get_sock_addr(socket_t* sock, sockaddr_t* addr) {
    if (sock == NULL) return ACLSHMEM_BOOTSTRAP_ERROR;
    struct sockaddr_storage temp_storage;
    std::fill(reinterpret_cast<char*>(&temp_storage), 
              reinterpret_cast<char*>(&temp_storage) + sizeof(temp_storage), 
              0);
    struct sockaddr* temp_addr = reinterpret_cast<struct sockaddr*>(&temp_storage);
    socklen_t addr_len = 0;
    int ret = socket_get_sainfo(sock, temp_addr, &addr_len);
    if (ret != 0) {
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (temp_storage.ss_family == AF_INET) {
        addr->type = ADDR_IPv4;
        const struct sockaddr_in* ipv4_src = reinterpret_cast<const struct sockaddr_in*>(&temp_storage);
        std::copy_n(reinterpret_cast<const char*>(ipv4_src), 
                    sizeof(struct sockaddr_in), 
                    reinterpret_cast<char*>(&addr->addr.addr4));
    } else if (temp_storage.ss_family == AF_INET6) {
        addr->type = ADDR_IPv6;
        const struct sockaddr_in6* ipv6_src = reinterpret_cast<const struct sockaddr_in6*>(&temp_storage);
        std::copy_n(reinterpret_cast<const char*>(ipv6_src), 
                    sizeof(struct sockaddr_in6), 
                    reinterpret_cast<char*>(&addr->addr.addr6));
    } else {
        SHM_LOG_ERROR("Unknown address type is not within IPv4 or IPv6.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    return ACLSHMEM_SUCCESS;
}

// Network Initialization (Locating Local Interface Matching Subnet / initialize root node UID information when is_arg_init == false)
static int aclshmemi_bootstrap_net_init(aclshmemi_bootstrap_uid_state_t* uid_args, bool is_arg_init = true) {
    SHM_LOG_INFO(" Network Initialization, Finding Interfaces Matching Subnets");
    pthread_mutex_lock(&priv_info.bootstrap_netlock);

    if (!is_arg_init) {
        SHM_LOG_INFO("net_init uid_args is NULL, get uid arg");
        bool is_from_ifa = false;

        if (env_ip_port != nullptr) {
            SHM_LOG_INFO("Environment variable SHMEM_UID_SESSION_ID has been set.");
            ACLSHMEM_CHECK_RET(aclshmemi_get_ip_from_env(uid_args, env_ip_port), 
                        "No available addresses were found with env_ip_port.");
        } else {
            SHM_LOG_INFO("Environment variable SHMEM_UID_SESSION_ID is not set, automatically obtaining ipPort.");
            is_from_ifa = true;
            ACLSHMEM_CHECK_RET(aclshmemi_get_ip_from_ifa(uid_args, env_ifname), 
                        "No available addresses were found with auto.");
        }

        SHM_LOG_INFO("Get uid arg success.");
        is_arg_init = true;
    }

    if (priv_info.bootstrap_netinitdone) {
        // Initialized, printing currently saved information
        SHM_LOG_INFO(" priv_info already inited: " << " bootstrap_netifname: "
            << (priv_info.bootstrap_netifname[0] != '\0' ? priv_info.bootstrap_netifname : "nullptr"));
        if (priv_info.bootstrap_netifaddr.type == ADDR_IPv4) {
            char ip_str[INET_ADDRSTRLEN] = {0};
            ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &priv_info.bootstrap_netifaddr.addr.addr4.sin_addr, ip_str, sizeof(ip_str)) == nullptr, "convert bootstrap_netifaddr ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
            SHM_LOG_INFO(" bootstrap_netifaddr (IPv4): " << ip_str << ":" << ntohs(priv_info.bootstrap_netifaddr.addr.addr4.sin_port));
        } else if (priv_info.bootstrap_netifaddr.type == ADDR_IPv6) {
            char ip_str[INET6_ADDRSTRLEN] = {0};
            ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &priv_info.bootstrap_netifaddr.addr.addr6.sin6_addr, ip_str, sizeof(ip_str)) == nullptr, "convert bootstrap_netifaddr ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
            SHM_LOG_INFO(" bootstrap_netifaddr (IPv6): " << ip_str << ":" << ntohs(priv_info.bootstrap_netifaddr.addr.addr6.sin6_port));
        } else {
            SHM_LOG_ERROR(" bootstrap_netifaddr: Unknown address type is not within IPv4 or IPv6.");
            return ACLSHMEM_BOOTSTRAP_ERROR;
        }

        pthread_mutex_unlock(&priv_info.bootstrap_netlock);
        return ACLSHMEM_SUCCESS;
    }

    // Print the root node address to be matched (uid_args->addr)
    if (uid_args->addr.type == ADDR_IPv4) {
        char ip_str[INET_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &uid_args->addr.addr.addr4.sin_addr, ip_str, sizeof(ip_str)) == nullptr, "convert uid_args addr ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        SHM_LOG_INFO(" Root address (IPv4): " << ip_str << ":" << ntohs(uid_args->addr.addr.addr4.sin_port));
    } else if (uid_args->addr.type == ADDR_IPv6) {
        char ip_str[INET6_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &uid_args->addr.addr.addr6.sin6_addr, ip_str, sizeof(ip_str)) == nullptr, "convert uid_args addr ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        SHM_LOG_INFO(" Root address (IPv6): " << ip_str << ":" << ntohs(uid_args->addr.addr.addr6.sin6_port));
    } else {
        SHM_LOG_ERROR(" Root address: Unknown address type is not within IPv4 or IPv6.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    int find_result = find_interface_match_subnet(priv_info.bootstrap_netifname, 
                                                &priv_info.bootstrap_netifaddr, 
                                                &uid_args->addr);
    if (find_result != 0) {
        SHM_LOG_ERROR(" Failed to find matching interface.");
        pthread_mutex_unlock(&priv_info.bootstrap_netlock);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    // Print the information of priv_info.
    if (priv_info.bootstrap_netifaddr.type == ADDR_IPv4) {
        char ip_str[INET_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &priv_info.bootstrap_netifaddr.addr.addr4.sin_addr, ip_str, sizeof(ip_str)) == nullptr, "convert bootstrap_netifaddr ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        SHM_LOG_INFO(" bootstrap_netifaddr (IPv4): " << ip_str 
                  << ":" << ntohs(priv_info.bootstrap_netifaddr.addr.addr4.sin_port));
    } else if (priv_info.bootstrap_netifaddr.type == ADDR_IPv6) {
        char ip_str[INET6_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &priv_info.bootstrap_netifaddr.addr.addr6.sin6_addr, ip_str, sizeof(ip_str)) == nullptr, "convert bootstrap_netifaddr ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        SHM_LOG_INFO(" bootstrap_netifaddr (IPv6): " << ip_str 
                  << ":" << ntohs(priv_info.bootstrap_netifaddr.addr.addr6.sin6_port));
    } else {
        SHM_LOG_ERROR(" Root bootstrap_netifaddr: Unknown address type is not within IPv4 or IPv6.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    priv_info.bootstrap_netinitdone = 1;
    pthread_mutex_unlock(&priv_info.bootstrap_netlock);
    SHM_LOG_INFO(" Net init success, priv_info.bootstrap_netinitdone = 1");
    return ACLSHMEM_SUCCESS;
}

static int set_files_limit() {
    struct rlimit files_limit, old_limit;

    ACLSHMEM_CHECK_RET(getrlimit(RLIMIT_NOFILE, &old_limit), "getrlimit failed", ACLSHMEM_BOOTSTRAP_ERROR);
    SHM_LOG_DEBUG("Original file descriptor limit - soft limit: " << old_limit.rlim_cur << ", hard limit: " << old_limit.rlim_max);

    files_limit = old_limit;
    files_limit.rlim_cur = files_limit.rlim_max;
    ACLSHMEM_CHECK_RET(setrlimit(RLIMIT_NOFILE, &files_limit), "setrlimit failed", ACLSHMEM_BOOTSTRAP_ERROR);

    struct rlimit new_limit;
    getrlimit(RLIMIT_NOFILE, &new_limit);
    SHM_LOG_DEBUG("Updated file descriptor limit - soft limit: " << new_limit.rlim_cur << ", hard limit: " << new_limit.rlim_max);

    return ACLSHMEM_SUCCESS;
}

static void* bootstrap_root(void* rargs) {
    struct bootstrap_root_args* args = (struct bootstrap_root_args*)rargs;
    if (args == NULL || args->listen_sock == NULL) {
        SHM_LOG_ERROR("bootstrap_root: invalid args");
        return NULL;
    }

    socket_t* listen_sock = args->listen_sock;
    uint64_t magic = args->magic;
    int root_version = args->version;
    int nranks = 0;
    int c = 0;  // Number of received nodes.
    bootstrap_ext_info info;
    sockaddr_t* zero_addr = nullptr;
    ACLSHMEM_BOOTSTRAP_CALLOC(&zero_addr, 1);
    sockaddr_t* rank_addrs = NULL;  // Store the common listening addresses of all nodes.
    sockaddr_t* rank_addrs_root = NULL;  // Store the dedicated root addresses for all nodes.

    if (zero_addr == NULL) {
        SHM_LOG_ERROR("bootstrap_root: calloc zero_addr failed");
        ACLSHMEM_BOOTSTRAP_PTR_FREE(args);
        return NULL;
    }

    // Adjusting file descriptor limits (the root node needs to handle multiple connections)
    if (set_files_limit() != 0) {
        SHM_LOG_ERROR("bootstrap_root: set_files_limit failed");
        ACLSHMEM_BOOTSTRAP_PTR_FREE(zero_addr);
        ACLSHMEM_BOOTSTRAP_PTR_FREE(args);
        return NULL;
    }

    // Continuously receive connections and information from all slave nodes
    while (1) {
        socket_t client_sock;
        // Initialize client socket (for receiving connections from a single slave node)
        if (socket_init(&client_sock, SOCKET_TYPE_BOOTSTRAP, SOCKET_MAGIC, NULL) != 0) {
            SHM_LOG_ERROR("bootstrap_root: socket_init failed");
            break;
        }

        // Accept connections from the node (blocking wait)
        if (socket_accept(&client_sock, listen_sock) != 0) {
            SHM_LOG_ERROR("bootstrap_root: socket_accept failed");
            socket_close(&client_sock);
            break;
        }

        // Version verification
        int peer_version;
        if (socket_recv(&client_sock, &peer_version, sizeof(peer_version)) != 0) {
            SHM_LOG_ERROR("bootstrap_root: recv peer_version failed");
            socket_close(&client_sock);
            break;
        }
        if (socket_send(&client_sock, &root_version, sizeof(root_version)) != 0) {
            SHM_LOG_ERROR("bootstrap_root: send root_version failed");
            socket_close(&client_sock);
            break;
        }
        if (peer_version != root_version) {
            SHM_LOG_ERROR("bootstrap_root: version mismatch");
            socket_close(&client_sock);
            break;
        }

        // Receive address information from the node
        if (socket_recv(&client_sock, &info, sizeof(info)) != 0) {
            SHM_LOG_ERROR("bootstrap_root: recv info failed");
            socket_close(&client_sock);
            break;
        }
        socket_close(&client_sock);

        // Initialize the address array upon first reception
        if (c == 0) {
            nranks = info.nranks;
            if (nranks <= 0) {
                SHM_LOG_ERROR("bootstrap_root: invalid nranks");
                break;
            }
            ACLSHMEM_BOOTSTRAP_CALLOC(&rank_addrs, nranks);
            ACLSHMEM_BOOTSTRAP_CALLOC(&rank_addrs_root, nranks);
            if (rank_addrs == NULL || rank_addrs_root == NULL) {
                SHM_LOG_ERROR("bootstrap_root: calloc addr arrays failed");
                break;
            }
        }

        if (info.nranks != nranks || info.rank < 0 || info.rank >= nranks) {
            SHM_LOG_ERROR("bootstrap_root: invalid info from rank " << info.rank);
            break;
        }
        // Check if the rank is duplicated
        if (memcmp(zero_addr, &rank_addrs_root[info.rank], sizeof(sockaddr_t)) != 0) {
            SHM_LOG_ERROR("bootstrap_root: duplicate rank " << info.rank);
            break;
        }

        std::copy_n(reinterpret_cast<const char*>(&info.ext_address_listen_root), 
                    sizeof(sockaddr_t), 
                    reinterpret_cast<char*>(&rank_addrs_root[info.rank]));
        std::copy_n(reinterpret_cast<const char*>(&info.ext_addr_listen), 
                    sizeof(sockaddr_t), 
                    reinterpret_cast<char*>(&rank_addrs[info.rank]));
        c++;

        if (c >= nranks) {
            SHM_LOG_INFO("bootstrap_root: Address receiving completed");
            break;
        }
    }

    if (c == nranks && rank_addrs != NULL && rank_addrs_root != NULL) {
        SHM_LOG_INFO("bootstrap_root: Start distributing addresses.");
        for (int r = 0; r < nranks; r++) {
            int next_rank = (r + 1) % nranks;
            socket_t send_sock;

            if (socket_init(&send_sock, SOCKET_TYPE_BOOTSTRAP, magic, &rank_addrs_root[r]) != 0) {
                SHM_LOG_ERROR("bootstrap_root: init send_sock for rank " << r << " failed");
                break;
            }

            if (socket_connect(&send_sock) != 0) {
                SHM_LOG_ERROR("bootstrap_root: connect to rank " << r << " failed");
                socket_close(&send_sock);
                break;
            }

            if (socket_send(&send_sock, &rank_addrs[next_rank], sizeof(sockaddr_t)) != 0) {
                SHM_LOG_ERROR("bootstrap_root: send next_addr to rank " << r << " failed");
                socket_close(&send_sock);
                break;
            }

            socket_close(&send_sock);
        }
    }

    ACLSHMEM_BOOTSTRAP_PTR_FREE(zero_addr);
    ACLSHMEM_BOOTSTRAP_PTR_FREE(rank_addrs);
    ACLSHMEM_BOOTSTRAP_PTR_FREE(rank_addrs_root);
    if (listen_sock != NULL) {
        socket_close(listen_sock);
        ACLSHMEM_BOOTSTRAP_PTR_FREE(listen_sock);
    }
    ACLSHMEM_BOOTSTRAP_PTR_FREE(args);
    return NULL;
}

static int bootstrap_create_root(aclshmemi_bootstrap_uid_state_t* uid_args) {
    if (uid_args == NULL) {
        SHM_LOG_ERROR("bootstrap_create_root: invalid uid_args");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    // 1. Create a dedicated listening socket for the root node.
    socket_t* listen_sock_root = nullptr;
    ACLSHMEM_CHECK_RET(ACLSHMEM_BOOTSTRAP_CALLOC(&listen_sock_root, 1), "bootstrap_create_root: malloc listen_sock_root failed");

    // 2. Initialize the listening socket (using the global network interface address)
    ACLSHMEM_CHECK_RET(socket_init(listen_sock_root, SOCKET_TYPE_BOOTSTRAP, uid_args->magic, &uid_args->addr), "bootstrap_create_root: socket_init failed");

    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_listen(listen_sock_root), "Listen_sock_root failed while executing listen. fd=" << listen_sock_root->fd, *listen_sock_root);

    // 3. Write the root node's listening address into uid_args (for slave nodes to connect to).
    std::copy_n(reinterpret_cast<const char*>(&listen_sock_root->addr), 
                sizeof(sockaddr_t), 
                reinterpret_cast<char*>(&uid_args->addr));

    // 4. Prepare thread parameters
    struct bootstrap_root_args* args = nullptr;
    ACLSHMEM_CHECK_RET(ACLSHMEM_BOOTSTRAP_CALLOC(&args, 1), "bootstrap_create_root: malloc args failed");

    args->listen_sock = listen_sock_root;
    args->magic = uid_args->magic;
    args->version = uid_args->version;

    // 5. Create detached thread
    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    int ret = pthread_create(&priv_info.bootstrap_root, &attr, bootstrap_root, args);
    if (ret != 0) {
        SHM_LOG_ERROR("bootstrap_create_root: pthread_create failed");
        ACLSHMEM_BOOTSTRAP_PTR_FREE(args);
        socket_close(listen_sock_root);
        ACLSHMEM_BOOTSTRAP_PTR_FREE(listen_sock_root);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    pthread_attr_destroy(&attr);
    return ACLSHMEM_SUCCESS;
}



int aclshmemi_bootstrap_get_unique_id(void* uid) {
    aclshmemi_bootstrap_uid_state_t* uid_args = (aclshmemi_bootstrap_uid_state_t*)uid;

    if (env_ip_port == nullptr) {
        const char* envip = std::getenv("SHMEM_UID_SESSION_ID");
        if (envip != nullptr) {
            env_ip_port = envip;
            SHM_LOG_DEBUG("SHMEM_UID_SESSION_ID is: " << env_ip_port);
        } else {
            SHM_LOG_DEBUG("The environment variable SHMEM_UID_SESSION_ID is not set.");
        }
    }

    if (env_ifname == nullptr) {
        const char* envinfo = std::getenv("SHMEM_UID_SOCK_IFNAME");
        if (envinfo != nullptr) {
            env_ifname = envinfo;
            SHM_LOG_DEBUG("SHMEM_UID_SOCK_IFNAME is: " << env_ifname);
        } else {
            SHM_LOG_DEBUG("The environment variable SHMEM_UID_SOCK_IFNAME is not set.");
        }
    }
    
    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_net_init(uid_args, false), "pe 0: failed to init bootstrap net.");
    ACLSHMEM_CHECK_RET(bootstrap_create_root(uid_args), "pe 0: failed to create root thread");
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_bootstrap_get_unique_id_static_magic(void* uid, bool is_root) {
    aclshmemi_bootstrap_uid_state_t* uid_args = (aclshmemi_bootstrap_uid_state_t*)uid;

    if (env_ip_port == nullptr) {
        const char* envip = std::getenv("SHMEM_UID_SESSION_ID");
        if (envip != nullptr) {
            env_ip_port = envip;
            SHM_LOG_DEBUG("SHMEM_UID_SESSION_ID is: " << env_ip_port);
        } else {
            SHM_LOG_DEBUG("The environment variable SHMEM_UID_SESSION_ID is not set.");
        }
    } 
    if (env_ip_port == nullptr) {
        SHM_LOG_ERROR("Using method get_unique_id_static_magic requires setting SHMEM_UID_SESSION_ID.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    
    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_net_init(uid_args, false), "pe 0: failed to init bootstrap net.");
    uid_args->magic = SOCKET_MAGIC + static_magic_count;
    static_magic_count++;
    if (is_root) {
        ACLSHMEM_CHECK_RET(bootstrap_create_root(uid_args), "pe 0: failed to create root thread");
    }
    return ACLSHMEM_SUCCESS;
}

int aclshmemi_bootstrap_get_unique_id_by_ipport(void* uid, const char *ipport) {
    aclshmemi_bootstrap_uid_state_t* uid_args = (aclshmemi_bootstrap_uid_state_t*)uid;

    if (ipport != nullptr) {
        env_ip_port = ipport;
        SHM_LOG_DEBUG("The ipport param is: " << env_ip_port);
     } else {
        
        SHM_LOG_DEBUG("The ipport param is not set. Try to use SHMEM_UID_SESSION_ID.");
        const char* envip = std::getenv("SHMEM_UID_SESSION_ID");
        if (envip != nullptr) {
            env_ip_port = envip;
            SHM_LOG_DEBUG("SHMEM_UID_SESSION_ID is: " << env_ip_port);
        } else {
            SHM_LOG_DEBUG("The environment variable SHMEM_UID_SESSION_ID is not set.");
        }
    }
    if (env_ip_port == nullptr) {
        SHM_LOG_ERROR("Using method get_unique_id_by_ipport requires setting ipport or SHMEM_UID_SESSION_ID.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    
    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_net_init(uid_args, false), "pe 0: failed to init bootstrap net.");
    uid_args->magic = SOCKET_MAGIC + static_magic_count;
    static_magic_count++;
    if (uid_args->my_pe == 0) {
        ACLSHMEM_CHECK_RET(bootstrap_create_root(uid_args), "pe 0: failed to create root thread");
    }
    return ACLSHMEM_SUCCESS;
}

// Plugin pre-initialization entry function. 
int aclshmemi_bootstrap_plugin_pre_init(aclshmemi_bootstrap_handle_t* handle) {
    if (handle->pre_init_ops == nullptr) {
        SHM_LOG_DEBUG(" bootstrap plugin pre init start.");
        ACLSHMEM_CHECK_RET(ACLSHMEM_BOOTSTRAP_CALLOC(&handle->pre_init_ops, 1));
        handle->pre_init_ops->get_unique_id = aclshmemi_bootstrap_get_unique_id;
        handle->pre_init_ops->get_unique_id_static_magic = aclshmemi_bootstrap_get_unique_id_static_magic;
        handle->pre_init_ops->cookie = nullptr;
        SHM_LOG_DEBUG(" bootstrap plugin pre init end.");
    } else {
        SHM_LOG_DEBUG(" pre_init_ops had already prepared.");
    }
    return ACLSHMEM_SUCCESS;
}


int aclshmemi_bootstrap_plugin_init(void* comm, aclshmemi_bootstrap_handle_t* handle) {
    if (comm == nullptr || handle == nullptr) {
        SHM_LOG_ERROR(" aclshmemi_bootstrap_plugin_init: invalid arguments (nullptr)");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }
    socket_t sock, listen_sock_root;
    uid_bootstrap_state* state = nullptr;
    ACLSHMEM_CHECK_RET(ACLSHMEM_BOOTSTRAP_CALLOC(&state, 1));
    aclshmemi_bootstrap_uid_state_t* uid_args = (aclshmemi_bootstrap_uid_state_t*)comm;
    sockaddr_t next_addr;
    bootstrap_ext_info info = {};
    int rank = uid_args->my_pe;
    int nranks = uid_args->n_pes;
    
    if (handle->use_attr_ipport && handle->ipport[0] != '\0') {
        SHM_LOG_DEBUG("aclshmemi_bootstrap_get_unique_id_by_ipport start. ipport: " << handle->ipport);
        aclshmemi_bootstrap_get_unique_id_by_ipport(comm, handle->ipport);
    }
    uint64_t magic = uid_args->magic;

    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_net_init(uid_args), " rank: " << rank << ": network interface init failed.");

    if (state == nullptr) {
        SHM_LOG_ERROR(" rank: " << rank << ": failed to allocate uid_bootstrap_state");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    state->rank = rank;
    state->nranks = nranks;
    state->magic = magic;

    ACLSHMEM_CHECK_RET(ACLSHMEM_BOOTSTRAP_CALLOC(&state->peer_addrs, nranks));

    if (state->peer_addrs == nullptr) {
        SHM_LOG_ERROR(" rank: " << rank << ": failed to allocate peer_addrs");
        ACLSHMEM_BOOTSTRAP_PTR_FREE(state);
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    handle->bootstrap_state = state;
    handle->mype = rank;
    handle->npes = nranks;

    ACLSHMEM_CHECK_RET(socket_init(&state->listen_sock, SOCKET_TYPE_BOOTSTRAP, state->magic, &priv_info.bootstrap_netifaddr), "State's listen_sock failed while executing init. fd=" << state->listen_sock.fd);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_listen(&state->listen_sock), "State's listen_sock failed while executing listen. fd=" << state->listen_sock.fd, state->listen_sock);
    ACLSHMEM_CHECK_RET(bootstrap_get_sock_addr(&state->listen_sock, &info.ext_addr_listen), "Get addr failed, the listen_sock in state maybe null. fd=" << state->listen_sock.fd);

    ACLSHMEM_CHECK_RET(socket_init(&listen_sock_root, SOCKET_TYPE_BOOTSTRAP, state->magic, &priv_info.bootstrap_netifaddr), "Listen_sock_root failed while executing init. fd=" << listen_sock_root.fd);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_listen(&listen_sock_root), "listen_sock_root failed while executing listen. fd=" << listen_sock_root.fd, listen_sock_root);
    ACLSHMEM_CHECK_RET(bootstrap_get_sock_addr(&listen_sock_root, &info.ext_address_listen_root), "Get addr failed, the listen_sock_root maybe null. fd=" << listen_sock_root.fd);


    ACLSHMEM_CHECK_RET(socket_init(&sock, SOCKET_TYPE_BOOTSTRAP, magic, &uid_args->addr), "Sock failed while executing init. fd=" << sock.fd);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_connect(&sock), "Sock failed while executing connect. fd=" << sock.fd, sock);
    int peer_version = uid_args->version;
    int root_version;
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_send(&sock, &peer_version, sizeof(peer_version)), "Sock failed while executing send peer_version. fd=" << sock.fd, sock);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_recv(&sock, &root_version, sizeof(root_version)), "Sock failed while executing recv root_version. fd=" << sock.fd, sock);
    ACLSHMEM_CHECK_RET(peer_version != root_version, " rank: " << rank << " . version mismatch with root", ACLSHMEM_SMEM_ERROR);

    info.rank = rank;
    info.nranks = nranks;

    if (info.ext_addr_listen.type == ADDR_IPv4) {
        struct sockaddr_in* ipv4 = &info.ext_addr_listen.addr.addr4;
        char ip_str[INET_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &ipv4->sin_addr, ip_str, sizeof(ip_str)) == nullptr, "convert ext_addr_listen ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        uint16_t port = ntohs(ipv4->sin_port);
        SHM_LOG_INFO(" Ext_addr_listen socket: Type: IPv4, IP: " << ip_str << ", Port: " << port << ", sa_family: " << ipv4->sin_family);

    } else if (info.ext_addr_listen.type == ADDR_IPv6) {
        struct sockaddr_in6* ipv6 = &info.ext_addr_listen.addr.addr6;
        char ip_str[INET6_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &ipv6->sin6_addr, ip_str, sizeof(ip_str)) == nullptr, "convert ext_addr_listen ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        uint16_t port = ntohs(ipv6->sin6_port);
        SHM_LOG_INFO(" Ext_addr_listen socket: Type: IPv6, IP: " << ip_str << ", Port: " << port << ", sa_family: " << ipv6->sin6_family);
    } else {
        SHM_LOG_ERROR(" Ext_address_listen_root socket: Type: Unknown address type is not within IPv4 or IPv6. (type=" << info.ext_addr_listen.type << ")");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    if (info.ext_address_listen_root.type == ADDR_IPv4) {
        struct sockaddr_in* ipv4 = &info.ext_address_listen_root.addr.addr4;
        char ip_str[INET_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &ipv4->sin_addr, ip_str, sizeof(ip_str)) == nullptr, "convert ext_address_listen_root ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        uint16_t port = ntohs(ipv4->sin_port);
        SHM_LOG_INFO(" Ext_address_listen_root socket: Type: IPv4, IP: " << ip_str << ", Port: " << port << ", sa_family: " << ipv4->sin_family);

    } else if (info.ext_address_listen_root.type == ADDR_IPv6) {
        struct sockaddr_in6* ipv6 = &info.ext_address_listen_root.addr.addr6;
        char ip_str[INET6_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &ipv6->sin6_addr, ip_str, sizeof(ip_str)) == nullptr, "convert ext_address_listen_root ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        uint16_t port = ntohs(ipv6->sin6_port);
        SHM_LOG_INFO(" Ext_address_listen_root socket: Type: IPv6, IP: " << ip_str << ", Port: " << port << ", sa_family: " << ipv6->sin6_family);
    } else {
        SHM_LOG_ERROR(" Ext_address_listen_root socket: Type: Unknown address type is not within IPv4 or IPv6. (type=" << info.ext_address_listen_root.type << ")");
        return ACLSHMEM_BOOTSTRAP_ERROR;

    }


    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_send(&sock, &info, sizeof(info)), "Sock failed while executing send info. fd=" << sock.fd, sock);
    ACLSHMEM_CHECK_RET(socket_close(&sock), "Sock failed while executing close. fd=" << sock.fd);


    ACLSHMEM_CHECK_RET(socket_init(&sock, SOCKET_TYPE_BOOTSTRAP, SOCKET_MAGIC, nullptr), "Sock failed while executing init. fd=" << sock.fd);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_accept(&sock, &listen_sock_root), "Sock failed while executing accept listen_sock_root. fd=" << sock.fd, sock);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_recv(&sock, &next_addr, sizeof(next_addr)), "Sock failed while executing recv next_addr. fd=" << sock.fd, sock);
    ACLSHMEM_CHECK_RET(socket_close(&sock), "Sock failed while executing close. fd=" << sock.fd);
    ACLSHMEM_CHECK_RET(socket_close(&listen_sock_root), "Listen_sock_root failed while executing close. fd=" << listen_sock_root.fd);

    
    if (next_addr.type == ADDR_IPv4) {
        char ip_str[INET_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET, &next_addr.addr.addr4.sin_addr, ip_str, sizeof(ip_str)) == nullptr, "convert next_addr ipv4 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        uint16_t port = ntohs(next_addr.addr.addr4.sin_port);
        SHM_LOG_INFO(" Received next socket: Type: IPv4, IP: " << ip_str << ", Port: " << port);
    } else if (next_addr.type == ADDR_IPv6) {
        char ip_str[INET6_ADDRSTRLEN] = {0};
        ACLSHMEM_CHECK_RET(inet_ntop(AF_INET6, &next_addr.addr.addr6.sin6_addr, ip_str, sizeof(ip_str)) == nullptr, "convert next_addr ipv6 to string failed. ", ACLSHMEM_BOOTSTRAP_ERROR);
        uint16_t port = ntohs(next_addr.addr.addr6.sin6_port);
        SHM_LOG_INFO(" Received next socket: Type: IPv6, IP: " << ip_str << ", Port: " << port);
    } else {
        SHM_LOG_ERROR(" Received next socket: Type: Unknown address type is not within IPv4 or IPv6.");
        return ACLSHMEM_BOOTSTRAP_ERROR;
    }

    // Initialize ring send socket
    ACLSHMEM_CHECK_RET(socket_init(&state->ring_send_sock, SOCKET_TYPE_BOOTSTRAP, magic, &next_addr), "State's ring_send_sock failed while executing init. fd=" << state->ring_send_sock.fd);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_connect(&state->ring_send_sock), "State's ring_send_sock failed while executing connect. fd=" << state->ring_send_sock.fd, state->ring_send_sock);
    ACLSHMEM_CHECK_RET(socket_init(&state->ring_recv_sock, SOCKET_TYPE_BOOTSTRAP, SOCKET_MAGIC, nullptr), "State's ring_recv_sock failed while executing init. fd=" << state->ring_recv_sock.fd);
    ACLSHMEM_CHECK_RET_CLOSE_SOCK(socket_accept(&state->ring_recv_sock, &state->listen_sock),"State's ring_recv_sock failed while executing accept State's listen_sock. fd=" << state->ring_recv_sock.fd, state->ring_recv_sock);
    ACLSHMEM_CHECK_RET(bootstrap_get_sock_addr(&state->listen_sock, state->peer_addrs + handle->mype), "Get addr failed, the listen_sock in state maybe null. fd=" << state->listen_sock.fd);

    ACLSHMEM_CHECK_RET(aclshmemi_bootstrap_uid_allgather(BOOTSTRAP_IN_PLACE, state->peer_addrs, sizeof(sockaddr_t), handle), "Bootstrap_uid_allgather failed");

    handle->allgather = aclshmemi_bootstrap_uid_allgather;
    handle->barrier = aclshmemi_bootstrap_uid_barrier_v2;
    handle->finalize = aclshmemi_bootstrap_uid_finalize;
    handle->alltoall = nullptr;
    handle->global_exit = nullptr;

    SHM_LOG_INFO("pe " << rank << ": bootstrap plugin initialized successfully");
    return ACLSHMEM_SUCCESS;
}
