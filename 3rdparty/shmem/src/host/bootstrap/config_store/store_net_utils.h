/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEMFABRIC_NET_UTIL_H
#define MEMFABRIC_NET_UTIL_H

#include <algorithm>
#include <random>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <fstream>
#include <ifaddrs.h>
#include <net/if.h>

#include "shmemi_logger.h"
#include "shmemi_host_def.h"

namespace shm {
namespace utils {

constexpr int MIN_PORT = 1024;
constexpr int MAX_PORT = 65536;
constexpr int MAX_ATTEMPTS = 1000;
constexpr int MAX_IFCONFIG_LENGTH = 23;
constexpr int MAX_IP = 48;
constexpr int DEFAULT_IFNAME_LNEGTH = 4;

inline int32_t shmem_get_uid_magic(shmemx_bootstrap_uid_state_t *innerUId)
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

inline int32_t bind_tcp_port_v4(int &sockfd, int port, shmemx_bootstrap_uid_state_t *innerUId, char *ip_str)
{
    sockfd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd != -1) {
        int on_v4 = 1;
        if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on_v4, sizeof(on_v4)) == 0) {
            innerUId->addr.addr.addr4.sin_port = htons(port);
            sockaddr *cur_addr = reinterpret_cast<sockaddr *>(&innerUId->addr.addr.addr4);
            if (::bind(sockfd, cur_addr, sizeof(innerUId->addr.addr.addr4)) == 0) {
                SHM_LOG_INFO("bind ipv4 success " << ", fd:" << sockfd << ", " << ip_str << ":" << port);
                return 0;
            } else {
                SHM_LOG_ERROR("bind socket fail:" << errno << "," << ip_str << ":" << port);
            }
        } else {
            SHM_LOG_ERROR("set socket opt fail:" << errno << ","  << ip_str << ":" << port);
        }
        close(sockfd);
        sockfd = -1;
    } else {
        SHM_LOG_ERROR("create socket fail:" << errno << ", " << ip_str << ":" << port);
    }
    return -1;
}

inline int32_t bind_tcp_port_v6(int &sockfd, int port, shmemx_bootstrap_uid_state_t *innerUId, char *ip_str)
{
    sockfd = ::socket(AF_INET6, SOCK_STREAM, 0);
    if (sockfd != -1) {
        int on_v6 = 1;
        if (::setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &on_v6, sizeof(on_v6)) == 0) {
            innerUId->addr.addr.addr6.sin6_port = htons(port);
            sockaddr *cur_addr = reinterpret_cast<sockaddr *>(&innerUId->addr.addr.addr6);
            if (::bind(sockfd, cur_addr, sizeof(innerUId->addr.addr.addr6)) == 0) {
                SHM_LOG_INFO("bind ipv6 success " << ", fd:" << sockfd << ", " << ip_str << ":" << port);
                return 0;
            } else {
                SHM_LOG_ERROR("bind socket6 fail:" << errno << "," << ip_str << ":" << port);
            }
        } else {
            SHM_LOG_ERROR("set socket6 opt fail:" << errno << "," << ip_str << ":" << port);
        }
        close(sockfd);
        sockfd = -1;
    } else {
        SHM_LOG_ERROR("create socket6 fail:" << errno << "," << ip_str << ":" << port);
    }
    return -1;
}

inline int32_t aclshmemi_get_port_magic(shmemx_bootstrap_uid_state_t *innerUId, char *ip_str)
{
    static std::random_device rd;
    const int min_port = MIN_PORT;
    const int max_port = MAX_PORT;
    const int max_attempts = MAX_ATTEMPTS;
    const int offset_bit = 32;
    uint64_t seed = 1;
    seed |= static_cast<uint64_t>(getpid()) << offset_bit;
    seed |= static_cast<uint64_t>(std::chrono::system_clock::now().time_since_epoch().count() & 0xFFFFFFFF);
    static std::mt19937_64 gen(seed);
    std::uniform_int_distribution<> dis(min_port, max_port);

    int sockfd = -1;
    int32_t ret;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        int port = dis(gen);
        if (innerUId->addr.type == ADDR_IPv4) {
            ret = bind_tcp_port_v4(sockfd, port, innerUId, ip_str);
            if (ret == 0) {
                innerUId->inner_sockFd = sockfd;
                return 0;
            }
        } else {
            ret = bind_tcp_port_v6(sockfd, port, innerUId, ip_str);
            if (ret == 0) {
                innerUId->inner_sockFd = sockfd;
                return 0;
            }
        }
    }
    SHM_LOG_ERROR("Not find a available tcp port");
    return -1;
}

inline int32_t aclshmemi_using_env_port(aclshmemi_bootstrap_uid_state_t *innerUId, char *ip_str, uint16_t envPort)
{
    if (envPort < MIN_PORT) {   // envPort > MAX_PORT always false
        SHM_LOG_ERROR("env port is invalid. " << envPort);
        return ACLSHMEM_INVALID_PARAM;
    }

    int sockfd = -1;
    int32_t ret;
    if (innerUId->addr.type == ADDR_IPv4) {
        ret = bind_tcp_port_v4(sockfd, envPort, innerUId, ip_str);
        if (ret == 0) {
            innerUId->inner_sockFd = sockfd;
            return 0;
        }
    } else {
        ret = bind_tcp_port_v6(sockfd, envPort, innerUId, ip_str);
        if (ret == 0) {
            innerUId->inner_sockFd = sockfd;
            return 0;
        }
    }
    SHM_LOG_ERROR("init with env port fialed " << envPort << ", ret=" << ret);
    return ret;
}

inline int32_t parse_interface_with_type(const char *ipInfo, char *IP, sa_family_t &sockType, bool &flag)
{
    const char *delim = ":";
    const char *sep = strchr(ipInfo, delim[0]);
    if (sep != nullptr) {
        size_t leftLen = sep - ipInfo;
        if (leftLen >= MAX_IFCONFIG_LENGTH - 1 || leftLen == 0) {
            return ACLSHMEM_INVALID_VALUE;
        }
        size_t actualCopyLen = std::min(strlen(ipInfo), static_cast<size_t>(leftLen));
        std::copy(ipInfo, ipInfo + actualCopyLen, IP);
        if (actualCopyLen < leftLen) {
            std::fill(IP + actualCopyLen, IP + leftLen, '\0');
        }
        sockType = (strcmp(sep + 1, "inet6") != 0) ? AF_INET : AF_INET6;
        flag = true;
    }
    return ACLSHMEM_SUCCESS;
}

inline int32_t aclshmemi_auto_get_ip(struct sockaddr *ifaAddr, char *local, sa_family_t &sockType)
{
    sockType = ifaAddr->sa_family;
    if (sockType == AF_INET) {
        auto localIp = reinterpret_cast<struct sockaddr_in *>(ifaAddr)->sin_addr;
        if (inet_ntop(sockType, &localIp, local, MAX_IP) == nullptr) {
            SHM_LOG_ERROR("convert local ipv4 to string failed. ");
            return ACLSHMEM_INVALID_PARAM;
        }
        return ACLSHMEM_SUCCESS;
    } else if (sockType == AF_INET6) {
        auto localIp = reinterpret_cast<struct sockaddr_in6 *>(ifaAddr)->sin6_addr;
        if (inet_ntop(sockType, &localIp, local, MAX_IP) == nullptr) {
            SHM_LOG_ERROR("convert local ipv6 to string failed. ");
            return ACLSHMEM_INVALID_PARAM;
        }
        return ACLSHMEM_SUCCESS;
    }
    return ACLSHMEM_INVALID_PARAM;
}

inline bool aclshmemi_check_ifa(struct ifaddrs *ifa, sa_family_t sockType, bool flag, char *ifaName, size_t ifaLen)
{
    if (ifa->ifa_addr == nullptr || ifa->ifa_netmask == nullptr || ifa->ifa_name == nullptr) {
        SHM_LOG_DEBUG("loop ifa_addr/ifa_netmask/ifa_name is nullptr");
        return false;
    }

    // socket type match and input env ifa valid
    if (ifa->ifa_addr->sa_family != sockType && flag) {
        SHM_LOG_DEBUG("sa family is not match, get " << ifa->ifa_addr->sa_family << ", expect " << sockType);
        return false;
    }

    //  prefix match with input ifa name
    if (strncmp(ifa->ifa_name, ifaName, ifaLen) != 0) {
        SHM_LOG_DEBUG("ifa name prefix un-match, get " << ifa->ifa_name << ", expect " << ifaName);
        return false;
    }

    // ignore ifa which is down or loopback or not running
    if ((ifa->ifa_flags & IFF_LOOPBACK) || !(ifa->ifa_flags & IFF_RUNNING) || !(ifa->ifa_flags & IFF_UP)) {
        SHM_LOG_DEBUG("ifa flag un-match, flag=" << ifa->ifa_flags);
        return false;
    }

    if (sockType == AF_INET6) {
        struct sockaddr_in6 *sa6 = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr);
        if (IN6_IS_ADDR_LINKLOCAL(&sa6->sin6_addr)) {
            SHM_LOG_DEBUG("ifa is scope link addr " << ifaName);
            return false;
        }
    }
    return true;
}

inline int32_t aclshmemi_get_ip_from_ifa(char *local, sa_family_t &sockType, const char *ipInfo)
{
    struct ifaddrs *ifaddr;
    char ifaName[MAX_IFCONFIG_LENGTH];
    sockType = AF_INET;
    bool flag = false;
    if (ipInfo == nullptr) {
        std::string ifaNameStr = "eth";
        ifaNameStr.resize(DEFAULT_IFNAME_LNEGTH, '\0');
        std::copy(ifaNameStr.begin(), ifaNameStr.end(), ifaName);
        ifaName[DEFAULT_IFNAME_LNEGTH - 1] = '\0';
        SHM_LOG_INFO("use default if to find IP:" << ifaName);
    } else if (parse_interface_with_type(ipInfo, ifaName, sockType, flag) != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("IP size set in SHMEM_CONF_STORE_MASTER_IF format has wrong length");
        return ACLSHMEM_INVALID_PARAM;
    }
    if (getifaddrs(&ifaddr) == -1) {
        SHM_LOG_ERROR("get local net interfaces failed: " << errno);
        return ACLSHMEM_INVALID_PARAM;
    }
    int32_t result = ACLSHMEM_INVALID_PARAM;
    for (auto ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if (!aclshmemi_check_ifa(ifa, sockType, flag, ifaName, strlen(ifaName))) {
            continue;
        }
        if (sockType == AF_INET && flag) {
            auto localIp = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr)->sin_addr;
            if (inet_ntop(sockType, &localIp, local, 64) == nullptr) {
                SHM_LOG_ERROR("convert local ipv4 to string failed. ");
                continue;
            }
            result = ACLSHMEM_SUCCESS;
            break;
        } else if (sockType == AF_INET6 && flag) {
            auto localIp = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr)->sin6_addr;
            if (inet_ntop(sockType, &localIp, local, 64) == nullptr) {
                SHM_LOG_ERROR("convert local ipv6 to string failed. ");
                continue;
            }
            result = ACLSHMEM_SUCCESS;
            break;
        } else {
            auto ret = aclshmemi_auto_get_ip(ifa->ifa_addr, local, sockType);
            if (ret != ACLSHMEM_SUCCESS) {
                continue;
            }
            result = ACLSHMEM_SUCCESS;
            break;
        }
    }
    freeifaddrs(ifaddr);
    return result;
}

inline int32_t aclshmemi_get_ip_from_env(char *ip, uint16_t &port, sa_family_t &sockType, const char *ipPort)
{
    if (ipPort == nullptr) {
        return ACLSHMEM_INVALID_PARAM;
    }
    SHM_LOG_DEBUG("get env SHMEM_UID_SESSION_ID value:" << ipPort);
    std::string ipPortStr = ipPort;

    if (ipPort[0] == '[') {
        sockType = AF_INET6;
        size_t found = ipPortStr.find_last_of(']');
        if (found == std::string::npos || ipPortStr.length() - found <= 1) {
            SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
            return ACLSHMEM_INVALID_PARAM;
        }
        std::string ipStr = ipPortStr.substr(1, found - 1);
        std::string portStr = ipPortStr.substr(found + 2);

        std::snprintf(ip, MAX_IP, "%s", ipStr.c_str());

        port = std::stoi(portStr);
    } else {
        sockType = AF_INET;
        size_t found = ipPortStr.find_last_of(':');
        if (found == std::string::npos || ipPortStr.length() - found <= 1) {
            SHM_LOG_ERROR("get env SHMEM_UID_SESSION_ID is invalid");
            return ACLSHMEM_INVALID_PARAM;
        }
        std::string ipStr = ipPortStr.substr(0, found);
        std::string portStr = ipPortStr.substr(found + 1);

        std::snprintf(ip, MAX_IP, "%s", ipStr.c_str());

        port = std::stoi(portStr);
    }
    return ACLSHMEM_SUCCESS;
}

inline int32_t aclshmemi_set_ip_info(aclshmemx_uniqueid_t *uid, sa_family_t &sock_type, char *pta_env_ip, uint16_t pta_env_port,
                          bool is_from_ifa)
{
    // init default uid
    SHM_ASSERT_RETURN(uid != nullptr, ACLSHMEM_INVALID_PARAM);
    aclshmemx_uniqueid_t default_uid{};
    default_uid.version = ACLSHMEM_UNIQUEID_VERSION;
    *uid = default_uid;
    shmemx_bootstrap_uid_state_t *innerUID = reinterpret_cast<shmemx_bootstrap_uid_state_t *>(uid);
    if (sock_type == AF_INET) {
        innerUID->addr.addr.addr4.sin_family = AF_INET;
        if (inet_pton(AF_INET, pta_env_ip, &(innerUID->addr.addr.addr4.sin_addr)) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv4 failed");
            return ACLSHMEM_NOT_INITED;
        }
        innerUID->addr.type = ADDR_IPv4;
    } else if (sock_type == AF_INET6) {
        innerUID->addr.addr.addr6.sin6_family = AF_INET6;
        if (inet_pton(AF_INET6, pta_env_ip, &(innerUID->addr.addr.addr6.sin6_addr)) <= 0) {
            SHM_LOG_ERROR("inet_pton IPv6 failed");
            return ACLSHMEM_NOT_INITED;
        }
        innerUID->addr.type = ADDR_IPv6;
    } else {
        SHM_LOG_ERROR("IP Type is not IPv4 or IPv6");
        return ACLSHMEM_INVALID_PARAM;
    }

    // fill ip port as part of uid
    if (is_from_ifa) {
        int32_t ret = aclshmemi_get_port_magic(innerUID, pta_env_ip);
        if (ret != 0) {
            SHM_LOG_ERROR("get available port failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
    } else {
        int32_t ret = aclshmemi_using_env_port(innerUID, pta_env_ip, pta_env_port);
        if (ret != 0) {
            SHM_LOG_ERROR("using env port failed.");
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    SHM_LOG_INFO("gen unique id success.");
    return ACLSHMEM_SUCCESS;
}

}  // namespace utils
}  // namespace shm

#endif