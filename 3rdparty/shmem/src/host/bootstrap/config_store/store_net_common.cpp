/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>
#include <ifaddrs.h>
#include <net/if.h>

#include <vector>
#include <map>
#include <regex>
#include "shmemi_string_util.h"
#include "store_net_common.h"
#include "host/shmem_host_def.h"
#include "shmemi_logger.h"

namespace shm {
namespace store {

const std::string PROTOCOL_TCP4 = "tcp://";
const std::string PROTOCOL_TCP6 = "tcp6://";
enum class PROTOCOLTYPE {
    PROTOCOLV4,
    PROTOCOLV6,
    IPNONE,
};
static PROTOCOLTYPE type = PROTOCOLTYPE::IPNONE;

inline void Split(const std::string &src, const std::string &sep, std::vector<std::string> &out)
{
    int COUNT = 1;
    std::string::size_type pos1 = 0;
    std::string::size_type pos2 = src.find_last_of(sep);

    std::string tmpStr;
    if (src[0] != '[') {
        while (pos2 != std::string::npos) {
            tmpStr = src.substr(pos1, pos2 - pos1);
            out.emplace_back(tmpStr);
            pos1 = pos2 + sep.size();
            pos2 = src.find(sep, pos1);
        }

        if (pos1 != src.length()) {
            tmpStr = src.substr(pos1);
            out.emplace_back(tmpStr);
        }
    } else {
        if (std::count(src.begin(), src.end(), sep[0]) > COUNT) {
            const int diff = 2;
            tmpStr = src.substr(pos1 + 1, pos2 - pos1 - diff);
            out.emplace_back(tmpStr);
            pos1 = pos2 + sep.size();
            pos2 = src.find(sep, pos1);
            if (pos1 != src.length()) {
                tmpStr = src.substr(pos1);
                out.emplace_back(tmpStr);
            }
        }
    }
}

bool IsValidIp(const std::string &address)
{
    // 校验输入长度，防止正则表达式栈溢出
    if (type == PROTOCOLTYPE::PROTOCOLV4) {
        constexpr size_t maxIpLenV4 = 15;
        if (address.size() > maxIpLenV4) {
            return false;
        }
        return true;
    } else if (type == PROTOCOLTYPE::PROTOCOLV6) {
        constexpr size_t maxIpLenV6 = 39;
        if (address.size() > maxIpLenV6) {
            return false;
        }
    } else {
        return false;
    }
    return true;
}

Result ExtractTcpURL(const std::string &url, std::map<std::string, std::string> &details)
{
    /* remove tcp:// or tcp6:// */
    std::string tmpUrl;
    if (url.compare(0, PROTOCOL_TCP6.size(), PROTOCOL_TCP6) == 0) {
        type = PROTOCOLTYPE::PROTOCOLV6;
        tmpUrl = url.substr(PROTOCOL_TCP6.length(), url.length() - PROTOCOL_TCP6.length());
    } else if (url.compare(0, PROTOCOL_TCP4.size(), PROTOCOL_TCP4) == 0) {
        type = PROTOCOLTYPE::PROTOCOLV4;
        tmpUrl = url.substr(PROTOCOL_TCP4.length(), url.length() - PROTOCOL_TCP4.length());
    } else {
        return ACLSHMEM_INVALID_PARAM;
    }

    /* split */
    std::vector<std::string> splits;
    Split(tmpUrl, ":", splits);
    if (splits.size() != UN2) {
        return ACLSHMEM_INVALID_PARAM;
    }

    /* assign port */
    details["port"] = splits[1];
    if (splits[0].find('/') == std::string::npos) {
        /* assign ip */
        details["ip"] = splits[0];
        return ACLSHMEM_SUCCESS;
    }

    /* get ip mask */
    tmpUrl = splits[0];
    splits.clear();
    Split(tmpUrl, "/", splits);
    if (splits.size() != UN2) {
        return ACLSHMEM_INVALID_PARAM;
    }

    details["ip"] = splits[0];
    details["mask"] = splits[1];
    return ACLSHMEM_SUCCESS;
}

Result UrlExtraction::ExtractIpPortFromUrl(const std::string &url)
{
    std::map<std::string, std::string> details;
    /* extract to vector */
    auto result = ExtractTcpURL(url, details);

    auto iterMask = details.find("mask");
    std::string ipStr = details["ip"];
    std::string portStr = details["port"];

    /* covert port */
    long tmpPort = 0;
    if (!StrToLong(portStr, tmpPort)) {
        SHM_LOG_ERROR("Invalid portStr :" << portStr << ", which is invalid");
        return ACLSHMEM_INVALID_PARAM;
    }

    if (!IsValidIp(ipStr) || tmpPort <= N1024 || tmpPort > UINT16_MAX) {
        SHM_LOG_ERROR("Invalid ipStr :" << ipStr << " or port :" << tmpPort << ", which is invalid");
        return ACLSHMEM_INVALID_PARAM;
    }

    /* set ip and port */
    ip = ipStr;
    port = tmpPort;
    return ACLSHMEM_SUCCESS;
}

static Result GetLocalIpWithTargetWhenIpv6(struct in6_addr &localIp, char *localResultIp, int size,
                                           mf_ip_addr &ipaddr, std::string &local)
{
    Result result = ACLSHMEM_SMEM_ERROR;
    if (inet_ntop(AF_INET6, &localIp, localResultIp, size) == nullptr) {
        SHM_LOG_ERROR("convert local ipv6 to string failed. ");
        result = ACLSHMEM_SMEM_ERROR;
    } else {
        ipaddr.type = IpV6;
        std::copy(std::begin(localIp.s6_addr), std::end(localIp.s6_addr), std::begin(ipaddr.addr.addrv6));
        local = std::string(localResultIp);
        result = ACLSHMEM_SUCCESS;
    }
    return result;
}

static Result GetLocalIpWithTargetWhenIpv4(struct in_addr &localIp, char *localResultIp, int size,
                                           mf_ip_addr &ipaddr, std::string &local)
{
    Result result = ACLSHMEM_SMEM_ERROR;
    if (inet_ntop(AF_INET, &localIp, localResultIp, size) == nullptr) {
        SHM_LOG_ERROR("convert local ipv4 to string failed. ");
        result = ACLSHMEM_SMEM_ERROR;
    } else {
        ipaddr.type = IpV4;
        ipaddr.addr.addrv4 = ntohl(localIp.s_addr);
        local = std::string(localResultIp);
        result = ACLSHMEM_SUCCESS;
    }
    return result;
}

static Result DetermineTargetIpType(const std::string &target, struct in_addr &targetIpV4,
                                    struct in6_addr &targetIpV6, bool &isTargetV6)
{
    if (inet_pton(AF_INET, target.c_str(), &targetIpV4) == 1) {
        isTargetV6 = false;
    } else if (inet_pton(AF_INET6, target.c_str(), &targetIpV6) == 1) {
        isTargetV6 = true;
    } else {
        SHM_LOG_ERROR("target ip address invalid.");
        return ACLSHMEM_INVALID_PARAM;
    }
    return ACLSHMEM_SUCCESS;
}

static bool IsSameNetwork(const struct in6_addr &localIp, const struct in6_addr &localMask,
                          const struct in6_addr &targetIp)
{
    constexpr int SIZE = 16;
    for (int i = 0; i < SIZE; i++) {
        if ((localIp.s6_addr[i] & localMask.s6_addr[i]) != (targetIp.s6_addr[i] & localMask.s6_addr[i])) {
            return false;
        }
    }
    return true;
}

Result GetLocalIpWithTarget(const std::string &target, std::string &local, mf_ip_addr &ipaddr)
{
    struct ifaddrs *ifaddr;
    const int SIZE = 64;
    char localResultIp[SIZE];
    Result result = ACLSHMEM_SMEM_ERROR;

    bool isTargetV6 = false;
    struct in_addr targetIpV4;
    struct in6_addr targetIpV6;
    if (DetermineTargetIpType(target, targetIpV4, targetIpV6, isTargetV6) != ACLSHMEM_SUCCESS) {
        return ACLSHMEM_INVALID_PARAM;
    }

    if (getifaddrs(&ifaddr) == -1) {
        SHM_LOG_ERROR("get local net interfaces failed: " << errno << ": " << strerror(errno));
        return ACLSHMEM_SMEM_ERROR;
    }

    for (auto ifa = ifaddr; ifa != nullptr; ifa = ifa->ifa_next) {
        if ((ifa->ifa_addr == nullptr) || ((ifa->ifa_addr->sa_family != AF_INET) &&
            (ifa->ifa_addr->sa_family != AF_INET6)) || (ifa->ifa_netmask == nullptr)) {
            continue;
        }

        if (!isTargetV6 && ifa->ifa_addr->sa_family == AF_INET) {
            auto localIp = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr)->sin_addr;
            auto localMask = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_netmask)->sin_addr;
            if ((localIp.s_addr & localMask.s_addr) != (targetIpV4.s_addr & localMask.s_addr)) {
                continue;
            }
            result = GetLocalIpWithTargetWhenIpv4(localIp, localResultIp, SIZE, ipaddr, local);
            break;
        } else if (isTargetV6 && ifa->ifa_addr->sa_family == AF_INET6) {
            auto localIp = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_addr)->sin6_addr;
            auto localMask = reinterpret_cast<struct sockaddr_in6 *>(ifa->ifa_netmask)->sin6_addr;

            if (!IsSameNetwork(localIp, localMask, targetIpV6)) {
                continue;
            }
            result = GetLocalIpWithTargetWhenIpv6(localIp, localResultIp, SIZE, ipaddr, local);
            break;
        }
    }

    freeifaddrs(ifaddr);
    return result;
}
}
}
