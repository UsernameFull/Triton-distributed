/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SHM_NET_UTIL_H
#define SHMEM_SHM_NET_UTIL_H

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

inline std::string get_default_network_interface()
{
    std::string routeFileName{"/proc/net/route"};
    std::ifstream input(routeFileName);
    if (!input.is_open()) {
        SHM_LOG_ERROR("open route file failed: " << strerror(errno));
        return "";
    }

    std::string ifname;
    uint32_t destination;
    uint32_t temp;
    uint32_t mask;
    std::string line;
    std::getline(input, line);  // skip header line
    while (std::getline(input, line)) {
        std::stringstream ss{line};
        ss >> ifname >> std::hex;  // Iface
        ss >> destination;         // Destination
        ss >> temp;                // Gateway
        ss >> temp;                // Flags
        ss >> temp;                // RefCnt
        ss >> temp;                // Use
        ss >> temp;                // Metric
        ss >> mask;                // Mask
        if (destination == 0U && mask == 0U) {
            SHM_LOG_INFO("default route network : " << ifname);
            return ifname;
        }
    }
    return "";
}

inline std::vector<uint32_t> NetworkGetIpAddresses() noexcept
{
    std::vector<uint32_t> addresses;
    struct ifaddrs *ifa;
    struct ifaddrs *p;
    if (getifaddrs(&ifa) < 0) {
        SHM_LOG_ERROR("getifaddrs() failed: " << errno << " : " << strerror(errno));
        return addresses;
    }

    uint32_t routeIp = 0;
    auto routeName = get_default_network_interface();
    for (p = ifa; p != nullptr; p = p->ifa_next) {
        if (p->ifa_addr == nullptr) {
            continue;
        }

        if (p->ifa_addr->sa_family != AF_INET && p->ifa_addr->sa_family != AF_INET6) {
            continue;
        }

        if ((p->ifa_flags & IFF_LOOPBACK) != 0) {
            continue;
        }

        if ((p->ifa_flags & IFF_UP) == 0 || (p->ifa_flags & IFF_RUNNING) == 0) {
            continue;
        }

        std::string ifname{p->ifa_name};
        if (p->ifa_addr->sa_family == AF_INET) {
            auto sin = reinterpret_cast<struct sockaddr_in *>(p->ifa_addr);
            uint32_t ip = ntohl(sin->sin_addr.s_addr);
            if (routeName == ifname) {
                routeIp = ip;
                SHM_LOG_INFO("find route ip address: " << ifname << " -> " << inet_ntoa(sin->sin_addr));
            } else {
                addresses.emplace_back(ip);
                SHM_LOG_INFO("find ip address: " << ifname << " -> " << inet_ntoa(sin->sin_addr));
            }
        } else if (p->ifa_addr->sa_family == AF_INET6) {
            auto sin6 = reinterpret_cast<struct sockaddr_in6 *>(p->ifa_addr);
            char addr_str[INET6_ADDRSTRLEN];
            inet_ntop(AF_INET6, &(sin6->sin6_addr), addr_str, INET6_ADDRSTRLEN);
            auto ip_address = addr_str;

            const uint64_t* latter_ptr = reinterpret_cast<const uint64_t*>(&(sin6->sin6_addr.s6_addr[8]));
            uint64_t latter_id = *latter_ptr;
            std::hash<uint64_t> hasher;
            uint32_t ipv6_derived_id = static_cast<uint32_t>(hasher(latter_id));

            if (routeName == ifname) {
                routeIp = ipv6_derived_id;
                SHM_LOG_INFO("find route ip6 address: " << ifname << " -> " << ip_address);
            } else {
                addresses.emplace_back(ipv6_derived_id);
                SHM_LOG_INFO("find ip6 address: " << ifname << " -> " << ip_address);
            }
        }
    }

    freeifaddrs(ifa);
    std::sort(addresses.begin(), addresses.end(), std::less<uint32_t>());
    if (routeIp != 0) {
        addresses.insert(addresses.begin(), routeIp);
    }
    return addresses;
}

}  // namespace utils
}  // namespace shm

#endif