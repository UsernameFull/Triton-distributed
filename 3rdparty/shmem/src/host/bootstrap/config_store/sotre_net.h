/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONFIG_STORE_NET_H
#define CONFIG_STORE_NET_H

#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <cstring>
#include <functional>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

using Result = int32_t;

enum IpType {
    IpV4,
    IpV6,
    IPNONE,
};

typedef struct {
    union {
        sockaddr_in ipv4;
        sockaddr_in6 ipv6;
    } ip;
    IpType type;
} mf_sockaddr;

typedef struct {
    union {
        uint32_t addrv4;
        uint8_t addrv6[16];
    } addr;
    IpType type;
} mf_ip_addr;

struct net_addr_t {
    union {
        struct in_addr ipv4;
        struct in6_addr ipv6;
    } ip {};
    IpType type;

    net_addr_t() : type(IPNONE) {}

    static net_addr_t from_ipv4(const struct in_addr& addr)
    {
        net_addr_t result;
        result.type = IpV4;
        result.ip.ipv4 = addr;
        return result;
    }

    static net_addr_t from_ipv6(const struct in6_addr& addr)
    {
        net_addr_t result;
        result.type = IpV6;
        result.ip.ipv6 = addr;
        return result;
    }

    bool operator==(const net_addr_t& other) const
    {
        if (type != other.type) return false;

        if (type == IpV4) {
            return ip.ipv4.s_addr == other.ip.ipv4.s_addr;
        } else if (type == IpV6) {
            return std::memcmp(&ip.ipv6, &other.ip.ipv6, sizeof(struct in6_addr)) == 0;
        }

        return true;
    }
};

#ifdef __cplusplus
}
#endif

namespace std {
    template<>
    struct hash<net_addr_t> {
        size_t operator()(const net_addr_t& addr) const
        {
            size_t result = 0;
            
            hash_combine(result, static_cast<int>(addr.type));
            
            if (addr.type == IpV4) {
                hash_combine(result, addr.ip.ipv4.s_addr);
            } else if (addr.type == IpV6) {
                const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&addr.ip.ipv6);
                for (size_t i = 0; i < sizeof(struct in6_addr); ++i) {
                    hash_combine(result, bytes[i]);
                }
            }
            
            return result;
        }
        
    private:
        static void hash_combine(size_t& seed, size_t value)
        {
            constexpr size_t SHIFT_LEFT = 6;
            constexpr size_t SHIFT_RIGHT = 2;
            seed ^= value + 0x9e3779b9 + (seed << SHIFT_LEFT) + (seed >> SHIFT_RIGHT);
        }
    };
}

#endif // CONFIG_STORE_NET_H