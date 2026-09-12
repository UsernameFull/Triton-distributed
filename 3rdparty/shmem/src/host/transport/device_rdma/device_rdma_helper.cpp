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
#include <cstdlib>
#include <cerrno>
#include <sstream>
#include <regex>
#include <limits>
#include "shmemi_logger.h"
#include "shmemi_string_util.h"
#include "shmemi_num_util.h"
#include "device_rdma_helper.h"

namespace shm {
namespace transport {
namespace device {
Result ParseDeviceNic(const std::string &nic, uint16_t &port)
{
    if (!utils::StringUtil::String2Uint(nic, port) || port == 0) {
        SHM_LOG_ERROR("failed to convert nic : " << nic << " to uint16_t, or port is 0.");
        return ACLSHMEM_INVALID_PARAM;
    }
    return ACLSHMEM_SUCCESS;
}

Result ParseDeviceNic(const std::string &nic, mf_sockaddr &address)
{
    SHM_LOG_DEBUG("parse device nic input nic(" << nic << ").");
    size_t proto_end = nic.find("://");
    if (proto_end == std::string::npos) {
        SHM_LOG_ERROR("input nic(" << nic << ") not matches - no protocol separator.");
        return ACLSHMEM_INVALID_PARAM;
    }

    std::string protocol = nic.substr(0, proto_end);

    if (protocol.length() > 16) {
        SHM_LOG_ERROR("protocol name too long: " << protocol);
        return ACLSHMEM_INVALID_PARAM;
    }

    for (char c : protocol) {
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_')) {
            SHM_LOG_ERROR("invalid protocol character: " << c);
            return ACLSHMEM_INVALID_PARAM;
        }
    }

    std::string remaining = nic.substr(proto_end + 3);

    if (remaining.length() >= 4 && remaining[0] == '[') {
        size_t bracket_end = remaining.find(']');
        if (bracket_end == std::string::npos) {
            SHM_LOG_ERROR("IPv6 format error - missing closing bracket: " << nic);
            return ACLSHMEM_INVALID_PARAM;
        }

        std::string ip_str = remaining.substr(1, bracket_end - 1);
        std::string port_str = remaining.substr(bracket_end + 2);

        size_t port_start = remaining.find(':', bracket_end);
        if (port_start == std::string::npos) {
            SHM_LOG_ERROR("IPv6 format error - missing port: " << nic);
            return ACLSHMEM_INVALID_PARAM;
        }

        std::string actual_port = remaining.substr(port_start + 1);

        if (actual_port.length() > 5) {
            SHM_LOG_ERROR("port number too long: " << actual_port);
            return ACLSHMEM_INVALID_PARAM;
        }

        for (char c : actual_port) {
            if (c < '0' || c > '9') {
                SHM_LOG_ERROR("invalid port characters: " << actual_port);
                return ACLSHMEM_INVALID_PARAM;
            }
        }

        if (inet_pton(AF_INET6, ip_str.c_str(), &address.ip.ipv6.sin6_addr) != 1) {
            SHM_LOG_ERROR("parse ip for nic: " << nic << " failed.");
            return ACLSHMEM_INVALID_PARAM;
        }

        uint16_t port_val;
        if (!utils::StringUtil::String2Uint(actual_port, port_val)) {
            SHM_LOG_ERROR("failed to convert str : " << actual_port << " to uint16_t, or sin_port is 0.");
            return ACLSHMEM_INVALID_PARAM;
        }

        address.type = IpV6;
        address.ip.ipv6.sin6_family = AF_INET6;
        address.ip.ipv6.sin6_port = htons(port_val);

        return ACLSHMEM_SUCCESS;
    } else {
        size_t last_colon = remaining.rfind(':');
        if (last_colon == std::string::npos) {
            SHM_LOG_ERROR("IPv4 format error - missing port: " << nic);
            return ACLSHMEM_INVALID_PARAM;
        }
        std::string ip_str = remaining.substr(0, last_colon);
        std::string port_str = remaining.substr(last_colon + 1);

        if (port_str.length() > 5) {
            SHM_LOG_ERROR("port number too long: " << port_str);
            return ACLSHMEM_INVALID_PARAM;
        }

        for (char c : port_str) {
            if (c < '0' || c > '9') {
                SHM_LOG_ERROR("invalid port characters: " << port_str);
                return ACLSHMEM_INVALID_PARAM;
            }
        }

        if (inet_aton(ip_str.c_str(), &address.ip.ipv4.sin_addr) == 0) {
            SHM_LOG_ERROR("parse ip for nic: " << nic << " failed.");
            return ACLSHMEM_INVALID_PARAM;
        }

        uint16_t port_val;
        if (!utils::StringUtil::String2Uint(port_str, port_val)) {
            SHM_LOG_ERROR("failed to convert str : " << port_str << " to uint16_t, or sin_port is 0.");
            return ACLSHMEM_INVALID_PARAM;
        }

        address.type = IpV4;
        address.ip.ipv4.sin_family = AF_INET;
        address.ip.ipv4.sin_port = port_val;

        return ACLSHMEM_SUCCESS;
    }
}

std::string GenerateDeviceNic(net_addr_t ip, uint16_t port)
{
    std::stringstream ss;
    if (ip.type == IpV4) {
        ss << "tcp://" << inet_ntoa(ip.ip.ipv4) << ":" << port;
    } else {
        char ipv6Str[INET6_ADDRSTRLEN];
        inet_ntop(AF_INET6, &ip.ip.ipv6, ipv6Str, INET6_ADDRSTRLEN);
        ss << "tcp6://[" << ipv6Str << "]:" << port;
    }
    return ss.str();
}
}
}
}