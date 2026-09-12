/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */


#ifndef STORE_NET_COMMON_H
#define STORE_NET_COMMON_H

#include "sotre_net.h"
#include "store_utils.h"

namespace shm {
namespace store {

using Result = int32_t;

struct UrlExtraction {
    std::string ip;           /* ip */
    uint16_t port = 9980L;    /* listen port */

    Result ExtractIpPortFromUrl(const std::string &url);
};

Result GetLocalIpWithTarget(const std::string &target, std::string &local, mf_ip_addr &ipaddr);

inline bool CharToLong(const char* src, long &value)
{
    char *remain = nullptr;
    errno = 0;
    value = std::strtol(src, &remain, 10L);  // 10 is decimal digits
    if ((value == 0 && std::strcmp(src, "0") != 0) || remain == nullptr || strlen(remain) > 0 || errno == ERANGE) {
        return false;
    }
    return true;
}

inline bool StrToLong(const std::string &src, long &value)
{
    return CharToLong(src.c_str(), value);
}
}
}
#endif // STORE_NET_COMMON_H