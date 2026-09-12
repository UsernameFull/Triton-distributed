/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_ACLSHMEMI_HOST_DEF_H
#define SHMEM_ACLSHMEMI_HOST_DEF_H

#include <sys/socket.h>
#include <netinet/in.h>
constexpr size_t MAX_ENV_STRING_LEN = 12800;
typedef enum {
    ADDR_IPv4,
    ADDR_IPv6
} addr_type_t;

// shmem unique id
typedef struct {
    union {
        struct sockaddr sa;
        struct sockaddr_in addr4;  // IPv4地址（含端口）
        struct sockaddr_in6 addr6; // IPv6地址（含端口）
    } addr;
    addr_type_t type;
} sockaddr_t;

typedef struct {
    int32_t version;
    int my_pe;
    int n_pes;
    int32_t inner_sockFd;          // for mf backend              
    sockaddr_t addr;              // 动态传入的地址（含端口）
    uint64_t magic;
} aclshmemi_bootstrap_uid_state_t;
#define shmemx_bootstrap_uid_state_t aclshmemi_bootstrap_uid_state_t
#endif // ACLSHMEM_ACLSHMEMI_HOST_DEF_H