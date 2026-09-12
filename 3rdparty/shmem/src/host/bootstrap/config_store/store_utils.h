/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SHMEM_STORE_UTILS_H
#define SHMEM_STORE_UTILS_H

#include <cstdint>

namespace shm {
namespace store {
using Result = int32_t;

enum SMErrorCode : int32_t {
    SM_OK = 0,
    SM_ERROR = -1,
    SM_INVALID_PARAM = -2000,
    SM_MALLOC_FAILED = -2001,
    SM_NEW_OBJECT_FAILED = -2002,
    SM_NOT_STARTED = -2003,
    SM_TIMEOUT = -2004,
    SM_REPEAT_CALL = -2005,
    SM_DUPLICATED_OBJECT = -2006,
    SM_OBJECT_NOT_EXISTS = -2007,
    SM_NOT_INITIALIZED = -2008,
    SM_RESOURCE_IN_USE = -2009,
};

constexpr int32_t N16 = 16;
constexpr int32_t N64 = 64;
constexpr int32_t N256 = 256;
constexpr int32_t N1024 = 1024;
constexpr int32_t N8192 = 8192;

constexpr uint32_t UN2 = 2;
constexpr uint32_t UN32 = 32;
constexpr uint32_t UN58 = 58;
constexpr uint32_t UN128 = 128;
constexpr uint32_t UN65536 = 65536;
constexpr uint32_t UN16777216 = 16777216;

constexpr uint32_t SMEM_DEFAUT_WAIT_TIME = 120; // 120s
constexpr uint32_t SMEM_WORLD_SIZE_MAX = 16384; // refs to SHMEM_MAX_RANKS
constexpr uint32_t SMEM_ID_MAX = 63;
constexpr uint64_t SMEM_LOCAL_SIZE_MAX = 40 * 1024 * 1024 * 1024L;
constexpr uint32_t MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB

typedef int (*smem_decrypt_handler)(const char *cipherText, size_t cipherTextLen, char *plainText,
                                    size_t &plainTextLen);
} // namespace store
} // namespace shm

#endif // SHMEM_STORE_UTILS_H
