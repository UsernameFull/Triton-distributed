/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACC_LINKS_ACC_OUT_LOGGER_H
#define ACC_LINKS_ACC_OUT_LOGGER_H

#include <ctime>
#include <iomanip>
#include <cstring>
#include <iostream>
#include <mutex>
#include <unistd.h>
#include <sstream>
#include <sys/time.h>
#include <sys/syscall.h>

#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"

#define LOG_DEBUG(ARGS) SHM_OUT_LOG(aclshmem_log::DEBUG_LEVEL, ARGS)
#define LOG_INFO(ARGS) SHM_OUT_LOG(aclshmem_log::INFO_LEVEL, ARGS)
#define LOG_WARN(ARGS) SHM_OUT_LOG(aclshmem_log::WARN_LEVEL, ARGS)
#define LOG_ERROR(ARGS) SHM_OUT_LOG(aclshmem_log::ERROR_LEVEL, ARGS)

#ifndef ENABLE_TRACE_LOG
#define LOG_TRACE(x)
#elif
#define LOG_TRACE(x) ACC_LINKS_OUT_LOG(DEBUG_LEVEL, AccLinksTrace, ARGS)
#endif

#define ASSERT_RETURN(ARGS, RET)           \
    do {                                   \
        if (UNLIKELY(!(ARGS))) {           \
            LOG_ERROR("Assert " << #ARGS); \
            return RET;                    \
        }                                  \
    } while (0)

#define ASSERT_RET_VOID(ARGS)              \
    do {                                   \
        if (UNLIKELY(!(ARGS))) {           \
            LOG_ERROR("Assert " << #ARGS); \
            return;                        \
        }                                  \
    } while (0)

#define ASSERT(ARGS)                       \
    do {                                   \
        if (UNLIKELY(!(ARGS))) {           \
            LOG_ERROR("Assert " << #ARGS); \
        }                                  \
    } while (0)

#define VALIDATE_RETURN(ARGS, msg, RET)          \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            LOG_ERROR(msg);                      \
            return RET;                          \
        }                                        \
    } while (0)

#define LOG_ERROR_RETURN_IT_IF_NOT_OK(result, msg) \
    do {                                           \
        auto innerResult = (result);               \
        if (UNLIKELY(innerResult != ACC_OK)) {     \
            LOG_ERROR(msg);                        \
            return innerResult;                    \
        }                                          \
    } while (0)

#endif  // ACC_LINKS_ACC_OUT_LOGGER_H
