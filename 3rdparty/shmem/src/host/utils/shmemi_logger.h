/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SHM_OUT_LOGGER_H
#define SHMEM_SHM_OUT_LOGGER_H

#include <ctime>
#include <climits>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <unistd.h>
#include <sstream>
#include <sys/time.h>
#include <sys/syscall.h>
#include <iostream>
#include "host/shmem_host_def.h"


namespace aclshmem_log {

class log_file_sink;

using external_log = void (*)(int32_t, const char *);

enum log_level : int32_t {
    DEBUG_LEVEL = 0,
    INFO_LEVEL,
    WARN_LEVEL,
    ERROR_LEVEL,
    FATAL_LEVEL,
    BUTT_LEVEL /* no use */
};

class aclshmem_out_logger {
public:
    static aclshmem_out_logger &Instance();
    aclshmem_error_code_t set_log_level(log_level level);
    void set_extern_log_func(external_log func, bool force_update = false);
    void log(int32_t level, const std::ostringstream &oss);

    aclshmem_out_logger(const aclshmem_out_logger &) = delete;
    aclshmem_out_logger(aclshmem_out_logger &&) = delete;
    ~aclshmem_out_logger();

private:
    aclshmem_out_logger();
    std::string build_log_content(int32_t level, const std::ostringstream &oss);
    const std::string &log_level_desc(int32_t level);

private:
    const std::string aclshmem_log_level_desc[BUTT_LEVEL] = {"debug", "info", "warn", "error", "fatal"};
    log_level aclshmem_log_level = WARN_LEVEL;
    external_log aclshmem_log_func = nullptr;
    log_file_sink* aclshmem_file_sink;
    bool is_log_stdout = false;
};
}  // namespace shm

// macro for gcc optimization for prediction of if/else
#ifndef LIKELY
#define LIKELY(x) (__builtin_expect(!!(x), 1) != 0)
#endif

#ifndef UNLIKELY
#define UNLIKELY(x) (__builtin_expect(!!(x), 0) != 0)
#endif

#ifndef SHM_LOG_FILENAME_SHORT
#define SHM_LOG_FILENAME_SHORT (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#endif

#define SHM_OUT_LOG(LEVEL, ARGS)                                                            \
    do {                                                                                    \
        std::ostringstream oss;                                                             \
        oss << "[SHM_SHMEM " << SHM_LOG_FILENAME_SHORT << ":" << __LINE__ << "] " << ARGS;  \
        aclshmem_log::aclshmem_out_logger::Instance().log(LEVEL, oss);                                    \
    } while (0)

#define SHM_LOG_DEBUG(ARGS) SHM_OUT_LOG(aclshmem_log::DEBUG_LEVEL, ARGS)
#define SHM_LOG_INFO(ARGS) SHM_OUT_LOG(aclshmem_log::INFO_LEVEL, ARGS)
#define SHM_LOG_WARN(ARGS) SHM_OUT_LOG(aclshmem_log::WARN_LEVEL, ARGS)
#define SHM_LOG_ERROR(ARGS) SHM_OUT_LOG(aclshmem_log::ERROR_LEVEL, ARGS)

#define SHM_CHECK_CONDITION_RET(condition, RET)     \
    do {                                            \
        if (condition) {                            \
            return RET;                             \
        }                                           \
    } while (0)

#define SHM_VALIDATE_RETURN(ARGS, msg, RET)         \
    do {                                            \
        if (__builtin_expect(!(ARGS), 0) != 0) {    \
            SHM_LOG_ERROR(msg);                     \
            return RET;                             \
        }                                           \
    } while (0)

#define SHM_ASSERT_LOG_AND_RETURN(ARGS, msg, RET)   \
    do {                                            \
        if (__builtin_expect(!(ARGS), 0) != 0) {    \
            SHM_LOG_ERROR(msg);                     \
            return RET;                             \
        }                                           \
    } while (0)

#define SHM_ASSERT_RETURN(ARGS, RET)             \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
            return RET;                          \
        }                                        \
    } while (0)

#define SHM_ASSERT_RET_VOID(ARGS)                \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
            return;                              \
        }                                        \
    } while (0)

#define SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(result, msg) \
    do {                                              \
        auto innerResult = (result);                  \
        if (UNLIKELY(innerResult != 0)) {             \
            SHM_LOG_ERROR(msg);                        \
            return innerResult;                       \
        }                                             \
    } while (0)

#define SHM_ASSERT_RETURN_NOLOG(ARGS, RET)       \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            return RET;                          \
        }                                        \
    } while (0)

#define SHM_ASSERT(ARGS)                         \
    do {                                         \
        if (__builtin_expect(!(ARGS), 0) != 0) { \
            SHM_LOG_ERROR("Assert " << #ARGS);   \
        }                                        \
    } while (0)

#define SHM_ASSERT_MULTIPLY_OVERFLOW(A, B, MAX, RET)                           \
    do {                                                                       \
        if ((A) <= 0 || (B) <= 0 || (MAX) <= 0) {                              \
            SHM_LOG_ERROR("INVALID PARAM " << #A << " " << #B << " " << #MAX); \
            return RET;                                                        \
        }                                                                      \
        if ((A) > (MAX) / (B)) {                                               \
            SHM_LOG_ERROR("OVERFLOW " << #A << " * " << #B << " > " << #MAX);  \
            return RET;                                                        \
        }                                                                      \
    } while (0)

#define ACLSHMEM_CHECK(x)                                                                          \
    do {                                                                                        \
        int32_t check_ret = x;                                                                  \
        if (check_ret != 0) {                                                                   \
            SHM_LOG_ERROR(" return shmem error: " << check_ret << " - " << #x << " failed.");   \
            return ;                                                                            \
        }                                                                                       \
    } while (0)

#define ACLSHMEM_CHECK_RET(...) \
    _SHMEM_CHECK_RET_HELPER(__VA_ARGS__, _SHMEM_CHECK_RET_WITH_LOG_AND_ERR_CODE, _SHMEM_CHECK_RET_WITH_LOG, _SHMEM_CHECK_RET)(__VA_ARGS__)

#define _SHMEM_CHECK_RET(x)                                                                     \
    do {                                                                                        \
        int32_t check_ret = (x);                                                                \
        if (check_ret != 0) {                                                                   \
            SHM_LOG_ERROR(" return shmem error: " << check_ret << " - " << #x << " failed.");   \
            return check_ret;                                                                   \
        }                                                                                       \
    } while (0)

#define _SHMEM_CHECK_RET_WITH_LOG(x, LOG_STR)                                       \
    do {                                                                            \
        int32_t check_ret = (x);                                                    \
        if (check_ret != 0) {                                                       \
            SHM_LOG_ERROR(" " << LOG_STR << " return shmem error: " << check_ret);  \
            return check_ret;                                                       \
        }                                                                           \
    } while (0)

#define _SHMEM_CHECK_RET_WITH_LOG_AND_ERR_CODE(x, LOG_STR, ERR_CODE)                \
    do {                                                                            \
        int32_t check_ret = (x);                                                    \
        if (check_ret != 0) {                                                       \
            SHM_LOG_ERROR(" " << LOG_STR << " return shmem error: " << ERR_CODE);   \
            return ERR_CODE;                                                        \
        }                                                                           \
    } while (0)

#define _SHMEM_CHECK_RET_HELPER(_1, _2, _3, FUNC, ...) FUNC

#define NEW_AND_CHECK_PTR(PTR_NAME, TYPE_NAME, RET_VAL, ...) \
    do { \
        PTR_NAME = new (std::nothrow) TYPE_NAME(__VA_ARGS__); \
        if (PTR_NAME == nullptr) { \
            SHM_LOG_ERROR("New object failed: " #TYPE_NAME " (" #PTR_NAME " is nullptr) " \
                                 << "(" << __FILE__ << ":" << __LINE__ << ")"); \
            return RET_VAL; \
        } \
    } while(0)


#endif  // SHMEM_SHM_OUT_LOGGER_H
