/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEM_SHM_FUNCTION_H
#define SHMEM_SHM_FUNCTION_H

#include <cstdint>
#include "shmemi_logger.h"

namespace shm {
class funci {
public:
    /**
     * @brief Get real path
     *
     * @param path         [in/out] input path, converted realpath
     * @return true if successful
     */
    static bool get_real_path(std::string &path);

    /**
     * @brief Get real path of a library and check if exists
     *
     * @param lib_dir_path   [in] dir path of the library
     * @param lib_name      [in] library name
     * @param real_path     [out] realpath of the library
     * @return true if successful
     */
    static bool get_library_real_path(const std::string &lib_dir_path, const std::string &lib_name,
                                      std::string &real_path);
};

inline bool funci::get_real_path(std::string &path)
{
    if (path.empty() || path.size() > PATH_MAX) {
        SHM_LOG_ERROR("Failed to get realpath, path is invalid");
        return false;
    }

    /* It will allocate memory to store path */
    char *real_path = realpath(path.c_str(), nullptr);
    if (real_path == nullptr) {
        SHM_LOG_ERROR("Failed to get realpath, error " << errno);
        return false;
    }

    path = real_path;
    free(real_path);
    real_path = nullptr;
    return true;
}

inline bool funci::get_library_real_path(const std::string &lib_dir_path, const std::string &lib_name,
                                         std::string &real_path)
{
    std::string tmp_full_path = lib_dir_path;
    if (!get_real_path(tmp_full_path)) {
        return false;
    }

    if (tmp_full_path.back() != '/') {
        tmp_full_path.push_back('/');
    }

    tmp_full_path.append(lib_name);
    auto ret = ::access(tmp_full_path.c_str(), F_OK);
    if (ret != 0) {
        SHM_LOG_ERROR(tmp_full_path << " cannot be accessed, ret: " << ret);
        return false;
    }

    real_path = tmp_full_path;
    return true;
}

#define HYBM_API __attribute__((visibility("default")))

#define DL_LOAD_SYM(TARGET_FUNC_VAR, TARGET_FUNC_TYPE, FILE_HANDLE, SYMBOL_NAME)                      \
    do {                                                                                              \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME);                          \
        if ((TARGET_FUNC_VAR) == nullptr) {                                                           \
            SHM_LOG_ERROR("Failed to call dlsym to load " << (SYMBOL_NAME) << ", error" << dlerror()); \
            dlclose(FILE_HANDLE);                                                                     \
            return ACLSHMEM_DL_FUNC_FAILED;                                                             \
        }                                                                                             \
    } while (0)

#define DL_LOAD_SYM_NON_TERMINATE(TARGET_FUNC_VAR, TARGET_FUNC_TYPE, FILE_HANDLE, SYMBOL_NAME)        \
    do {                                                                                              \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME);                          \
        if ((TARGET_FUNC_VAR) == nullptr) {                                                           \
            SHM_LOG_ERROR("Failed to call dlsym to load " << (SYMBOL_NAME) << ", error" << dlerror()); \
            return ACLSHMEM_DL_FUNC_FAILED;                                                             \
        }                                                                                             \
    } while (0)

#define DL_COLLECT_RESULT(ret, func)       \
    do {                                   \
        Result tmp_ret = (func);           \
        if (tmp_ret != ACLSHMEM_SUCCESS) { \
            (ret) = tmp_ret;               \
        }                                  \
    } while (0)

#define DL_LOAD_SYM_ALT(TARGET_FUNC_VAR, TARGET_FUNC_TYPE, FILE_HANDLE, SYMBOL_NAME, SYMBOL_NAME_ALT) \
    do {                                                                                              \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME);                          \
        if ((TARGET_FUNC_VAR) != nullptr) {                                                           \
            SHM_LOG_DEBUG("Loaded symbol " << (SYMBOL_NAME) << " successfully");                       \
            break;                                                                                    \
        }                                                                                             \
        TARGET_FUNC_VAR = (TARGET_FUNC_TYPE)dlsym(FILE_HANDLE, SYMBOL_NAME_ALT);                      \
        if ((TARGET_FUNC_VAR) != nullptr) {                                                           \
            SHM_LOG_DEBUG("Loaded symbol " << (SYMBOL_NAME_ALT) << " successfully");                   \
            break;                                                                                    \
        }                                                                                             \
        SHM_LOG_ERROR("Failed to call dlsym to load " << (SYMBOL_NAME) << " or " << (SYMBOL_NAME_ALT)  \
                                                     << ", error" << dlerror());                      \
        dlclose(FILE_HANDLE);                                                                         \
        FILE_HANDLE = nullptr;                                                                        \
        return ACLSHMEM_DL_FUNC_FAILED;                                                                 \
    } while (0)

} // namespace shm

#endif  // ACLSHMEM_SHM_FUNCTION_H
