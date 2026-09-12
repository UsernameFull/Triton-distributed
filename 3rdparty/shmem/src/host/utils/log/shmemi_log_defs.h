/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef SHMEMI_LOG_DEFS_H
#define SHMEMI_LOG_DEFS_H

#include <cstdint>
#include <string>
#include <sys/stat.h>

namespace aclshmem_log {

// 日志文件管理核心常量
constexpr size_t MAX_LOG_FILE_COUNT = 50;                               // 最多保留50个日志文件
constexpr size_t MAX_FILE_NAME_LEN = 255;                               // 文件名最大长度（不含\0）
constexpr uint64_t MAX_FILE_SIZE_THRESHOLD = 1024 * 1024 * 1024;        // 单个日志文件最大1GB
constexpr uint64_t DISK_AVAILABLE_LIMIT = 10 * MAX_FILE_SIZE_THRESHOLD; // 磁盘剩余空间门限10GB

// 内部辅助函数声明（仅在cpp中使用）
std::string get_home_dir();
bool is_invalid_path(const std::string& path);
std::string normalize_path(const std::string& path);
void make_dir_recursive(const std::string& dir);
bool is_disk_available(const std::string& dir);
bool starts_with(const std::string& str, const std::string& prefix);
bool ends_with(const std::string& str, const std::string& suffix);
bool is_all_digit(const std::string& str);

} // namespace aclshmem_log

#endif // ACLSHMEMI_LOG_DEFS_H