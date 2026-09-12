/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef UTILS_H
#define UTILS_H

#include <acl/acl.h>
#include <climits>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <string>
#include <iomanip>
#include <cstring>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
#include <runtime/rt_ffts.h>
#include "shmem.h"

#define INFO_LOG(fmt, args...) fprintf(stdout, "[INFO] " fmt "\n", ##args)
#define WARN_LOG(fmt, args...) fprintf(stdout, "[WARN] " fmt "\n", ##args)
#define ERROR_LOG(fmt, args...) fprintf(stdout, "[ERROR] " fmt "\n", ##args)

// Macro function for unwinding acl errors.
#define ACL_CHECK(status)                                                                    \
    do {                                                                                     \
        aclError error = status;                                                             \
        if (error != ACL_ERROR_NONE) {                                                       \
            std::cerr << __FILE__ << ":" << __LINE__ << " aclError:" << error << std::endl;  \
        }                                                                                    \
    } while (0)

// ACL错误检查宏
#define ACL_CHECK_WITH_RET(cond, log_func, return_expr)                                                           \
    do {                                                                                                 \
        if ((cond) != ACL_ERROR_NONE) {                                                                  \
            log_func;                                                                                    \
            return_expr;                                                                                 \
        }                                                                                                \
    } while (0)

// Macro function for unwinding rt errors.
#define RT_CHECK(status)                                                                     \
    do {                                                                                     \
        rtError_t error = status;                                                            \
        if (error != RT_ERROR_NONE) {                                                        \
            std::cerr << __FILE__ << ":" << __LINE__ << " rtError:" << error << std::endl;   \
        }                                                                                    \
    } while (0)

inline bool ReadFile(const std::string &filePath, void *buffer, size_t bufferSize)
{
    struct stat sBuf;
    int fileStatus = stat(filePath.data(), &sBuf);
    if (fileStatus == -1) {
        ERROR_LOG("Failed to get file");
        return false;
    }
    if (S_ISREG(sBuf.st_mode) == 0) {
        ERROR_LOG("%s is not a file, please enter a file.", filePath.c_str());
        return false;
    }

    std::ifstream file;
    file.open(filePath, std::ios::binary);
    if (!file.is_open()) {
        ERROR_LOG("Open file failed. path = %s.", filePath.c_str());
        return false;
    }

    std::filebuf *buf = file.rdbuf();
    size_t size = buf->pubseekoff(0, std::ios::end, std::ios::in);
    if (size == 0) {
        ERROR_LOG("File size is 0");
        file.close();
        return false;
    }
    if (size > bufferSize) {
        ERROR_LOG("File size is larger than buffer size.");
        file.close();
        return false;
    }
    buf->pubseekpos(0, std::ios::in);
    buf->sgetn(static_cast<char *>(buffer), size);
    file.close();
    return true;
}

inline bool WriteFile(const std::string &filePath, const void *buffer, size_t size, size_t offset = 0)
{
    if (buffer == nullptr) {
        ERROR_LOG("Write file failed. Buffer is nullptr.");
        return false;
    }

    int fd = open(filePath.c_str(), O_RDWR | O_CREAT, 0666);
    if (!fd) {
        ERROR_LOG("Open file failed. path = %s", filePath.c_str());
        return false;
    }

    // lock
    if (flock(fd, LOCK_EX) == -1) {
        std::cerr << "Failed to acquire lock: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // move ptr to specified offset
    if (lseek(fd, offset, SEEK_SET) == -1) {
        std::cerr << "Failed to seek in file: " << strerror(errno) << std::endl;
        close(fd);
        return false;
    }

    // write data
    if (write(fd, static_cast<const char *>(buffer), size) != static_cast<ssize_t>(size)) {
        std::cerr << "Failed to write to file: " << strerror(errno) << std::endl;
    }

    // unlock
    flock(fd, LOCK_UN);

    close(fd);
    return true;
}

inline int32_t test_set_attr(int32_t my_pe, int32_t n_pes, uint64_t local_mem_size, const char *ip_port, aclshmemx_uniqueid_t default_flag_uid,
                       aclshmemx_init_attr_t *attributes)
{
    size_t ip_len = 0;
    if (ip_port != nullptr) {
        ip_len = std::min(strlen(ip_port), static_cast<size_t>(ACLSHMEM_MAX_IP_PORT_LEN) - 1);
        std::copy_n(ip_port, ip_len, attributes->ip_port);
        if (attributes->ip_port[0] == '\0') {
            return ACLSHMEM_INVALID_VALUE;
        }
    }
    int attr_version = (1 << 16) + sizeof(aclshmemx_init_attr_t);
    attributes->my_pe = my_pe;
    attributes->n_pes = n_pes;
    attributes->ip_port[ip_len] = '\0';
    attributes->local_mem_size = local_mem_size;
    attributes->option_attr = {attr_version, ACLSHMEM_DATA_OP_MTE, DEFAULT_TIMEOUT, 
                               DEFAULT_TIMEOUT, DEFAULT_TIMEOUT};
    attributes->comm_args = reinterpret_cast<void *>(&default_flag_uid);
    return ACLSHMEM_SUCCESS;
}

inline bool make_dir(const std::string& path) {
    if (path.empty()) return true;

    // 尝试创建目录，0755 权限
    if (mkdir(path.c_str(), 0755) == 0) {
        std::cout << "[INFO] Directory created successfully: " << path << std::endl;
        return true;
    }

    // 目录已存在也算成功，打印提示
    if (errno == EEXIST) {
        std::cout << "[INFO] Directory already exists: " << path << std::endl;
        return true;
    }

    // 目录创建失败，打印错误原因
    std::cerr << "[ERROR] Failed to create directory '" << path 
              << "' (errno: " << errno << "): ";
    switch (errno) {
        case EACCES: std::cerr << "Permission denied"; break;
        case EROFS:  std::cerr << "Read-only file system"; break;
        case ENAMETOOLONG: std::cerr << "Path name too long"; break;
        default: std::cerr << "Unknown error";
    }
    std::cerr << std::endl;
    return false;
}

inline std::string get_dir(const std::string& filename) {
    size_t pos = filename.find_last_of("/");
    if (pos == std::string::npos) return "";
    return filename.substr(0, pos);
}


inline void write_csv(const std::string& filename, const std::vector<std::vector<std::string>>& data) {

    std::string dir = get_dir(filename);
    if (!dir.empty() && !make_dir(dir)) {
        std::cerr << "Error: create dir failed: " << dir << std::endl;
        return;
    }

    std::ofstream out_file(filename);

    if (!out_file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return;
    }

    for (const auto& row : data) {
        for (size_t i = 0; i < row.size(); ++i) {
            out_file << row[i];
            if (i < row.size() - 1) {
                out_file << ",";
            }
        }
        out_file << "\n";
    }

    out_file.close();
}

inline std::string int_to_string(int value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

inline std::string float_to_string(float value) {
    std::ostringstream oss;
    oss << value;
    return oss.str();
}

inline std::string double_to_string(double value) {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4) << value;
    return oss.str();
}

inline void collect_prof_data_to_csv(aclshmem_prof_pe_t *out_profs, int frame_id, int block_size, 
                                    uint64_t datasize, int g_npus, int ub_size,
                                    std::vector<std::vector<std::string>>& prof_csv_data)
{
    if (out_profs == nullptr) {
        return;
    }
    
    const char *soc_name = aclrtGetSocName();
    int64_t cycle2us = 50;
    std::string soc_str;
    if (soc_name != nullptr) {
        soc_str = std::string(soc_name);
        if (soc_str.find("Ascend950") != std::string::npos) {
            cycle2us = 1000;
        }
    }
    INFO_LOG("SocName: %s, cycle2us: %lld", soc_str.c_str(), (long long)cycle2us);
    
    double max_core_time = 0.0;
    std::vector<double> core_times;
    
    int actual_blocks = std::min(block_size, ACLSHMEM_CYCLE_PROF_MAX_BLOCK);
    
    for (int32_t block_id = 0; block_id < actual_blocks; block_id++) {
        aclshmem_prof_block_t *prof = &out_profs->block_prof[block_id];

        if (prof->ccount[frame_id] == 0) {
            continue;
        }

        double avg_us = (double)prof->cycles[frame_id] / prof->ccount[frame_id] / cycle2us;
        
        if (avg_us > max_core_time) {
            max_core_time = avg_us;
        }
        core_times.push_back(avg_us);
    }
    
    double bandwidth = 0.0;
    if (max_core_time > 0) {
        bandwidth = (double)datasize * (double)block_size / max_core_time * 1000000.0 / 1024.0 / 1024.0 / 1024.0;
    }
    
    std::vector<std::string> sub_data = {
        int_to_string(datasize), 
        int_to_string(g_npus), 
        int_to_string(block_size), 
        int_to_string(ub_size),
        double_to_string(bandwidth),
        double_to_string(max_core_time)
    };
    
    for (size_t i = 0; i < core_times.size(); i++) {
        sub_data.push_back(double_to_string(core_times[i]));
    }
    
    prof_csv_data.push_back(sub_data);
}

#endif // UTILS_H
