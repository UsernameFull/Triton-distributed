/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_C_API_ADAPTER_HPP
#define ACLSHMEMI_C_API_ADAPTER_HPP

#include "aclshmemi_topo_addr_info_manager.h"
#include <cstring>

namespace shm {
namespace topo {

class aclshmemi_c_api_adapter_t {
public:
    static int get_size(int phy_id, size_t* size);
    static int get_topo_file_path(int phy_id, char* file_path, size_t buf_size);
    static int get_root_info(int phy_id, char* rank_info, size_t* buf_size);
};

inline int aclshmemi_c_api_adapter_t::get_size(int phy_id, size_t* size) {
    if (size == nullptr) {
        return -1;
    }
    
    auto result = aclshmemi_topo_addr_info_manager_t::instance().get_size(phy_id);
    if (!result) {
        return -1;
    }
    
    *size = *result;
    return 0;
}

inline int aclshmemi_c_api_adapter_t::get_topo_file_path(int phy_id, char* file_path, size_t buf_size) {
    if (file_path == nullptr || buf_size == 0) {
        return -1;
    }
    
    auto result = aclshmemi_topo_addr_info_manager_t::instance().get_topo_file_path(phy_id);
    if (!result) {
        return -1;
    }
    
    const std::string& path = *result;
    if (path.size() + 1 > buf_size) {
        return -1;
    }
    
    std::memcpy(file_path, path.c_str(), path.size());
    file_path[path.size()] = '\0';
    return 0;
}

inline int aclshmemi_c_api_adapter_t::get_root_info(int phy_id, char* rank_info, size_t* buf_size) {
    if (rank_info == nullptr || buf_size == nullptr || *buf_size == 0) {
        return -1;
    }
    
    auto result = aclshmemi_topo_addr_info_manager_t::instance().get_root_info_json(phy_id);
    if (!result) {
        return -1;
    }
    
    const std::string& json = *result;
    if (json.size() + 1 > *buf_size) {
        *buf_size = json.size() + 1;
        return -1;
    }
    
    std::memcpy(rank_info, json.c_str(), json.size());
    rank_info[json.size()] = '\0';
    *buf_size = json.size();
    return 0;
}

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_C_API_ADAPTER_HPP