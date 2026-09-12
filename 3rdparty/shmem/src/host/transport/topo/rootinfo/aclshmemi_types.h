/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_TYPES_HPP
#define ACLSHMEMI_TYPES_HPP

#include <string>
#include <vector>
#include <cstdint>
#include "dl_dcmi_types.h"

namespace shm {
namespace topo {

/**
 * @brief 地址信息结构体
 */
struct aclshmemi_addr_t {
    std::string addr_type;
    std::string addr;
    std::vector<std::string> ports;
    std::string plane_id;
};

/**
 * @brief 网络层信息结构体
 */
struct aclshmemi_net_layer_t {
    int level;
    std::string instance_id;
    std::string net_type;
    std::string net_attr;
    std::vector<aclshmemi_addr_t> addr_list;
};

/**
 * @brief Rank 信息结构体
 */
struct aclshmemi_rank_t {
    int device_id;
    int local_id;
    std::vector<aclshmemi_net_layer_t> level_list;
};

/**
 * @brief RootInfo 信息结构体
 */
struct aclshmemi_root_info_t {
    std::string version;
    std::string status;
    std::string topo_file_path;
    std::vector<aclshmemi_rank_t> ranks;
};

int aclshmemi_addr_set_eid(aclshmemi_addr_t& addr, const dcmi_urma_eid_t& eid);
int aclshmemi_addr_set_ip(aclshmemi_addr_t& addr, const std::string& ip);
int aclshmemi_addr_set_plane_id(aclshmemi_addr_t& addr, const std::string& plane_id);
int aclshmemi_addr_add_port(aclshmemi_addr_t& addr, const std::string& port);
std::string aclshmemi_addr_to_string(const aclshmemi_addr_t& addr);

int aclshmemi_net_layer_init(aclshmemi_net_layer_t& layer, int level, const std::string& instance_id);
int aclshmemi_net_layer_set_net_type(aclshmemi_net_layer_t& layer, const std::string& net_type);
int aclshmemi_net_layer_set_net_attr(aclshmemi_net_layer_t& layer, const std::string& net_attr);
int aclshmemi_net_layer_add_addr(aclshmemi_net_layer_t& layer, const aclshmemi_addr_t& addr);
int aclshmemi_net_layer_set_addr_at(aclshmemi_net_layer_t& layer, const aclshmemi_addr_t& addr, int index);
std::string aclshmemi_net_layer_to_string(const aclshmemi_net_layer_t& layer);

int aclshmemi_rank_init(aclshmemi_rank_t& rank, int device_id, int local_id);
int aclshmemi_rank_add_net_layer(aclshmemi_rank_t& rank, const aclshmemi_net_layer_t& layer);
std::string aclshmemi_rank_to_string(const aclshmemi_rank_t& rank);

int aclshmemi_root_info_init(aclshmemi_root_info_t& root_info);
int aclshmemi_root_info_set_topo_file_path(aclshmemi_root_info_t& root_info, const std::string& path);
int aclshmemi_root_info_add_rank(aclshmemi_root_info_t& root_info, const aclshmemi_rank_t& rank);
std::string aclshmemi_root_info_to_string(const aclshmemi_root_info_t& root_info);
std::string aclshmemi_root_info_get_topo_file_path(const aclshmemi_root_info_t& root_info);

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_TYPES_HPP