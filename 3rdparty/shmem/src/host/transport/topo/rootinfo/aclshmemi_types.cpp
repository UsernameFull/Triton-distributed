/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_types.h"

#include <iomanip>
#include <sstream>

#include <securec.h>

namespace shm {
namespace topo {

int aclshmemi_addr_set_eid(aclshmemi_addr_t& addr, const dcmi_urma_eid_t& eid)
{
    addr.addr_type = "EID";
    addr.addr.clear();
    addr.addr.reserve(DCMI_URMA_EID_SIZE * 2);
    for (int i = 0; i < DCMI_URMA_EID_SIZE; ++i) {
        char buf[3] = {0};
        snprintf_s(buf, sizeof(buf), sizeof(buf) - 1, "%02x", eid.raw[i]);
        addr.addr += buf;
    }
    return 0;
}

int aclshmemi_addr_set_ip(aclshmemi_addr_t& addr, const std::string& ip)
{
    addr.addr_type = "IPV4";
    addr.addr = ip;
    return 0;
}

int aclshmemi_addr_set_plane_id(aclshmemi_addr_t& addr, const std::string& plane_id)
{
    addr.plane_id = plane_id;
    return 0;
}

int aclshmemi_addr_add_port(aclshmemi_addr_t& addr, const std::string& port)
{
    addr.ports.push_back(port);
    return 0;
}

std::string aclshmemi_addr_to_string(const aclshmemi_addr_t& addr)
{
    std::ostringstream oss;
    oss << "{\"addr_type\": \"" << addr.addr_type << "\","
        << "\"addr\": \"" << addr.addr << "\","
        << "\"plane_id\": \"" << addr.plane_id << "\",";

    oss << "\"ports\": [";
    for (size_t i = 0; i < addr.ports.size(); ++i) {
        oss << "\"" << addr.ports[i] << "\"";
        if (i < addr.ports.size() - 1) {
            oss << ",";
        }
    }
    oss << "]}";
    return oss.str();
}

int aclshmemi_net_layer_init(aclshmemi_net_layer_t& layer, int level, const std::string& instance_id)
{
    layer.level = level;
    layer.instance_id = instance_id;
    layer.net_type.clear();
    layer.net_attr.clear();
    layer.addr_list.clear();
    return 0;
}

int aclshmemi_net_layer_set_net_type(aclshmemi_net_layer_t& layer, const std::string& net_type)
{
    layer.net_type = net_type;
    return 0;
}

int aclshmemi_net_layer_set_net_attr(aclshmemi_net_layer_t& layer, const std::string& net_attr)
{
    layer.net_attr = net_attr;
    return 0;
}

int aclshmemi_net_layer_add_addr(aclshmemi_net_layer_t& layer, const aclshmemi_addr_t& addr)
{
    layer.addr_list.push_back(addr);
    return 0;
}

int aclshmemi_net_layer_set_addr_at(aclshmemi_net_layer_t& layer, const aclshmemi_addr_t& addr, int index)
{
    if (index >= 0 && static_cast<size_t>(index) < layer.addr_list.size()) {
        layer.addr_list[index] = addr;
    } else if (index == static_cast<int>(layer.addr_list.size())) {
        layer.addr_list.push_back(addr);
    }
    return 0;
}

std::string aclshmemi_net_layer_to_string(const aclshmemi_net_layer_t& layer)
{
    std::ostringstream oss;
    oss << "{\"net_layer\": " << layer.level << ","
        << "\"net_instance_id\": \"" << layer.instance_id << "\","
        << "\"net_type\": \"" << layer.net_type << "\","
        << "\"net_attr\": \"" << layer.net_attr << "\",";

    oss << "\"rank_addr_list\": [";
    for (size_t i = 0; i < layer.addr_list.size(); ++i) {
        oss << aclshmemi_addr_to_string(layer.addr_list[i]);
        if (i < layer.addr_list.size() - 1) {
            oss << ",";
        }
    }
    oss << "]}";

    return oss.str();
}

int aclshmemi_rank_init(aclshmemi_rank_t& rank, int device_id, int local_id)
{
    rank.device_id = device_id;
    rank.local_id = local_id;
    rank.level_list.clear();
    return 0;
}

int aclshmemi_rank_add_net_layer(aclshmemi_rank_t& rank, const aclshmemi_net_layer_t& layer)
{
    rank.level_list.push_back(layer);
    return 0;
}

std::string aclshmemi_rank_to_string(const aclshmemi_rank_t& rank)
{
    std::ostringstream oss;
    oss << "{\"device_id\": " << rank.device_id << ","
        << "\"local_id\": " << rank.local_id << ",";

    oss << "\"level_list\": [";
    for (size_t i = 0; i < rank.level_list.size(); ++i) {
        oss << aclshmemi_net_layer_to_string(rank.level_list[i]);
        if (i < rank.level_list.size() - 1) {
            oss << ", ";
        }
    }
    oss << "]}";

    return oss.str();
}

int aclshmemi_root_info_init(aclshmemi_root_info_t& root_info)
{
    root_info.version = "2.0";
    root_info.topo_file_path.clear();
    root_info.ranks.clear();
    return 0;
}

int aclshmemi_root_info_set_topo_file_path(aclshmemi_root_info_t& root_info, const std::string& path)
{
    root_info.topo_file_path = path;
    return 0;
}

int aclshmemi_root_info_add_rank(aclshmemi_root_info_t& root_info, const aclshmemi_rank_t& rank)
{
    root_info.ranks.push_back(rank);
    return 0;
}

std::string aclshmemi_root_info_to_string(const aclshmemi_root_info_t& root_info)
{
    std::ostringstream oss;
    oss << "{\"version\": \"" << root_info.version << "\","
        << "\"topo_file_path\": \"" << root_info.topo_file_path << "\","
        << "\"rank_count\": " << root_info.ranks.size() << ",";

    oss << "\"rank_list\": [";
    for (size_t i = 0; i < root_info.ranks.size(); ++i) {
        oss << aclshmemi_rank_to_string(root_info.ranks[i]);
        if (i < root_info.ranks.size() - 1) {
            oss << ",";
        }
    }
    oss << "]}";

    return oss.str();
}

std::string aclshmemi_root_info_get_topo_file_path(const aclshmemi_root_info_t& root_info)
{
    return root_info.topo_file_path;
}

} // namespace topo
} // namespace shm