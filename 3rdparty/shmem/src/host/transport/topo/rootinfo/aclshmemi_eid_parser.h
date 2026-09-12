/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_EID_PARSER_HPP
#define ACLSHMEMI_EID_PARSER_HPP

#include <string>
#include <cstdint>
#include <optional>
#include "dl_dcmi_types.h"

namespace shm {
namespace topo {

class aclshmemi_eid_parser_t {
public:
    static std::optional<int> get_fe_id(const std::string& eid_hex_str);
    static std::optional<int> get_port_id(const std::string& eid_hex_str);
    static std::optional<int> get_die_id(const std::string& eid_hex_str);
    
    static int get_fe_id(const dcmi_urma_eid_t& eid);
    static int get_port_id(const dcmi_urma_eid_t& eid);
    static int get_die_id(const dcmi_urma_eid_t& eid);
    static int get_low_bit_port(const dcmi_urma_eid_t& eid);
    static int get_pod_die_id(const dcmi_urma_eid_t& eid);
    static int get_server_die_id(const dcmi_urma_eid_t& eid);
    
    static std::optional<int> get_max_fe_id(const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt);
    
    static int get_ub_entity_id(const UBEntity& ue);
    static int get_server_port_group_idx(const UBEntity& ue);
    static int get_max_entity_id(const UEList& ue_list);
};

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_EID_PARSER_HPP