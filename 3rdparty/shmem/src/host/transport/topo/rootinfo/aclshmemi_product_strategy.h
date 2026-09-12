/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_PRODUCT_STRATEGY_HPP
#define ACLSHMEMI_PRODUCT_STRATEGY_HPP

#include <memory>
#include <string>
#include <optional>
#include <vector>
#include "aclshmemi_types.h"
#include "aclshmemi_hal.h"
#include "aclshmemi_eid_parser.h"

namespace shm {
namespace topo {

#define ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH   0x68
#define ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH   0x6a
#define ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH   0x6c
#define ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH 0x25
#define ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1  0x23
#define ACLSHMEMI_MAIN_BOARD_ID_POD           0x07
#define ACLSHMEMI_MAIN_BOARD_ID_POD_2D        0x03

struct aclshmemi_ub_entity_rule_t {
    uint32_t mainboard_id;
    int level;
    int die_id;
    int ue_id;
    std::vector<int> ports;
};

struct aclshmemi_pod_ub_entity_rule_t {
    uint32_t mainboard_id;
    int level;
    int die_id;
    int ue_id;
    std::vector<int> ports;
    std::vector<int> npus;
    std::string plane_id;
};

class aclshmemi_product_strategy_t {
public:
    virtual ~aclshmemi_product_strategy_t() = default;
    
    virtual std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) = 0;
    virtual size_t get_root_info_size() const = 0;
    
    static std::unique_ptr<aclshmemi_product_strategy_t> create(uint32_t mainboard_id);
};

class aclshmemi_card_product_t : public aclshmemi_product_strategy_t {
public:
    std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) override;
    size_t get_root_info_size() const override { return 2048; }
    
private:
    std::optional<aclshmemi_net_layer_t> process_mesh_layer(int phy_id, 
                                                            const dcmi_urma_eid_info_t* eid_list, 
                                                            size_t eid_cnt);
    std::optional<aclshmemi_net_layer_t> process_mesh2p_layer(int phy_id,
                                                              const dcmi_urma_eid_info_t* eid_list,
                                                              size_t eid_cnt);
    std::optional<aclshmemi_net_layer_t> process_roce_layer(int phy_id);
};

class aclshmemi_server_product_t : public aclshmemi_product_strategy_t {
public:
    std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) override;
    size_t get_root_info_size() const override { return 2048; }
    
private:
    std::optional<aclshmemi_net_layer_t> process_mesh_layer(int phy_id, const UEList& ue_list, 
                                                            const dcmi_spod_info& spod_info);
    std::optional<aclshmemi_net_layer_t> process_clos_layer(int phy_id, uint32_t mainboard_id,
                                                            const UEList& ue_list, const dcmi_spod_info& spod_info);
};

class aclshmemi_pod_product_t : public aclshmemi_product_strategy_t {
public:
    std::optional<aclshmemi_root_info_t> get_root_info(int phy_id, uint32_t mainboard_id) override;
    size_t get_root_info_size() const override { return 2048; }
    
private:
    std::optional<aclshmemi_net_layer_t> process_mesh_layer(int phy_id, const UEList& ue_list, 
                                                            const dcmi_spod_info& spod_info);
    std::optional<aclshmemi_net_layer_t> process_clos_layer(int phy_id, uint32_t mainboard_id,
                                                            const UEList& ue_list, const dcmi_spod_info& spod_info);
};

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_PRODUCT_STRATEGY_HPP