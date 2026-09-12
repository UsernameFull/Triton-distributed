/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_topo_addr_info_manager.h"

namespace shm {
namespace topo {

aclshmemi_topo_addr_info_manager_t& aclshmemi_topo_addr_info_manager_t::instance()
{
    static aclshmemi_topo_addr_info_manager_t inst;
    return inst;
}

std::optional<size_t> aclshmemi_topo_addr_info_manager_t::get_size(int phy_id)
{
    auto mainboard_id = hal_.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        return std::nullopt;
    }

    auto strategy = aclshmemi_product_strategy_t::create(*mainboard_id);
    if (!strategy) {
        return std::nullopt;
    }

    return strategy->get_root_info_size();
}

std::optional<std::string> aclshmemi_topo_addr_info_manager_t::get_topo_file_path(int phy_id)
{
    auto mainboard_id = hal_.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        return std::nullopt;
    }

    std::string driver_path = hal_.get_driver_install_path();

    switch (*mainboard_id) {
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH:
            return driver_path + "/driver/topo/950/atlas_350_1.json";
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH:
            return driver_path + "/driver/topo/950/atlas_350_2.json";
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH:
            return driver_path + "/driver/topo/950/atlas_350_3.json";
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1:
            return driver_path + "/driver/topo/950/atlas_850_1.json";
        case ACLSHMEMI_MAIN_BOARD_ID_POD:
        case ACLSHMEMI_MAIN_BOARD_ID_POD_2D:
            return driver_path + "/driver/topo/950/atlas_950_1.json";
        default:
            return std::nullopt;
    }
}

std::optional<std::string> aclshmemi_topo_addr_info_manager_t::get_root_info_json(int phy_id)
{
    auto mainboard_id = hal_.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        return std::nullopt;
    }

    auto strategy = aclshmemi_product_strategy_t::create(*mainboard_id);
    if (!strategy) {
        return std::nullopt;
    }

    auto root_info = strategy->get_root_info(phy_id, *mainboard_id);
    if (!root_info) {
        return std::nullopt;
    }

    return aclshmemi_root_info_to_string(*root_info);
}

} // namespace topo
} // namespace shm
