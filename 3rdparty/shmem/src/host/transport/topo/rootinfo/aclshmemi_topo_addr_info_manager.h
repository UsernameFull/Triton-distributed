/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_TOPO_ADDR_INFO_MANAGER_HPP
#define ACLSHMEMI_TOPO_ADDR_INFO_MANAGER_HPP

#include <string>
#include <optional>
#include <memory>
#include "aclshmemi_types.h"
#include "aclshmemi_hal.h"
#include "aclshmemi_product_strategy.h"

namespace shm {
namespace topo {

class aclshmemi_topo_addr_info_manager_t {
public:
    static aclshmemi_topo_addr_info_manager_t& instance();
    
    std::optional<size_t> get_size(int phy_id);
    std::optional<std::string> get_topo_file_path(int phy_id);
    std::optional<std::string> get_root_info_json(int phy_id);
    
private:
    aclshmemi_topo_addr_info_manager_t() = default;
    aclshmemi_topo_addr_info_manager_t(const aclshmemi_topo_addr_info_manager_t&) = delete;
    aclshmemi_topo_addr_info_manager_t& operator=(const aclshmemi_topo_addr_info_manager_t&) = delete;
    
    aclshmemi_hal_t& hal_ = aclshmemi_hal_t::instance();
};

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_TOPO_ADDR_INFO_MANAGER_HPP