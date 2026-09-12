/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ACLSHMEMI_HAL_HPP
#define ACLSHMEMI_HAL_HPP

#include <mutex>
#include <memory>
#include <string>
#include <cstdint>
#include <optional>
#include "dl_dcmi_types.h"
#include "dl_dcmi_api.h"

namespace shm {
namespace topo {

#define ACLSHMEMI_MAX_EID_NUM 32
#define ACLSHMEMI_MAX_NPU_COUNT 64

class aclshmemi_dcmi_api_t {
public:
    static aclshmemi_dcmi_api_t& instance();
    
    bool initialize();
    bool is_initialized() const;
    
    std::optional<uint32_t> get_mainboard_id(int phy_id);
    std::optional<uint32_t> get_logic_id_from_phy_id(uint32_t phy_id);
    std::optional<int> get_npu_count();
    std::optional<uint32_t> get_urma_device_cnt(int phy_id);
    
    int get_eid_list(int phy_id, dcmi_urma_eid_info_t* eid_list, size_t* eid_cnt);
    int get_ue_list(int phy_id, UEList* ue_list);
    int get_eid_list_by_dev_index(int phy_id, int dev_index, dcmi_urma_eid_info_t* eid_list, int* eid_cnt);
    int get_device_pcie_info(int phy_id, struct dcmi_pcie_info_all* pcie_info);
    int get_spod_info(int phy_id, struct dcmi_spod_info* spod_info);
    
private:
    aclshmemi_dcmi_api_t() = default;
    ~aclshmemi_dcmi_api_t() = default;
    
    aclshmemi_dcmi_api_t(const aclshmemi_dcmi_api_t&) = delete;
    aclshmemi_dcmi_api_t& operator=(const aclshmemi_dcmi_api_t&) = delete;
    
    std::mutex mutex_;
    bool is_initialized_ = false;
};

class aclshmemi_hal_t {
public:
    static aclshmemi_hal_t& instance();
    
    std::optional<uint32_t> get_mainboard_id(int phy_id);
    std::optional<int> get_npu_count();
    std::string get_server_id();
    std::string get_driver_install_path();
    
    int get_ue_list(int phy_id, UEList* ue_list);
    int get_device_pcie_info(int phy_id, struct dcmi_pcie_info_all* pcie_info);
    int get_spod_info(int phy_id, struct dcmi_spod_info* spod_info);
    
private:
    aclshmemi_hal_t() = default;
    aclshmemi_hal_t(const aclshmemi_hal_t&) = delete;
    aclshmemi_hal_t& operator=(const aclshmemi_hal_t&) = delete;
    
    aclshmemi_dcmi_api_t& dcmi_ = aclshmemi_dcmi_api_t::instance();
};

} // namespace topo
} // namespace shm

#endif // ACLSHMEMI_HAL_HPP