/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mock_hal.h"
#include "aclshmemi_hal.h"

namespace shm {
namespace topo {

aclshmemi_dcmi_api_t& aclshmemi_dcmi_api_t::instance() {
    static aclshmemi_dcmi_api_t inst;
    return inst;
}

bool aclshmemi_dcmi_api_t::initialize() {
    return MockDcmiApi::instance().initialize();
}

bool aclshmemi_dcmi_api_t::is_initialized() const {
    return MockDcmiApi::instance().is_initialized();
}

std::optional<uint32_t> aclshmemi_dcmi_api_t::get_mainboard_id(int phy_id) {
    return MockDcmiApi::instance().get_mainboard_id(phy_id);
}

std::optional<uint32_t> aclshmemi_dcmi_api_t::get_logic_id_from_phy_id(uint32_t phy_id) {
    return MockDcmiApi::instance().get_logic_id_from_phy_id(phy_id);
}

std::optional<int> aclshmemi_dcmi_api_t::get_npu_count() {
    return MockDcmiApi::instance().get_npu_count();
}

std::optional<uint32_t> aclshmemi_dcmi_api_t::get_urma_device_cnt(int phy_id) {
    return MockDcmiApi::instance().get_urma_device_cnt(phy_id);
}

int aclshmemi_dcmi_api_t::get_eid_list(int phy_id, dcmi_urma_eid_info_t* eid_list, size_t* eid_cnt) {
    return MockDcmiApi::instance().get_eid_list(phy_id, eid_list, eid_cnt);
}

int aclshmemi_dcmi_api_t::get_ue_list(int phy_id, UEList* ue_list) {
    return MockDcmiApi::instance().get_ue_list(phy_id, ue_list);
}

int aclshmemi_dcmi_api_t::get_eid_list_by_dev_index(int phy_id, int dev_index, 
                                                     dcmi_urma_eid_info_t* eid_list, int* eid_cnt) {
    return MockDcmiApi::instance().get_eid_list_by_dev_index(phy_id, dev_index, eid_list, eid_cnt);
}

int aclshmemi_dcmi_api_t::get_device_pcie_info(int phy_id, struct dcmi_pcie_info_all* pcie_info) {
    return MockDcmiApi::instance().get_device_pcie_info(phy_id, pcie_info);
}

int aclshmemi_dcmi_api_t::get_spod_info(int phy_id, struct dcmi_spod_info* spod_info) {
    return MockDcmiApi::instance().get_spod_info(phy_id, spod_info);
}


aclshmemi_hal_t& aclshmemi_hal_t::instance() {
    static aclshmemi_hal_t inst;
    return inst;
}

std::optional<uint32_t> aclshmemi_hal_t::get_mainboard_id(int phy_id) {
    return MockHal::instance().get_mainboard_id(phy_id);
}

std::optional<int> aclshmemi_hal_t::get_npu_count() {
    return MockHal::instance().get_npu_count();
}

std::string aclshmemi_hal_t::get_server_id() {
    return MockHal::instance().get_server_id();
}

std::string aclshmemi_hal_t::get_driver_install_path() {
    return MockHal::instance().get_driver_install_path();
}

int aclshmemi_hal_t::get_ue_list(int phy_id, UEList* ue_list) {
    return MockHal::instance().get_ue_list(phy_id, ue_list);
}

int aclshmemi_hal_t::get_device_pcie_info(int phy_id, struct dcmi_pcie_info_all* pcie_info) {
    return MockHal::instance().get_device_pcie_info(phy_id, pcie_info);
}

int aclshmemi_hal_t::get_spod_info(int phy_id, struct dcmi_spod_info* spod_info) {
    return MockHal::instance().get_spod_info(phy_id, spod_info);
}

} // namespace topo
} // namespace shm