/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the software repository for the full text of the License.
 */
#ifndef MOCK_HAL_H
#define MOCK_HAL_H

#include <gmock/gmock.h>
#include <optional>
#include <string>
#include "dl_dcmi_types.h"

namespace shm {
namespace topo {

class MockDcmiApi {
public:
    static MockDcmiApi& instance() {
        static MockDcmiApi inst;
        return inst;
    }
    
    MOCK_METHOD(bool, initialize, ());
    MOCK_METHOD(bool, is_initialized, (), (const));
    MOCK_METHOD(std::optional<uint32_t>, get_mainboard_id, (int phy_id));
    MOCK_METHOD(std::optional<uint32_t>, get_logic_id_from_phy_id, (uint32_t phy_id));
    MOCK_METHOD(std::optional<int>, get_npu_count, ());
    MOCK_METHOD(std::optional<uint32_t>, get_urma_device_cnt, (int phy_id));
    MOCK_METHOD(int, get_eid_list, (int phy_id, dcmi_urma_eid_info_t* eid_list, size_t* eid_cnt));
    MOCK_METHOD(int, get_ue_list, (int phy_id, UEList* ue_list));
    MOCK_METHOD(int, get_eid_list_by_dev_index, (int phy_id, int dev_index, dcmi_urma_eid_info_t* eid_list, int* eid_cnt));
    MOCK_METHOD(int, get_device_pcie_info, (int phy_id, struct dcmi_pcie_info_all* pcie_info));
    MOCK_METHOD(int, get_spod_info, (int phy_id, struct dcmi_spod_info* spod_info));
};

class MockHal {
public:
    static MockHal& instance() {
        static MockHal inst;
        return inst;
    }
    
    MOCK_METHOD(std::optional<uint32_t>, get_mainboard_id, (int phy_id));
    MOCK_METHOD(std::optional<int>, get_npu_count, ());
    MOCK_METHOD(std::string, get_server_id, ());
    MOCK_METHOD(std::string, get_driver_install_path, ());
    MOCK_METHOD(int, get_ue_list, (int phy_id, UEList* ue_list));
    MOCK_METHOD(int, get_device_pcie_info, (int phy_id, struct dcmi_pcie_info_all* pcie_info));
    MOCK_METHOD(int, get_spod_info, (int phy_id, struct dcmi_spod_info* spod_info));
};

} // namespace topo
} // namespace shm

#endif // MOCK_HAL_H