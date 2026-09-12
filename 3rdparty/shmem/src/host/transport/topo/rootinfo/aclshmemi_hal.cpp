/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_hal.h"

#include <cctype>
#include <cstring>
#include <fstream>
#include <sstream>
#include <securec.h>

#include <dlfcn.h>
#include <unistd.h>

#include "utils/shmemi_logger.h"

#include <securec.h>

namespace shm {
namespace topo {

static constexpr const char* DRIVER_DEFAULT_INSTALL_PATH = "/usr/local/Ascend";
static constexpr const char* DRIVER_INSTALL_INFO_FILE = "/etc/ascend_install.info";
static constexpr const char* DRIVER_INSTALL_PATH_KEY = "Driver_Install_Path_Param";
static constexpr int MAX_NPU_COUNT_FOR_CHECK = 64;
static constexpr int MAX_DAVINCI_DEV_LEN = 64;

static std::string trim_whitespace(const std::string& str)
{
    size_t start = 0;
    while (start < str.size() && std::isspace(static_cast<unsigned char>(str[start]))) {
        start++;
    }
    size_t end = str.size();
    while (end > start && std::isspace(static_cast<unsigned char>(str[end - 1]))) {
        end--;
    }
    return str.substr(start, end - start);
}

aclshmemi_dcmi_api_t& aclshmemi_dcmi_api_t::instance()
{
    static aclshmemi_dcmi_api_t inst;
    return inst;
}

bool aclshmemi_dcmi_api_t::initialize()
{
    if (is_initialized_) {
        return true;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    if (is_initialized_) {
        return true;
    }

    if (shm::DlDcmiApi::LoadLibrary() != 0) {
        SHM_LOG_ERROR("aclshmemi_dcmi_api_t initialize failed: DlDcmiApi::LoadLibrary error");
        return false;
    }

    is_initialized_ = true;
    return true;
}

bool aclshmemi_dcmi_api_t::is_initialized() const { return is_initialized_; }

std::optional<uint32_t> aclshmemi_dcmi_api_t::get_logic_id_from_phy_id(uint32_t phy_id)
{
    if (!is_initialized_) {
        if (!initialize()) {
            return std::nullopt;
        }
    }

    uint32_t logic_id = 0;
    int ret = shm::DlDcmiApi::GetLogicIdFromPhyId(phy_id, &logic_id);
    if (ret != 0) {
        SHM_LOG_ERROR("get_logic_id_from_phy_id failed: ret=" << ret << " for phy_id=" << phy_id);
        return std::nullopt;
    }
    return logic_id;
}

std::optional<uint32_t> aclshmemi_dcmi_api_t::get_mainboard_id(int phy_id)
{
    if (!is_initialized_) {
        if (!initialize()) {
            return std::nullopt;
        }
    }

    auto logic_id = get_logic_id_from_phy_id(static_cast<uint32_t>(phy_id));
    if (!logic_id) {
        return std::nullopt;
    }

    uint32_t mainboard_id = 0;
    int ret = shm::DlDcmiApi::Dcmiv2GetMainboardId(static_cast<int>(*logic_id), &mainboard_id);
    if (ret != 0) {
        SHM_LOG_ERROR("get_mainboard_id failed: ret=" << ret << " for phy_id=" << phy_id);
        return std::nullopt;
    }
    return mainboard_id;
}

std::optional<int> aclshmemi_dcmi_api_t::get_npu_count()
{
    int count = 0;
    for (int i = 0; i < MAX_NPU_COUNT_FOR_CHECK; ++i) {
        char davinci_dev[MAX_DAVINCI_DEV_LEN] = {0};
        int ret = snprintf_s(davinci_dev, MAX_DAVINCI_DEV_LEN, MAX_DAVINCI_DEV_LEN - 1, "/dev/davinci%d", i);
        if (ret < 0) {
            SHM_LOG_ERROR("get_npu_count failed: snprintf_s error");
            continue;
        }
        if (access(davinci_dev, F_OK) == 0) {
            count++;
        }
    }

    return count > 0 ? std::optional<int>(count) : std::nullopt;
}

std::optional<uint32_t> aclshmemi_dcmi_api_t::get_urma_device_cnt(int phy_id)
{
    if (!is_initialized_) {
        if (!initialize()) {
            return std::nullopt;
        }
    }

    auto logic_id = get_logic_id_from_phy_id(static_cast<uint32_t>(phy_id));
    if (!logic_id) {
        return std::nullopt;
    }

    uint32_t dev_cnt = 0;
    int ret = shm::DlDcmiApi::Dcmiv2GetUrmaDeviceCnt(static_cast<int>(*logic_id), &dev_cnt);
    if (ret != 0) {
        SHM_LOG_ERROR("get_urma_device_cnt failed: ret=" << ret << " for phy_id=" << phy_id);
        return std::nullopt;
    }
    return dev_cnt;
}

int aclshmemi_dcmi_api_t::get_eid_list(int phy_id, dcmi_urma_eid_info_t* eid_list, size_t* eid_cnt)
{
    if (!is_initialized_) {
        if (!initialize()) {
            return -1;
        }
    }

    auto logic_id = get_logic_id_from_phy_id(static_cast<uint32_t>(phy_id));
    if (!logic_id) {
        return -1;
    }

    uint32_t dev_cnt = 0;
    int ret = shm::DlDcmiApi::Dcmiv2GetUrmaDeviceCnt(static_cast<int>(*logic_id), &dev_cnt);
    if (ret != 0) {
        SHM_LOG_ERROR("get_eid_list failed: get_urma_device_cnt ret=" << ret);
        return -1;
    }

    size_t total_cnt = 0;
    for (uint32_t i = 0; i < dev_cnt && total_cnt < *eid_cnt; ++i) {
        int eid_num = static_cast<int>(*eid_cnt - total_cnt);
        ret = shm::DlDcmiApi::Dcmiv2GetEidListByUrmaDevIndex(static_cast<int>(*logic_id), i, eid_list + total_cnt, &eid_num);
        if (ret != 0) {
            SHM_LOG_ERROR("get_eid_list failed: Dcmiv2GetEidListByUrmaDevIndex ret=" << ret << " for dev_index=" << i);
            continue;
        }
        total_cnt += eid_num;
    }
    *eid_cnt = total_cnt;
    return 0;
}

int aclshmemi_dcmi_api_t::get_eid_list_by_dev_index(
    int phy_id, int dev_index, dcmi_urma_eid_info_t* eid_list, int* eid_cnt)
{
    if (!is_initialized_) {
        if (!initialize()) {
            return -1;
        }
    }

    auto logic_id = get_logic_id_from_phy_id(static_cast<uint32_t>(phy_id));
    if (!logic_id) {
        return -1;
    }

    int ret = shm::DlDcmiApi::Dcmiv2GetEidListByUrmaDevIndex(static_cast<int>(*logic_id), dev_index, eid_list, eid_cnt);
    if (ret != 0) {
        SHM_LOG_ERROR("get_eid_list_by_dev_index failed: ret=" << ret << " for phy_id=" << phy_id << " dev_index=" << dev_index);
        return -1;
    }
    return 0;
}

int aclshmemi_dcmi_api_t::get_ue_list(int phy_id, UEList* ue_list)
{
    if (!is_initialized_) {
        if (!initialize()) {
            return -1;
        }
    }

    auto logic_id = get_logic_id_from_phy_id(static_cast<uint32_t>(phy_id));
    if (!logic_id) {
        return -1;
    }

    size_t eid_cnt = ACLSHMEMI_MAX_EID_NUM;
    int ret = get_eid_list(phy_id, ue_list->ueList[0].eidList, &eid_cnt);
    if (ret != 0) {
        SHM_LOG_ERROR("get_ue_list failed: get_eid_list ret=" << ret << " for phy_id=" << phy_id);
        return -1;
    }

    ue_list->ueNum = 1;
    ue_list->ueList[0].eidNum = static_cast<uint32_t>(eid_cnt);
    return 0;
}

int aclshmemi_dcmi_api_t::get_device_pcie_info(int phy_id, struct dcmi_pcie_info_all* pcie_info)
{
    if (!is_initialized_) {
        if (!initialize()) {
            return -1;
        }
    }

    auto logic_id = get_logic_id_from_phy_id(static_cast<uint32_t>(phy_id));
    if (!logic_id) {
        return -1;
    }

    int ret = shm::DlDcmiApi::Dcmiv2GetDevicePcieInfo(static_cast<int>(*logic_id), pcie_info);
    if (ret != 0) {
        SHM_LOG_ERROR("get_device_pcie_info failed: ret=" << ret << " for phy_id=" << phy_id);
        return -1;
    }
    return 0;
}

int aclshmemi_dcmi_api_t::get_spod_info(int phy_id, struct dcmi_spod_info* spod_info)
{
    if (!is_initialized_) {
        if (!initialize()) {
            return -1;
        }
    }

    auto logic_id = get_logic_id_from_phy_id(static_cast<uint32_t>(phy_id));
    if (!logic_id) {
        return -1;
    }

    uint32_t buf_size = sizeof(struct dcmi_spod_info);
    return shm::DlDcmiApi::Dcmiv2GetDeviceInfo(
        static_cast<int>(*logic_id), DCMI_MAIN_CMD_CHIP_INF, DCMI_CHIP_INFO_SUB_CMD_SPOD_INFO, spod_info, &buf_size);
}

aclshmemi_hal_t& aclshmemi_hal_t::instance()
{
    static aclshmemi_hal_t inst;
    return inst;
}

std::optional<uint32_t> aclshmemi_hal_t::get_mainboard_id(int phy_id) { return dcmi_.get_mainboard_id(phy_id); }
std::optional<int> aclshmemi_hal_t::get_npu_count() { return dcmi_.get_npu_count(); }

std::string aclshmemi_hal_t::get_server_id()
{
    std::string driver_path = get_driver_install_path();
    std::ifstream file(driver_path + "/version.info");
    if (!file.is_open()) {
        file.open(driver_path + "/driver/version.info");
        if (!file.is_open()) {
            return "unknown";
        }
    }

    std::string line;
    std::string server_id;
    while (std::getline(file, line)) {
        size_t pos = line.find("Version=");
        if (pos != std::string::npos) {
            server_id = trim_whitespace(line.substr(pos + 8));
            break;
        }
    }
    return server_id.empty() ? "unknown" : server_id;
}

std::string aclshmemi_hal_t::get_driver_install_path()
{
    std::ifstream file(DRIVER_INSTALL_INFO_FILE);
    if (!file.is_open()) {
        return DRIVER_DEFAULT_INSTALL_PATH;
    }

    std::string line;
    std::string path;
    while (std::getline(file, line)) {
        if (line.find(DRIVER_INSTALL_PATH_KEY) != std::string::npos) {
            size_t pos = line.find('=');
            if (pos != std::string::npos) {
                path = trim_whitespace(line.substr(pos + 1));
                break;
            }
        }
    }
    return path.empty() ? DRIVER_DEFAULT_INSTALL_PATH : path;
}

int aclshmemi_hal_t::get_ue_list(int phy_id, UEList* ue_list) { return dcmi_.get_ue_list(phy_id, ue_list); }
int aclshmemi_hal_t::get_device_pcie_info(int phy_id, struct dcmi_pcie_info_all* pcie_info)
{
    return dcmi_.get_device_pcie_info(phy_id, pcie_info);
}
int aclshmemi_hal_t::get_spod_info(int phy_id, struct dcmi_spod_info* spod_info) { return dcmi_.get_spod_info(phy_id, spod_info); }

} // namespace topo
} // namespace shm