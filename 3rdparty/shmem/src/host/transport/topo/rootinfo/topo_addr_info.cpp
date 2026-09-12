/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_addr_info.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <memory>
#include <securec.h>

#include "aclshmemi_hal.h"
#include "aclshmemi_product_strategy.h"
#include "utils/shmemi_logger.h"

namespace {
constexpr int MAX_NPU_COUNT_LIMIT = 64; // 与 ACLSHMEMI_MAX_NPU_COUNT 一致
constexpr int DEFAULT_RANKINFO_SIZE = 4096;
}

int topo_addr_info_get_size(int phy_id, size_t* size)
{
    if (size == NULL) {
        SHM_LOG_ERROR("topo_addr_info_get_size failed: size param is NULL");
        return -1;
    }

    auto& hal = shm::topo::aclshmemi_hal_t::instance();
    auto mainboard_id = hal.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        SHM_LOG_ERROR("topo_addr_info_get_size failed: get_mainboard_id returned null for phy_id=" << phy_id);
        return -1;
    }

    auto strategy = shm::topo::aclshmemi_product_strategy_t::create(*mainboard_id);
    if (strategy) {
        *size = strategy->get_root_info_size();
        SHM_LOG_INFO(
            "topo_addr_info_get_size: strategy created for mainboard_id=0x" << std::hex << *mainboard_id
                                                                            << ", size=" << std::dec << *size);
        return 0;
    }

    SHM_LOG_WARN(
        "topo_addr_info_get_size: strategy creation failed for mainboard_id=0x"
        << std::hex << *mainboard_id << ", using default size=" << std::dec << DEFAULT_RANKINFO_SIZE);
    *size = DEFAULT_RANKINFO_SIZE;
    return 0;
}

int topo_addr_info_get_topo_file_path(int phy_id, char* file_path, size_t buf_size)
{
    if (file_path == NULL) {
        SHM_LOG_ERROR("topo_addr_info_get_topo_file_path failed: file_path param is NULL");
        return -1;
    }

    auto& hal = shm::topo::aclshmemi_hal_t::instance();
    auto mainboard_id = hal.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        SHM_LOG_ERROR("topo_addr_info_get_topo_file_path failed: get_mainboard_id returned null for phy_id=" << phy_id);
        return -1;
    }

    auto strategy = shm::topo::aclshmemi_product_strategy_t::create(*mainboard_id);
    if (!strategy) {
        SHM_LOG_ERROR(
            "topo_addr_info_get_topo_file_path failed: strategy creation failed for mainboard_id=0x" << std::hex
                                                                                                     << *mainboard_id);
        return -1;
    }

    auto root_info = strategy->get_root_info(phy_id, *mainboard_id);
    if (!root_info) {
        SHM_LOG_ERROR(
            "topo_addr_info_get_topo_file_path failed: get_root_info returned null for phy_id="
            << phy_id << ", mainboard_id=0x" << std::hex << *mainboard_id);
        return -1;
    }

    auto topo_path = shm::topo::aclshmemi_root_info_get_topo_file_path(*root_info);
    if (topo_path.size() >= buf_size) {
        SHM_LOG_ERROR(
            "topo_addr_info_get_topo_file_path failed: path size " << topo_path.size() << " >= buffer size "
                                                                    << buf_size);
        return -1;
    }

    int snprintf_ret = snprintf_s(file_path, buf_size, buf_size - 1, "%s", topo_path.c_str());
    if (snprintf_ret < 0) {
        SHM_LOG_ERROR("topo_addr_info_get_topo_file_path failed: snprintf_s error");
        return -1;
    }
    SHM_LOG_INFO(
        "topo_addr_info_get_topo_file_path succeeded: phy_id=" << phy_id << ", mainboard_id=0x" << std::hex
                                                                << *mainboard_id << ", path=" << file_path);
    return 0;
}

int topo_addr_info_get(int phy_id, char* rank_info, size_t* buf_size)
{
    if (rank_info == NULL || buf_size == NULL) {
        SHM_LOG_ERROR("topo_addr_info_get failed: rank_info=" << rank_info << ", buf_size=" << buf_size);
        return -1;
    }

    auto& hal = shm::topo::aclshmemi_hal_t::instance();
    auto mainboard_id = hal.get_mainboard_id(phy_id);
    if (!mainboard_id) {
        SHM_LOG_ERROR("topo_addr_info_get failed: get_mainboard_id returned null for phy_id=" << phy_id);
        return -1;
    }

    auto strategy = shm::topo::aclshmemi_product_strategy_t::create(*mainboard_id);
    if (!strategy) {
        SHM_LOG_ERROR(
            "topo_addr_info_get failed: strategy creation failed for phy_id=" << phy_id << ", mainboard_id=0x"
                                                                              << std::hex << *mainboard_id);
        return -1;
    }

    auto root_info = strategy->get_root_info(phy_id, *mainboard_id);
    if (!root_info) {
        SHM_LOG_ERROR(
            "topo_addr_info_get failed: get_root_info returned null for phy_id=" << phy_id << ", mainboard_id=0x"
                                                                                 << std::hex << *mainboard_id);
        return -1;
    }

    auto json_str = shm::topo::aclshmemi_root_info_to_string(*root_info);
    if (json_str.size() >= *buf_size) {
        SHM_LOG_ERROR(
            "topo_addr_info_get failed: json size " << json_str.size() << " >= buffer size " << *buf_size
                                                    << " for phy_id=" << phy_id);
        return -1;
    }

    int snprintf_ret = snprintf_s(rank_info, *buf_size, *buf_size - 1, "%s", json_str.c_str());
    if (snprintf_ret < 0) {
        SHM_LOG_ERROR("topo_addr_info_get failed: snprintf_s error");
        return -1;
    }
    *buf_size = json_str.size();
    SHM_LOG_INFO(
        "topo_addr_info_get succeeded: phy_id=" << phy_id << ", mainboard_id=0x" << std::hex << *mainboard_id
                                                << ", json_size=" << std::dec << *buf_size);
    return 0;
}

int topo_addr_info_get_all(char* rank_info, size_t* buf_size)
{
    if (rank_info == NULL || buf_size == NULL) {
        SHM_LOG_ERROR("topo_addr_info_get_all failed: rank_info=" << rank_info << ", buf_size=" << buf_size);
        return -1;
    }

    auto& hal = shm::topo::aclshmemi_hal_t::instance();
    auto npu_count = hal.get_npu_count();
    if (!npu_count) {
        SHM_LOG_ERROR("topo_addr_info_get_all failed: get_npu_count returned null");
        return -1;
    }

    // 边界校验：防止负值或超大值导致内存过度分配
    if (*npu_count <= 0 || *npu_count > MAX_NPU_COUNT_LIMIT) {
        SHM_LOG_ERROR(
            "topo_addr_info_get_all failed: invalid npu_count=" << *npu_count << ", expected range [1, "
                                                                << MAX_NPU_COUNT_LIMIT << "]");
        return -1;
    }

    shm::topo::aclshmemi_root_info_t merged_root_info;
    shm::topo::aclshmemi_root_info_init(merged_root_info);

    bool first_root_info = true;
    int successful_phy_ids = 0;

    for (int phy_id = 0; phy_id < *npu_count; ++phy_id) {
        auto mainboard_id = hal.get_mainboard_id(phy_id);
        if (!mainboard_id) {
            SHM_LOG_WARN(
                "topo_addr_info_get_all: get_mainboard_id returned null for phy_id=" << phy_id << ", skipping");
            continue;
        }

        auto strategy = shm::topo::aclshmemi_product_strategy_t::create(*mainboard_id);
        if (!strategy) {
            SHM_LOG_WARN(
                "topo_addr_info_get_all: strategy creation failed for phy_id="
                << phy_id << ", mainboard_id=0x" << std::hex << *mainboard_id << ", skipping");
            continue;
        }

        auto root_info = strategy->get_root_info(phy_id, *mainboard_id);
        if (!root_info) {
            SHM_LOG_WARN(
                "topo_addr_info_get_all: get_root_info returned null for phy_id="
                << phy_id << ", mainboard_id=0x" << std::hex << *mainboard_id << ", skipping");
            continue;
        }

        if (first_root_info) {
            merged_root_info.version = root_info->version;
            merged_root_info.topo_file_path = root_info->topo_file_path;
            first_root_info = false;
        }

        for (const auto& rank : root_info->ranks) {
            merged_root_info.ranks.push_back(rank);
        }

        successful_phy_ids++;
        SHM_LOG_DEBUG(
            "topo_addr_info_get_all: added root_info for phy_id=" << phy_id
                                                                  << ", ranks_count=" << root_info->ranks.size());
    }

    if (successful_phy_ids == 0) {
        SHM_LOG_ERROR("topo_addr_info_get_all failed: no successful phy_ids");
        return -1;
    }

    auto json_str = shm::topo::aclshmemi_root_info_to_string(merged_root_info);
    if (json_str.size() >= *buf_size) {
        SHM_LOG_ERROR(
            "topo_addr_info_get_all failed: merged json size " << json_str.size() << " >= buffer size " << *buf_size);
        return -1;
    }

    int snprintf_ret = snprintf_s(rank_info, *buf_size, *buf_size - 1, "%s", json_str.c_str());
    if (snprintf_ret < 0) {
        SHM_LOG_ERROR("topo_addr_info_get_all failed: snprintf_s error");
        return -1;
    }
    *buf_size = json_str.size();
    SHM_LOG_INFO(
        "topo_addr_info_get_all succeeded: npu_count=" << *npu_count << ", successful_phy_ids=" << successful_phy_ids
                                                        << ", total_ranks=" << merged_root_info.ranks.size()
                                                        << ", json_size=" << *buf_size);
    return 0;
}
