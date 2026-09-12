/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_product_strategy.h"

#include <securec.h>

#include "aclshmemi_eid_parser.h"
#include "aclshmemi_hal.h"
#include "host_rdma.h"
#include "utils/shmemi_logger.h"

namespace shm {
namespace topo {

static constexpr const char* TOPO_FILE_DIR_PATH = "driver/topo/950";

static std::string build_topo_file_path(const std::string& driver_path, const std::string& topo_filename)
{
    return driver_path + "/" + TOPO_FILE_DIR_PATH + "/" + topo_filename;
}

static const aclshmemi_ub_entity_rule_t g_ub_rules[] = {
    {ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1, 1, 0, 3, {4, 5, 6, 7}},
    {ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1, 1, 1, 2, {5, 6}},
    {ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH, 1, 0, 3, {1, 2, 3, 4, 5, 6, 7, 8}},
};

static const aclshmemi_pod_ub_entity_rule_t g_pod_ub_rules[] = {
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 1, 2, {0, 1, 2, 3, 5, 6}, {0, 1, 2, 3}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 0, 2, {1, 2}, {0, 1, 2, 3}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 0, 2, {0, 1, 2, 3, 4, 5}, {4, 5, 6, 7}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD, 1, 1, 2, {1, 2}, {4, 5, 6, 7}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 1, 2, {0, 1, 2, 3, 5, 6}, {0, 1, 2, 3}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 0, 2, {1, 2}, {0, 1, 2, 3}, "plane_pg_1"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 0, 2, {0, 1, 2, 3, 4, 5}, {4, 5, 6, 7}, "plane_pg_0"},
    {ACLSHMEMI_MAIN_BOARD_ID_POD_2D, 1, 1, 2, {1, 2}, {4, 5, 6, 7}, "plane_pg_1"},
};

std::unique_ptr<aclshmemi_product_strategy_t> aclshmemi_product_strategy_t::create(uint32_t mainboard_id)
{
    switch (mainboard_id) {
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH:
            return std::make_unique<aclshmemi_card_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH:
        case ACLSHMEMI_MAIN_BOARD_ID_SERVER_TYPE1:
            return std::make_unique<aclshmemi_server_product_t>();
        case ACLSHMEMI_MAIN_BOARD_ID_POD:
        case ACLSHMEMI_MAIN_BOARD_ID_POD_2D:
            return std::make_unique<aclshmemi_pod_product_t>();
        default:
            return nullptr;
    }
}
std::optional<aclshmemi_root_info_t> aclshmemi_card_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    auto& hal = aclshmemi_hal_t::instance();

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    std::string driver_path = hal.get_driver_install_path();
    aclshmemi_root_info_set_topo_file_path(root_info, build_topo_file_path(driver_path, "atlas_300a.json"));

    UEList ue_list;
    memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list));
    int ret = hal.get_ue_list(phy_id, &ue_list);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_card_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, phy_id);

    dcmi_urma_eid_info_t eid_list[MAX_EID_NUM];
    memset_s(eid_list, sizeof(eid_list), 0, sizeof(eid_list));
    size_t eid_cnt = MAX_EID_NUM;
    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum && eid_cnt > 0; ++j) {
            if (eid_cnt > 0) {
                eid_list[MAX_EID_NUM - eid_cnt] = ue_list.ueList[i].eidList[j];
                eid_cnt--;
            }
        }
    }
    eid_cnt = MAX_EID_NUM - eid_cnt;

    if (mainboard_id == ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH) {
        auto roce_layer = process_roce_layer(phy_id);
        if (roce_layer) {
            aclshmemi_rank_add_net_layer(rank, *roce_layer);
        }
    } else if (mainboard_id == ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH) {
        auto mesh_layer = process_mesh2p_layer(phy_id, eid_list, eid_cnt);
        if (mesh_layer) {
            aclshmemi_rank_add_net_layer(rank, *mesh_layer);
        }
        auto roce_layer = process_roce_layer(phy_id);
        if (roce_layer) {
            aclshmemi_rank_add_net_layer(rank, *roce_layer);
        }
    } else {
        auto mesh_layer = process_mesh_layer(phy_id, eid_list, eid_cnt);
        if (mesh_layer) {
            aclshmemi_rank_add_net_layer(rank, *mesh_layer);
        }
        auto roce_layer = process_roce_layer(phy_id);
        if (roce_layer) {
            aclshmemi_rank_add_net_layer(rank, *roce_layer);
        }
    }

    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_mesh_layer(
    int phy_id, const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    if (eid_cnt == 0) {
        return std::nullopt;
    }

    auto& hal = aclshmemi_hal_t::instance();
    std::string server_id = hal.get_server_id();

    char instance_id[64] = {0};
    snprintf_s(instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "%s", server_id.c_str());

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "TOPO_FILE_DESC");

    for (size_t i = 0; i < eid_cnt; ++i) {
        int port_id = aclshmemi_eid_parser_t::get_port_id(eid_list[i].eid);
        if (port_id > 9) {
            continue;
        }
        int die_id = aclshmemi_eid_parser_t::get_die_id(eid_list[i].eid);

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, eid_list[i].eid);

        char port[16] = {0};
        char plane_id[16] = {0};
        snprintf_s(port, sizeof(port), sizeof(port) - 1, "%d/%d", die_id, port_id);
        snprintf_s(plane_id, sizeof(plane_id), sizeof(plane_id) - 1, "plane_%d", die_id);
        aclshmemi_addr_add_port(addr, port);
        aclshmemi_addr_set_plane_id(addr, plane_id);

        aclshmemi_net_layer_add_addr(layer, addr);
    }

    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_mesh2p_layer(
    int phy_id, const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    if (eid_cnt == 0) {
        return std::nullopt;
    }

    auto& hal = aclshmemi_hal_t::instance();
    std::string server_id = hal.get_server_id();

    char instance_id[64] = {0};
    snprintf_s(instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "%s_%d", server_id.c_str(), phy_id / 2);

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "TOPO_FILE_DESC");

    for (size_t i = 0; i < eid_cnt; ++i) {
        int port_id = aclshmemi_eid_parser_t::get_port_id(eid_list[i].eid);
        if (port_id <= 9) {
            continue;
        }

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, eid_list[i].eid);

        const int ports[] = {4, 5, 6, 8};
        for (int j = 0; j < 4; ++j) {
            char port[16] = {0};
            snprintf_s(port, sizeof(port), sizeof(port) - 1, "0/%d", ports[j]);
            aclshmemi_addr_add_port(addr, port);
        }
        aclshmemi_addr_set_plane_id(addr, "plane_0");
        aclshmemi_net_layer_add_addr(layer, addr);
    }

    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_card_product_t::process_roce_layer(int phy_id)
{
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 3, "cluster");
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    aclshmemi_addr_t addr;
    char ip_addr[64] = {0};
    if (get_npu_host_rdma_ip(phy_id, ip_addr, sizeof(ip_addr)) != 0) {
        return std::nullopt;
    }
    aclshmemi_addr_set_ip(addr, ip_addr);
    aclshmemi_addr_add_port(addr, "d2h");

    aclshmemi_net_layer_add_addr(layer, addr);
    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_server_product_t::process_mesh_layer(
    int phy_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    snprintf_s(
        instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "sp%d_srv%d", spod_info.super_pod_id,
        spod_info.server_index);
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "MESH");

    int mesh_entity_id = aclshmemi_eid_parser_t::get_max_entity_id(ue_list);
    constexpr uint32_t MIN_MESH_EID_NUM = 7;
    constexpr int MESH_DIE = 1;
    constexpr int MAX_MESH_PORT_ID = 9;

    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        if (fe != mesh_entity_id || ue_list.ueList[i].eidNum < MIN_MESH_EID_NUM) {
            continue;
        }

        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum; ++j) {
            int phy_port_id = aclshmemi_eid_parser_t::get_port_id(ue_list.ueList[i].eidList[j].eid);
            if (phy_port_id > MAX_MESH_PORT_ID) {
                continue;
            }

            aclshmemi_addr_t addr;
            aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[j].eid);

            char port[16] = {0};
            char plane_id[16] = {0};
            snprintf_s(port, sizeof(port), sizeof(port) - 1, "%d/%d", MESH_DIE, phy_port_id - 1);
            snprintf_s(plane_id, sizeof(plane_id), sizeof(plane_id) - 1, "plane_%d", MESH_DIE);

            aclshmemi_addr_add_port(addr, port);
            aclshmemi_addr_set_plane_id(addr, plane_id);
            aclshmemi_net_layer_add_addr(layer, addr);
        }
    }

    return layer;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_server_product_t::process_clos_layer(
    int phy_id, uint32_t mainboard_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    snprintf_s(instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "superpod_%d", spod_info.super_pod_id);
    aclshmemi_net_layer_init(layer, 1, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    constexpr size_t RULE_COUNT = sizeof(g_ub_rules) / sizeof(g_ub_rules[0]);

    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        int port_group_idx = aclshmemi_eid_parser_t::get_server_port_group_idx(ue_list.ueList[i]);
        if (port_group_idx < 0) {
            continue;
        }

        int die = aclshmemi_eid_parser_t::get_server_die_id(ue_list.ueList[i].eidList[port_group_idx].eid);

        for (size_t r = 0; r < RULE_COUNT; ++r) {
            if (mainboard_id == g_ub_rules[r].mainboard_id && die == g_ub_rules[r].die_id &&
                fe == g_ub_rules[r].ue_id) {
                aclshmemi_addr_t addr;
                aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[port_group_idx].eid);

                for (size_t p = 0; p < g_ub_rules[r].ports.size(); ++p) {
                    char port[16] = {0};
                    snprintf_s(port, sizeof(port), sizeof(port) - 1, "%d/%d", die, g_ub_rules[r].ports[p]);
                    aclshmemi_addr_add_port(addr, port);
                }

                char plane_id[16] = {0};
                snprintf_s(plane_id, sizeof(plane_id), sizeof(plane_id) - 1, "plane_%d", die);
                aclshmemi_addr_set_plane_id(addr, plane_id);

                aclshmemi_net_layer_add_addr(layer, addr);
            }
        }
    }

    return layer;
}
std::optional<aclshmemi_root_info_t> aclshmemi_server_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    auto& hal = aclshmemi_hal_t::instance();

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    std::string driver_path = hal.get_driver_install_path();
    aclshmemi_root_info_set_topo_file_path(root_info, build_topo_file_path(driver_path, "atlas_850_1.json"));

    UEList ue_list;
    memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list));
    int ret = hal.get_ue_list(phy_id, &ue_list);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_server_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    dcmi_spod_info spod_info;
    memset_s(&spod_info, sizeof(spod_info), 0, sizeof(spod_info));
    ret = hal.get_spod_info(phy_id, &spod_info);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_server_product_t::get_root_info failed: get_spod_info failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, phy_id);

    auto mesh_layer = process_mesh_layer(phy_id, ue_list, spod_info);
    if (mesh_layer) {
        aclshmemi_rank_add_net_layer(rank, *mesh_layer);
    }

    auto clos_layer = process_clos_layer(phy_id, mainboard_id, ue_list, spod_info);
    if (clos_layer) {
        aclshmemi_rank_add_net_layer(rank, *clos_layer);
    }

    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_pod_product_t::process_mesh_layer(
    int phy_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    snprintf_s(
        instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "sp%d_chassis%d", spod_info.super_pod_id,
        spod_info.chassis_id);
    aclshmemi_net_layer_init(layer, 0, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "MESH");

    int mesh_entity_id = aclshmemi_eid_parser_t::get_max_entity_id(ue_list);
    constexpr uint32_t MIN_MESH_EID_NUM = 7;
    constexpr int MAX_MESH_PORT_ID = 9;
    constexpr int NPU_NUM = 8;

    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        if (fe != mesh_entity_id || ue_list.ueList[i].eidNum < MIN_MESH_EID_NUM) {
            continue;
        }

        for (uint32_t j = 0; j < ue_list.ueList[i].eidNum; ++j) {
            int phy_port_id = aclshmemi_eid_parser_t::get_port_id(ue_list.ueList[i].eidList[j].eid);
            if (phy_port_id > MAX_MESH_PORT_ID) {
                continue;
            }

            aclshmemi_addr_t addr;
            aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[j].eid);

            int die = (phy_id % NPU_NUM) < 4 ? 0 : 1;
            int transformed_port = phy_port_id < 2 ? (phy_port_id - 1) : (phy_port_id + 2);

            char port[16] = {0};
            snprintf_s(port, sizeof(port), sizeof(port) - 1, "%d/%d", die, transformed_port);
            aclshmemi_addr_add_port(addr, port);
            aclshmemi_addr_set_plane_id(addr, "plane_0");

            aclshmemi_net_layer_add_addr(layer, addr);
        }
    }

    return layer;
}

static const aclshmemi_pod_ub_entity_rule_t* get_pod_ub_rule(
    int npu_id, int level, uint32_t mainboard_id, int die, int fe_id)
{
    constexpr size_t RULE_COUNT = sizeof(g_pod_ub_rules) / sizeof(g_pod_ub_rules[0]);
    constexpr int NPU_NUM = 8;

    for (size_t i = 0; i < RULE_COUNT; ++i) {
        if (g_pod_ub_rules[i].level != level) {
            continue;
        }

        int npu_in_rule = 0;
        for (size_t j = 0; j < g_pod_ub_rules[i].npus.size(); ++j) {
            if (g_pod_ub_rules[i].npus[j] == (npu_id % NPU_NUM)) {
                npu_in_rule = 1;
                break;
            }
        }
        if (npu_in_rule == 0) {
            continue;
        }

        if (mainboard_id == g_pod_ub_rules[i].mainboard_id && die == g_pod_ub_rules[i].die_id &&
            fe_id == g_pod_ub_rules[i].ue_id) {
            return &g_pod_ub_rules[i];
        }
    }
    return nullptr;
}

std::optional<aclshmemi_net_layer_t> aclshmemi_pod_product_t::process_clos_layer(
    int phy_id, uint32_t mainboard_id, const UEList& ue_list, const dcmi_spod_info& spod_info)
{
    aclshmemi_net_layer_t layer;
    char instance_id[64] = {0};
    snprintf_s(instance_id, sizeof(instance_id), sizeof(instance_id) - 1, "superpod_%d", spod_info.super_pod_id);
    aclshmemi_net_layer_init(layer, 1, instance_id);
    aclshmemi_net_layer_set_net_type(layer, "CLOS");

    int addr_6port_idx = 0;
    int addr_2port_idx = 0;

    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        int fe = aclshmemi_eid_parser_t::get_ub_entity_id(ue_list.ueList[i]);
        int port_group_idx = aclshmemi_eid_parser_t::get_server_port_group_idx(ue_list.ueList[i]);
        if (port_group_idx < 0) {
            continue;
        }

        int die = aclshmemi_eid_parser_t::get_pod_die_id(ue_list.ueList[i].eidList[port_group_idx].eid);

        const aclshmemi_pod_ub_entity_rule_t* rule = get_pod_ub_rule(phy_id, 1, mainboard_id, die, fe);
        if (rule == nullptr) {
            continue;
        }

        aclshmemi_addr_t addr;
        aclshmemi_addr_set_eid(addr, ue_list.ueList[i].eidList[port_group_idx].eid);

        for (size_t p = 0; p < rule->ports.size(); ++p) {
            char port[16] = {0};
            snprintf_s(port, sizeof(port), sizeof(port) - 1, "%d/%d", die, rule->ports[p]);
            aclshmemi_addr_add_port(addr, port);
        }

        aclshmemi_addr_set_plane_id(addr, rule->plane_id);

        constexpr int PRIMARY_PORT_NUM = 6;
        if (rule->ports.size() >= PRIMARY_PORT_NUM) {
            aclshmemi_net_layer_set_addr_at(layer, addr, addr_6port_idx++);
        } else {
            aclshmemi_net_layer_set_addr_at(layer, addr, addr_2port_idx++);
        }
    }

    return layer;
}
std::optional<aclshmemi_root_info_t> aclshmemi_pod_product_t::get_root_info(int phy_id, uint32_t mainboard_id)
{
    auto& hal = aclshmemi_hal_t::instance();

    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    std::string driver_path = hal.get_driver_install_path();
    aclshmemi_root_info_set_topo_file_path(root_info, build_topo_file_path(driver_path, "atlas_950_1.json"));

    UEList ue_list;
    memset_s(&ue_list, sizeof(ue_list), 0, sizeof(ue_list));
    int ret = hal.get_ue_list(phy_id, &ue_list);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: get_ue_list failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    dcmi_spod_info spod_info;
    memset_s(&spod_info, sizeof(spod_info), 0, sizeof(spod_info));
    ret = hal.get_spod_info(phy_id, &spod_info);
    if (ret != 0) {
        SHM_LOG_ERROR("aclshmemi_pod_product_t::get_root_info failed: get_spod_info failed for phy_id=" << phy_id);
        return std::nullopt;
    }

    constexpr int NPU_NUM = 8;
    int local_id = (phy_id % NPU_NUM) + ((spod_info.server_index % NPU_NUM) * NPU_NUM);

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, phy_id, local_id);

    auto mesh_layer = process_mesh_layer(phy_id, ue_list, spod_info);
    if (mesh_layer) {
        aclshmemi_rank_add_net_layer(rank, *mesh_layer);
    }

    auto clos_layer = process_clos_layer(phy_id, mainboard_id, ue_list, spod_info);
    if (clos_layer) {
        aclshmemi_rank_add_net_layer(rank, *clos_layer);
    }

    aclshmemi_root_info_add_rank(root_info, rank);
    return root_info;
}

} // namespace topo
} // namespace shm