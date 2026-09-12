/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "aclshmemi_eid_parser.h"
#include <cstdlib>
#include <cctype>

namespace shm {
namespace topo {

namespace {
constexpr int UE_ID_POS = 14;
constexpr int PORT_POS = 30;
constexpr int DIE_POS = 39;
constexpr int HEX_BASE = 16;
constexpr int FE_OFFSET = 5;
constexpr int PORT_LEFT_OFFSET = 1;
constexpr int PORT_RIGHT_OFFSET = 4;
constexpr int DIE_OFFSET = 3;
constexpr int DIE_MASK = 4;
constexpr int SERVER_DIE_OFFSET = 7;
constexpr int LOW_BIT_PORT_MASK = 0x7F;
constexpr int PORT_MASK = 0x80;
} // namespace

std::optional<int> aclshmemi_eid_parser_t::get_fe_id(const std::string& eid_hex_str)
{
    if (eid_hex_str.size() < static_cast<size_t>(UE_ID_POS + 1)) {
        return std::nullopt;
    }
    char fe = eid_hex_str[UE_ID_POS];
    int fe_id = static_cast<int>(strtol(&fe, nullptr, HEX_BASE)) >> 1;
    return fe_id;
}

std::optional<int> aclshmemi_eid_parser_t::get_port_id(const std::string& eid_hex_str)
{
    if (eid_hex_str.size() < static_cast<size_t>(PORT_POS + 1)) {
        return std::nullopt;
    }
    uint8_t end = static_cast<uint8_t>(strtoul(&eid_hex_str[PORT_POS], nullptr, HEX_BASE));
    uint8_t tmp = static_cast<uint8_t>(~PORT_MASK & end);
    uint8_t port = tmp >> 3;
    return static_cast<int>(port);
}

std::optional<int> aclshmemi_eid_parser_t::get_die_id(const std::string& eid_hex_str)
{
    if (eid_hex_str.size() < static_cast<size_t>(DIE_POS + 1)) {
        return std::nullopt;
    }
    char c = eid_hex_str[DIE_POS];
    if (!isxdigit(c)) {
        return std::nullopt;
    }

    uint8_t d = 0;
    char upper_c = toupper(c);
    if (isdigit(upper_c)) {
        d = static_cast<uint8_t>(upper_c - '0');
    } else {
        d = static_cast<uint8_t>(upper_c - 'A' + 10);
    }

    return static_cast<int>((d & 0x08) >> DIE_OFFSET);
}

int aclshmemi_eid_parser_t::get_fe_id(const dcmi_urma_eid_t& eid)
{
    constexpr int FE_POS = 7;
    int fe = eid.raw[FE_POS];
    return fe >> FE_OFFSET;
}

int aclshmemi_eid_parser_t::get_port_id(const dcmi_urma_eid_t& eid)
{
    uint8_t last = eid.raw[DCMI_URMA_EID_SIZE - 1];
    last = static_cast<uint8_t>(last << PORT_LEFT_OFFSET);
    last = static_cast<uint8_t>(last >> PORT_RIGHT_OFFSET);
    return static_cast<int>(last);
}

int aclshmemi_eid_parser_t::get_die_id(const dcmi_urma_eid_t& eid)
{
    uint8_t last = eid.raw[DCMI_URMA_EID_SIZE - 1];
    int die_id = (DIE_MASK & last) == 0 ? 0 : 1;
    return die_id;
}

int aclshmemi_eid_parser_t::get_low_bit_port(const dcmi_urma_eid_t& eid)
{
    uint16_t lower = eid.raw[DCMI_URMA_EID_SIZE - 1];
    return lower & LOW_BIT_PORT_MASK;
}

int aclshmemi_eid_parser_t::get_pod_die_id(const dcmi_urma_eid_t& eid)
{
    uint8_t bit = eid.raw[DCMI_URMA_EID_SIZE - 2];
    return (bit & DIE_MASK) > 0 ? 1 : 0;
}

int aclshmemi_eid_parser_t::get_server_die_id(const dcmi_urma_eid_t& eid)
{
    uint8_t bit = eid.raw[DCMI_URMA_EID_SIZE - 1];
    return bit >> SERVER_DIE_OFFSET;
}

std::optional<int> aclshmemi_eid_parser_t::get_max_fe_id(const dcmi_urma_eid_info_t* eid_list, size_t eid_cnt)
{
    int max_fe_id = -1;
    for (size_t i = 0; i < eid_cnt; ++i) {
        int fe_id = get_fe_id(eid_list[i].eid);
        if (fe_id > max_fe_id) {
            max_fe_id = fe_id;
        }
    }
    return max_fe_id >= 0 ? std::optional<int>(max_fe_id) : std::nullopt;
}

int aclshmemi_eid_parser_t::get_ub_entity_id(const UBEntity& ue)
{
    if (ue.eidNum == 0) {
        return -1;
    }
    return get_fe_id(ue.eidList[0].eid);
}

int aclshmemi_eid_parser_t::get_server_port_group_idx(const UBEntity& ue)
{
    constexpr int MAX_PHY_PORT_ID = 9;
    for (uint32_t i = 0; i < ue.eidNum; ++i) {
        int port = get_port_id(ue.eidList[i].eid);
        if (port <= MAX_PHY_PORT_ID) {
            continue;
        }
        return static_cast<int>(i);
    }
    return -1;
}

int aclshmemi_eid_parser_t::get_max_entity_id(const UEList& ue_list)
{
    int max_id = -1;
    for (uint32_t i = 0; i < ue_list.ueNum; ++i) {
        if (ue_list.ueList[i].eidNum == 0) {
            continue;
        }
        int id = get_fe_id(ue_list.ueList[i].eidList[0].eid);
        if (id > max_id) {
            max_id = id;
        }
    }
    return max_id;
}

} // namespace topo
} // namespace shm