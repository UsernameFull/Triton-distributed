/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_TRANSPORT_TOPO_READER_H
#define MF_HYBRID_TRANSPORT_TOPO_READER_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"

namespace shm {
namespace transport {

constexpr std::size_t URMA_EID_RAW_SIZE = 16;
constexpr std::size_t URMA_EID_HEX_SIZE = URMA_EID_RAW_SIZE * 2;
constexpr uint32_t URMA_EID_IPV4_PREFIX = 0x0000;

struct RankInfo {
    uint32_t device_id{0};
    uint32_t local_id{0};
};

struct RootInfo {
    std::string topo_file_path;
    // The parsed view is scoped to the current device/phyId.
    std::vector<RankInfo> rank_list;
    // deviceId -> localId
    std::unordered_map<uint32_t, uint32_t> deviceLocalIdMap;
    // eid count for the current device only
    uint32_t eidCount{0};
    // localId -> port -> eidIndex
    std::unordered_map<uint32_t, std::unordered_map<std::string, uint32_t>> portEidMap;
    // localId -> eidIndex -> eid raw bytes from rootinfo addr
    std::unordered_map<uint32_t, std::map<uint32_t, std::array<uint8_t, URMA_EID_RAW_SIZE>>> eidAddrMap;
};

struct TopoEdge {
    uint32_t local_a{0};
    std::vector<std::string> local_a_ports;
    uint32_t local_b{0};
    std::vector<std::string> local_b_ports;
};

struct TopoInfo {
    std::vector<TopoEdge> edge_list;
};

class TopoReader {
public:
    static constexpr const char* ROOTINFO_PATH = "/etc/hccl_rootinfo.json";
    static bool ParseRootInfo(uint32_t phyId, RootInfo& out);
    static bool ParseTopoInfo(const std::string& path, TopoInfo& out);
    static bool GetLocalEidRouteForPeer(
        const RootInfo& root, const TopoInfo& topo, uint32_t myLocalId, uint32_t peerLocalId, uint32_t& localEidIndex,
        std::array<uint8_t, URMA_EID_RAW_SIZE>& localEidRaw);
    static bool GetLocalId(const RootInfo& root, uint32_t deviceId, uint32_t& localId);

    static bool GetEidCount(const RootInfo& root, uint32_t& count);

private:
    static bool ParseRootInfoJson(const nlohmann::json& rootInfoJson, uint32_t phyId, RootInfo& out);
    static bool ParseRankAddrRaw(
        const nlohmann::json& rankAddrJson, uint32_t localId, uint32_t eidIndex,
        std::array<uint8_t, URMA_EID_RAW_SIZE>& raw);
    static bool ParseEidRaw(const nlohmann::json& jsonValue, std::array<uint8_t, URMA_EID_RAW_SIZE>& raw);
    static bool ParseIpv4EidRaw(const std::string& addr, std::array<uint8_t, URMA_EID_RAW_SIZE>& raw);
    static bool ParseIpv6EidRaw(const std::string& addr, std::array<uint8_t, URMA_EID_RAW_SIZE>& raw);
    static bool ParseUint(const nlohmann::json& jsonValue, uint32_t& value);
};

} // namespace transport
} // namespace shm

#endif // MF_HYBRID_TRANSPORT_TOPO_READER_H
