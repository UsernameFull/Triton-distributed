/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <arpa/inet.h>
#include <cctype>
#include <cstring>
#include <exception>
#include <fstream>
#include <utility>
#include <vector>

#include "rootinfo/topo_addr_info.h"
#include "shmemi_file_util.h"
#include "shmemi_host_common.h"
#include "topo_reader.h"

namespace shm {
namespace transport {

namespace {

bool CheckTopoFilePath(const std::string& path, std::string& realPath)
{
    realPath = path;
    if (utils::FileUtil::IsSymlink(realPath) || !utils::FileUtil::Realpath(realPath) ||
        !utils::FileUtil::IsFile(realPath) || !utils::FileUtil::CheckFileSize(realPath)) {
        SHM_LOG_ERROR("Topo file path check failed: " << path);
        return false;
    }
    return true;
}

} // namespace

bool TopoReader::ParseRootInfo(uint32_t phyId, RootInfo& out)
{
    out = RootInfo{};
    nlohmann::json rootInfoJson;

    bool shouldFallback = false;
    std::ifstream rootInfoFile(ROOTINFO_PATH);
    if (!rootInfoFile.is_open()) {
        SHM_LOG_WARN("Rootinfo file not found at " << ROOTINFO_PATH
                                                    << ", fallback to generated rootinfo for phyId " << phyId);
        shouldFallback = true;
    } else {
        try {
            rootInfoFile >> rootInfoJson;
        } catch (const std::exception& ex) {
            SHM_LOG_ERROR("Parse rootinfo file failed, the path is " << ROOTINFO_PATH << ", error: " << ex.what());
            SHM_LOG_WARN("Rootinfo file parse failed, fallback to generated rootinfo for phyId " << phyId);
            shouldFallback = true;
        }
    }

    if (!shouldFallback) {
        SHM_LOG_INFO("Load rootinfo from file " << ROOTINFO_PATH);
        if (ParseRootInfoJson(rootInfoJson, phyId, out)) {
            return true;
        }
        out = RootInfo{};
        SHM_LOG_WARN("Rootinfo file content is unusable, fallback to generated rootinfo for phyId " << phyId);
        shouldFallback = true;
    }

    size_t rootInfoSize = 0;
    int ret = topo_addr_info_get_size(static_cast<int>(phyId), &rootInfoSize);
    if (ret != 0 || rootInfoSize == 0) {
        SHM_LOG_ERROR("Failed to get generated rootinfo size for phyId " << phyId << ", ret = " << ret
                                                                          << ", size = " << rootInfoSize);
        return false;
    }
    SHM_LOG_INFO("Generated rootinfo size for phyId " << phyId << " is " << rootInfoSize);

    std::vector<char> rootInfoBuffer(rootInfoSize + 1, '\0');
    size_t actualSize = rootInfoSize;
    ret = topo_addr_info_get(static_cast<int>(phyId), rootInfoBuffer.data(), &actualSize);
    if (ret != 0 || actualSize == 0) {
        SHM_LOG_ERROR("Failed to get generated rootinfo for phyId " << phyId << ", ret = " << ret
                                                                     << ", actualSize = " << actualSize);
        return false;
    }
    if (actualSize > rootInfoBuffer.size() - 1) {
        SHM_LOG_ERROR("Generated rootinfo size overflow, actualSize " << actualSize << ", capacity "
                                                                       << rootInfoBuffer.size());
        return false;
    }
    rootInfoBuffer[actualSize] = '\0';

    try {
        rootInfoJson = nlohmann::json::parse(rootInfoBuffer.data(), rootInfoBuffer.data() + actualSize);
#ifdef DEBUG_MODE
        SHM_LOG_DEBUG("Generated rootinfo json for phyId " << phyId << ":\n" << rootInfoJson.dump(2));
#endif
    } catch (const std::exception& ex) {
        SHM_LOG_ERROR("Failed to parse generated rootinfo json for phyId " << phyId << ", error: " << ex.what());
        return false;
    }

    SHM_LOG_INFO("Use generated rootinfo fallback for phyId " << phyId);
    if (!ParseRootInfoJson(rootInfoJson, phyId, out)) {
        SHM_LOG_ERROR("Generated rootinfo content is unusable for phyId " << phyId);
        return false;
    }
    return true;
}

bool TopoReader::ParseRootInfoJson(const nlohmann::json& rootInfoJson, uint32_t phyId, RootInfo& out)
{
    if (!rootInfoJson.contains("topo_file_path") || !rootInfoJson["topo_file_path"].is_string()) {
        SHM_LOG_ERROR("Topo file path not found in rootinfo.");
        return false;
    }
    out.topo_file_path = rootInfoJson["topo_file_path"].get<std::string>();

    if (!rootInfoJson.contains("rank_list") || !rootInfoJson["rank_list"].is_array()) {
        SHM_LOG_ERROR("Rootinfo: rank_list not found.");
        return false;
    }

    const auto& rankListJson = rootInfoJson["rank_list"];
    for (const auto& rankJson : rankListJson) {
        if (!rankJson.contains("device_id") || !rankJson.contains("local_id")) {
            continue;
        }

        RankInfo rankInfo;
        if (!ParseUint(rankJson["device_id"], rankInfo.device_id) ||
            !ParseUint(rankJson["local_id"], rankInfo.local_id)) {
            SHM_LOG_ERROR("Rootinfo: invalid device_id or local_id entry.");
            return false;
        }

        if (rankInfo.device_id != phyId) {
            continue;
        }

        out.rank_list.push_back(rankInfo);
        out.deviceLocalIdMap[rankInfo.device_id] = rankInfo.local_id;

        if (!rankJson.contains("level_list") || !rankJson["level_list"].is_array() || rankJson["level_list"].empty()) {
            SHM_LOG_ERROR("Rootinfo: missing level_list for current phyId " << phyId << ", localId "
                                                                      << rankInfo.local_id);
            return false;
        }

        const auto& levelListJson = rankJson["level_list"];
        bool foundTopoLevel = false;
        for (const auto& levelJson : levelListJson) {
            if (!levelJson.contains("net_type") || !levelJson["net_type"].is_string()) {
                continue;
            }
            const auto& netType = levelJson["net_type"].get_ref<const std::string&>();
            if (netType == "TOPO_FILE_DESC" || netType == "MESH") {
                foundTopoLevel = true;
                if (!levelJson.contains("rank_addr_list") || !levelJson["rank_addr_list"].is_array()) {
                    SHM_LOG_ERROR("Rootinfo: missing rank_addr_list for current phyId " << phyId << ", localId "
                                                                                          << rankInfo.local_id);
                    return false;
                }
                const auto& rankAddrListJson = levelJson["rank_addr_list"];
                out.eidCount = static_cast<uint32_t>(rankAddrListJson.size());
                uint32_t eidIndex = 0;
                for (const auto& rankAddrJson : rankAddrListJson) {
                    std::array<uint8_t, URMA_EID_RAW_SIZE> rawAddr{};
                    if (ParseRankAddrRaw(rankAddrJson, rankInfo.local_id, eidIndex, rawAddr)) {
                        out.eidAddrMap[rankInfo.local_id][eidIndex] = rawAddr;
                    } else {
                        SHM_LOG_WARN(
                            "Rootinfo: failed to parse rank_addr_list addr, localId " << rankInfo.local_id
                                                                                      << ", eidIndex " << eidIndex);
                    }
                    if (rankAddrJson.contains("ports") && rankAddrJson["ports"].is_array()) {
                        for (const auto& port : rankAddrJson["ports"]) {
                            if (port.is_string()) {
                                out.portEidMap[rankInfo.local_id][port.get<std::string>()] = eidIndex;
                            }
                        }
                    }
                    ++eidIndex;
                }
                break;
            }
        }
        if (!foundTopoLevel) {
            SHM_LOG_ERROR("Rootinfo: missing topo level for current phyId " << phyId << ", localId "
                                                                             << rankInfo.local_id);
            return false;
        }

        if (out.eidCount == 0) {
            SHM_LOG_ERROR("Rootinfo: invalid eid count for current phyId " << phyId << ", localId "
                                                                            << rankInfo.local_id);
            return false;
        }

        SHM_LOG_INFO("Parse rootinfo success for phyId " << phyId << ", localId " << rankInfo.local_id
                                                          << ", eidCount " << out.eidCount);
        return true;
    }

    SHM_LOG_ERROR("Rootinfo: no rank entry found for phyId " << phyId);
    return false;
}

bool TopoReader::ParseTopoInfo(const std::string& path, TopoInfo& out)
{
    std::string realPath;
    if (!CheckTopoFilePath(path, realPath)) {
        return false;
    }

    std::ifstream topoFile(realPath);
    if (!topoFile.is_open()) {
        SHM_LOG_ERROR("Failed to open topo file: " << realPath);
        return false;
    }
    nlohmann::json topoInfoJson;
    try {
        topoFile >> topoInfoJson;
    } catch (const nlohmann::json::exception& e) {
        SHM_LOG_ERROR("Topo parse failed: " << e.what());
        return false;
    }
    SHM_LOG_INFO("Read topo json from " << realPath);

    if (!topoInfoJson.contains("edge_list") || !topoInfoJson["edge_list"].is_array()) {
        SHM_LOG_ERROR("Topo parse failed: edge_list not found.");
        return false;
    }
    const auto& edgeListJson = topoInfoJson["edge_list"];
    out.edge_list.reserve(edgeListJson.size());
    for (const auto& edgeObj : edgeListJson) {
        if (!edgeObj.contains("local_a") || !edgeObj.contains("local_b")) {
            continue;
        }
        TopoEdge edge;
        if (!ParseUint(edgeObj["local_a"], edge.local_a) || !ParseUint(edgeObj["local_b"], edge.local_b)) {
            SHM_LOG_ERROR("Topo parse failed: local_a or local_b is invalid.");
            return false;
        }

        if (edgeObj.contains("local_a_ports") && edgeObj["local_a_ports"].is_array()) {
            edge.local_a_ports = edgeObj["local_a_ports"].get<std::vector<std::string>>();
        }
        if (edgeObj.contains("local_b_ports") && edgeObj["local_b_ports"].is_array()) {
            edge.local_b_ports = edgeObj["local_b_ports"].get<std::vector<std::string>>();
        }
        out.edge_list.push_back(std::move(edge));
    }

    if (out.edge_list.empty()) {
        SHM_LOG_ERROR("Topo parse failed: no valid edge entries parsed.");
        return false;
    }
    return true;
}

// Resolves the local route used to connect from myLocalId to peerLocalId.
// The remote EID index must be provided by the peer, because rootinfo only contains
// port -- EID
bool TopoReader::GetLocalEidRouteForPeer(
    const RootInfo& root, const TopoInfo& topo, uint32_t myLocalId, uint32_t peerLocalId, uint32_t& localEidIndex,
    std::array<uint8_t, URMA_EID_RAW_SIZE>& localEidRaw)
{
    std::string localPort;
    for (const auto& edge : topo.edge_list) {
        if (edge.local_a == myLocalId && edge.local_b == peerLocalId && !edge.local_a_ports.empty()) {
            localPort = edge.local_a_ports[0];
            break;
        }
        if (edge.local_b == myLocalId && edge.local_a == peerLocalId && !edge.local_b_ports.empty()) {
            localPort = edge.local_b_ports[0];
            break;
        }
    }
    if (localPort.empty()) {
        SHM_LOG_ERROR(
            "Failed to get local eid route, no usable edge between localId " << myLocalId << " and " << peerLocalId);
        return false;
    }

    const auto localRankItem = root.portEidMap.find(myLocalId);
    if (localRankItem == root.portEidMap.end()) {
        SHM_LOG_ERROR("Failed to get local eid index, localId " << myLocalId << " not found in portEidMap");
        return false;
    }
    const auto localPortItem = localRankItem->second.find(localPort);
    if (localPortItem == localRankItem->second.end()) {
        SHM_LOG_ERROR("Failed to get local eid index, port " << localPort << " not found for localId " << myLocalId);
        return false;
    }
    localEidIndex = localPortItem->second;

    const auto localIt = root.eidAddrMap.find(myLocalId);
    if (localIt == root.eidAddrMap.end()) {
        SHM_LOG_ERROR("Failed to get eid route, localId " << myLocalId << " not found in eidAddrMap");
        return false;
    }
    const auto eidIt = localIt->second.find(localEidIndex);
    if (eidIt == localIt->second.end()) {
        SHM_LOG_ERROR("Failed to get eid route, eidIndex " << localEidIndex << " not found for localId " << myLocalId);
        return false;
    }
    localEidRaw = eidIt->second;
    SHM_LOG_INFO(
        "Get local eid route success, myLocalId: " << myLocalId << ", peerLocalId: " << peerLocalId << ", localPort: "
                                                   << localPort << ", localEidIndex: " << localEidIndex);
    return true;
}

bool TopoReader::GetLocalId(const RootInfo& root, uint32_t deviceId, uint32_t& localId)
{
    const auto it = root.deviceLocalIdMap.find(deviceId);
    if (it != root.deviceLocalIdMap.end()) {
        localId = it->second;
        return true;
    }
    SHM_LOG_ERROR("RootInfo is invalid or incomplete, failed to find localId for deviceId " << deviceId);
    return false;
}

bool TopoReader::GetEidCount(const RootInfo& root, uint32_t& count)
{
    if (root.eidCount > 0) {
        count = root.eidCount;
        return true;
    }

    SHM_LOG_ERROR("RootInfo is invalid or incomplete, failed to find a valid eid count.");
    return false;
}

bool TopoReader::ParseRankAddrRaw(
    const nlohmann::json& rankAddrJson, uint32_t localId, uint32_t eidIndex,
    std::array<uint8_t, URMA_EID_RAW_SIZE>& raw)
{
    if (!rankAddrJson.contains("addr")) {
        SHM_LOG_ERROR("Rootinfo rank_addr_list entry does not contain addr, localId " << localId << ", eidIndex "
                                                                                       << eidIndex);
        return false;
    }

    std::string addrType = "EID";
    if (rankAddrJson.contains("addr_type")) {
        if (!rankAddrJson["addr_type"].is_string()) {
            SHM_LOG_ERROR("Rootinfo addr_type format is unsupported, localId " << localId << ", eidIndex " << eidIndex);
            return false;
        }
        addrType = rankAddrJson["addr_type"].get<std::string>();
        std::transform(addrType.begin(), addrType.end(), addrType.begin(), [](unsigned char ch) {
            return static_cast<char>(std::toupper(ch));
        });
    }

    if (addrType == "EID") {
        return ParseEidRaw(rankAddrJson["addr"], raw);
    }

    if (!rankAddrJson["addr"].is_string()) {
        SHM_LOG_ERROR("Rootinfo " << addrType << " addr must be a string, localId " << localId << ", eidIndex "
                                  << eidIndex);
        return false;
    }

    const auto& addr = rankAddrJson["addr"].get_ref<const std::string&>();
    if (addrType == "IPV4") {
        const bool ret = ParseIpv4EidRaw(addr, raw);
        if (ret) {
            SHM_LOG_INFO(
                "Rootinfo IPV4 addr converted to EID, localId " << localId << ", eidIndex " << eidIndex << ", addr "
                                                                << addr);
        } else {
            SHM_LOG_ERROR(
                "Rootinfo IPV4 addr conversion failed, localId " << localId << ", eidIndex " << eidIndex << ", addr "
                                                                 << addr);
        }
        return ret;
    }
    if (addrType == "IPV6") {
        const bool ret = ParseIpv6EidRaw(addr, raw);
        if (ret) {
            SHM_LOG_INFO(
                "Rootinfo IPV6 addr converted to EID, localId " << localId << ", eidIndex " << eidIndex << ", addr "
                                                                << addr);
        } else {
            SHM_LOG_ERROR(
                "Rootinfo IPV6 addr conversion failed, localId " << localId << ", eidIndex " << eidIndex << ", addr "
                                                                 << addr);
        }
        return ret;
    }

    SHM_LOG_ERROR("Rootinfo addr_type is unsupported: " << addrType << ", localId " << localId << ", eidIndex "
                                                        << eidIndex);
    return false;
}

bool TopoReader::ParseEidRaw(const nlohmann::json& jsonValue, std::array<uint8_t, URMA_EID_RAW_SIZE>& raw)
{
    raw.fill(0);
    if (jsonValue.is_array()) {
        if (jsonValue.size() != raw.size()) {
            SHM_LOG_ERROR("Rootinfo addr array size is invalid: " << jsonValue.size());
            return false;
        }
        for (size_t i = 0; i < raw.size(); ++i) {
            if (!jsonValue[i].is_number_unsigned() && !jsonValue[i].is_number_integer()) {
                SHM_LOG_ERROR("Rootinfo addr array contains non-numeric value at index " << i);
                return false;
            }
            auto value = jsonValue[i].get<int32_t>();
            if (value < 0 || value > 0xff) {
                SHM_LOG_ERROR("Rootinfo addr array value out of byte range at index " << i << ", value " << value);
                return false;
            }
            raw[i] = static_cast<uint8_t>(value);
        }
        return true;
    }

    if (!jsonValue.is_string()) {
        SHM_LOG_ERROR("Rootinfo addr format is unsupported.");
        return false;
    }

    const std::string& jsonString = jsonValue.get_ref<const std::string&>();
    std::string normalized;
    normalized.reserve(jsonString.size());
    for (const char ch : jsonString) {
        if (std::isxdigit(static_cast<unsigned char>(ch))) {
            normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
        }
    }

    if (normalized.size() != URMA_EID_HEX_SIZE) {
        SHM_LOG_ERROR("Rootinfo addr hex length is invalid: " << normalized.size());
        return false;
    }

    try {
        for (size_t i = 0; i < raw.size(); ++i) {
            const std::string byteStr = normalized.substr(i * 2, 2);
            raw[i] = static_cast<uint8_t>(std::stoul(byteStr, nullptr, 16));
        }
    } catch (const std::exception& ex) {
        SHM_LOG_ERROR("Failed to parse rootinfo addr hex string, error: " << ex.what());
        return false;
    }
    return true;
}

bool TopoReader::ParseIpv4EidRaw(const std::string& addr, std::array<uint8_t, URMA_EID_RAW_SIZE>& raw)
{
    in_addr ipv4Addr {};
    if (inet_pton(AF_INET, addr.c_str(), &ipv4Addr) != 1) {
        SHM_LOG_ERROR("Rootinfo IPV4 addr is invalid: " << addr);
        return false;
    }

    raw.fill(0);
    raw[10] = static_cast<uint8_t>((URMA_EID_IPV4_PREFIX >> 8) & 0xff);
    raw[11] = static_cast<uint8_t>(URMA_EID_IPV4_PREFIX & 0xff);

    const auto *ipv4Bytes = reinterpret_cast<const uint8_t *>(&ipv4Addr.s_addr);
    std::copy_n(ipv4Bytes, sizeof(ipv4Addr), raw.begin() + 12);
    return true;
}

bool TopoReader::ParseIpv6EidRaw(const std::string& addr, std::array<uint8_t, URMA_EID_RAW_SIZE>& raw)
{
    raw.fill(0);
    if (inet_pton(AF_INET6, addr.c_str(), raw.data()) != 1) {
        SHM_LOG_ERROR("Rootinfo IPV6 addr is invalid: " << addr);
        return false;
    }
    return true;
}

bool TopoReader::ParseUint(const nlohmann::json& jsonValue, uint32_t& value)
{
    try {
        if (jsonValue.is_string()) {
            value = static_cast<uint32_t>(std::stoul(jsonValue.get<std::string>()));
            return true;
        }
        value = jsonValue.get<uint32_t>();
        return true;
    } catch (const std::exception& ex) {
        SHM_LOG_ERROR("Failed to parse uint value, error: " << ex.what());
        return false;
    }
}

} // namespace transport
} // namespace shm
