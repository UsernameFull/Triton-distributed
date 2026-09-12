/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <string>
#include "aclshmemi_types.h"
#include "dl_dcmi_types.h"

namespace shm {
namespace topo {

class TypesTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

static void hex32_to_bin16(const char* hex_str, uint8_t* bin_out) {
    for (int i = 0; i < 16; i++) {
        char c1 = hex_str[2 * i];
        char c2 = hex_str[2 * i + 1];
        int h1 = (c1 >= '0' && c1 <= '9') ? c1 - '0' : (c1 >= 'a' && c1 <= 'f') ? 10 + c1 - 'a' : 10 + c1 - 'A';
        int h2 = (c2 >= '0' && c2 <= '9') ? c2 - '0' : (c2 >= 'a' && c2 <= 'f') ? 10 + c2 - 'a' : 10 + c2 - 'A';
        bin_out[i] = (h1 << 4) | h2;
    }
}

TEST_F(TypesTest, AddrSetEid) {
    aclshmemi_addr_t addr;
    dcmi_urma_eid_t eid;
    hex32_to_bin16("000000000000002000100000dfdf0020", eid.raw);

    int ret = aclshmemi_addr_set_eid(addr, eid);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(addr.addr_type, "EID");
    EXPECT_EQ(addr.addr, "000000000000002000100000dfdf0020");
}

TEST_F(TypesTest, AddrSetIp) {
    aclshmemi_addr_t addr;
    int ret = aclshmemi_addr_set_ip(addr, "192.168.1.1");
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(addr.addr_type, "IPV4");
    EXPECT_EQ(addr.addr, "192.168.1.1");
}

TEST_F(TypesTest, AddrSetPlaneId) {
    aclshmemi_addr_t addr;
    int ret = aclshmemi_addr_set_plane_id(addr, "plane0");
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(addr.plane_id, "plane0");
}

TEST_F(TypesTest, AddrAddPort) {
    aclshmemi_addr_t addr;
    int ret = aclshmemi_addr_add_port(addr, "port0");
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(addr.ports.size(), 1);
    EXPECT_EQ(addr.ports[0], "port0");

    ret = aclshmemi_addr_add_port(addr, "port1");
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(addr.ports.size(), 2);
    EXPECT_EQ(addr.ports[1], "port1");
}

TEST_F(TypesTest, AddrToString) {
    aclshmemi_addr_t addr;
    addr.addr_type = "EID";
    addr.addr = "000000000000002000100000dfdf0020";
    addr.ports.push_back("4");
    addr.ports.push_back("5");
    addr.plane_id = "plane0";

    std::string str = aclshmemi_addr_to_string(addr);
    EXPECT_TRUE(str.find("\"addr_type\": \"EID\"") != std::string::npos);
    EXPECT_TRUE(str.find("\"addr\": \"000000000000002000100000dfdf0020\"") != std::string::npos);
    EXPECT_TRUE(str.find("\"ports\": [\"4\",\"5\"]") != std::string::npos);
    EXPECT_TRUE(str.find("\"plane_id\": \"plane0\"") != std::string::npos);
}

TEST_F(TypesTest, NetLayerInit) {
    aclshmemi_net_layer_t layer;
    int ret = aclshmemi_net_layer_init(layer, 0, "instance0");
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(layer.level, 0);
    EXPECT_EQ(layer.instance_id, "instance0");
    EXPECT_TRUE(layer.net_type.empty());
    EXPECT_TRUE(layer.net_attr.empty());
    EXPECT_TRUE(layer.addr_list.empty());
}

TEST_F(TypesTest, NetLayerSetNetType) {
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, "instance0");
    int ret = aclshmemi_net_layer_set_net_type(layer, "MESH");
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(layer.net_type, "MESH");
}

TEST_F(TypesTest, NetLayerSetNetAttr) {
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, "instance0");
    int ret = aclshmemi_net_layer_set_net_attr(layer, "FULL");
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(layer.net_attr, "FULL");
}

TEST_F(TypesTest, NetLayerAddAddr) {
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, "instance0");

    aclshmemi_addr_t addr;
    addr.addr_type = "EID";
    addr.addr = "000000000000002000100000dfdf0020";

    int ret = aclshmemi_net_layer_add_addr(layer, addr);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(layer.addr_list.size(), 1);
}

TEST_F(TypesTest, NetLayerSetAddrAt) {
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, "instance0");

    aclshmemi_addr_t addr1;
    addr1.addr_type = "EID";
    addr1.addr = "000000000000002000100000dfdf0020";

    aclshmemi_addr_t addr2;
    addr2.addr_type = "EID";
    addr2.addr = "000000000000002000100000dfdf0028";

    aclshmemi_net_layer_add_addr(layer, addr1);
    int ret = aclshmemi_net_layer_set_addr_at(layer, addr2, 0);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(layer.addr_list[0].addr, "000000000000002000100000dfdf0028");

    ret = aclshmemi_net_layer_set_addr_at(layer, addr1, 1);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(layer.addr_list.size(), 2);
}

TEST_F(TypesTest, NetLayerToString) {
    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, "instance0");
    aclshmemi_net_layer_set_net_type(layer, "MESH");
    aclshmemi_net_layer_set_net_attr(layer, "FULL");

    aclshmemi_addr_t addr;
    addr.addr_type = "EID";
    addr.addr = "000000000000002000100000dfdf0020";
    addr.plane_id = "plane0";
    aclshmemi_net_layer_add_addr(layer, addr);

    std::string str = aclshmemi_net_layer_to_string(layer);
    EXPECT_TRUE(str.find("\"net_layer\": 0") != std::string::npos);
    EXPECT_TRUE(str.find("\"net_instance_id\": \"instance0\"") != std::string::npos);
    EXPECT_TRUE(str.find("\"net_type\": \"MESH\"") != std::string::npos);
    EXPECT_TRUE(str.find("\"net_attr\": \"FULL\"") != std::string::npos);
    EXPECT_TRUE(str.find("000000000000002000100000dfdf0020") != std::string::npos);
}

TEST_F(TypesTest, RankInit) {
    aclshmemi_rank_t rank;
    int ret = aclshmemi_rank_init(rank, 0, 1);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rank.device_id, 0);
    EXPECT_EQ(rank.local_id, 1);
    EXPECT_TRUE(rank.level_list.empty());
}

TEST_F(TypesTest, RankAddNetLayer) {
    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, 0, 1);

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, "instance0");

    int ret = aclshmemi_rank_add_net_layer(rank, layer);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(rank.level_list.size(), 1);
}

TEST_F(TypesTest, RankToString) {
    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, 0, 1);

    aclshmemi_net_layer_t layer;
    aclshmemi_net_layer_init(layer, 0, "instance0");
    aclshmemi_net_layer_set_net_type(layer, "MESH");
    aclshmemi_rank_add_net_layer(rank, layer);

    std::string str = aclshmemi_rank_to_string(rank);
    EXPECT_TRUE(str.find("\"device_id\": 0") != std::string::npos);
    EXPECT_TRUE(str.find("\"local_id\": 1") != std::string::npos);
    EXPECT_TRUE(str.find("\"net_type\": \"MESH\"") != std::string::npos);
}

TEST_F(TypesTest, RootInfoInit) {
    aclshmemi_root_info_t root_info;
    int ret = aclshmemi_root_info_init(root_info);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(root_info.version, "2.0");
    EXPECT_TRUE(root_info.topo_file_path.empty());
    EXPECT_TRUE(root_info.ranks.empty());
}

TEST_F(TypesTest, RootInfoSetTopoFilePath) {
    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);

    int ret = aclshmemi_root_info_set_topo_file_path(root_info, "/usr/local/Ascend/driver/topo/950/atlas_300a.json");
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(root_info.topo_file_path, "/usr/local/Ascend/driver/topo/950/atlas_300a.json");
}

TEST_F(TypesTest, RootInfoAddRank) {
    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, 0, 1);

    int ret = aclshmemi_root_info_add_rank(root_info, rank);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(root_info.ranks.size(), 1);
}

TEST_F(TypesTest, RootInfoToString) {
    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    aclshmemi_root_info_set_topo_file_path(root_info, "/usr/local/Ascend/driver/topo/test.json");

    aclshmemi_rank_t rank;
    aclshmemi_rank_init(rank, 0, 1);
    aclshmemi_root_info_add_rank(root_info, rank);

    std::string str = aclshmemi_root_info_to_string(root_info);
    EXPECT_TRUE(str.find("\"version\": \"2.0\"") != std::string::npos);
    EXPECT_TRUE(str.find("\"topo_file_path\": \"/usr/local/Ascend/driver/topo/test.json\"") != std::string::npos);
    EXPECT_TRUE(str.find("\"rank_count\": 1") != std::string::npos);
    EXPECT_TRUE(str.find("\"device_id\": 0") != std::string::npos);
}

TEST_F(TypesTest, RootInfoGetTopoFilePath) {
    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    aclshmemi_root_info_set_topo_file_path(root_info, "/test/path.json");

    std::string path = aclshmemi_root_info_get_topo_file_path(root_info);
    EXPECT_EQ(path, "/test/path.json");
}

TEST_F(TypesTest, CompleteRootInfoSerialization) {
    aclshmemi_root_info_t root_info;
    aclshmemi_root_info_init(root_info);
    aclshmemi_root_info_set_topo_file_path(root_info, "/usr/local/Ascend/driver/topo/950/atlas_850_1.json");

    aclshmemi_rank_t rank1;
    aclshmemi_rank_init(rank1, 0, 0);
    aclshmemi_net_layer_t layer1;
    aclshmemi_net_layer_init(layer1, 0, "mesh0");
    aclshmemi_net_layer_set_net_type(layer1, "MESH");
    aclshmemi_addr_t addr1;
    dcmi_urma_eid_t eid1;
    hex32_to_bin16("000000000000002000100000dfdf0020", eid1.raw);
    aclshmemi_addr_set_eid(addr1, eid1);
    aclshmemi_addr_add_port(addr1, "4");
    aclshmemi_net_layer_add_addr(layer1, addr1);
    aclshmemi_rank_add_net_layer(rank1, layer1);
    aclshmemi_root_info_add_rank(root_info, rank1);

    aclshmemi_rank_t rank2;
    aclshmemi_rank_init(rank2, 1, 1);
    aclshmemi_net_layer_t layer2;
    aclshmemi_net_layer_init(layer2, 0, "mesh1");
    aclshmemi_net_layer_set_net_type(layer2, "MESH");
    aclshmemi_addr_t addr2;
    dcmi_urma_eid_t eid2;
    hex32_to_bin16("000000000000002000100000dfdf0028", eid2.raw);
    aclshmemi_addr_set_eid(addr2, eid2);
    aclshmemi_addr_add_port(addr2, "5");
    aclshmemi_net_layer_add_addr(layer2, addr2);
    aclshmemi_rank_add_net_layer(rank2, layer2);
    aclshmemi_root_info_add_rank(root_info, rank2);

    std::string json_str = aclshmemi_root_info_to_string(root_info);

    EXPECT_TRUE(json_str.find("\"version\": \"2.0\"") != std::string::npos);
    EXPECT_TRUE(json_str.find("\"rank_count\": 2") != std::string::npos);
    EXPECT_TRUE(json_str.find("000000000000002000100000dfdf0020") != std::string::npos);
    EXPECT_TRUE(json_str.find("000000000000002000100000dfdf0028") != std::string::npos);
    EXPECT_TRUE(json_str.find("\"net_type\": \"MESH\"") != std::string::npos);
    EXPECT_TRUE(json_str.find("\"device_id\": 0") != std::string::npos);
    EXPECT_TRUE(json_str.find("\"device_id\": 1") != std::string::npos);
}

} // namespace topo
} // namespace shm