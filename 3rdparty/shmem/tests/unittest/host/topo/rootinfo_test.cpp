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
#include <gmock/gmock.h>
#include <cstring>
#include <cstdlib>
#include <string>
#include "topo_addr_info.h"
#include "mock_hal.h"
#include "aclshmemi_hal.h"

namespace shm {
namespace topo {

using testing::Return;
using testing::_;
using testing::Invoke;
using testing::NiceMock;

class RootInfoTest : public testing::Test {
protected:
    void SetUp() override {
        testing::Mock::AllowLeak(&MockHal::instance());
    }
    
    void TearDown() override {
        testing::Mock::VerifyAndClearExpectations(&MockHal::instance());
    }
};

TEST_F(RootInfoTest, GetSizeInvalidParams) {
    size_t size = 0;
    int ret = topo_addr_info_get_size(0, nullptr);
    EXPECT_EQ(ret, -1);
}

TEST_F(RootInfoTest, GetTopoFilePathInvalidParams) {
    char file_path[256];
    int ret = topo_addr_info_get_topo_file_path(0, nullptr, 256);
    EXPECT_EQ(ret, -1);
}

TEST_F(RootInfoTest, GetInvalidParams) {
    char rank_info[4096];
    size_t buf_size = 4096;
    int ret = topo_addr_info_get(0, nullptr, &buf_size);
    EXPECT_EQ(ret, -1);

    ret = topo_addr_info_get(0, rank_info, nullptr);
    EXPECT_EQ(ret, -1);
}

TEST_F(RootInfoTest, GetSizeWithMock) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::optional<uint32_t>(0x6c)));
    
    size_t size = 0;
    int ret = topo_addr_info_get_size(0, &size);
    EXPECT_EQ(ret, 0);
    EXPECT_EQ(size, 2048);
}

TEST_F(RootInfoTest, GetSizeNoDevice) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::nullopt));
    
    size_t size = 0;
    int ret = topo_addr_info_get_size(0, &size);
    EXPECT_EQ(ret, -1);
}

TEST_F(RootInfoTest, GetTopoFilePathWithMock) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::optional<uint32_t>(0x6c)));
    
    char file_path[256];
    int ret = topo_addr_info_get_topo_file_path(0, file_path, 256);
    EXPECT_EQ(ret, 0);
}

TEST_F(RootInfoTest, GetTopoFilePathNoDevice) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::nullopt));
    
    char file_path[256];
    int ret = topo_addr_info_get_topo_file_path(0, file_path, 256);
    EXPECT_EQ(ret, -1);
}

TEST_F(RootInfoTest, GetWithMockCard4PMesh) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::optional<uint32_t>(0x6c)));
    EXPECT_CALL(MockHal::instance(), get_ue_list(_, _))
        .WillOnce(Invoke([](int phy_id, UEList* ue_list) {
            memset(ue_list, 0, sizeof(UEList));
            ue_list->ueNum = 1;
            ue_list->ueList[0].eidNum = 3;
            uint8_t eids[][16] = {
                {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x20,0x00,0x10,0x00,0x00,0xdf,0xdf,0x00,0x20},
                {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x20,0x00,0x10,0x00,0x00,0xdf,0xdf,0x00,0x28},
                {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x20,0x00,0x10,0x00,0x00,0xdf,0xdf,0x00,0x30},
            };
            for (int i = 0; i < 3; i++) {
                memcpy(ue_list->ueList[0].eidList[i].eid.raw, eids[i], 16);
            }
            return 0;
        }));
    
    char rank_info[4096];
    size_t buf_size = 4096;
    int ret = topo_addr_info_get(0, rank_info, &buf_size);
    EXPECT_EQ(ret, 0);
}

TEST_F(RootInfoTest, GetNoDevice) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::nullopt));
    
    char rank_info[4096];
    size_t buf_size = 4096;
    int ret = topo_addr_info_get(0, rank_info, &buf_size);
    EXPECT_EQ(ret, -1);
}

TEST_F(RootInfoTest, GetBufferTooSmall) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::optional<uint32_t>(0x6c)));
    
    char rank_info[10];
    size_t buf_size = 10;
    int ret = topo_addr_info_get(0, rank_info, &buf_size);
    EXPECT_EQ(ret, -1);
}

TEST_F(RootInfoTest, GetWithServer8PMesh) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::optional<uint32_t>(0x25)));
    EXPECT_CALL(MockHal::instance(), get_ue_list(_, _))
        .WillOnce(Invoke([](int phy_id, UEList* ue_list) {
            memset(ue_list, 0, sizeof(UEList));
            ue_list->ueNum = 4;
            ue_list->ueList[0].eidNum = 1;
            ue_list->ueList[1].eidNum = 2;
            ue_list->ueList[2].eidNum = 1;
            ue_list->ueList[3].eidNum = 2;
            return 0;
        }));
    EXPECT_CALL(MockHal::instance(), get_spod_info(_, _))
        .WillOnce(Invoke([](int phy_id, struct dcmi_spod_info* spod_info) {
            memset(spod_info, 0, sizeof(dcmi_spod_info));
            spod_info->sdid = 0;
            spod_info->super_pod_size = 128;
            spod_info->super_pod_id = 1;
            spod_info->server_index = 1;
            return 0;
        }));
    
    char rank_info[4096];
    size_t buf_size = 4096;
    int ret = topo_addr_info_get(0, rank_info, &buf_size);
    EXPECT_EQ(ret, 0);
}

TEST_F(RootInfoTest, GetWithPod) {
    EXPECT_CALL(MockHal::instance(), get_mainboard_id(_))
        .WillOnce(Return(std::optional<uint32_t>(0x07)));
    EXPECT_CALL(MockHal::instance(), get_ue_list(_, _))
        .WillOnce(Invoke([](int phy_id, UEList* ue_list) {
            memset(ue_list, 0, sizeof(UEList));
            ue_list->ueNum = 2;
            ue_list->ueList[0].eidNum = 1;
            ue_list->ueList[1].eidNum = 1;
            return 0;
        }));
    EXPECT_CALL(MockHal::instance(), get_spod_info(_, _))
        .WillOnce(Invoke([](int phy_id, struct dcmi_spod_info* spod_info) {
            memset(spod_info, 0, sizeof(dcmi_spod_info));
            spod_info->sdid = 0;
            spod_info->super_pod_size = 128;
            spod_info->super_pod_id = 1;
            spod_info->server_index = 1;
            return 0;
        }));
    
    char rank_info[4096];
    size_t buf_size = 4096;
    int ret = topo_addr_info_get(0, rank_info, &buf_size);
    EXPECT_EQ(ret, 0);
}

} // namespace topo
} // namespace shm