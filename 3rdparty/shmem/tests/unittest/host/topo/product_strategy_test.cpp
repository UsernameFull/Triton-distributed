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
#include <memory>
#include "aclshmemi_product_strategy.h"

namespace shm {
namespace topo {

class ProductStrategyTest : public testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(ProductStrategyTest, CreateCardNoMesh) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 2048);
}

TEST_F(ProductStrategyTest, CreateCard2PMesh) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 2048);
}

TEST_F(ProductStrategyTest, CreateCard4PMesh) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 2048);
}

TEST_F(ProductStrategyTest, CreateServer8PMesh) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 2048);
}

TEST_F(ProductStrategyTest, CreatePod) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_POD);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 2048);
}

TEST_F(ProductStrategyTest, CreatePod2D) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_POD_2D);
    EXPECT_TRUE(strategy != nullptr);
    EXPECT_EQ(strategy->get_root_info_size(), 2048);
}

TEST_F(ProductStrategyTest, CreateInvalidMainboardId) {
    auto strategy = aclshmemi_product_strategy_t::create(0xFFFFFFFF);
    EXPECT_TRUE(strategy == nullptr);
}

TEST_F(ProductStrategyTest, CreateUnknownMainboardId) {
    auto strategy = aclshmemi_product_strategy_t::create(0x99);
    EXPECT_TRUE(strategy == nullptr);
}

TEST_F(ProductStrategyTest, RootInfoSizeAllTypes) {
    uint32_t mainboard_ids[] = {
        ACLSHMEMI_MAIN_BOARD_ID_CARD_NOMESH,
        ACLSHMEMI_MAIN_BOARD_ID_CARD_2PMESH,
        ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH,
        ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH,
        ACLSHMEMI_MAIN_BOARD_ID_POD,
        ACLSHMEMI_MAIN_BOARD_ID_POD_2D
    };

    for (auto id : mainboard_ids) {
        auto strategy = aclshmemi_product_strategy_t::create(id);
        if (strategy) {
            EXPECT_EQ(strategy->get_root_info_size(), 2048);
        }
    }
}

TEST_F(ProductStrategyTest, CardProductTypeCheck) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_CARD_4PMESH);
    EXPECT_TRUE(strategy != nullptr);
    
    aclshmemi_card_product_t* card_strategy = dynamic_cast<aclshmemi_card_product_t*>(strategy.get());
    EXPECT_TRUE(card_strategy != nullptr);
}

TEST_F(ProductStrategyTest, ServerProductTypeCheck) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_SERVER_8PMESH);
    EXPECT_TRUE(strategy != nullptr);
    
    aclshmemi_server_product_t* server_strategy = dynamic_cast<aclshmemi_server_product_t*>(strategy.get());
    EXPECT_TRUE(server_strategy != nullptr);
}

TEST_F(ProductStrategyTest, PodProductTypeCheck) {
    auto strategy = aclshmemi_product_strategy_t::create(ACLSHMEMI_MAIN_BOARD_ID_POD);
    EXPECT_TRUE(strategy != nullptr);
    
    aclshmemi_pod_product_t* pod_strategy = dynamic_cast<aclshmemi_pod_product_t*>(strategy.get());
    EXPECT_TRUE(pod_strategy != nullptr);
}

} // namespace topo
} // namespace shm