/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef PCIE_NIC_H
#define PCIE_NIC_H

#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

int get_npu_roce_ip(int phy_id, char* ip_addr, size_t addr_len);

int get_ip_by_if_name(const char* ifname, char* ip, size_t addr_len);

#ifdef __cplusplus
}
#endif

#endif // PCIE_NIC_H
