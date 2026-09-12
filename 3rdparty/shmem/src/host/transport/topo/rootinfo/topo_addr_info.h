/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_ADDR_INFO_H
#define TOPO_ADDR_INFO_H

#include <cstdio>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * 获取topo addr info的大小，用于提前分配内存
 * param[in] phy_id   NPU物理ID
 * param[out] size   rankinfo的大小, 注意不是精确大小，能保证装下rankinfo
 */
int topo_addr_info_get_size(int phy_id, size_t* size);

/**
 * 获取拓扑文件的路径
 * param[in] phy_id   NPU物理ID
 * param[out] file_path  拓扑文件的路径
 * param[out] buf_size   拓扑文件路径的最大长度
 */

int topo_addr_info_get_topo_file_path(int phy_id, char* file_path, size_t buf_size);
/**
 * 获取rankinfo的内容，用于集合通信感知每层组网的地址信息
 * param[in] phy_id   NPU物理ID
 * param[out] rank_info   rankinfo的内容，为json格式的字符串
 * param[out] buf_size   实际大小
 */
int topo_addr_info_get(int phy_id, char* rank_info, size_t* buf_size);

/**
 * 获取所有NPU的rootinfo信息，包含所有phy_id的rank信息
 * param[out] rank_info   所有NPU的rootinfo内容，为json格式的字符串
 * param[out] buf_size   实际大小
 *
 * JSON格式示例:
 * {
 *   "version": "2.0",
 *   "topo_file_path": "/path/to/topo",
 *   "rank_count": 8,  // 所有phy_id的rank总数
 *   "rank_list": [    // 包含所有phy_id的rank信息
 *     {"device_id": 0, "local_id": 0, ...},
 *     {"device_id": 1, "local_id": 1, ...},
 *     ...
 *   ]
 * }
 */
int topo_addr_info_get_all(char* rank_info, size_t* buf_size);

#ifdef __cplusplus
}
#endif

#endif // TOPO_ADDR_INFO_H
