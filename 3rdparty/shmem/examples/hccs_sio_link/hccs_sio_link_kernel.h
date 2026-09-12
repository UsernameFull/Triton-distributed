/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef HCCS_SIO_LINK_KERNEL_H
#define HCCS_SIO_LINK_KERNEL_H

#include <cstdint>

template <typename T>
void launch_mte_get(uint32_t block_dim, void *stream, uint8_t *local_buf, uint8_t *src_addr,
                    int elements, int peer_pe, int ub_size_kb);

template <typename T>
void launch_mte_get_hccs(uint32_t block_dim, void *stream, uint8_t *local_buf, uint8_t *hccs_src,
                          int elements, int ub_size_kb);

template <typename T>
void launch_mte_get_mixed(uint32_t block_dim, void *stream,
                           uint8_t *sio_dst, uint8_t *sio_src, int sio_elements, int peer_pe,
                           uint8_t *hccs_dst, uint8_t *hccs_src, int hccs_elements,
                           int ub_size_kb);

#endif
