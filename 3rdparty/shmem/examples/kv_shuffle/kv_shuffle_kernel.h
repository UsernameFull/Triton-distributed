/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef KVSHUFFLE_H
#define KVSHUFFLE_H

class KVShuffleOps {
public:
    // 默认构造函数
    KVShuffleOps(uint32_t block_dims, void* stream);

    ~KVShuffleOps();
    
    // 接受 Tensor 的函数
    void compute(
        uint8_t* k_cache,
        uint8_t* v_cache,
        uint8_t* global_shuffle_table,
        uint8_t* src_block_table,
        uint8_t* dst_block_table,
        int64_t block_nums,
        int64_t kv_head_num, int64_t page_size, int64_t head_dim);
private:
    void* sync_ptr_;
    int32_t count_;
    uint32_t block_dims_;
    void* stream_;
    uint64_t fftsAddr_;
};

#endif // KVSHUFFLE_H