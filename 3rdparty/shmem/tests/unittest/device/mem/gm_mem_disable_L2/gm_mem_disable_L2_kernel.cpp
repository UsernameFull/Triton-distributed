/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#undef L2_CACHE_HINT
#include "kernel_operator.h"

#include "shmem.h"

const int length = 16;
const int ub_size = 256;
const int buffer_bytes = 4096;
const int capaticy = 2;

class kernel_shmemx_mte_put_num {
public:
    __aicore__ inline kernel_shmemx_mte_put_num() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ int8_t *)gva;
        dev_gm = (__gm__ int8_t *)dev;

        /* set GM Buffer */
        src_gm.SetGlobalBuffer(dev_gm);
        dst_gm.SetGlobalBuffer(gva_gm);

        rank = aclshmem_my_pe();
        rank_size = aclshmem_n_pes();

        /* 1x4096 Bytes Buffer */
        pipe.InitBuffer(buf_queue, 1, buffer_bytes);
    }
    __aicore__ inline void Process(uint64_t config)
    {
        util_set_ffts_config(config);
        AscendC::LocalTensor<int8_t> buf_tensor = buf_queue.AllocTensor<int8_t>();
        uintptr_t addr = static_cast<uintptr_t>(buf_tensor.address_.bufferAddr);
        __ubuf__ int8_t *buf = (__ubuf__ int8_t *)addr;
        aclshmemx_mte_put_nbi(gva_gm, dev_gm, buf, (uint32_t)ub_size, rank_size * length / 4U, rank, EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        aclshmemx_mte_put_nbi(dst_gm[rank_size * length / 4U], src_gm[rank_size * length / 4U], buf_tensor,
            rank_size * length / 4U, rank, EVENT_ID0);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        aclshmemx_mte_put_nbi(gva_gm + rank_size * length / 2U, dev_gm + rank_size * length / 2U,
            rank_size * length / 4U, rank, false);
        AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        aclshmemx_mte_put_nbi(gva_gm + rank_size * length * 3U / 4U, dev_gm + rank_size * length * 3U / 4U,
            rank_size * length / 4U, rank, false);
        aclshmemx_barrier_all_vec();
        buf_queue.FreeTensor(buf_tensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, capaticy> buf_queue;

    __gm__ int8_t *gva_gm;
    __gm__ int8_t *dev_gm;
    AscendC::GlobalTensor<int8_t> src_gm, dst_gm;

    int64_t rank;
    int64_t rank_size;
};


extern "C" __global__ __aicore__ void put_shmemx_mte_num_test(GM_ADDR gva, GM_ADDR dev, uint64_t config)
{
    kernel_shmemx_mte_put_num op;
    op.Init(gva, dev);
    op.Process(config);
}

void test_shmemx_mte_put(uint32_t block_dim, void* stream, uint64_t config, uint8_t* gva, uint8_t* dev)
{
    put_shmemx_mte_num_test<<<block_dim, nullptr, stream>>>(gva, dev, config);
}


class kernel_shmemx_mte_get_num {
public:
    __aicore__ inline kernel_shmemx_mte_get_num() {}
    __aicore__ inline void Init(GM_ADDR gva, GM_ADDR dev)
    {
        gva_gm = (__gm__ int8_t *)gva;
        dev_gm = (__gm__ int8_t *)dev;

        /* set GM Buffer */
        src_gm.SetGlobalBuffer(gva_gm);
        dst_gm.SetGlobalBuffer(dev_gm);

        rank = aclshmem_my_pe();
        rank_size = aclshmem_n_pes();

        /* 1x4096 Bytes Buffer */
        pipe.InitBuffer(buf_queue, 1, buffer_bytes);
    }
    __aicore__ inline void Process(uint64_t config)
    {
        util_set_ffts_config(config);
        AscendC::LocalTensor<int8_t> buf_tensor = buf_queue.AllocTensor<int8_t>();
        uintptr_t addr = static_cast<uintptr_t>(buf_tensor.address_.bufferAddr);
        __ubuf__ int8_t *buf = (__ubuf__ int8_t *)addr;

        for (int i = 0; i < rank_size / 2U; i++) {
            aclshmemx_mte_get_nbi(dev_gm + length * i, gva_gm, buf, (uint32_t)ub_size,
                length / 2U, i % rank_size, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            aclshmemx_mte_get_nbi(dst_gm[length * i + length / 2U], src_gm, buf_tensor,
                length / 2U, i % rank_size, EVENT_ID0);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        for (int i = rank_size / 2U; i < rank_size; i++) {
            aclshmemx_mte_get_nbi(dev_gm + length * i, gva_gm, length, i % rank_size, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            aclshmemx_mte_get_nbi(dev_gm + length * i + length / 2U, gva_gm, length / 2U, i % rank_size, true);
            AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
            AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(EVENT_ID0);
        }

        aclshmemx_barrier_all_vec();
        buf_queue.FreeTensor(buf_tensor);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 2U> buf_queue;
    __gm__ int8_t *gva_gm;
    __gm__ int8_t *dev_gm;
    AscendC::GlobalTensor<int8_t> src_gm, dst_gm;

    int64_t rank;
    int64_t rank_size;
};

extern "C" __global__ __aicore__ void get_shmemx_mte_num_test(GM_ADDR gva, GM_ADDR dev, uint64_t config)
{
    kernel_shmemx_mte_get_num op;
    op.Init(gva, dev);
    op.Process(config);
}

void test_shmemx_mte_get(uint32_t block_dim, void* stream, uint64_t config, uint8_t* gva, uint8_t* dev)
{
    get_shmemx_mte_num_test<<<block_dim, nullptr, stream>>>(gva, dev, config);
}
