/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "shmemi_logger.h"
#include "mstx_mem_register.h"
namespace shm {

class mstx_mem_register_empty : public mstx_mem_register_base {
public:
    mstx_mem_register_empty()
    {
        SHM_LOG_DEBUG("MSTX empty impl: mstx_mem_register created (no op)");
    }

    ~mstx_mem_register_empty() noexcept override
    {
        SHM_LOG_DEBUG("MSTX empty impl: mstx_mem_register destroyed (no op)");
    }

    void mstx_mem_regions_register(int group = 0) override
    {
        SHM_LOG_DEBUG("MSTX empty impl: mstx_mem_regions_register group=" << group);
    }

    void mstx_mem_regions_unregister(int group = 0) override
    {
        SHM_LOG_DEBUG("MSTX empty impl: mstx_mem_regions_unregister group=" << group);
    }

    void add_mem_regions(void *ptr, uint64_t size, int group = 0) override
    {
        SHM_LOG_DEBUG("MSTX empty impl: add_mem_regions group=" << group);
    }

    void add_mem_regions_multi_pe_align(void *ptr, uint64_t size, uint64_t align_size, uint32_t pe_size, int group = 0) override
    {
        SHM_LOG_DEBUG("MSTX empty impl: add_mem_regions_multi_pe_align (group=" << group 
                    << ", base ptr=" << ptr 
                    << ", size per region=" << size 
                    << ", align size=" << align_size 
                    << ", PE count=" << pe_size << ")");
    }
};

extern "C" mstx_mem_register_base* create_mstx_mem_register_instance(void)
{
    return new mstx_mem_register_empty();
}

} /* namespace shm */