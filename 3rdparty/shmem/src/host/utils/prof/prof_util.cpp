/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <iostream>
#include <string>
#include "acl/acl.h"
#include "utils/shmemi_logger.h"
#include "host/shmem_host_def.h"
#include "prof_util.h"

int32_t prof_get_col_pe(char *tmp_pe)
{
    long pe_id = std::strtol(tmp_pe, nullptr, 10L);
    if (pe_id >= 0 && pe_id < ACLSHMEM_MAX_PES) {
        return pe_id;
    } else {
        SHM_LOG_WARN("invalid input tmp_pe: " << pe_id << " , set default pe_id to 0.");
        pe_id = 0;
    }

    SHM_LOG_INFO("set prof rank " << pe_id);
    return static_cast<int32_t>(pe_id);
}

int32_t prof_util_init(aclshmem_prof_pe_t *host_profs, aclshmem_device_host_state_t *global_state)
{
    char *tmp_pe = std::getenv("SHMEM_CYCLE_PROF_PE");
    if (tmp_pe == nullptr) {
        host_profs->pe_id = -1;
        SHM_LOG_INFO("no prof rank set");
        return ACLSHMEM_SUCCESS;
    }

    bzero(host_profs, sizeof(aclshmem_prof_pe_t));
    host_profs->pe_id = prof_get_col_pe(tmp_pe);
    if (host_profs->pe_id != global_state->mype) {
        SHM_LOG_INFO("not prof rank " << host_profs->pe_id << ", skip init.");
        return ACLSHMEM_SUCCESS;
    }

    void *prof_ptr = nullptr;
    auto ret = aclrtMalloc(&prof_ptr, sizeof(aclshmem_prof_pe_t), ACL_MEM_MALLOC_HUGE_FIRST);
    if (ret != 0) {
        SHM_LOG_ERROR("aclrtMalloc rank prof mem failed, ret: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    auto copy_size = sizeof(aclshmem_prof_pe_t);
    ret = aclrtMemcpy(prof_ptr, copy_size, host_profs, copy_size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != 0) {
        SHM_LOG_ERROR("aclrtMemcpy rank prof info failed, ret: " << ret);
        if (aclrtFree(prof_ptr) != 0) {
            SHM_LOG_ERROR("aclrtFree rank prof info failed.");
        }
        prof_ptr = nullptr;
        return ACLSHMEM_INNER_ERROR;
    }

    global_state->profs = (aclshmem_prof_pe_t *)prof_ptr;
    SHM_LOG_INFO("shmemi prof init succuss, pe=" << host_profs->pe_id << ", block=" << ACLSHMEM_CYCLE_PROF_MAX_BLOCK << ", prof_ptr: " << prof_ptr << ", size: " << copy_size);
    return ACLSHMEM_SUCCESS;
}

void prof_data_print(aclshmem_prof_pe_t *host_profs, aclshmem_device_host_state_t *global_state, 
                     aclshmem_prof_pe_t **out_profs, bool verbose)
{
    if (host_profs->pe_id != global_state->mype) {
        SHM_LOG_INFO("not collect profiling data on this rank, skip print.");
        return;
    }
    
    if (out_profs != nullptr) {
        *out_profs = host_profs;
    }
    
    auto copySize = sizeof(aclshmem_prof_pe_t);
    auto ret = aclrtMemcpy(host_profs, copySize, global_state->profs, copySize, ACL_MEMCPY_DEVICE_TO_HOST);

    if (ret != 0) {
        std::cout << "copy device profs to host failed " << ret << std::endl;
        return;
    }


    if (!verbose) {
        return;
    }
    const char *soc_name = aclrtGetSocName();
    int64_t cycle2us = 50;
    std::string soc_str;
    if (soc_name != nullptr) {
        soc_str = std::string(soc_name);
        if (soc_str.find("Ascend950") != std::string::npos) {
            cycle2us = 1000;
        }
    }
    SHM_LOG_INFO("SocName: " << soc_str << ", cycle2us: " << cycle2us);
    const int width_1 = 10;
    const int width_2 = 15;
    std::cout << std::fixed << std::setprecision(3);

    std::cout << "============================================================" << std::endl;
    std::cout << std::left
              << std::setw(width_1) << "BlockID"
              << std::setw(width_1) << "FrameID"
              << std::setw(width_2) << "Cycles"
              << std::setw(width_2) << "Count"
              << std::setw(width_2) << "AvgTime(us)" << std::endl;
    std::cout << "------------------------------------------------------------" << std::endl;

    for (int32_t block_id = 0; block_id < ACLSHMEM_CYCLE_PROF_MAX_BLOCK; block_id++) {
        aclshmem_prof_block_t* prof = &host_profs->block_prof[block_id];

        for (int32_t frame_id = 0; frame_id < ACLSHMEM_CYCLE_PROF_FRAME_CNT; frame_id++) {
            if (prof->ccount[frame_id] == 0) {
                continue;
            }

            double avg_us = (double)prof->cycles[frame_id] / prof->ccount[frame_id] / cycle2us;

            std::cout << std::left
                      << std::setw(width_1) << block_id
                      << std::setw(width_1) << frame_id
                      << std::setw(width_2) << prof->cycles[frame_id]
                      << std::setw(width_2) << prof->ccount[frame_id]
                      << std::setw(width_2) << avg_us << std::endl;
        }
    }

    std::cout << "============================================================" << std::endl;
    std::cout << std::resetiosflags(std::ios::fixed) << std::right;
}