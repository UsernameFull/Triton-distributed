/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mstx_mem_register.h"
#include <unistd.h>
#include <syscall.h>
#include <mstx/ms_tools_ext.h>
#include <mstx/ms_tools_ext_mem.h>
#include "acl/acl.h"
#include "shmemi_logger.h"

#define CHECK_MSTX_ENABLE() \
    do { \
        if (!is_mstx_enable()) { \
            SHM_LOG_INFO("MSTX tool not supported"); \
            return; \
        } \
    } while (0)

namespace shm {

class mstx_mem_register_real : public mstx_mem_register_base {
public:
    mstx_mem_register_real() :
        ranges_desc_(MSTX_REGISTER_GROUP_NUM),
        region_handles_(MSTX_REGISTER_GROUP_NUM)
    {
        init_domain();

        if (domain_ != nullptr) {
            SHM_LOG_INFO("mstx_mem_register created, domain init success");
        } else {
            SHM_LOG_INFO("mstx_mem_register created, domain init failed");
        }
    }

    ~mstx_mem_register_real() noexcept override
    {
        CHECK_MSTX_ENABLE();
        mstx_mem_regions_unregister(-1);
        clear_mstx_mem_regions(-1);
    }

    void mstx_mem_regions_register(int group = 0) override
    {
        CHECK_MSTX_ENABLE();
        if (group == -1) {
            for (int i = 0; i < MSTX_REGISTER_GROUP_NUM; ++i) {
                mstx_mem_regions_register(i);
            }
            return;
        }

        if (group < 0 || group >= MSTX_REGISTER_GROUP_NUM) {
            SHM_LOG_ERROR("Invalid group index: " << group);
            return;
        }

        auto& target_ranges = ranges_desc_[group];
        auto& target_handles = region_handles_[group];

        if (target_ranges.empty()) {
            SHM_LOG_INFO("No regions to register for group=" << group);
            return;
        }

        SHM_LOG_INFO("Register group " << group << ", total regions: " << target_ranges.size());
        for (size_t i = 0; i < target_ranges.size(); ++i) {
            const auto& region = target_ranges[i];
            SHM_LOG_INFO("  Region " << i
                         << ": ptr=" << region.ptr
                         << ", size=" << region.size
                         << ", deviceId=" << static_cast<unsigned int>(region.deviceId));
        }

        target_handles.resize(target_ranges.size());

        mstxMemRegionsRegisterBatch_t regions_desc = {0};
        regions_desc.heap = nullptr;
        regions_desc.regionType = MSTX_MEM_TYPE_VIRTUAL_ADDRESS;
        regions_desc.regionCount = target_ranges.size();
        regions_desc.regionDescArray = target_ranges.data();
        regions_desc.regionHandleArrayOut = target_handles.data();
        mstxMemRegionsRegister(domain_, &regions_desc);
        SHM_LOG_INFO("Registered " << target_ranges.size() << " regions for group " << group);

        std::vector<mstxMemPermissionsAssignRegionsDesc_t> permAssignDesc(target_handles.size());
        for (size_t i = 0; i < target_handles.size(); ++i) {
            permAssignDesc[i].flags = MSTX_MEM_PERMISSIONS_REGION_FLAGS_SHARED;
            permAssignDesc[i].region.refType = MSTX_MEM_REGION_REF_TYPE_HANDLE;
            permAssignDesc[i].region.handle = target_handles[i];
        }

        mstxMemPermissionsAssignBatch_t permAssignBatch{};
        permAssignBatch.regionCount = permAssignDesc.size();
        permAssignBatch.regionDescArray = permAssignDesc.data();
        mstxMemPermissionsAssign(domain_, &permAssignBatch);
        SHM_LOG_INFO("Registered " << permAssignDesc.size() << " permissions for group " << group);
    }

    void mstx_mem_regions_unregister(int group = 0) override
    {
        CHECK_MSTX_ENABLE();
        if (group == -1) {
            for (int i = 0; i < MSTX_REGISTER_GROUP_NUM; ++i) {
                mstx_mem_regions_unregister(i);
            }
            return;
        }

        if (group < 0 || group >= MSTX_REGISTER_GROUP_NUM) {
            SHM_LOG_ERROR("Invalid group index: " << group);
            return;
        }

        auto& target_ranges = ranges_desc_[group];
        auto& target_handles = region_handles_[group];

        if (target_handles.empty()) {
            SHM_LOG_INFO("No regions to unregister for group=" << group);
            return;
        }

        size_t arr_size = target_handles.size();
        mstxMemRegionRef_t *region_ref = new mstxMemRegionRef_t[arr_size]();
        for (size_t i = 0; i < target_handles.size(); ++i) {
            region_ref[i].refType = MSTX_MEM_REGION_REF_TYPE_HANDLE;
            region_ref[i].handle = target_handles[i];
        }

        mstxMemRegionsUnregisterBatch_t refs_desc = {0};
        refs_desc.refCount = target_handles.size();
        refs_desc.refArray = region_ref;
        mstxMemRegionsUnregister(domain_, &refs_desc);
        
        delete[] region_ref;
        region_ref = nullptr;

        target_handles.clear();
        SHM_LOG_INFO("Unregistered " << target_ranges.size() << " regions for group " << group);
    }

    void add_mem_regions(void *ptr, uint64_t size, int group = 0) override
    {
        CHECK_MSTX_ENABLE();
        if (group < 0 || group >= MSTX_REGISTER_GROUP_NUM) {
            SHM_LOG_ERROR("Invalid group index: " << group << " when adding regions");
            return;
        }

        SHM_LOG_INFO("Add mem region: group=" << group 
                     << ", ptr=" << ptr
                     << ", size=" << size);

        if (get_mstx_device() >= 0) {
            mstxMemVirtualRangeDesc_t region_info = {0};
            region_info.deviceId = static_cast<uint32_t>(get_mstx_device());
            region_info.ptr = ptr;
            region_info.size = size;
            ranges_desc_[group].push_back(region_info);
        }
    }

    void add_mem_regions_multi_pe_align(void *ptr, uint64_t size, uint64_t align_size, uint32_t pe_size, int group = 0) override
    {
        CHECK_MSTX_ENABLE();
        if (group < 0 || group >= MSTX_REGISTER_GROUP_NUM) {
            SHM_LOG_ERROR("Invalid group index: " << group << " when adding multi-PE aligned regions");
            return;
        }
        if (pe_size == 0) {
            SHM_LOG_WARN("PE size is 0, skip adding multi-PE aligned regions (group=" << group << ")");
            return;
        }
        if (align_size < size) {
            SHM_LOG_ERROR("Align size (" << align_size << ") cannot be less than region size (" << size 
                        << ") for multi-PE aligned regions (group=" << group << ")");
            return;
        }

        int32_t device_id = get_mstx_device();
        if (device_id < 0) {
            SHM_LOG_ERROR("Get MSTX device failed, skip adding multi-PE aligned regions (group=" << group << ")");
            return;
        }

        SHM_LOG_INFO("Add multi-PE aligned mem regions: group=" << group 
                    << ", base ptr=" << ptr
                    << ", size per region=" << size 
                    << ", align size=" << align_size 
                    << ", PE count=" << pe_size);

        for (uint32_t pe_idx = 0; pe_idx < pe_size; ++pe_idx) {
            void *current_ptr = (void *)((uintptr_t)ptr + (pe_idx * align_size));

            mstxMemVirtualRangeDesc_t region_info = {0};
            region_info.deviceId = static_cast<uint32_t>(device_id);
            region_info.ptr = current_ptr;
            region_info.size = size;

            ranges_desc_[group].push_back(region_info);

            SHM_LOG_INFO("PE " << pe_idx 
                        << ": ptr=" << current_ptr
                        << ", size=" << size);
        }

        SHM_LOG_INFO("Added total " << pe_size << " multi-PE aligned regions (group=" << group << ")");
    }

private:
    static constexpr int MSTX_REGISTER_GROUP_NUM = 3;

    void init_domain()
    {
        domain_ = mstxDomainCreateA("shmem");
        if (domain_ == nullptr) {
            is_mstx_enable_ = false;
            SHM_LOG_ERROR("mstxDomainCreateA failed, MSTX tool not supported");
        } else {
            is_mstx_enable_ = true;
        }
    }

    bool is_mstx_enable() const
    {
        return is_mstx_enable_ && (domain_ != nullptr);
    }

    void clear_mstx_mem_regions(int group = 0) noexcept
    {
        CHECK_MSTX_ENABLE();
        if (group == -1) {
            std::vector<std::vector<mstxMemVirtualRangeDesc_t>>().swap(ranges_desc_);
            std::vector<std::vector<mstxMemRegionHandle_t>>().swap(region_handles_);
            SHM_LOG_INFO("Cleared all groups");
            return;
        }

        if (group < 0 || group >= MSTX_REGISTER_GROUP_NUM) {
            SHM_LOG_ERROR("Invalid group index: " << group);
            return;
        }

        std::vector<mstxMemVirtualRangeDesc_t>().swap(ranges_desc_[group]);
        std::vector<mstxMemRegionHandle_t>().swap(region_handles_[group]);
        SHM_LOG_INFO("Cleared group " << group);
    }

    int32_t get_mstx_device()
    {
        int32_t device_id = -1;
        aclError ret = aclrtGetDevice(&device_id);
        if (ret != ACL_SUCCESS) {
            SHM_LOG_ERROR("get device id error! acl error: " << ret);
            device_id = -1;
        }
        return device_id;
    }

    bool is_mstx_enable_ = true;
    mstxDomainHandle_t domain_ = nullptr;
    std::vector<std::vector<mstxMemVirtualRangeDesc_t>> ranges_desc_;
    std::vector<std::vector<mstxMemRegionHandle_t>> region_handles_;
};

extern "C" mstx_mem_register_base* create_mstx_mem_register_instance(void)
{
    return new mstx_mem_register_real();
}

} /* namespace shm */
