/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "shmemi_logger.h"
#include "mem_entity_inter.h"
#include "dl_api.h"
#include "dl_hal_api.h"
#include "dl_acl_api.h"
#include "dl_comm_def.h"
#include "devmm_svm_gva.h"
#include "devmm_cmd.h"
#include "hybm_ex_info_transfer.h"
#include "shmemi_file_util.h"
#include "mem_entity_factory.h"
#include "mem_entity_def.h"
#include "acl/acl.h"
#include "acl/acl_rt.h"

#include "runtime/kernel.h"
#include "runtime/mem.h"
#include "runtime/dev.h"
#include "runtime/rt_ffts.h"

#include "mem_entity_entry.h"

using namespace shm;

namespace {
static uint64_t g_baseAddr = 0ULL;
bool initialized = false;
int32_t inited_device_id = -1;
int32_t initedLogicDeviceId = -1;
drv_mem_handle_t *alloc_handle = nullptr;
AscendSocType soc_type = ASCEND_UNKNOWN;

std::mutex initMutex;
}

int32_t HybmGetInitDeviceId()
{
    return inited_device_id;
}

static inline int hybm_load_library()
{
    char *path = std::getenv("ASCEND_HOME_PATH");
    SHM_VALIDATE_RETURN(path != nullptr, "Environment ASCEND_HOME_PATH not set.", ACLSHMEM_INNER_ERROR);

    std::string libPath = std::string(path).append("/lib64");
    
    if (!shm::utils::FileUtil::Realpath(libPath) || !shm::utils::FileUtil::IsDir(libPath)) {
        SHM_LOG_ERROR("Environment ASCEND_HOME_PATH check failed: realpath or isdir validation failed.");
        return ACLSHMEM_INNER_ERROR;
    }
    
    if (!shm::utils::FileUtil::IsOwnedByCurrentUserOrRoot(libPath)) {
        SHM_LOG_ERROR("Security check failed: ASCEND_HOME_PATH/lib64 is not owned by current user or root.");
        return ACLSHMEM_INNER_ERROR;
    }
    
    if (!shm::utils::FileUtil::HasSecurePermissions(libPath)) {
        SHM_LOG_ERROR("Security check failed: ASCEND_HOME_PATH/lib64 has insecure permissions.");
        return ACLSHMEM_INNER_ERROR;
    }
    
    auto ret = DlApi::LoadLibrary(libPath);
    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(ret, "load library from path failed: " << ret);
    return 0;
}

static inline int32_t init_meta_memory_for_modern(void** globalMemoryBase, size_t allocSize)
{
    size_t alignSize = ALIGN_UP(allocSize, DEVMM_HEAP_SIZE);
    uint64_t va = (HYBM_DEVICE_VA_START + HYBM_DEVICE_VA_SIZE - DEVMM_HEAP_SIZE) - alignSize;
    auto ret = DlHalApi::HalMemAddressReserve(globalMemoryBase, alignSize, 0, reinterpret_cast<void *>(va), 0);
    if (ret != 0) {
        DlApi::CleanupLibrary();
        SHM_LOG_ERROR("prepare virtual memory size(" << alignSize << ") failed. ret: " << ret);
        return ACLSHMEM_MALLOC_FAILED;
    }
    drv_mem_prop memprop;
    memprop.side = MEM_DEV_SIDE;
    memprop.devid = initedLogicDeviceId;
    memprop.module_id = 0;
    memprop.pg_type = MEM_NORMAL_PAGE_TYPE;
    memprop.mem_type = MEM_HBM_TYPE;
    memprop.reserve = 0;
    ret = DlHalApi::HalMemCreate(&alloc_handle, allocSize, &memprop, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        DlApi::CleanupLibrary();
        SHM_LOG_ERROR("HalMemCreate failed: " << ret);
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    ret = DlHalApi::HalMemMap(reinterpret_cast<void *>(HYBM_DEVICE_META_ADDR), allocSize, 0, alloc_handle, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        DlApi::CleanupLibrary();
        SHM_LOG_ERROR("HalMemMap failed: " << ret);
        DlHalApi::HalMemRelease(alloc_handle);
        DlHalApi::HalMemAddressFree(reinterpret_cast<void *>(*globalMemoryBase));
        alloc_handle = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }
    return ACLSHMEM_SUCCESS;
}


static inline int32_t init_meta_memory_for_legacy(void** globalMemoryBase, size_t allocSize, uint64_t flags)
{
    drv::DevmmInitialize(initedLogicDeviceId, DlHalApi::GetFd());
    auto ret = drv::HalGvaReserveMemory((uint64_t *)globalMemoryBase, allocSize, initedLogicDeviceId, flags);
    if (ret != ACLSHMEM_SUCCESS) {
        DlApi::CleanupLibrary();
        SHM_LOG_ERROR("initialize mete memory with size: " << allocSize << ", flag: " << flags << " failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    ret = drv::HalGvaAlloc(HYBM_DEVICE_META_ADDR, allocSize, 0);
    if (ret != ACLSHMEM_SUCCESS) {
        DlApi::CleanupLibrary();
        int32_t hal_ret = drv::HalGvaUnreserveMemory((uint64_t)*globalMemoryBase);
        SHM_LOG_ERROR("HalGvaAlloc hybm meta memory failed: " << ret << ", un-reserve memory " << hal_ret);
        return ACLSHMEM_MALLOC_FAILED;
    }
    return ACLSHMEM_SUCCESS;
}

HYBM_API int32_t hybm_init(int32_t deviceId, uint64_t flags)
{
    std::unique_lock<std::mutex> lockGuard{initMutex};
    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(HalGvaPrecheck(), "the current version of ascend driver does not support!");
    SHM_LOG_ERROR_RETURN_IT_IF_NOT_OK(hybm_load_library(), "load library failed");
    auto ret = DlAclApi::RtGetLogicDevIdByUserDevId(deviceId, &initedLogicDeviceId);
    if (ret != 0 || initedLogicDeviceId < 0) {
        SHM_LOG_ERROR("fail to get logic device id " << deviceId << ", ret=" << ret);
        return ACLSHMEM_INNER_ERROR;
    }
    SHM_LOG_INFO("success to get logic device user id=" << deviceId << ", logic deviceId = " << initedLogicDeviceId);
    ret = DlAclApi::AclrtSetDevice(deviceId);
    if (ret != ACLSHMEM_SUCCESS) {
        DlApi::CleanupLibrary();
        SHM_LOG_ERROR("set device id to be " << deviceId << " failed: " << ret);
        return ACLSHMEM_INNER_ERROR;
    }

    void *globalMemoryBase = nullptr;
    size_t allocSize = HYBM_DEVICE_INFO_SIZE;  // 申请meta空间
    soc_type = DlApi::GetAscendSocType();
    if ((soc_type == AscendSocType::ASCEND_950) || (HybmGetGvaVersion() == HYBM_GVA_V4)) {
        ret = init_meta_memory_for_modern(&globalMemoryBase, allocSize);
    } else {
        ret = init_meta_memory_for_legacy(&globalMemoryBase, allocSize, flags);
    }
    if (ret != ACLSHMEM_SUCCESS) {
        return ret;
    }
    inited_device_id = deviceId;
    SHM_LOG_INFO("hybm_init end device id " << deviceId << ", logic device id " << initedLogicDeviceId);
    initialized = true;
    g_baseAddr = (uint64_t)globalMemoryBase;
    SHM_LOG_INFO("hybm init successfully.");
    return 0;
}

HYBM_API void hybm_uninit(void)
{
    std::unique_lock<std::mutex> lockGuard{initMutex};
    if (!initialized) {
        SHM_LOG_WARN("hybm not initialized.");
        return;
    }
    int ret = 0;
    if ((soc_type == AscendSocType::ASCEND_950) || (HybmGetGvaVersion() == HYBM_GVA_V4)) {
        if (g_baseAddr != 0ULL) {
            ret = DlHalApi::HalMemUnmap(reinterpret_cast<void *>(HYBM_DEVICE_META_ADDR));
            SHM_LOG_INFO("unmap meta info res: " << ret);
            if (alloc_handle != nullptr) {
                ret = DlHalApi::HalMemRelease(alloc_handle);
                SHM_LOG_INFO("release meta memory handle res: " << ret);
            }
            ret = DlHalApi::HalMemAddressFree(reinterpret_cast<void *>(g_baseAddr));
            SHM_LOG_INFO("free meta memory res: " << ret);
        }
    } else {
        drv::HalGvaFree(HYBM_DEVICE_META_ADDR, HYBM_DEVICE_INFO_SIZE);
        ret = drv::HalGvaUnreserveMemory(g_baseAddr);
    }

    g_baseAddr = 0ULL;
    alloc_handle = nullptr;
    SHM_LOG_INFO("uninitialize GVA memory return: " << ret);
    initialized = false;
}

HYBM_API hybm_entity_t hybm_create_entity(uint16_t id, const hybm_options *options, uint32_t flags)
{
    auto &factory = MemEntityFactory::Instance();
    auto entity = factory.GetOrCreateEngine(id, flags);
    if (entity == nullptr) {
        SHM_LOG_ERROR("create entity failed.");
        return nullptr;
    }

    auto ret = entity->Initialize(options);
    if (ret != 0) {
        MemEntityFactory::Instance().RemoveEngine(entity.get());
        SHM_LOG_ERROR("initialize entity failed: " << ret);
        return nullptr;
    }

    return entity.get();
}

HYBM_API void hybm_destroy_entity(hybm_entity_t e, uint32_t flags)
{
    SHM_ASSERT_RET_VOID(e != nullptr);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RET_VOID(entity != nullptr);
    entity->UnInitialize();
    MemEntityFactory::Instance().RemoveEngine(e);
}

HYBM_API int32_t hybm_reserve_mem_space(hybm_entity_t e, uint32_t flags, void **reservedMem)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(reservedMem != nullptr, ACLSHMEM_INVALID_PARAM);
    return entity->ReserveMemorySpace(reservedMem);
}

HYBM_API int32_t hybm_unreserve_mem_space(hybm_entity_t e, uint32_t flags, void *reservedMem)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    return entity->UnReserveMemorySpace();
}

HYBM_API void *hybm_get_memory_ptr(hybm_entity_t e, hybm_mem_type mType)
{
    auto entity = static_cast<MemEntity *>(e);
    SHM_ASSERT_RETURN(entity != nullptr, nullptr);
    return entity->GetReservedMemoryPtr(mType);
}

HYBM_API hybm_mem_slice_t hybm_alloc_local_memory(hybm_entity_t e, hybm_mem_type mType, uint64_t size, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, nullptr);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, nullptr);
    hybm_mem_slice_t slice;
    auto ret = entity->AllocLocalMemory(size, mType, flags, slice);
    if (ret != 0) {
        SHM_LOG_ERROR("allocate slice with size: " << size << " failed: " << ret);
        return nullptr;
    }

    return slice;
}

HYBM_API int32_t hybm_free_local_memory(hybm_entity_t e, hybm_mem_slice_t slice, uint32_t count, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(slice != nullptr, ACLSHMEM_INVALID_PARAM);
    return entity->FreeLocalMemory(slice, flags);
}

HYBM_API hybm_mem_slice_t hybm_register_local_memory(hybm_entity_t e, hybm_mem_type mType, const void *ptr,
                                                     uint64_t size, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, nullptr);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, nullptr);

    hybm_mem_slice_t slice;
    auto ret = entity->RegisterLocalMemory(ptr, size, flags, slice);
    if (ret != 0) {
        SHM_LOG_ERROR("register slice with size: " << size << " failed: " << ret);
        return nullptr;
    }

    return slice;
}

HYBM_API int32_t hybm_export(hybm_entity_t e, hybm_mem_slice_t slice, uint32_t flags, hybm_exchange_info *exInfo)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(exInfo != nullptr, ACLSHMEM_INVALID_PARAM);

    ExchangeInfoWriter writer(exInfo);
    auto ret = entity->ExportExchangeInfo(slice, writer, flags);
    if (ret != 0) {
        SHM_LOG_ERROR("export slices failed: " << ret);
        return ret;
    }

    return ACLSHMEM_SUCCESS;
}

HYBM_API int32_t hybm_export_slice_size(hybm_entity_t e, size_t *size)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(size != nullptr, ACLSHMEM_INVALID_PARAM);

    auto ret = entity->GetExportSliceInfoSize(*size);
    return ret;
}

HYBM_API int32_t hybm_import(hybm_entity_t e, const hybm_exchange_info allExInfo[], uint32_t count, void *addresses[],
                             uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(allExInfo != nullptr, ACLSHMEM_INVALID_PARAM);
    std::vector<ExchangeInfoReader> readers(count);
    for (auto i = 0U; i < count; i++) {
        readers[i].Reset(allExInfo + i);
    }

    return entity->ImportExchangeInfo(readers.data(), count, addresses, flags);
}

HYBM_API int32_t hybm_mmap(hybm_entity_t e, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    return entity->Mmap();
}

HYBM_API int32_t hybm_entity_reach_types(hybm_entity_t e, uint32_t rank, hybm_data_op_type &reachTypes, uint32_t flags)
{
    auto entity = (MemEntity *)e;
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);

    reachTypes = entity->CanReachDataOperators(rank);
    return ACLSHMEM_SUCCESS;
}

HYBM_API int32_t hybm_remove_imported(hybm_entity_t e, uint32_t rank, uint32_t flags)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);

    std::vector<uint32_t> ranks = {rank};
    return entity->RemoveImported(ranks);
}

HYBM_API int32_t hybm_set_extra_context(hybm_entity_t e, const void *context, uint32_t size)
{
    SHM_ASSERT_RETURN(e != nullptr, ACLSHMEM_INVALID_PARAM);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RETURN(entity != nullptr, ACLSHMEM_INVALID_PARAM);
    SHM_ASSERT_RETURN(context != nullptr, ACLSHMEM_INVALID_PARAM);
    auto ret = entity->SetExtraContext(context, size);
    if (ret != ACLSHMEM_SUCCESS) {
        SHM_LOG_ERROR("SetExtraContext failed, ret: " << ret);
        return ret;
    }
    return ACLSHMEM_SUCCESS;
}

HYBM_API void hybm_unmap(hybm_entity_t e, uint32_t flags)
{
    SHM_ASSERT_RET_VOID(e != nullptr);
    auto entity = MemEntityFactory::Instance().FindEngineByPtr(e);
    SHM_ASSERT_RET_VOID(entity != nullptr);
    entity->Unmap();
}
