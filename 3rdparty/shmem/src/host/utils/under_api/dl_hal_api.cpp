/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <dlfcn.h>
#include <mutex>
#include "shmemi_functions.h"
#include "dl_comm_def.h"
#include "dl_hal_api.h"
#include "utils/shmemi_logger.h"

namespace shm {
bool DlHalApi::gLoaded = false;
std::mutex DlHalApi::gMutex;
void *DlHalApi::halHandle;

const char *DlHalApi::gAscendHalLibName = "libascend_hal.so";

halSvmModuleAllocedSizeIncFunc DlHalApi::pSvmModuleAllocedSizeInc = nullptr;
halVirtAllocMemFromBaseFunc DlHalApi::pVirtAllocMemFromBase = nullptr;
halIoctlEnableHeapFunc DlHalApi::pIoctlEnableHeap = nullptr;
halGetHeapListByTypeFunc DlHalApi::pGetHeapListByType = nullptr;
halVirtSetHeapIdleFunc DlHalApi::pVirtSetHeapIdle = nullptr;
halVirtDestroyHeapV1Func DlHalApi::pVirtDestroyHeapV1 = nullptr;
halVirtDestroyHeapV2Func DlHalApi::pVirtDestroyHeapV2 = nullptr;
halVirtGetHeapMgmtFunc DlHalApi::pVirtGetHeapMgmt = nullptr;
halIoctlFreePagesFunc DlHalApi::pIoctlFreePages = nullptr;
halVaToHeapIdxFunc DlHalApi::pVaToHeapIdx = nullptr;
halVirtGetHeapFromQueueFunc DlHalApi::pVirtGetHeapFromQueue = nullptr;
halVirtNormalHeapUpdateInfoFunc DlHalApi::pVirtNormalHeapUpdateInfo = nullptr;
halVaToHeapFunc DlHalApi::pVaToHeap = nullptr;
int *DlHalApi::pHalFd = nullptr;

halAssignNodeDataFunc DlHalApi::pAssignNodeData = nullptr;
halInsertIdleSizeTreeFunc DlHalApi::pInsertIdleSizeTree = nullptr;
halInsertIdleVaTreeFunc DlHalApi::pInsertIdleVaTree = nullptr;
halAllocRbtreeNodeFunc DlHalApi::pAllocRbtreeNode = nullptr;
halEraseIdleVaTreeFunc DlHalApi::pEraseIdleVaTree = nullptr;
halEraseIdleSizeTreeFunc DlHalApi::pEraseIdleSizeTree = nullptr;
halGetAllocedNodeInRangeFunc DlHalApi::pGetAllocedNodeInRange = nullptr;
halGetIdleVaNodeInRangeFunc DlHalApi::pGetIdleVaNodeInRange = nullptr;
halInsertAllocedTreeFunc DlHalApi::pInsertAllocedTree = nullptr;

halMemAddressReserveFunc DlHalApi::pHalMemAddressReserve = nullptr;
halMemAddressFreeFunc DlHalApi::pHalMemAddressFree = nullptr;
halMemCreateFunc DlHalApi::pHalMemCreate = nullptr;
halMemReleaseFunc DlHalApi::pHalMemRelease = nullptr;
halMemMapFunc DlHalApi::pHalMemMap = nullptr;
halMemUnmapFunc DlHalApi::pHalMemUnmap = nullptr;
halMemExportFunc DlHalApi::pHalMemExport = nullptr;
halMemImportFunc DlHalApi::pHalMemImport = nullptr;
halMemTransShareableHandleFunc DlHalApi::pHalMemTransShareableHandle = nullptr;
halMemGetAllocationGranularityFunc DlHalApi::pHalMemGetAllocationGranularity = nullptr;
halMemShareHandleSetAttributeFunc DlHalApi::pHalMemShareHandleSetAttribute = nullptr;

Result DlHalApi::loadModernLibrary(AscendSocType socType)
{
    DL_LOAD_SYM(pHalMemAddressReserve, halMemAddressReserveFunc, halHandle, "halMemAddressReserve");
    DL_LOAD_SYM(pHalMemAddressFree, halMemAddressFreeFunc, halHandle, "halMemAddressFree");
    DL_LOAD_SYM(pHalMemCreate, halMemCreateFunc, halHandle, "halMemCreate");
    DL_LOAD_SYM(pHalMemRelease, halMemReleaseFunc, halHandle, "halMemRelease");
    DL_LOAD_SYM(pHalMemMap, halMemMapFunc, halHandle, "halMemMap");
    DL_LOAD_SYM(pHalMemUnmap, halMemUnmapFunc, halHandle, "halMemUnmap");
    DL_LOAD_SYM(pHalMemExport, halMemExportFunc, halHandle, "halMemExportToShareableHandleV2");
    DL_LOAD_SYM(pHalMemImport, halMemImportFunc, halHandle, "halMemImportFromShareableHandleV2");
    DL_LOAD_SYM(pHalMemShareHandleSetAttribute, halMemShareHandleSetAttributeFunc, halHandle,
                    "halMemShareHandleSetAttribute");
    DL_LOAD_SYM(pHalMemTransShareableHandle, halMemTransShareableHandleFunc, halHandle, "halMemTransShareableHandle");
    DL_LOAD_SYM(pHalMemGetAllocationGranularity, halMemGetAllocationGranularityFunc, halHandle,
                "halMemGetAllocationGranularity");
    return ACLSHMEM_SUCCESS;
}

Result DlHalApi::LoadLegacyLibrary()
{
    /* load sym */
    DL_LOAD_SYM(pHalFd, int *, halHandle, "g_devmm_mem_dev");
    DL_LOAD_SYM(pVirtAllocMemFromBase, halVirtAllocMemFromBaseFunc, halHandle, "devmm_virt_alloc_mem_from_base");
    DL_LOAD_SYM(pIoctlEnableHeap, halIoctlEnableHeapFunc, halHandle, "devmm_ioctl_enable_heap");
    DL_LOAD_SYM(pGetHeapListByType, halGetHeapListByTypeFunc, halHandle, "devmm_get_heap_list_by_type");
    DL_LOAD_SYM(pVirtSetHeapIdle, halVirtSetHeapIdleFunc, halHandle, "devmm_virt_set_heap_idle");
    DL_LOAD_SYM(pVirtGetHeapMgmt, halVirtGetHeapMgmtFunc, halHandle, "devmm_virt_get_heap_mgmt");
    DL_LOAD_SYM(pIoctlFreePages, halIoctlFreePagesFunc, halHandle, "devmm_ioctl_free_pages");
    DL_LOAD_SYM(pVaToHeapIdx, halVaToHeapIdxFunc, halHandle, "devmm_va_to_heap_idx");
    DL_LOAD_SYM(pVirtGetHeapFromQueue, halVirtGetHeapFromQueueFunc, halHandle, "devmm_virt_get_heap_from_queue");
    DL_LOAD_SYM(pVirtNormalHeapUpdateInfo, halVirtNormalHeapUpdateInfoFunc, halHandle,
                "devmm_virt_normal_heap_update_info");
    DL_LOAD_SYM(pVaToHeap, halVaToHeapFunc, halHandle, "devmm_va_to_heap");
    if (HybmGetGvaVersion() == HYBM_GVA_V1 or HybmGetGvaVersion() == HYBM_GVA_V2) {
        if (HybmGetGvaVersion() == HYBM_GVA_V1) {
            DL_LOAD_SYM(pVirtDestroyHeapV1, halVirtDestroyHeapV1Func, halHandle, "devmm_virt_destroy_heap");
        } else {
            DL_LOAD_SYM(pSvmModuleAllocedSizeInc, halSvmModuleAllocedSizeIncFunc, halHandle,
                        "svm_module_alloced_size_inc");
        }
        DL_LOAD_SYM(pAssignNodeData, halAssignNodeDataFunc, halHandle, "devmm_assign_rbtree_node_data");
        DL_LOAD_SYM(pInsertIdleSizeTree, halInsertIdleSizeTreeFunc, halHandle,
                    "devmm_rbtree_insert_idle_size_tree");
        DL_LOAD_SYM(pInsertIdleVaTree, halInsertIdleVaTreeFunc, halHandle,
                    "devmm_rbtree_insert_idle_va_tree");
        DL_LOAD_SYM(pAllocRbtreeNode, halAllocRbtreeNodeFunc, halHandle, "devmm_alloc_rbtree_node");
        DL_LOAD_SYM(pEraseIdleVaTree, halEraseIdleVaTreeFunc, halHandle, "devmm_rbtree_erase_idle_va_tree");
        DL_LOAD_SYM(pEraseIdleSizeTree, halEraseIdleSizeTreeFunc, halHandle,
                    "devmm_rbtree_erase_idle_size_tree");
        DL_LOAD_SYM(pGetAllocedNodeInRange, halGetAllocedNodeInRangeFunc, halHandle,
                    "devmm_rbtree_get_alloced_node_in_range");
        DL_LOAD_SYM(pGetIdleVaNodeInRange, halGetIdleVaNodeInRangeFunc, halHandle,
                    "devmm_rbtree_get_idle_va_node_in_range");
        DL_LOAD_SYM(pInsertAllocedTree, halInsertAllocedTreeFunc, halHandle,
                    "devmm_rbtree_insert_alloced_tree");
    } else { // HYBM_GVA_V3
        DL_LOAD_SYM(pSvmModuleAllocedSizeInc, halSvmModuleAllocedSizeIncFunc, halHandle, "svm_module_alloced_size_inc");
        DL_LOAD_SYM(pVirtDestroyHeapV2, halVirtDestroyHeapV2Func, halHandle, "devmm_virt_destroy_heap");
    }
    return ACLSHMEM_SUCCESS;
}

Result DlHalApi::LoadLibrary(AscendSocType socType)
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return ACLSHMEM_SUCCESS;
    }

    halHandle = dlopen(gAscendHalLibName, RTLD_NOW);
    if (halHandle == nullptr) {
        SHM_LOG_ERROR("Failed to open library [" << gAscendHalLibName << "], please source ascend-toolkit set_env.sh,"
                     << " or add ascend driver lib path into LD_LIBRARY_PATH," << " error: " << dlerror());
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    SHM_ASSERT_RETURN(HybmGetGvaVersion() != HYBM_GVA_UNKNOWN, ACLSHMEM_NOT_INITED);
    Result ret = 0;
    if ((socType == AscendSocType::ASCEND_950) || (HybmGetGvaVersion() == HYBM_GVA_V4)) {
        ret = DlHalApi::loadModernLibrary(socType);
    } else {
        ret = DlHalApi::LoadLegacyLibrary();
    }
    if (ret != 0) {
        return ret;
    }

    gLoaded = true;
    return ACLSHMEM_SUCCESS;
}

void DlHalApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    pHalFd = nullptr;
    pSvmModuleAllocedSizeInc = nullptr;
    pVirtAllocMemFromBase = nullptr;
    pIoctlEnableHeap = nullptr;
    pGetHeapListByType = nullptr;
    pVirtSetHeapIdle = nullptr;
    pVirtDestroyHeapV1 = nullptr;
    pVirtDestroyHeapV2 = nullptr;
    pVirtGetHeapMgmt = nullptr;
    pIoctlFreePages = nullptr;
    pVaToHeapIdx = nullptr;
    pVirtGetHeapFromQueue = nullptr;
    pVirtNormalHeapUpdateInfo = nullptr;
    pVaToHeap = nullptr;

    pAssignNodeData = nullptr;
    pInsertIdleSizeTree = nullptr;
    pInsertIdleVaTree = nullptr;
    pAllocRbtreeNode = nullptr;
    pEraseIdleVaTree = nullptr;
    pEraseIdleSizeTree = nullptr;
    pGetAllocedNodeInRange = nullptr;
    pGetIdleVaNodeInRange = nullptr;
    pInsertAllocedTree = nullptr;

    pHalMemAddressReserve = nullptr;
    pHalMemAddressFree = nullptr;
    pHalMemCreate = nullptr;
    pHalMemRelease = nullptr;
    pHalMemMap = nullptr;
    pHalMemUnmap = nullptr;
    pHalMemExport = nullptr;
    pHalMemImport = nullptr;
    pHalMemGetAllocationGranularity = nullptr;
    pHalMemTransShareableHandle = nullptr;

    if (halHandle != nullptr) {
        dlclose(halHandle);
        halHandle = nullptr;
    }

    gLoaded = false;
}
}
