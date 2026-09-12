/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"
#include "shmemi_functions.h"
#include "dl_hccp_v2_def.h"
#include "dl_hccp_v2_api.h"

namespace shm {

std::mutex DlHccpV2Api::gMutex;
bool DlHccpV2Api::gLoaded = false;
const char* DlHccpV2Api::gHcclLibraryName = "libhccl_v2.so";
void* DlHccpV2Api::gHcclLibraryHandler = nullptr;
const char* DlHccpV2Api::gRaLibraryName = "libra.so";
void* DlHccpV2Api::gRaLibraryHandler = nullptr;
const char* DlHccpV2Api::gTsdLibraryName = "libtsdclient.so";
void* DlHccpV2Api::gTsdLibraryHandler = nullptr;
const char* DlHccpV2Api::gHcclV1LibraryName = "libhccl.so";
void* DlHccpV2Api::gHcclV1LibraryHandler = nullptr;

RaInitFunc               DlHccpV2Api::gRaInit               = nullptr;
RaDeinitFunc             DlHccpV2Api::gRaDeinit             = nullptr;
TsdProcessOpenFunc       DlHccpV2Api::gTsdProcessOpen       = nullptr;
TsdProcessCloseFunc      DlHccpV2Api::gTsdProcessClose      = nullptr;
RaGetDevEidInfoNumFunc   DlHccpV2Api::gRaGetDevEidInfoNum   = nullptr;
RaGetDevEidInfoListFunc  DlHccpV2Api::gRaGetDevEidInfoList  = nullptr;
RaCtxInitFunc            DlHccpV2Api::gRaCtxInit            = nullptr;
RaGetDevBaseAttrFunc     DlHccpV2Api::gRaGetDevBaseAttr     = nullptr;
RaGetAsyncReqResultFunc  DlHccpV2Api::gRaGetAsyncReqResult  = nullptr;
RaCtxChanCreateFunc      DlHccpV2Api::gRaCtxChanCreate      = nullptr;
RaCtxCqCreateFunc        DlHccpV2Api::gRaCtxCqCreate        = nullptr;
RaCtxQpCreateFunc        DlHccpV2Api::gRaCtxQpCreate        = nullptr;
RaCtxTokenIdAllocFunc    DlHccpV2Api::gRaCtxTokenIdAlloc    = nullptr;
RaCtxQpImportFunc        DlHccpV2Api::gRaCtxQpImport        = nullptr;
RaCtxQpBindFunc          DlHccpV2Api::gRaCtxQpBind          = nullptr;
RaCtxLmemRegisterFunc    DlHccpV2Api::gRaCtxLmemRegister    = nullptr;
RaCtxRmemImportFunc      DlHccpV2Api::gRaCtxRmemImport      = nullptr;
RaCtxRmemUnimportFunc    DlHccpV2Api::gRaCtxRmemUnimport    = nullptr;
RaCtxLmemUnregisterFunc  DlHccpV2Api::gRaCtxLmemUnregister  = nullptr;
RaCtxQpUnbindFunc        DlHccpV2Api::gRaCtxQpUnbind        = nullptr;
RaCtxQpUnimportFunc      DlHccpV2Api::gRaCtxQpUnimport      = nullptr;
RaCtxTokenIdFreeFunc     DlHccpV2Api::gRaCtxTokenIdFree     = nullptr;
RaCtxQpDestroyFunc       DlHccpV2Api::gRaCtxQpDestroy       = nullptr;
RaCtxCqDestroyFunc       DlHccpV2Api::gRaCtxCqDestroy       = nullptr;
RaCtxChanDestroyFunc     DlHccpV2Api::gRaCtxChanDestroy     = nullptr;
RaCtxDeinitFunc          DlHccpV2Api::gRaCtxDeinit          = nullptr;
RaCtxQpQueryBatchFunc    DlHccpV2Api::gRaCtxQpQueryBatch    = nullptr;
RaBatchSendWrFunc        DlHccpV2Api::gRaBatchSendWr        = nullptr;
RaCtxUpdateCiFunc        DlHccpV2Api::gRaCtxUpdateCi        = nullptr;
RaCustomChannelFunc      DlHccpV2Api::gRaCustomChannel      = nullptr;
RaGetTpInfoListAsyncFunc DlHccpV2Api::gRaGetTpInfoListAsync = nullptr;
RaGetIfNumFunc           DlHccpV2Api::gRaGetIfNum           = nullptr;
RaGetIfAddrsFunc         DlHccpV2Api::gRaGetIfAddrs         = nullptr;

Result DlHccpV2Api::LoadLibrary()
{
    std::lock_guard<std::mutex> lock(gMutex);
    if (gLoaded)
        return ACLSHMEM_SUCCESS;

    gHcclV1LibraryHandler = dlopen(gHcclV1LibraryName, RTLD_NOW);
    if (gHcclV1LibraryHandler == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gHcclV1LibraryName
            << "], please source ascend-toolkit set_env.sh, or add ascend driver lib path into LD_LIBRARY_PATH,"
            << " error: " << dlerror()
        );
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    gHcclLibraryHandler = dlopen(gHcclLibraryName, RTLD_NOW);
    if (gHcclLibraryHandler == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gHcclLibraryName
            << "], please source ascend-toolkit set_env.sh, or add ascend driver lib path into LD_LIBRARY_PATH,"
            << " error: " << dlerror()
        );
        dlclose(gHcclV1LibraryHandler);
        gHcclV1LibraryHandler = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    gRaLibraryHandler = dlopen(gRaLibraryName, RTLD_NOW);
    if (gRaLibraryHandler == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gRaLibraryName
            << "], please source ascend-toolkit set_env.sh, or add ascend driver lib path into LD_LIBRARY_PATH,"
            << " error: " << dlerror()
        );
        dlclose(gHcclV1LibraryHandler);
        gHcclV1LibraryHandler = nullptr;
        dlclose(gHcclLibraryHandler);
        gHcclLibraryHandler = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    gTsdLibraryHandler = dlopen(gTsdLibraryName, RTLD_NOW);
    if (gTsdLibraryHandler == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gTsdLibraryName
            << "], please source ascend-toolkit set_env.sh, or add ascend driver lib path into LD_LIBRARY_PATH,"
            << " error: " << dlerror()
        );
        dlclose(gHcclV1LibraryHandler);
        gHcclV1LibraryHandler = nullptr;
        dlclose(gHcclLibraryHandler);
        gHcclLibraryHandler = nullptr;
        dlclose(gRaLibraryHandler);
        gRaLibraryHandler = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    DL_LOAD_SYM_ALT(gRaInit,               RaInitFunc,               gHcclLibraryHandler, "RaInit",               "ra_init");
    DL_LOAD_SYM_ALT(gRaDeinit,             RaDeinitFunc,             gHcclLibraryHandler, "RaDeinit",             "ra_deinit");
    DL_LOAD_SYM_ALT(gTsdProcessOpen,       TsdProcessOpenFunc,       gTsdLibraryHandler,  "TsdProcessOpen",       "tsd_process_open");
    DL_LOAD_SYM_ALT(gTsdProcessClose,      TsdProcessCloseFunc,      gTsdLibraryHandler,  "TsdProcessClose",      "tsd_process_close");
    DL_LOAD_SYM_ALT(gRaGetDevEidInfoNum,   RaGetDevEidInfoNumFunc,   gHcclLibraryHandler, "RaGetDevEidInfoNum",   "ra_get_dev_eid_info_num");
    DL_LOAD_SYM_ALT(gRaGetDevEidInfoList,  RaGetDevEidInfoListFunc,  gHcclLibraryHandler, "RaGetDevEidInfoList",  "ra_get_dev_eid_info_list");
    DL_LOAD_SYM_ALT(gRaCtxInit,            RaCtxInitFunc,            gHcclLibraryHandler, "RaCtxInit",            "ra_ctx_init");
    DL_LOAD_SYM_ALT(gRaGetDevBaseAttr,     RaGetDevBaseAttrFunc,     gHcclLibraryHandler, "RaGetDevBaseAttr",     "ra_get_dev_base_attr");
    DL_LOAD_SYM_ALT(gRaGetAsyncReqResult,  RaGetAsyncReqResultFunc,  gHcclLibraryHandler, "RaGetAsyncReqResult",  "ra_get_async_req_result");
    DL_LOAD_SYM_ALT(gRaCtxChanCreate,      RaCtxChanCreateFunc,      gRaLibraryHandler,   "RaCtxChanCreate",      "ra_ctx_chan_create");
    DL_LOAD_SYM_ALT(gRaCtxCqCreate,        RaCtxCqCreateFunc,        gHcclLibraryHandler, "RaCtxCqCreate",        "ra_ctx_cq_create");
    DL_LOAD_SYM_ALT(gRaCtxQpCreate,        RaCtxQpCreateFunc,        gHcclLibraryHandler, "RaCtxQpCreate",        "ra_ctx_qp_create");
    DL_LOAD_SYM_ALT(gRaCtxTokenIdAlloc,    RaCtxTokenIdAllocFunc,    gHcclLibraryHandler, "RaCtxTokenIdAlloc",    "ra_ctx_token_id_alloc");
    DL_LOAD_SYM_ALT(gRaCtxQpImport,        RaCtxQpImportFunc,        gHcclLibraryHandler, "RaCtxQpImport",        "ra_ctx_qp_import");
    DL_LOAD_SYM_ALT(gRaCtxQpBind,          RaCtxQpBindFunc,          gHcclLibraryHandler, "RaCtxQpBind",          "ra_ctx_qp_bind");
    DL_LOAD_SYM_ALT(gRaCtxLmemRegister,    RaCtxLmemRegisterFunc,    gHcclLibraryHandler, "RaCtxLmemRegister",    "ra_ctx_lmem_register");
    DL_LOAD_SYM_ALT(gRaCtxRmemImport,      RaCtxRmemImportFunc,      gHcclLibraryHandler, "RaCtxRmemImport",      "ra_ctx_rmem_import");
    DL_LOAD_SYM_ALT(gRaCtxRmemUnimport,    RaCtxRmemUnimportFunc,    gHcclLibraryHandler, "RaCtxRmemUnimport",    "ra_ctx_rmem_unimport");
    DL_LOAD_SYM_ALT(gRaCtxLmemUnregister,  RaCtxLmemUnregisterFunc,  gHcclLibraryHandler, "RaCtxLmemUnregister",  "ra_ctx_lmem_unregister");
    DL_LOAD_SYM_ALT(gRaCtxQpUnbind,        RaCtxQpUnbindFunc,        gHcclLibraryHandler, "RaCtxQpUnbind",        "ra_ctx_qp_unbind");
    DL_LOAD_SYM_ALT(gRaCtxQpUnimport,      RaCtxQpUnimportFunc,      gHcclLibraryHandler, "RaCtxQpUnimport",      "ra_ctx_qp_unimport");
    DL_LOAD_SYM_ALT(gRaCtxTokenIdFree,     RaCtxTokenIdFreeFunc,     gHcclLibraryHandler, "RaCtxTokenIdFree",     "ra_ctx_token_id_free");
    DL_LOAD_SYM_ALT(gRaCtxQpDestroy,       RaCtxQpDestroyFunc,       gHcclLibraryHandler, "RaCtxQpDestroy",       "ra_ctx_qp_destroy");
    DL_LOAD_SYM_ALT(gRaCtxCqDestroy,       RaCtxCqDestroyFunc,       gHcclLibraryHandler, "RaCtxCqDestroy",       "ra_ctx_cq_destroy");
    DL_LOAD_SYM_ALT(gRaCtxChanDestroy,     RaCtxChanDestroyFunc,     gRaLibraryHandler,   "RaCtxChanDestroy",     "ra_ctx_chan_destroy");
    DL_LOAD_SYM_ALT(gRaCtxDeinit,          RaCtxDeinitFunc,          gHcclLibraryHandler, "RaCtxDeinit",          "ra_ctx_deinit");
    DL_LOAD_SYM_ALT(gRaCtxQpQueryBatch,    RaCtxQpQueryBatchFunc,    gRaLibraryHandler,   "RaCtxQpQueryBatch",    "ra_ctx_qp_query_batch");
    DL_LOAD_SYM_ALT(gRaBatchSendWr,        RaBatchSendWrFunc,        gHcclLibraryHandler, "RaBatchSendWr",        "ra_batch_send_wr");
    DL_LOAD_SYM_ALT(gRaCtxUpdateCi,        RaCtxUpdateCiFunc,        gHcclLibraryHandler, "RaCtxUpdateCi",        "ra_ctx_update_ci");
    DL_LOAD_SYM_ALT(gRaCustomChannel,      RaCustomChannelFunc,      gHcclLibraryHandler, "RaCustomChannel",      "ra_custom_channel");
    DL_LOAD_SYM_ALT(gRaGetTpInfoListAsync, RaGetTpInfoListAsyncFunc, gHcclLibraryHandler, "RaGetTpInfoListAsync", "ra_get_tp_info_list_async");
    DL_LOAD_SYM_ALT(gRaGetIfNum,           RaGetIfNumFunc,           gRaLibraryHandler,   "RaGetIfnum",           "ra_get_if_num");
    DL_LOAD_SYM_ALT(gRaGetIfAddrs,         RaGetIfAddrsFunc,         gRaLibraryHandler,   "RaGetIfaddrs",         "ra_get_if_addrs");

    gLoaded = true;
    return ACLSHMEM_SUCCESS;
}

Result DlHccpV2Api::CleanUpLibrary()
{
    std::lock_guard<std::mutex> lock(gMutex);
    if (!gLoaded)
        return ACLSHMEM_SUCCESS;

    int ret;

    gRaInit = nullptr;
    gRaDeinit = nullptr;
    gTsdProcessOpen = nullptr;
    gTsdProcessClose = nullptr;
    gRaGetDevEidInfoNum = nullptr;
    gRaGetDevEidInfoList = nullptr;
    gRaCtxInit = nullptr;
    gRaGetDevBaseAttr = nullptr;
    gRaGetAsyncReqResult = nullptr;
    gRaCtxChanCreate = nullptr;
    gRaCtxCqCreate = nullptr;
    gRaCtxQpCreate = nullptr;
    gRaCtxTokenIdAlloc = nullptr;
    gRaCtxQpImport = nullptr;
    gRaCtxQpBind = nullptr;
    gRaCtxLmemRegister = nullptr;
    gRaCtxRmemImport = nullptr;
    gRaCtxRmemUnimport = nullptr;
    gRaCtxLmemUnregister = nullptr;
    gRaCtxQpUnbind = nullptr;
    gRaCtxQpUnimport = nullptr;
    gRaCtxTokenIdFree = nullptr;
    gRaCtxQpDestroy = nullptr;
    gRaCtxCqDestroy = nullptr;
    gRaCtxChanDestroy = nullptr;
    gRaCtxDeinit = nullptr;
    gRaCtxQpQueryBatch = nullptr;
    gRaBatchSendWr = nullptr;
    gRaCtxUpdateCi = nullptr;
    gRaCustomChannel = nullptr;
    gRaGetTpInfoListAsync = nullptr;
    gRaGetIfNum = nullptr;
    gRaGetIfAddrs = nullptr;

    if (gTsdLibraryHandler != nullptr) {
        ret = dlclose(gTsdLibraryHandler);
        if (ret != 0) {
            SHM_LOG_ERROR("Failed to close " << gTsdLibraryName << " library handler" << dlerror());
        }
        gTsdLibraryHandler = nullptr;
    }
    if (gRaLibraryHandler != nullptr) {
        ret = dlclose(gRaLibraryHandler);
        if (ret != 0) {
            SHM_LOG_ERROR("Failed to close " << gRaLibraryName << " library handler" << dlerror());
        }
        gRaLibraryHandler = nullptr;
    }
    if (gHcclLibraryHandler != nullptr) {
        ret = dlclose(gHcclLibraryHandler);
        if (ret != 0) {
            SHM_LOG_ERROR("Failed to close " << gHcclLibraryName << " library handler" << dlerror());
        }
        gHcclLibraryHandler = nullptr;
    }
    if (gHcclV1LibraryHandler != nullptr) {
        ret = dlclose(gHcclV1LibraryHandler);
        if (ret != 0) {
            SHM_LOG_ERROR("Failed to close " << gHcclV1LibraryName << " library handler" << dlerror());
        }
        gHcclV1LibraryHandler = nullptr;
    }
    gLoaded = false;

    return ACLSHMEM_SUCCESS;
}

} // namespace shm
