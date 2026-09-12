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
#include "shmemi_functions.h"
#include "dl_hccp_api.h"
#include "host/shmem_host_def.h"
#include "utils/shmemi_logger.h"

namespace shm {
bool DlHccpApi::gLoaded = false;
std::mutex DlHccpApi::gMutex;
void *DlHccpApi::raHandle;
void *DlHccpApi::tsdHandle;

const char *DlHccpApi::gRaLibName = "libra.so";
const char *DlHccpApi::gTsdLibName = "libtsdclient.so";

raRdevGetHandleFunc DlHccpApi::gRaRdevGetHandle;
raInitFunc DlHccpApi::gRaInit;
raSocketInitFunc DlHccpApi::gRaSocketInit;
raSocketDeinitFunc DlHccpApi::gRaSocketDeinit;
raRdevInitV2Func DlHccpApi::gRaRdevInitV2;
raSocketBatchConnectFunc DlHccpApi::gRaSocketBatchConnect;
raSocketBatchCloseFunc DlHccpApi::gRaSocketBatchClose;
raSocketListenStartFunc DlHccpApi::gRaSocketListenStart;
raSocketListenStopFunc DlHccpApi::gRaSocketListenStop;
raGetSocketsFunc DlHccpApi::gRaGetSockets;
raGetIfNumFunc DlHccpApi::gRaGetIfNum;
raGetIfAddrsFunc DlHccpApi::gRaGetIfAddrs;
raSocketWhiteListAddFunc DlHccpApi::gRaSocketWhiteListAdd;
raSocketWhiteListDelFunc DlHccpApi::gRaSocketWhiteListDel;
raQpCreateFunc DlHccpApi::gRaQpCreate;
raQpAiCreateFunc DlHccpApi::gRaQpAiCreate;
raQpDestroyFunc DlHccpApi::gRaQpDestroy;
raGetQpStatusFunc DlHccpApi::gRaGetQpStatus;
raQpConnectAsyncFunc DlHccpApi::gRaQpConnectAsync;
raRegisterMrFunc DlHccpApi::gRaRegisterMR;
raDeregisterMrFunc DlHccpApi::gRaDeregisterMR;
raMrRegFunc DlHccpApi::gRaMrReg;
tsdOpenFunc DlHccpApi::gTsdOpen;

Result DlHccpApi::LoadLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (gLoaded) {
        return ACLSHMEM_SUCCESS;
    }

    raHandle = dlopen(gRaLibName, RTLD_NOW);
    if (raHandle == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gRaLibName
            << "], please source ascend-toolkit set_env.sh, or add ascend driver lib path into LD_LIBRARY_PATH,"
            << " error: " << dlerror());
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    tsdHandle = dlopen(gTsdLibName, RTLD_NOW);
    if (tsdHandle == nullptr) {
        SHM_LOG_ERROR(
            "Failed to open library ["
            << gTsdLibName
            << "], please source ascend-toolkit set_env.sh, or add ascend driver lib path into LD_LIBRARY_PATH,"
            << " error: " << dlerror());
        dlclose(raHandle);
        raHandle = nullptr;
        return ACLSHMEM_DL_FUNC_FAILED;
    }

    /* load sym */
    DL_LOAD_SYM_ALT(gRaSocketInit, raSocketInitFunc, raHandle, "ra_socket_init", "RaSocketInit");
    DL_LOAD_SYM_ALT(gRaInit, raInitFunc, raHandle, "ra_init", "RaInit");
    DL_LOAD_SYM_ALT(gRaSocketDeinit, raSocketDeinitFunc, raHandle, "ra_socket_deinit", "RaSocketDeinit");
    DL_LOAD_SYM_ALT(gRaRdevInitV2, raRdevInitV2Func, raHandle, "ra_rdev_init_v2", "RaRdevInitV2");
    DL_LOAD_SYM_ALT(gRaRdevGetHandle, raRdevGetHandleFunc, raHandle, "ra_rdev_get_handle", "RaRdevGetHandle");
    DL_LOAD_SYM_ALT(gRaSocketBatchConnect, raSocketBatchConnectFunc, raHandle,
        "ra_socket_batch_connect", "RaSocketBatchConnect");
    DL_LOAD_SYM_ALT(gRaSocketBatchClose, raSocketBatchCloseFunc, raHandle,
        "ra_socket_batch_close", "RaSocketBatchClose");
    DL_LOAD_SYM_ALT(gRaSocketListenStart, raSocketListenStartFunc, raHandle,
        "ra_socket_listen_start", "RaSocketListenStart");
    DL_LOAD_SYM_ALT(gRaSocketListenStop, raSocketListenStopFunc, raHandle,
        "ra_socket_listen_stop", "RaSocketListenStop");
    DL_LOAD_SYM_ALT(gRaGetSockets, raGetSocketsFunc, raHandle, "ra_get_sockets", "RaGetSockets");
    DL_LOAD_SYM_ALT(gRaGetIfNum, raGetIfNumFunc, raHandle, "ra_get_ifnum", "RaGetIfnum");
    DL_LOAD_SYM_ALT(gRaGetIfAddrs, raGetIfAddrsFunc, raHandle, "ra_get_ifaddrs", "RaGetIfaddrs");
    DL_LOAD_SYM_ALT(gRaSocketWhiteListAdd, raSocketWhiteListAddFunc, raHandle,
        "ra_socket_white_list_add", "RaSocketWhiteListAdd");
    DL_LOAD_SYM_ALT(gRaSocketWhiteListDel, raSocketWhiteListDelFunc, raHandle,
        "ra_socket_white_list_del", "RaSocketWhiteListDel");
    DL_LOAD_SYM_ALT(gRaQpCreate, raQpCreateFunc, raHandle, "ra_qp_create", "RaQpCreate");
    DL_LOAD_SYM_ALT(gRaQpAiCreate, raQpAiCreateFunc, raHandle, "ra_ai_qp_create", "RaAiQpCreate");
    DL_LOAD_SYM_ALT(gRaQpDestroy, raQpDestroyFunc, raHandle, "ra_qp_destroy", "RaQpDestroy");
    DL_LOAD_SYM_ALT(gRaGetQpStatus, raGetQpStatusFunc, raHandle, "ra_get_qp_status", "RaGetQpStatus");
    DL_LOAD_SYM_ALT(gRaQpConnectAsync, raQpConnectAsyncFunc, raHandle, "ra_qp_connect_async", "RaQpConnectAsync");
    DL_LOAD_SYM_ALT(gRaRegisterMR, raRegisterMrFunc, raHandle, "ra_register_mr", "RaRegisterMr");
    DL_LOAD_SYM_ALT(gRaDeregisterMR, raDeregisterMrFunc, raHandle, "ra_deregister_mr", "RaDeregisterMr");
    DL_LOAD_SYM_ALT(gRaMrReg, raMrRegFunc, raHandle, "ra_mr_reg", "RaMrReg");

    DL_LOAD_SYM(gTsdOpen, tsdOpenFunc, tsdHandle, "TsdOpen");
    SHM_LOG_INFO("LoadLibrary for DlHccpApi success");
    gLoaded = true;
    return ACLSHMEM_SUCCESS;
}

void DlHccpApi::CleanupLibrary()
{
    std::lock_guard<std::mutex> guard(gMutex);
    if (!gLoaded) {
        return;
    }

    gRaRdevGetHandle = nullptr;
    gRaInit = nullptr;
    gRaSocketInit = nullptr;
    gRaSocketDeinit = nullptr;
    gRaRdevInitV2 = nullptr;
    gRaSocketBatchConnect = nullptr;
    gRaSocketBatchClose = nullptr;
    gRaSocketListenStart = nullptr;
    gRaSocketListenStop = nullptr;
    gRaGetSockets = nullptr;
    gRaGetIfNum = nullptr;
    gRaGetIfAddrs = nullptr;
    gRaSocketWhiteListAdd = nullptr;
    gRaSocketWhiteListDel = nullptr;
    gRaQpCreate = nullptr;
    gRaQpAiCreate = nullptr;
    gRaQpDestroy = nullptr;
    gRaGetQpStatus = nullptr;
    gRaQpConnectAsync = nullptr;
    gRaRegisterMR = nullptr;
    gRaDeregisterMR = nullptr;
    gRaMrReg = nullptr;
    gTsdOpen = nullptr;

    if (raHandle != nullptr) {
        dlclose(raHandle);
        raHandle = nullptr;
    }

    if (tsdHandle != nullptr) {
        dlclose(tsdHandle);
        tsdHandle = nullptr;
    }
    gLoaded = false;
}
}