/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBM_CORE_DL_HCCP_API_H
#define MF_HYBM_CORE_DL_HCCP_API_H

#include "dl_hccp_def.h"

namespace shm {

using Result = int32_t;

using raRdevGetHandleFunc = int (*)(uint32_t, void **);

using raInitFunc = int (*)(const HccpRaInitConfig *);
using raSocketInitFunc = int (*)(HccpNetworkMode, HccpRdev, void **);
using raSocketDeinitFunc = int (*)(void *);
using raRdevInitV2Func = int (*)(HccpRdevInitInfo, HccpRdev, void **);
using raSocketBatchConnectFunc = int (*)(HccpSocketConnectInfo[], uint32_t);
using raSocketBatchCloseFunc = int (*)(HccpSocketCloseInfo[], uint32_t);
using raSocketListenStartFunc = int (*)(HccpSocketListenInfo[], uint32_t);
using raSocketListenStopFunc = int (*)(HccpSocketListenInfo[], uint32_t);
using raGetSocketsFunc = int (*)(uint32_t, HccpSocketInfo[], uint32_t, uint32_t *);
using raGetIfNumFunc = int (*)(const HccpRaGetIfAttr *, uint32_t *);
using raGetIfAddrsFunc = int (*)(const HccpRaGetIfAttr *, HccpInterfaceInfo[], uint32_t *);
using raSocketWhiteListAddFunc = int (*)(void *, const HccpSocketWhiteListInfo[], uint32_t num);
using raSocketWhiteListDelFunc = int (*)(void *, const HccpSocketWhiteListInfo[], uint32_t num);
using raQpCreateFunc = int (*)(void *, int, int, void **);
using raQpAiCreateFunc = int (*)(void *, const HccpQpExtAttrs *, HccpAiQpInfo *, void **);
using raQpDestroyFunc = int (*)(void *);
using raGetQpStatusFunc = int (*)(void *, int *);
using raQpConnectAsyncFunc = int (*)(void *, const void *);
using raRegisterMrFunc = int (*)(const void *, HccpMrInfo *, void **);
using raDeregisterMrFunc = int (*)(const void *, void *);
using raMrRegFunc = int (*)(void *, HccpMrInfo *);
using tsdOpenFunc = uint32_t (*)(uint32_t, uint32_t);

class DlHccpApi {
public:
    static Result LoadLibrary();
    static void CleanupLibrary();

    static inline int RaSocketInit(HccpNetworkMode mode, const HccpRdev &rdev, void *&socketHandle)
    {
        return gRaSocketInit(mode, rdev, &socketHandle);
    }

    static inline int RaInit(const HccpRaInitConfig &config)
    {
        return gRaInit(&config);
    }

    static inline int RaSocketDeinit(void *socketHandle)
    {
        return gRaSocketDeinit(socketHandle);
    }

    static inline int RaRdevInitV2(const HccpRdevInitInfo &info, const HccpRdev &rdev, void *&rdmaHandle)
    {
        return gRaRdevInitV2(info, rdev, &rdmaHandle);
    }

    static inline int RaRdevGetHandle(uint32_t deviceId, void *&rdmaHandle)
    {
        return gRaRdevGetHandle(deviceId, &rdmaHandle);
    }

    static inline int RaSocketBatchConnect(HccpSocketConnectInfo conn[], uint32_t num)
    {
        return gRaSocketBatchConnect(conn, num);
    }

    static inline int RaSocketBatchClose(HccpSocketCloseInfo conn[], uint32_t num)
    {
        return gRaSocketBatchClose(conn, num);
    }

    static inline int RaSocketListenStart(HccpSocketListenInfo conn[], uint32_t num)
    {
        return gRaSocketListenStart(conn, num);
    }

    static inline int RaSocketListenStop(HccpSocketListenInfo conn[], uint32_t num)
    {
        return gRaSocketListenStop(conn, num);
    }

    static inline int RaGetSockets(uint32_t role, HccpSocketInfo conn[], uint32_t num, uint32_t &connectedNum)
    {
        return gRaGetSockets(role, conn, num, &connectedNum);
    }

    static inline int RaGetIfNum(const HccpRaGetIfAttr &config, uint32_t &num)
    {
        return gRaGetIfNum(&config, &num);
    }

    static inline int RaGetIfAddrs(const HccpRaGetIfAttr &config, HccpInterfaceInfo infos[], uint32_t &num)
    {
        return gRaGetIfAddrs(&config, infos, &num);
    }

    static inline int RaSocketWhiteListAdd(void *socket, const HccpSocketWhiteListInfo list[], uint32_t num)
    {
        return gRaSocketWhiteListAdd(socket, list, num);
    }

    static inline int RaSocketWhiteListDel(void *socket, const HccpSocketWhiteListInfo list[], uint32_t num)
    {
        return gRaSocketWhiteListDel(socket, list, num);
    }

    static inline int RaQpCreate(void *rdmaHandle, int flag, int qpMode, void *&qpHandle)
    {
        return gRaQpCreate(rdmaHandle, flag, qpMode, &qpHandle);
    }

    static inline int RaQpAiCreate(void *rdmaHandle, const HccpQpExtAttrs &attrs, HccpAiQpInfo &info, void *&qpHandle)
    {
        return gRaQpAiCreate(rdmaHandle, &attrs, &info, &qpHandle);
    }

    static inline int RaQpDestroy(void *qpHandle)
    {
        return gRaQpDestroy(qpHandle);
    }

    static inline int RaGetQpStatus(void *qpHandle, int &status)
    {
        return gRaGetQpStatus(qpHandle, &status);
    }

    static inline int RaQpConnectAsync(void *qp, const void *socketFd)
    {
        return gRaQpConnectAsync(qp, socketFd);
    }

    static inline int RaRegisterMR(const void *rdmaHandle, HccpMrInfo *info, void *&mrHandle)
    {
        return gRaRegisterMR(rdmaHandle, info, &mrHandle);
    }

    static inline int RaDeregisterMR(const void *rdmaHandle, void *mrHandle)
    {
        return gRaDeregisterMR(rdmaHandle, mrHandle);
    }

    static inline int RaMrReg(void *qpHandle, HccpMrInfo &info)
    {
        return gRaMrReg(qpHandle, &info);
    }

    static inline uint32_t TsdOpen(uint32_t deviceId, uint32_t rankSize)
    {
        return gTsdOpen(deviceId, rankSize);
    }

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void *raHandle;
    static void *tsdHandle;
    static const char *gRaLibName;
    static const char *gTsdLibName;

    static raRdevGetHandleFunc gRaRdevGetHandle;

    static raInitFunc gRaInit;
    static raSocketInitFunc gRaSocketInit;
    static raSocketDeinitFunc gRaSocketDeinit;
    static raRdevInitV2Func gRaRdevInitV2;
    static raSocketBatchConnectFunc gRaSocketBatchConnect;
    static raSocketBatchCloseFunc gRaSocketBatchClose;
    static raSocketListenStartFunc gRaSocketListenStart;
    static raSocketListenStopFunc gRaSocketListenStop;
    static raGetSocketsFunc gRaGetSockets;
    static raGetIfNumFunc gRaGetIfNum;
    static raGetIfAddrsFunc gRaGetIfAddrs;
    static raSocketWhiteListAddFunc gRaSocketWhiteListAdd;
    static raSocketWhiteListDelFunc gRaSocketWhiteListDel;
    static raQpCreateFunc gRaQpCreate;
    static raQpAiCreateFunc gRaQpAiCreate;
    static raQpDestroyFunc gRaQpDestroy;
    static raGetQpStatusFunc gRaGetQpStatus;
    static raQpConnectAsyncFunc gRaQpConnectAsync;
    static raRegisterMrFunc gRaRegisterMR;
    static raDeregisterMrFunc gRaDeregisterMR;
    static raMrRegFunc gRaMrReg;

    static tsdOpenFunc gTsdOpen;
};

}

#endif  // MF_HYBM_CORE_DL_HCCP_API_H
