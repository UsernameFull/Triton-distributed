/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DL_HCCP_V2_API_H
#define DL_HCCP_V2_API_H

#include <mutex>
#include <cstdint>
#include "dl_hccp_v2_def.h"

namespace shm {

using Result = int32_t;

using RaInitFunc                  = int (*) (RaInitConfig*);
using RaDeinitFunc                = int (*) (RaInitConfig*);
using TsdProcessOpenFunc          = uint32_t (*) (const uint32_t, ProcOpenArgs*);
using TsdProcessCloseFunc         = uint32_t (*) (const uint32_t, const pid_t closePid);
using RaGetDevEidInfoNumFunc      = int (*) (RaInfo,               unsigned int*);
using RaGetDevEidInfoListFunc     = int (*) (RaInfo, DevEidInfo[], unsigned int*);
using RaCtxInitFunc               = int (*) (CtxInitCfg*, CtxInitAttr*, void**);
using RaGetDevBaseAttrFunc        = int (*) (void*, DevBaseAttrT*);
using RaGetAsyncReqResultFunc     = int (*) (void*, int*);
using RaCtxChanCreateFunc         = int (*) (void*, ChanInfoT*,                   void**);
using RaCtxCqCreateFunc           = int (*) (void*, CqInfoT*,                     void**);
using RaCtxQpCreateFunc           = int (*) (void*, QpCreateAttr*, QpCreateInfo*, void**);
using RaCtxTokenIdAllocFunc       = int (*) (void*, HccpTokenId*,                 void**);
using RaCtxQpImportFunc           = int (*) (void*, QpImportInfoT*,               void**);
using RaCtxQpBindFunc             = int (*) (void*,                               void*);
using RaCtxLmemRegisterFunc       = int (*) (void*, MrRegInfoT*,                  void**);
using RaCtxRmemImportFunc         = int (*) (void*, MrImportInfoT*,               void**);
using RaCtxRmemUnimportFunc       = int (*) (void*,                               void*);
using RaCtxLmemUnregisterFunc     = int (*) (void*,                               void*);
using RaCtxQpUnbindFunc           = int (*) (void*);
using RaCtxQpUnimportFunc         = int (*) (void*,                               void*);
using RaCtxTokenIdFreeFunc        = int (*) (void*,                               void*);
using RaCtxQpDestroyFunc          = int (*) (void*);
using RaCtxCqDestroyFunc          = int (*) (void*,                               void*);
using RaCtxChanDestroyFunc        = int (*) (void*,                               void*);
using RaCtxDeinitFunc             = int (*) (void*);
using RaCtxQpQueryBatchFunc       = int (*) (void*[], JettyAttr[], unsigned int*);
using RaBatchSendWrFunc           = int (*) (void*, SendWrData[], SendWrResp[], unsigned int, unsigned int*);
using RaCtxUpdateCiFunc           = int (*) (void*, uint16_t);
using RaCustomChannelFunc         = int (*) (RaInfo, CustomChanInfoIn*, CustomChanInfoOut*);
using RaGetTpInfoListAsyncFunc    = int (*) (void*, GetTpCfg*, TpInfo[], unsigned int*, void**);
using RaGetIfNumFunc              = int (*) (RaGetIfAttr*, unsigned int*);
using RaGetIfAddrsFunc            = int (*) (RaGetIfAttr*, InterfaceInfo[], unsigned int*);

class DlHccpV2Api {
public:
    static Result LoadLibrary();
    static Result CleanUpLibrary();
private:
    static std::mutex gMutex;
    static bool gLoaded;

    static const char* gHcclLibraryName;
    static void* gHcclLibraryHandler;
    static const char* gRaLibraryName;
    static void* gRaLibraryHandler;
    static const char* gTsdLibraryName;
    static void* gTsdLibraryHandler;

    // Loading hccl_v2.so directly causes undefined symbol errors.
    // Preload hccl.so first to avoid those errors.
    static const char* gHcclV1LibraryName;
    static void* gHcclV1LibraryHandler;
public:
    static int RaInit(RaInitConfig *config)
    {
        return gRaInit(config);
    }

    static int RaDeinit(RaInitConfig *config)
    {
        return gRaDeinit(config);
    }

    static uint32_t TsdProcessOpen(const uint32_t logicDeviceId, ProcOpenArgs* openArgs)
    {
        return gTsdProcessOpen(logicDeviceId, openArgs);
    }

    static uint32_t TsdProcessClose(const uint32_t logicDeviceId, const pid_t closePid)
    {
        return gTsdProcessClose(logicDeviceId, closePid);
    }

    static int RaGetDevEidInfoNum(RaInfo info, unsigned int* num)
    {
        return gRaGetDevEidInfoNum(info, num);
    }

    static int RaGetDevEidInfoList(RaInfo info, DevEidInfo infoList[], unsigned int* num)
    {
        return gRaGetDevEidInfoList(info, infoList, num);
    }

    static int RaCtxInit(CtxInitCfg* config, CtxInitAttr* attr, void** ctxHandle)
    {
        return gRaCtxInit(config, attr, ctxHandle);
    }

    static int RaGetDevBaseAttr(void* ctxHandle, DevBaseAttrT* attr)
    {
        return gRaGetDevBaseAttr(ctxHandle, attr);
    }

    static int RaGetAsyncReqResult(void* reqHandle, int* reqResult)
    {
        return gRaGetAsyncReqResult(reqHandle, reqResult);
    }

    static int RaCtxChanCreate(void* ctxHandle, ChanInfoT* chanInfo, void** chanHandle)
    {
        return gRaCtxChanCreate(ctxHandle, chanInfo, chanHandle);
    }

    static int RaCtxCqCreate(void* ctxHandle, CqInfoT* cqInfo, void** cqHandle)
    {
        return gRaCtxCqCreate(ctxHandle, cqInfo, cqHandle);
    }

    static int RaCtxQpCreate(void* ctxHandle, QpCreateAttr* attr, QpCreateInfo* info, void** qpHandle)
    {
        return gRaCtxQpCreate(ctxHandle, attr, info, qpHandle);
    }

    static int RaCtxTokenIdAlloc(void* ctxHandle, HccpTokenId* info, void** tokenIdHandle)
    {
        return gRaCtxTokenIdAlloc(ctxHandle, info, tokenIdHandle);
    }

    static int RaCtxQpImport(void* ctxHandle, QpImportInfoT* qpInfo, void** remQpHandle)
    {
        return gRaCtxQpImport(ctxHandle, qpInfo, remQpHandle);
    }

    static int RaCtxQpBind(void* qpHandle, void* remQpHandle)
    {
        return gRaCtxQpBind(qpHandle, remQpHandle);
    }

    static int RaCtxLmemRegister(void* ctxHandle, MrRegInfoT* lmemInfo, void** lmemHandle)
    {
        return gRaCtxLmemRegister(ctxHandle, lmemInfo, lmemHandle);
    }

    static int RaCtxRmemImport(void* ctxHandle, MrImportInfoT* rmemInfo, void** rmemHandle)
    {
        return gRaCtxRmemImport(ctxHandle, rmemInfo, rmemHandle);
    }

    static int RaCtxRmemUnimport(void* ctxHandle, void* rmemHandle)
    {
        return gRaCtxRmemUnimport(ctxHandle, rmemHandle);
    }

    static int RaCtxLmemUnregister(void* ctxHandle, void* lmemHandle)
    {
        return gRaCtxLmemUnregister(ctxHandle, lmemHandle);
    }

    static int RaCtxQpUnbind(void* qpHandle)
    {
        return gRaCtxQpUnbind(qpHandle);
    }

    static int RaCtxQpUnimport(void* ctxHandle, void* remQpHandle)
    {
        return gRaCtxQpUnimport(ctxHandle, remQpHandle);
    }

    static int RaCtxTokenIdFree(void* ctxHandle, void* tokenIdHandle)
    {
        return gRaCtxTokenIdFree(ctxHandle, tokenIdHandle);
    }

    static int RaCtxQpDestroy(void* qpHandle)
    {
        return gRaCtxQpDestroy(qpHandle);
    }

    static int RaCtxCqDestroy(void* ctxHandle, void* cqHandle)
    {
        return gRaCtxCqDestroy(ctxHandle, cqHandle);
    }

    static int RaCtxChanDestroy(void* ctxHandle, void* chanHandle)
    {
        return gRaCtxChanDestroy(ctxHandle, chanHandle);
    }

    static int RaCtxDeinit(void* ctxHandle)
    {
        return gRaCtxDeinit(ctxHandle);
    }

    static int RaCtxQpQueryBatch(void* qpHandle[], struct JettyAttr attr[], unsigned int* num)
    {
        return gRaCtxQpQueryBatch(qpHandle, attr, num);
    }

    static int RaBatchSendWr(void* qpHandle, SendWrData wrList[], SendWrResp opResp[], unsigned int num, unsigned int* completeNum)
    {
        return gRaBatchSendWr(qpHandle, wrList, opResp, num, completeNum);
    }

    static int RaCtxUpdateCi(void* qpHandle, uint16_t ci)
    {
        return gRaCtxUpdateCi(qpHandle, ci);
    }

    static int RaCustomChannel(RaInfo info, CustomChanInfoIn* in, CustomChanInfoOut* out)
    {
        return gRaCustomChannel(info, in, out);
    }

    static int RaGetTpInfoListAsync(void* ctxHandle, GetTpCfg* cfg, TpInfo infoList[], unsigned int* num, void** reqHandle)
    {
        return gRaGetTpInfoListAsync(ctxHandle, cfg, infoList, num, reqHandle);
    }

    static int RaGetIfNum(RaGetIfAttr* config, unsigned int* num)
    {
        return gRaGetIfNum(config, num);
    }

    static int RaGetIfAddrs(RaGetIfAttr* config, InterfaceInfo interfaceInfos[], unsigned int* num)
    {
        return gRaGetIfAddrs(config, interfaceInfos, num);
    }

private:
    static RaInitFunc gRaInit; 
    static RaDeinitFunc gRaDeinit; 
    static TsdProcessOpenFunc gTsdProcessOpen;
    static TsdProcessCloseFunc gTsdProcessClose;
    static RaGetDevEidInfoNumFunc gRaGetDevEidInfoNum; 
    static RaGetDevEidInfoListFunc gRaGetDevEidInfoList; 
    static RaCtxInitFunc gRaCtxInit; 
    static RaGetDevBaseAttrFunc gRaGetDevBaseAttr; 
    static RaGetAsyncReqResultFunc gRaGetAsyncReqResult; 
    static RaCtxChanCreateFunc gRaCtxChanCreate; 
    static RaCtxCqCreateFunc gRaCtxCqCreate; 
    static RaCtxQpCreateFunc gRaCtxQpCreate; 
    static RaCtxTokenIdAllocFunc gRaCtxTokenIdAlloc; 
    static RaCtxQpImportFunc gRaCtxQpImport; 
    static RaCtxQpBindFunc gRaCtxQpBind; 
    static RaCtxLmemRegisterFunc gRaCtxLmemRegister; 
    static RaCtxRmemImportFunc gRaCtxRmemImport; 
    static RaCtxRmemUnimportFunc gRaCtxRmemUnimport; 
    static RaCtxLmemUnregisterFunc gRaCtxLmemUnregister; 
    static RaCtxQpUnbindFunc gRaCtxQpUnbind; 
    static RaCtxQpUnimportFunc gRaCtxQpUnimport; 
    static RaCtxTokenIdFreeFunc gRaCtxTokenIdFree;
    static RaCtxQpDestroyFunc gRaCtxQpDestroy; 
    static RaCtxCqDestroyFunc gRaCtxCqDestroy; 
    static RaCtxChanDestroyFunc gRaCtxChanDestroy; 
    static RaCtxDeinitFunc gRaCtxDeinit; 
    static RaCtxQpQueryBatchFunc gRaCtxQpQueryBatch;
    static RaBatchSendWrFunc gRaBatchSendWr;
    static RaCtxUpdateCiFunc gRaCtxUpdateCi;
    static RaCustomChannelFunc gRaCustomChannel;
    static RaGetTpInfoListAsyncFunc gRaGetTpInfoListAsync; 
    static RaGetIfNumFunc gRaGetIfNum;
    static RaGetIfAddrsFunc gRaGetIfAddrs;
};
    
} // namespace shm

#endif // !DL_HCCP_V2_API_H
