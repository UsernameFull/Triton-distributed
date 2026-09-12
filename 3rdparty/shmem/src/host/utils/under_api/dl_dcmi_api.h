/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBM_CORE_DL_DCMI_API_H
#define MF_HYBM_CORE_DL_DCMI_API_H

#include <mutex>
#include "dl_dcmi_types.h"
namespace shm {
using dcmiInitFunc = int (*)(void);
using dcmiv2GetUrmaDeviceCntFunc = int (*)(int, unsigned int *);
using dcmiv2GetEidListByUrmaDevIndexFunc = int (*)(int, int, dcmi_urma_eid_info_t*, int*);
using dcmiv2GetMainboardIdFunc = int (*)(int, unsigned int*);
using dcmiv2GetDevicePcieInfoFunc = int (*)(int, struct dcmi_pcie_info_all*);
using dcmiv2GetDeviceInfoFunc = int (*)(int, enum dcmi_main_cmd, unsigned int, void*, unsigned int*);
using getLogicIdFromPhyIdFunc = int (*)(unsigned int, unsigned int*);

class DlDcmiApi {
public:
    static int LoadLibrary();
    static void CleanupLibrary();

    static inline int DcmiInit()
    {
        if (pDcmiInit == nullptr) {
            return -1;
        }
        return pDcmiInit();
    }

    static inline int Dcmiv2GetUrmaDeviceCnt(int npu_id, unsigned int *dev_cnt)
    {
        if (pDcmiv2GetUrmaDeviceCnt == nullptr) {
            return -1;
        }
        return pDcmiv2GetUrmaDeviceCnt(npu_id, dev_cnt);
    }

    static inline int Dcmiv2GetEidListByUrmaDevIndex(int npu_id, int urma_dev_index, dcmi_urma_eid_info_t* eid_list, int* eid_cnt)
    {
        if (pDcmiv2GetEidListByUrmaDevIndex == nullptr) {
            return -1;
        }
        return pDcmiv2GetEidListByUrmaDevIndex(npu_id, urma_dev_index, eid_list, eid_cnt);
    }

    static inline int Dcmiv2GetMainboardId(int npu_id, unsigned int* mainboard_id)
    {
        if (pDcmiv2GetMainboardId == nullptr) {
            return -1;
        }
        return pDcmiv2GetMainboardId(npu_id, mainboard_id);
    }

    static inline int Dcmiv2GetDevicePcieInfo(int npu_id, struct dcmi_pcie_info_all* pcie_info)
    {
        if (pDcmiv2GetDevicePcieInfo == nullptr) {
            return -1;
        }
        return pDcmiv2GetDevicePcieInfo(npu_id, pcie_info);
    }

    static inline int Dcmiv2GetDeviceInfo(int npu_id, enum dcmi_main_cmd main_cmd, unsigned int sub_cmd, void* buf, unsigned int* size)
    {
        if (pDcmiv2GetDeviceInfo == nullptr) {
            return -1;
        }
        return pDcmiv2GetDeviceInfo(npu_id, main_cmd, sub_cmd, buf, size);
    }

    static inline int GetLogicIdFromPhyId(unsigned int phy_id, unsigned int* logic_id)
    {
        if (pGetLogicIdFromPhyId == nullptr) {
            return -1;
        }
        return pGetLogicIdFromPhyId(phy_id, logic_id);
    }

private:
    static std::mutex gMutex;
    static bool gLoaded;
    static void* dcmiHandle;
    static const char* gDcmiLibName;

    static dcmiInitFunc pDcmiInit;
    static dcmiv2GetUrmaDeviceCntFunc pDcmiv2GetUrmaDeviceCnt;
    static dcmiv2GetEidListByUrmaDevIndexFunc pDcmiv2GetEidListByUrmaDevIndex;
    static dcmiv2GetMainboardIdFunc pDcmiv2GetMainboardId;
    static dcmiv2GetDevicePcieInfoFunc pDcmiv2GetDevicePcieInfo;
    static dcmiv2GetDeviceInfoFunc pDcmiv2GetDeviceInfo;
    static getLogicIdFromPhyIdFunc pGetLogicIdFromPhyId;
};

} // namespace shm

#endif  // MF_HYBM_CORE_DL_DCMI_API_H