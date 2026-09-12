/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dl_dcmi_api.h"
#include <dlfcn.h>
#include <unistd.h>

namespace shm {

std::mutex DlDcmiApi::gMutex;
bool DlDcmiApi::gLoaded = false;
void* DlDcmiApi::dcmiHandle = nullptr;
const char* DlDcmiApi::gDcmiLibName = "libdcmi.so";

dcmiInitFunc DlDcmiApi::pDcmiInit = nullptr;
dcmiv2GetUrmaDeviceCntFunc DlDcmiApi::pDcmiv2GetUrmaDeviceCnt = nullptr;
dcmiv2GetEidListByUrmaDevIndexFunc DlDcmiApi::pDcmiv2GetEidListByUrmaDevIndex = nullptr;
dcmiv2GetMainboardIdFunc DlDcmiApi::pDcmiv2GetMainboardId = nullptr;
dcmiv2GetDevicePcieInfoFunc DlDcmiApi::pDcmiv2GetDevicePcieInfo = nullptr;
dcmiv2GetDeviceInfoFunc DlDcmiApi::pDcmiv2GetDeviceInfo = nullptr;
getLogicIdFromPhyIdFunc DlDcmiApi::pGetLogicIdFromPhyId = nullptr;

int DlDcmiApi::LoadLibrary() {
    std::lock_guard<std::mutex> lock(gMutex);
    if (gLoaded) {
        return 0;
    }
    
    if (dcmiHandle != nullptr) {
        const int maxWaitTime = 10;
        for (int i = 0; i < maxWaitTime; i++) {
            if (gLoaded) {
                return 0;
            }
            sleep(1);
        }
        return 0;
    }

    dcmiHandle = dlopen(gDcmiLibName, RTLD_LAZY);
    if (dcmiHandle == nullptr) {
        return -1;
    }

    pDcmiInit = (dcmiInitFunc)dlsym(dcmiHandle, "dcmiv2_init");
    pDcmiv2GetUrmaDeviceCnt = (dcmiv2GetUrmaDeviceCntFunc)dlsym(dcmiHandle, "dcmiv2_get_urma_device_cnt");
    pDcmiv2GetEidListByUrmaDevIndex = (dcmiv2GetEidListByUrmaDevIndexFunc)dlsym(dcmiHandle, "dcmiv2_get_eid_list_by_urma_dev_index");
    pDcmiv2GetMainboardId = (dcmiv2GetMainboardIdFunc)dlsym(dcmiHandle, "dcmiv2_get_mainboard_id");
    pDcmiv2GetDevicePcieInfo = (dcmiv2GetDevicePcieInfoFunc)dlsym(dcmiHandle, "dcmiv2_get_device_pcie_info");
    pDcmiv2GetDeviceInfo = (dcmiv2GetDeviceInfoFunc)dlsym(dcmiHandle, "dcmiv2_get_device_info");
    pGetLogicIdFromPhyId = (getLogicIdFromPhyIdFunc)dlsym(dcmiHandle, "dcmiv2_get_dev_id_by_chip_phy_id");

    if (pGetLogicIdFromPhyId == NULL) {
        pGetLogicIdFromPhyId = (getLogicIdFromPhyIdFunc)dlsym(dcmiHandle, "dcmiv2_get_dev_id_from_chip_phyid");
    }

    if ((pDcmiInit == nullptr) ||
        (pDcmiv2GetUrmaDeviceCnt == nullptr) ||
        (pDcmiv2GetEidListByUrmaDevIndex == nullptr) ||
        (pDcmiv2GetMainboardId == nullptr) ||
        (pDcmiv2GetDevicePcieInfo == nullptr) ||
        (pDcmiv2GetDeviceInfo == nullptr) ||
        (pGetLogicIdFromPhyId == nullptr)) {
        dlclose(dcmiHandle);
        dcmiHandle = nullptr;
        return -1;
    }

    (void)pDcmiInit(); // dcmi_init may have been called before
    gLoaded = true;
    return 0;
}

void DlDcmiApi::CleanupLibrary() {
    std::lock_guard<std::mutex> lock(gMutex);
    if (dcmiHandle != nullptr) {
        dlclose(dcmiHandle);
        dcmiHandle = nullptr;
    }
    gLoaded = false;
    pDcmiInit = nullptr;
    pDcmiv2GetUrmaDeviceCnt = nullptr;
    pDcmiv2GetEidListByUrmaDevIndex = nullptr;
    pDcmiv2GetMainboardId = nullptr;
    pDcmiv2GetDevicePcieInfo = nullptr;
    pDcmiv2GetDeviceInfo = nullptr;
    pGetLogicIdFromPhyId = nullptr;
}

} // namespace shm