/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __DCMI_TYPES_H__
#define __DCMI_TYPES_H__

#include <stddef.h>

#define DCMI_URMA_EID_SIZE (16)
#define MAX_EID_NUM (32)
#define MAX_NPU_COUNT (64)

typedef union dcmi_urma_eid{
    unsigned char raw[DCMI_URMA_EID_SIZE];
    struct {
        unsigned long subnet_prefix;
        unsigned long interface_id;
    }in6;
}dcmi_urma_eid_t;

typedef struct dcmi_urma_eid_info {
    dcmi_urma_eid_t eid;
    unsigned int eid_index;
}dcmi_urma_eid_info_t;

#define MAX_EID_PER_UE (32)
typedef struct {
    dcmi_urma_eid_info_t eidList[MAX_EID_PER_UE];
    unsigned int eidNum;
}UBEntity;

#define MAX_UE_PER_NPU (8)
typedef struct {
    UBEntity ueList[MAX_UE_PER_NPU];
    unsigned int ueNum;
}UEList;

struct dcmi_pcie_info_all {
unsigned int venderid; //厂商ID
unsigned int subvenderid; //厂商子ID
unsigned int deviceid; //设备ID
unsigned int subdeviceid; //设备子ID
int domain; //pcie domain
unsigned int bdf_busid; //BDF（Bus，Device，Function）中的总线ID
unsigned int bdf_deviceid; //BDF（Bus，Device，Function）中的设备ID
unsigned int bdf_funcid; //BDF（Bus，Device，Function）中的功能ID
unsigned char reserve[32];
};

struct dcmi_spod_info {
    unsigned int sdid;
    unsigned int super_pod_size;
    unsigned int super_pod_id;
    unsigned int server_index;
    unsigned int chassis_id;
    unsigned int super_pod_type;
    unsigned int reserve[6];
};

#define MAIN_BOARD_ID_CARD_NOMESH (0x68)
#define MAIN_BOARD_ID_CARD_2PMESH (0x6a)
#define MAIN_BOARD_ID_CARD_4PMESH (0x6c)
#define MAIN_BOARD_ID_SERVER_TYPE1 (0x23)
#define MAIN_BOARD_ID_SERVER_8PMESH (0x25)
#define MAIN_BOARD_ID_SERVER_16PMESH (0x44)
#define MAIN_BOARD_ID_SERVER_UBX (0x44)
#define MAIN_BOARD_ID_POD         (0x07)
#define MAIN_BOARD_ID_POD_2D      (0x03)

enum dcmi_main_cmd {
    DCMI_MAIN_CMD_DVPP = 0,
    DCMI_MAIN_CMD_ISP,
    DCMI_MAIN_CMD_TS_GROUP_NUM,
    DCMI_MAIN_CMD_CAN,
    DCMI_MAIN_CMD_UART,
    DCMI_MAIN_CMD_UPGRADE = 5,
    DCMI_MAIN_CMD_UFS,
    DCMI_MAIN_CMD_OS_POWER,
    DCMI_MAIN_CMD_LP,
    DCMI_MAIN_CMD_MEMORY,
    DCMI_MAIN_CMD_RECOVERY,
    DCMI_MAIN_CMD_TS,
    DCMI_MAIN_CMD_CHIP_INF,
    DCMI_MAIN_CMD_QOS,
    DCMI_MAIN_CMD_SOC_INFO,
    DCMI_MAIN_CMD_HCCS = 16,
    DCMI_MAIN_CMD_TEMP = 50,
    DCMI_MAIN_CMD_SVM = 51,
    DCMI_MAIN_CMD_VDEV_MNG,
    DCMI_MAIN_CMD_SIO = 56,
    DCMI_MAIN_CMD_DEVICE_SHARE = 0x8001,
    DCMI_MAIN_CMD_MAX
};

typedef enum {
    DCMI_CHIP_INFO_SUB_CMD_CHIP_ID,
    DCMI_CHIP_INFO_SUB_CMD_SPOD_INFO,
    DCMI_CHIP_INFO_SUB_CMD_MAX = 0xFF,
}DCMI_CHIP_INFO_SUB_CMD;

#endif // __DCMI_TYPES_H__