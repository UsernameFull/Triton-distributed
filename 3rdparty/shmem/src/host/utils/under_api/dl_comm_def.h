/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef DL_VER_UTIL_INCLUDE_H
#define DL_VER_UTIL_INCLUDE_H

#include <string>

#include "host/shmem_host_def.h"

enum HybmGvaVersion : uint32_t {
    HYBM_GVA_V1 = 0,
    HYBM_GVA_V2 = 1,
    HYBM_GVA_V3 = 2,
    HYBM_GVA_V4 = 3,
    HYBM_GVA_UNKNOWN
};

enum AscendSocType {
    ASCEND_UNKNOWN = 0,
    ASCEND_910B,
    ASCEND_910C,
    ASCEND_950,
};

enum DeviceSystemInfoType {
    INFO_TYPE_PHY_CHIP_ID = 18,
    INFO_TYPE_PHY_DIE_ID,
    INFO_TYPE_SDID = 26,
    INFO_TYPE_SERVER_ID,
    INFO_TYPE_SCALE_TYPE,
    INFO_TYPE_SUPER_POD_ID,
    INFO_TYPE_ADDR_MODE,
};

typedef struct drv_mem_handle {
    int id;
    uint32_t side;
    uint32_t sdid;
    uint32_t devid;
    uint32_t module_id;
    uint64_t pg_num;
    uint32_t pg_type;
    uint32_t phy_mem_type;
} drv_mem_handle_t;

typedef enum {
    MEM_HANDLE_TYPE_NONE = 0x0,
    MEM_HANDLE_TYPE_FABRIC = 0x1,
    MEM_HANDLE_TYPE_MAX = 0x2,
} drv_mem_handle_type;

#define MEM_SHARE_HANDLE_LEN 128
struct MemShareHandle {
    uint8_t share_info[MEM_SHARE_HANDLE_LEN];
};

enum drv_mem_side {
    MEM_HOST_SIDE = 0,
    MEM_DEV_SIDE,
    MEM_HOST_NUMA_SIDE,
    MEM_MAX_SIDE
};

enum drv_mem_pg_type {
    MEM_NORMAL_PAGE_TYPE = 0,
    MEM_HUGE_PAGE_TYPE,
    MEM_GIANT_PAGE_TYPE,
    MEM_MAX_PAGE_TYPE
};

enum drv_mem_type {
    MEM_HBM_TYPE = 0,
    MEM_DDR_TYPE,
    MEM_P2P_HBM_TYPE,
    MEM_P2P_DDR_TYPE,
    MEM_TS_DDR_TYPE,
    MEM_MAX_TYPE
};

struct drv_mem_prop {
    uint32_t side;       /* enum drv_mem_side */
    uint32_t devid;
    uint32_t module_id;  /* module id defines in ascend_hal_define.h */

    uint32_t pg_type;    /* enum drv_mem_pg_type */
    uint32_t mem_type;   /* enum drv_mem_type */
    uint64_t reserve;
};

typedef enum {
    MEM_ALLOC_GRANULARITY_MINIMUM = 0x0,
    MEM_ALLOC_GRANULARITY_RECOMMENDED,
    MEM_ALLOC_GRANULARITY_INVALID,
} drv_mem_granularity_options;

enum ShareHandleAttrType {
    SHR_HANDLE_ATTR_NO_WLIST_IN_SERVER = 0,
    SHR_HANDLE_ATTR_TYPE_MAX
};

#define SHR_HANDLE_WLIST_ENABLE     0x0
#define SHR_HANDLE_NO_WLIST_ENABLE  0x1
struct ShareHandleAttr {
    unsigned int enableFlag;     /* wlist enable: 0 no wlist enable: 1 */
    unsigned int rsv[8];
};

#define MEM_RSV_TYPE_DEVICE_SHARE_BIT   8 /* mmap va in all opened devices, exclude host to avoid va conflict */
#define MEM_RSV_TYPE_DEVICE_SHARE       (0x1u << MEM_RSV_TYPE_DEVICE_SHARE_BIT)
#define MEM_RSV_TYPE_REMOTE_MAP_BIT     9 /* this va only map remote addr, not create double page table */
#define MEM_RSV_TYPE_REMOTE_MAP         (0x1u << MEM_RSV_TYPE_REMOTE_MAP_BIT)

HybmGvaVersion HybmGetGvaVersion();

static bool DriverVersionCheck(const std::string &ver);

int32_t HalGvaPrecheck(void);

#endif  // DL_VER_UTIL_INCLUDE_H