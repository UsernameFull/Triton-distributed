/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_UDMA_DEF_H
#define MF_HYBRID_DEVICE_UDMA_DEF_H

#include <cstdint>

namespace shm {
namespace transport {
namespace device {
struct ACLSHMEMUBmemInfo {
    bool token_value_valid;      /* token_en 表示是否使能token */
    uint32_t rmt_jetty_type : 2; /* 表示远端jetty的类型 */
    uint8_t target_hint;
    uint32_t tpn;             /* 对应着tp_id 区分传输层是简易传输层还是完整传输层 */
    uint32_t tid;             /* 对应着SQE的rmt_jetty_or_seg_id，来源是udma_seg->tid; */
    uint32_t rmt_token_value; /* 对应着SQE的rmt_token_value，来源是udma_seg->token_value.token; */
    uint32_t len;
    uint64_t addr; /* 来源urma_sge的addr，对应SQE的rmt_addr_l_or_token_id，rmt_addr_h_or_token_value */
    uint64_t eidAddr;
};

enum class ACLSHMEMUDMADBMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct ACLSHMEMUDMAWQCtx {
    uint32_t wqn;              /* work queue number */
    uint64_t bufAddr;          /* start address of ring buffer qbuf */
    uint32_t wqeShiftSize;     /* basic block size of each WQE */
    uint32_t depth;            /* depth of ring buffer qbufSize */
    uint64_t headAddr;         /* work queue head (Producer Index) address  pi */
    uint64_t tailAddr;         /* work queue tail (Consumer Index) address  ci  */
    ACLSHMEMUDMADBMode dbMode; /* dbtype */
    uint64_t dbAddr;           /* doorbell address  */
    uint32_t sl;               /* service level */
    uint64_t wqeCntAddr;       /* wqe count address*/
    uint64_t amoAddr;          /* amo address to store fetch data*/
};

struct ACLSHMEMUDMACqCtx {
    uint32_t cqn;               /* completion queue number */
    uint64_t bufAddr;           /* start address of ring buffer */
    uint32_t cqeShiftSize;      /* basic block size of each CQE */
    uint32_t depth;             /* depth of ring buffer */
    uint64_t headAddr;          /* work queue head (Producer Index) address */
    uint64_t tailAddr;          /* work queue tail (Consumer Index) address */
    ACLSHMEMUDMADBMode dbMode;  /* dbtype */
    uint64_t dbAddr;            /* doorbell address */
};

struct ACLSHMEMAIVUDMAInfo {
    uint32_t qpNum;  /* number of QP per connection */
    uint64_t sqPtr;  /* pointer to send queue address array of size [PE_NUM][qpNum] */
    uint64_t rqPtr;  /* pointer to receive queue address array of size [PE_NUM][qpNum] */
    uint64_t scqPtr; /* pointer to send completion queue address array of size [PE_NUM][qpNum] */
    uint64_t rcqPtr; /* pointer to receive completion queue address array of size [PE_NUM][qpNum] */
    uint64_t memPtr; /* pointer to memory region array of size [MAX_PE_NUM] */
};

}  // namespace device
}  // namespace transport
}  // namespace shm
#endif  // MF_HYBRID_DEVICE_UDMA_DEF_H