/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEMI_DEVICE_UDMA_H
#define SHMEMI_DEVICE_UDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"

enum class aclshmemi_udma_opcode_t : uint32_t {
    UDMA_OP_SEND = 0,
    UDMA_OP_SEND_WITH_IMM,
    UDMA_OP_SEND_WITH_INV,
    UDMA_OP_WRITE,
    UDMA_OP_WRITE_WITH_IMM,
    UDMA_OP_WRITE_WITH_NOTIFY,
    UDMA_OP_READ,
    UDMA_OP_CAS,
    UDMA_OP_ATOMIC_SWAP,
    UDMA_OP_ATOMIC_STORE,
    UDMA_OP_ATOMIC_LOAD,
    UDMA_OPCODE_FAA = 0xb,
    UDMA_OP_WRITE_WITH_REDUCE = 0x10, // self-defined opcode, will be converted to UDMA_OP_WRITE in actual usage
    UDMA_OPCODE_NOP = 0x11
};

struct ACLSHMEMAIVUDMAInfo {
    uint32_t qpNum;  // number of QP per connection
    uint64_t sqPtr;  // pointer to send queue address array of size [PE_NUM][qpNum]
    uint64_t rqPtr;  // pointer to receive queue address array of size [PE_NUM][qpNum]
    uint64_t scqPtr; // pointer to send completion queue address array of size [PE_NUM][qpNum]
    uint64_t rcqPtr; // pointer to receive completion queue address array of size [PE_NUM][qpNum]
    uint64_t memPtr; // pointer to memory region array of size [MAX_PE_NUM]
};

struct ACLSHMEMUBmemInfo {
    bool tokenValueValid;      // token_en 表示是否使能token
    uint32_t rmtJettyType : 2; // 表示远端jetty的类型
    uint8_t targetHint;        // jettygrp场景使用
    uint32_t tpn;              // 对应着tp_id 区分传输层是简易传输层还是完整传输层
    uint32_t tid;              // 对应着SQE的rmt_jetty_or_seg_id，来源是udma_seg->tid;
    uint32_t rmtTokenValue;    // 对应着SQE的rmt_token_value，来源是udma_seg->token_value.token;
    uint32_t len;
    uint64_t addr;             // 来源urma_sge的addr，对应SQE的rmt_addr_l_or_token_id，rmt_addr_h_or_token_value
    uint64_t eidAddr;
};

enum class ACLSHMEMUDMADBMode : int32_t { INVALID_DB = -1, HW_DB = 0, SW_DB };

struct ACLSHMEMUDMAWQCtx {
    uint32_t wqn;         // work queue number
    uint64_t bufAddr;     // start address of ring buffer
    uint32_t baseBkShift; // log2(size of each WQE)
    uint32_t depth;       // depth of ring buffer
    uint64_t headAddr;    // work queue head (Producer Index) address
    uint64_t tailAddr;    // work queue tail (Consumer Index) address
    ACLSHMEMUDMADBMode dbMode;
    uint64_t dbAddr;      // doorbell address
    uint32_t sl;          // service level
    uint64_t wqeCntAddr;  // wqe count address
    uint64_t amoAddr;     // amo address to store fetch data
};

struct ACLSHMEMUDMACqCtx {
    uint32_t cqn;         // completion queue number
    uint64_t bufAddr;     // start address of ring buffer
    uint32_t baseBkShift; // log2(size of each CQE)
    uint32_t depth;       // depth of ring buffer
    uint64_t headAddr;    // work queue head (Producer Index) address
    uint64_t tailAddr;    // work queue tail (Consumer Index) address
    ACLSHMEMUDMADBMode dbMode;
    uint64_t dbAddr;      // doorbell address
};

struct ACLSHMEMSqeCtx { // 对应着 ACLSHMEMwqeCtx
    /* byte 4 */
    uint32_t sqeBbIdx : 16;
    uint32_t flag : 8;
    uint32_t rsv0 : 3;
    uint32_t nf : 1;
    uint32_t tokenEn : 1;
    uint32_t rmtJettyType : 2;
    uint32_t owner : 1;
    /* byte 8 */
    uint32_t targetHint : 8;
    uint32_t opcode : 8;
    uint32_t rsv1 : 6;
    uint32_t inlineMsgLen : 10;
    /* byte 12 */
    uint32_t tpId : 24;
    uint32_t sgeNum : 8;
    /* byte 16 */
    uint32_t rmtJettyOrSegId : 20;
    uint32_t rsv2 : 12;
    /* byte 20 - 32 */
    // For better perf, use 2 uint64_t to represent int8[16]
    uint64_t rmtEidL;
    uint64_t rmtEidH;
    /* byte 36 */
    uint32_t rmtTokenValue;
    /* byte 40 */
    uint32_t udfType : 8;
    uint32_t reduceDataType : 4;
    uint32_t reduceOpcode : 4;
    uint32_t rsv3 : 16;
    /* byte 44 - 48*/
    uint32_t rmtAddrLOrTokenId;
    uint32_t rmtAddrHOrTokenValue;
};

struct ACLSHMEMSgeCtx { // 对应着ACLSHMEMsegCtx
    uint32_t len;
    uint32_t token_id;
    uint64_t va;
};

struct ACLSHMEMNotifyCtx {
    /* byte 48 - 52 */
    uint32_t notifyTokenId : 20;
    uint32_t rsv : 12;
    /* byte 52 - 56 */
    uint32_t notifyTokenValue;
    /* byte 56 - 60 */
    uint32_t notifyAddrL;
    /* byte 60 - 64 */
    uint32_t notifyAddrH;
    /* byte 64 - 68 */
    uint32_t notifyDataL;
    /* byte 68 - 72 */
    uint32_t notifyDataH;
    /* byte 72 - 80 */
    uint32_t rsv2[2];
};

struct ACLSHMEMJfcCqeCtx { // 对应ACLSHMEMcqeCtx
    /* DW0 */
    uint32_t sR : 1;
    uint32_t isJetty : 1;
    uint32_t owner : 1;
    uint32_t inlineEn : 1;
    uint32_t opcode : 3;
    uint32_t fd : 1;
    uint32_t rsv : 8;
    uint32_t substatus : 8;
    uint32_t status : 8;
    /* DW1 */
    uint32_t entryIdx : 16;
    uint32_t localNumL : 16;
    /* DW2 */
    uint32_t localNumH : 4;
    uint32_t rmtIdx : 20;
    uint32_t rsv1 : 8;
    /* DW3 */
    uint32_t tpn : 24;
    uint32_t rsv2 : 8;
    /* DW4 */
    uint32_t byteCnt;
    /* DW5 ~ DW6 */
    uint32_t userDataL;
    uint32_t userDataH;
    /* DW7 ~ DW10 */
    uint32_t rmtEid[4];
    /* DW11 ~ DW12 */
    uint32_t dataL;
    uint32_t dataH;
    /* DW13 ~ DW15 */
    uint32_t inlineData[3];
};

struct ACLSHMEMUDMADeviceMeta {
    uint32_t entityId;
    uint32_t rankId;
    uint32_t rankSize;
    uint32_t extraContextSize;
    uint64_t symmetricSize;
    uint64_t qpInfoAddress; // 对应着ACLSHMEMAIVUDMAInfo
    uint64_t reserved[12];  // total 128B, equal HYBM_DEVICE_PRE_META_SIZE
};

struct default_op_tag_t {}; // 默认类别的标记
struct atomic_op_tag_t {};  // atomic语义类别的标记
struct signal_op_tag_t {};  // 信号操作语义类别的标记
template <aclshmemi_udma_opcode_t OP_CODE>
struct aclshmemi_op_category_t {
    using type = default_op_tag_t;
};

template <>
struct aclshmemi_op_category_t<aclshmemi_udma_opcode_t::UDMA_OP_CAS> {
    using type = atomic_op_tag_t;
};

template <>
struct aclshmemi_op_category_t<aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_REDUCE> {
    using type = atomic_op_tag_t;
};

template <>
struct aclshmemi_op_category_t<aclshmemi_udma_opcode_t::UDMA_OPCODE_FAA> {
    using type = atomic_op_tag_t;
};

template <>
struct aclshmemi_op_category_t<aclshmemi_udma_opcode_t::UDMA_OP_WRITE_WITH_NOTIFY> {
    using type = signal_op_tag_t;
};

template <typename T, typename OP_TAG>
struct aclshmemi_udma_params_impl_t {};

template <typename T>
struct aclshmemi_udma_params_impl_t<T, atomic_op_tag_t> {
    T value;
    T cond;
};

template <typename T>
struct aclshmemi_udma_params_impl_t<T, signal_op_tag_t> {
    __gm__ uint64_t* sig_addr;
    uint64_t signal;
};

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
using aclshmemi_udma_params_t = aclshmemi_udma_params_impl_t<T, typename aclshmemi_op_category_t<OP_CODE>::type>;

ACLSHMEM_DEVICE __gm__ ACLSHMEMAIVUDMAInfo* aclshmemi_udma_qp_info_fetch();

/**
 * @brief UDMA Poll Completion Queue (CQ) function. Return status: 0 means success, non-zero means error.
 *
 * @param pe                     [in] destination PE ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param idx                    [in] expect completion queue consumer index after polling
 */
ACLSHMEM_DEVICE uint32_t aclshmemi_udma_poll_cq(uint32_t pe, uint32_t qpIdx, uint32_t idx);

ACLSHMEM_DEVICE void aclshmemi_udma_poll_cq_update_info(
    uint32_t curTail, uint32_t qpIdx, __gm__ ACLSHMEMUDMACqCtx* cqCtxEntry, __gm__ ACLSHMEMUDMAWQCtx* wqCtxEntry);

/**
 * @brief AIV direct UDMA helper function for post send, prepare WQE and ring doorbell.
 *
 * @param remoteAddr             [in] address in remote HBM
 * @param localAddr              [in] address in lcoal HBM
 * @param pe                     [in] destination PE ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param opcode                 [in] udma opcode in aclshmemi_udma_opcode_t enum class
 * @param messageLen             [in] message length in Bytes
 * @param params                 [in] extra parameters
 */
template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_post_send(
    __gm__ uint8_t* remoteAddr, __gm__ uint8_t* localAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    const aclshmemi_udma_params_t<T, OP_CODE>& params = {});

ACLSHMEM_DEVICE void aclshmemi_udma_post_send_update_info(uint32_t curHead, __gm__ ACLSHMEMUDMAWQCtx*& qpCtxEntry);

/**
 * @brief Asynchronous UDMA Write function.
 *
 * @param destDmaAddr            [in] destination address in remote HBM
 * @param srcDmaAddr             [in] source address in local HBM
 * @param pe                     [in] destination PE ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_write(
    __gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen);

template <typename T, aclshmemi_udma_opcode_t OP_CODE>
ACLSHMEM_DEVICE void aclshmemi_udma_write_notify(
    __gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t pe, uint32_t qpIdx, uint64_t messageLen,
    const aclshmemi_udma_params_t<T, OP_CODE>& params = {});

/**
 * @brief Asynchronous UDMA READ function.
 *
 * @param destDmaAddr            [in] destination address in local HBM
 * @param srcDmaAddr             [in] source address in remote HBM
 * @param srcPe                  [in] source PE ID
 * @param qpIdx                  [in] QP index in multi-QP scenario (default 0 for single QP)
 * @param messageLen             [in] message length in Bytes
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_read(
    __gm__ T* destDmaAddr, __gm__ T* srcDmaAddr, uint32_t srcPe, uint32_t qpIdx, uint64_t messageLen);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_get_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe);

template <typename T>
ACLSHMEM_DEVICE void aclshmemi_udma_put_nbi(__gm__ T* dst, __gm__ T* src, uint32_t elem_size, int pe);

#endif