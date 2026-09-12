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
#ifndef SHMEM_DEVICE_UDMA_H
#define SHMEM_DEVICE_UDMA_H

#include "kernel_operator.h"
#include "device/shmem_def.h"
#include "gm2gm/engine/shmem_device_udma.hpp"

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 * PE to address on the local device.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations
 * to the same PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param src               [in] Pointer on Symmetric memory of the source data.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe);

/**
 * @brief Asynchronous interface. Copy contiguous data on symmetric memory from the specified
 * PE to address on the local PE.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations
 * to the same PE are not supported.
 *
 * @param dst               [in] GlobalTensor on local device of the destination data.
 * @param src               [in] GlobalTensor on Symmetric memory of the source data.
 * @param buf               [in] LocalTensor on local UB, available space larger than 64 Bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_get_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same PE
 *        are not supported.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(__gm__ T* dst, __gm__ T* src, __ubuf__ T* buf, uint32_t elem_size, int pe);

/**
 * @brief Asynchronous interface. Copy contiguous data on local PE to symmetric address on the specified PE.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @param dst               [in] GlobalTensor on Symmetric memory of the destination data.
 * @param src               [in] GlobalTensor on local device of the source data.
 * @param buf               [in] Pointer on local UB, available space larger than 64 Bytes.
 * @param elem_size         [in] Number of elements in the destination and source arrays.
 * @param pe                [in] PE number of the remote PE.
 * @param sync_id           [in] ID used to sync S\\MTE3 Event.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_nbi(
    const AscendC::GlobalTensor<T>& dst, const AscendC::GlobalTensor<T>& src, const AscendC::LocalTensor<T>& buf,
    uint32_t elem_size, int pe, uint32_t sync_id);

/**
 * @brief Asynchronous interface. Copy a contiguous data from local to symmetric address on the specified PE and
 *        updating a remote signal flag on completion using UDMA.
 *        Template function for different data types.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 *        PE are not supported.
 *
 * @param dst               [in] Pointer on Symmetric memory of the destination data.
 * @param src               [in] Pointer on local device of the source data.
 * @param elem_size         [in] Number of elements in the dest and source arrays.
 * @param sig_addr          [in] Symmetric address of the signal word to be updated.
 * @param signal            [in] The value used to update sig_addr.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_put_signal_nbi(
    __gm__ T* dst, __gm__ T* src, uint32_t elem_size, __gm__ uint64_t* sig_addr, uint64_t signal, int pe);

/**
 * @brief UDMA Quiet function. This synchronous function ensures all previous UDMA WQEs are completed
 * (data has arrived at the destination PE).
 *
 * @param pe                [in] PE number of the remote PE.
 */
ACLSHMEM_DEVICE void aclshmemx_udma_quiet(int pe);

/**
 * @brief Asynchronous interface. Add value to dst (remote symmetric address) on the specified PE pe,
 * and atomically update the dst without returning the value. Supported types: int32, uint32, int64, uint64, float.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Add value to dst (remote symmetric address) on the specified PE pe,
 * and return the previous content of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous content of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_add(__gm__ T* dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Conditionally update dst (remote symmetric address) on the specified PE pe
 * and return the previous content of dst. If cond and the remote dst value are equal,
 * then value is swapped into the remote dst; otherwise, the remote dst is unchanged. In either case, the old
 * value of the remote dest is returned. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param cond              [in] condition for swap
 * @param value             [in] Operand of atomic add
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous content of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_compare_swap(__gm__ T* dst, T cond, T value, int32_t pe);


/**
 * @brief Synchronous interface. Fetch the contents of dst (remote symmetric address) on the specified PE pe
 * and return the contents. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch(__gm__ T *dst, int32_t pe);

/**
 * @brief Synchronous interface. Set value to dst (remote symmetric address) on the specified PE pe
 * without returning a value. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Value to be atomically written to the remote PE.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_set(__gm__ T *dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Swap value to dst (remote symmetric address) on the specified PE pe
 * and return the previous contents of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Value to be atomically written to the remote PE.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_swap(__gm__ T *dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Increment dst (remote symmetric address) on the specified PE pe by one
 * and return the previous contents of dst. Supported types: int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_inc(__gm__ T *dst, int32_t pe);

/**
 * @brief Synchronous interface. Increment dst (remote symmetric address) on the specified PE pe by one
 * without returning a value. Supported types: int32, uint32, int64, uint64, float.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_inc(__gm__ T *dst, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_and(__gm__ T *dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise AND operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_and(__gm__ T *dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_or(__gm__ T *dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise OR operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_or(__gm__ T *dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE.
 * @return                  Return the previous contents of dst.
 */
template <typename T>
ACLSHMEM_DEVICE T aclshmemx_udma_atomic_fetch_xor(__gm__ T *dst, T value, int32_t pe);

/**
 * @brief Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value. Supported types:
 * int32, uint32, int64, uint64.
 *        WARNING: When using UDMA as the underlying transport, concurrent RMA/AMO operations to the same
 * PE are not supported.
 *
 * @param dst               [in] Pointer on local device of the destination data.
 * @param value             [in] Operand of bitwise XOR operation.
 * @param pe                [in] PE number of the remote PE.
 */
template <typename T>
ACLSHMEM_DEVICE void aclshmemx_udma_atomic_xor(__gm__ T *dst, T value, int32_t pe);

#endif
