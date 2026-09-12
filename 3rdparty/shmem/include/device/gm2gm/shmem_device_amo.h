/**
 * @cond IGNORE_COPYRIGHT
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * @endcond
 */
#ifndef SHMEM_DEVICE_ATOMIC_H
#define SHMEM_DEVICE_ATOMIC_H

#include "kernel_operator.h"
#include "device/gm2gm/engine/shmem_device_mte.h"

/**
 * @brief Standard Atomic Add Types and Names
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |int8       | int8      |
 * |int16      | int16     |
 * |int32      | int32     |
 * |uint32     | uint32    |
 * |int64      | int64     |
 * |uint64     | uint64    |
 * |half       | half      |
 * |bfloat16   | bfloat16  |
 * |float      | float     |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(FUNC) \
    FUNC(int8, int8_t);                     \
    FUNC(int16, int16_t);                   \
    FUNC(int32, int32_t);                   \
    FUNC(uint32, uint32_t);                 \
    FUNC(int64, int64_t);                   \
    FUNC(uint64, uint64_t);                 \
    FUNC(half, half);                       \
    FUNC(bfloat16, bfloat16_t);             \
    FUNC(float, float)

/**
 * @brief  Automatically generates aclshmem atomic add functions for different data types
 *         (MTE supports int8, int16, int32, float, half, bfloat16; UDMA supports int32, uint32, int64, uint64, float).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_add(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Asynchronous interface. Perform contiguous data atomic add operation on
 * symmetric memory from the specified PE to address on the local PE.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Value atomic add to destination.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_ADD_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_add(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(ACLSHMEM_ATOMIC_ADD_TYPENAME);
/** \endcond */
#define shmem_int8_atomic_add aclshmem_int8_atomic_add
#define shmem_int16_atomic_add aclshmem_int16_atomic_add
#define shmem_int32_atomic_add aclshmem_int32_atomic_add
#define shmem_half_atomic_add aclshmem_half_atomic_add
#define shmem_bfloat16_atomic_add aclshmem_bfloat16_atomic_add
#define shmem_float_atomic_add aclshmem_float_atomic_add

/**
 * @brief Standard Atomic Fetch Types and Names (supported types for fetch_add and compare swap)
 *
 * |NAME       | TYPE      |
 * |-----------|-----------|
 * |int32      | int32     |
 * |uint32     | uint32    |
 * |int64      | int64     |
 * |uint64     | uint64    |
 */
#define ACLSHMEM_TYPE_FUNC_ATOMIC(FUNC) \
    FUNC(int32, int32_t);                     \
    FUNC(uint32, uint32_t);                   \
    FUNC(int64, int64_t);                     \
    FUNC(uint64, uint64_t)

/**
 * @brief  Automatically generates aclshmem atomic fetch add functions for different data types
 *         (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_add(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Add value to dst (remote symmetric address) on the specified PE pe,
 * and return the previous content of dst.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Value atomic add to destination.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous content of dst.
 */
#define ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_add(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_FETCH_ADD_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic compare swap functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_compare_swap(\_\_gm\_\_ TYPE *dst, TYPE cond, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Conditionally update dst (remote symmetric address) on the specified PE pe
 * and return the previous content of dst. If cond and the remote dst value are equal,
 * then value is swapped into the remote dst; otherwise, the remote dst is unchanged. In either case, the old
 * value of the remote dest is returned.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **cond**   - [in] Condition compared to the remote dst value.
 * - **value**  - [in] Value atomic swap to destination.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous content of dst.
 */
#define ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_compare_swap(__gm__ TYPE *dst, TYPE cond, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_COMPARE_SWAP_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch(\_\_gm\_\_ TYPE *dst, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Fetch the contents of dst (remote symmetric address) on the specified PE pe
 * and return the contents.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the contents of dst.
 */
#define ACLSHMEM_ATOMIC_FETCH_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch(__gm__ TYPE *dst, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_FETCH_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic set functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_set(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Set value to dst (remote symmetric address) on the specified PE pe
 * without returning a value.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Value to be atomically written to the remote PE.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_SET_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_set(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_SET_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic swap functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_swap(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Swap value to dst (remote symmetric address) on the specified PE pe
 * and return the previous contents of dst.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Value to be atomically written to the remote PE.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous contents of dst.
 */
#define ACLSHMEM_ATOMIC_SWAP_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_swap(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_SWAP_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch inc functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_inc(\_\_gm\_\_ TYPE *dst, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Increment dst (remote symmetric address) on the specified PE pe by one
 * and return the previous contents of dst.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous contents of dst.
 */
#define ACLSHMEM_ATOMIC_FETCH_INC_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_inc(__gm__ TYPE *dst, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_FETCH_INC_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic inc functions for different data types
 *        (e.g., int32, uint32, int64, uint64, float).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_inc(\_\_gm\_\_ TYPE *dst, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Increment dst (remote symmetric address) on the specified PE pe by one
 * without returning a value.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_INC_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_inc(__gm__ TYPE *dst, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC_ADD(ACLSHMEM_ATOMIC_INC_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch and functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_and(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Operand of bitwise AND operation.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous contents of dst.
 */
#define ACLSHMEM_ATOMIC_FETCH_AND_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_and(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_FETCH_AND_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic and functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_and(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Perform a bitwise AND operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Operand of bitwise AND operation.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_AND_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_and(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_AND_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch or functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_or(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Operand of bitwise OR operation.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous contents of dst.
 */
#define ACLSHMEM_ATOMIC_FETCH_OR_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_or(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_FETCH_OR_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic or functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_or(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Perform a bitwise OR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Operand of bitwise OR operation.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_OR_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_or(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_OR_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic fetch xor functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE TYPE aclshmem_NAME_atomic_fetch_xor(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, and return the previous contents of dst.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Operand of bitwise XOR operation.
 * - **pe**     - [in] PE number of the remote PE.
 *
 * @par Returns
 *      Return the previous contents of dst.
 */
#define ACLSHMEM_ATOMIC_FETCH_XOR_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE TYPE aclshmem_##NAME##_atomic_fetch_xor(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_FETCH_XOR_TYPENAME);
/** \endcond */

/**
 * @brief  Automatically generates aclshmem atomic xor functions for different data types
 *        (e.g., int32, uint32, int64, uint64).
 *        The macro parameters: NAME is the function name suffix, TYPE is the operation data type.
 *
 * \remark ACLSHMEM_DEVICE void aclshmem_NAME_atomic_xor(\_\_gm\_\_ TYPE *dst, TYPE value, int32_t pe)
 *
 * @par Function Description
 * Synchronous interface. Perform a bitwise XOR operation on dst (remote symmetric address) on the
 * specified PE pe with the operand value, without returning a value.
 *
 * @par Parameters
 * - **dst**    - [in] Pointer on local device of the destination data.
 * - **value**  - [in] Operand of bitwise XOR operation.
 * - **pe**     - [in] PE number of the remote PE.
 */
#define ACLSHMEM_ATOMIC_XOR_TYPENAME(NAME, TYPE) \
    ACLSHMEM_DEVICE void aclshmem_##NAME##_atomic_xor(__gm__ TYPE *dst, TYPE value, int32_t pe)

/** \cond */
ACLSHMEM_TYPE_FUNC_ATOMIC(ACLSHMEM_ATOMIC_XOR_TYPENAME);
/** \endcond */

#include "gm2gm/shmem_device_amo.hpp"
#endif