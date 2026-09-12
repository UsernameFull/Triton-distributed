/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef LAUNCH_MAP_H
#define LAUNCH_MAP_H

#include <unordered_map>

enum CocCommType {
    MATMUL_ALLREDUCE = 0,
    ALLGATHER_MATMUL,
    MATMUL_REDUCE_SCATTER,
    MATMUL_REDUCE_SCATTER_PADDING,
    ALLGATHER_MATMUL_WITH_GATHER_RESULT,
    ALLGATHER_MATMUL_PADDING,
    MATMUL_REDUCE_SCATTER_PADDING_AB,
    MATMUL_REDUCE_SCATTER_PADDING_A,
    MATMUL_REDUCE_SCATTER_PADDING_B,
    TYPE_NUM
};

enum CocDataType {
    FP16 = 1,
    BF16 = 27
};

using KernelFuncPtr = void (*)(void *, uint64_t, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,
    CocTilingParams &, uint32_t, uint32_t);

class KernelDispatcher {
public:
    static KernelFuncPtr GetKernelFunc(CocCommType commType, CocDataType dataType)
    {
        auto &kernelMap = GetKernelMap();
        if (auto it = kernelMap.find(Hash(commType, dataType)); it != kernelMap.end()) {
            return it->second;
        }
        return nullptr;
    }

    static void RegisterKernelFunc(CocCommType commType, CocDataType dataType, KernelFuncPtr func)
    {
        auto &kernelMap = GetKernelMap();
        kernelMap.insert({Hash(commType, dataType), func});
    }

private:
    static std::unordered_map<int, KernelFuncPtr> &GetKernelMap()
    {
        static std::unordered_map<int, KernelFuncPtr> kernelMap;
        return kernelMap;
    }

    static int Hash(CocCommType commType, CocDataType dataType)
    {
        const int SHIFT_BITS = 8;
        return (commType << SHIFT_BITS) | dataType;
    }
};

#define REGISTER_KERNEL_FUNC(kernelName, commType, dataType)                                                           \
    void Launch##kernelName##dataType(void *, uint64_t, uint8_t *, uint8_t *, uint8_t *, uint8_t *, uint8_t *,         \
        uint8_t *, CocTilingParams &, uint32_t, uint32_t);                                                             \
    namespace {                                                                                                        \
        struct AutoRegister##kernelName##dataType {                                                                    \
            AutoRegister##kernelName##dataType() {                                                                     \
                KernelDispatcher::RegisterKernelFunc(commType, dataType, &Launch##kernelName##dataType);               \
            }                                                                                                          \
        } s_autoRegister##kernelName##dataType;                                                                        \
    }

REGISTER_KERNEL_FUNC(MatmulAllReduce, MATMUL_ALLREDUCE, FP16);
REGISTER_KERNEL_FUNC(AllGatherMatmul, ALLGATHER_MATMUL, FP16);
REGISTER_KERNEL_FUNC(MatmulReduceScatter, MATMUL_REDUCE_SCATTER, FP16);
REGISTER_KERNEL_FUNC(MatmulReduceScatterPaddingAB, MATMUL_REDUCE_SCATTER_PADDING_AB, FP16);
REGISTER_KERNEL_FUNC(MatmulReduceScatterPaddingA, MATMUL_REDUCE_SCATTER_PADDING_A, FP16);
REGISTER_KERNEL_FUNC(MatmulReduceScatterPaddingB, MATMUL_REDUCE_SCATTER_PADDING_B, FP16);
REGISTER_KERNEL_FUNC(AllGatherMatmulWithGatherResult, ALLGATHER_MATMUL_WITH_GATHER_RESULT, FP16);
REGISTER_KERNEL_FUNC(AllGatherMatmulPadding, ALLGATHER_MATMUL_PADDING, FP16);

REGISTER_KERNEL_FUNC(MatmulAllReduce, MATMUL_ALLREDUCE, BF16);
REGISTER_KERNEL_FUNC(AllGatherMatmul, ALLGATHER_MATMUL, BF16);
REGISTER_KERNEL_FUNC(MatmulReduceScatter, MATMUL_REDUCE_SCATTER, BF16);
REGISTER_KERNEL_FUNC(MatmulReduceScatterPaddingAB, MATMUL_REDUCE_SCATTER_PADDING_AB, BF16);
REGISTER_KERNEL_FUNC(MatmulReduceScatterPaddingA, MATMUL_REDUCE_SCATTER_PADDING_A, BF16);
REGISTER_KERNEL_FUNC(MatmulReduceScatterPaddingB, MATMUL_REDUCE_SCATTER_PADDING_B, BF16);
REGISTER_KERNEL_FUNC(AllGatherMatmulWithGatherResult, ALLGATHER_MATMUL_WITH_GATHER_RESULT, BF16);
REGISTER_KERNEL_FUNC(AllGatherMatmulPadding, ALLGATHER_MATMUL_PADDING, BF16);

#undef REGISTER_KERNEL_FUNC

#endif // LAUNCH_MAP_H