/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ACLSHMEM_TORCH_REGISTER_H
#define ACLSHMEM_TORCH_REGISTER_H

#define REGISTER_SHMEM_OPS_CLASS(CLASS_NAME, ...) \
static auto registry_##CLASS_NAME = \
torch::jit::class_<ShmemOps::CLASS_NAME>("ShmemOps", #CLASS_NAME) \
.def(torch::jit::init<>()) \
REGISTER_SHMEM_OPS_FUNCS_HELPER(CLASS_NAME, ##__VA_ARGS__)

#define REGISTER_SHMEM_OPS_FUNCS_HELPER(CLASS, ...) \
    REGISTER_SHMEM_OPS_FUNCS_CHOOSER(__VA_ARGS__, REGISTER_SHMEM_OPS_FUNCS_6, REGISTER_SHMEM_OPS_FUNCS_5, REGISTER_SHMEM_OPS_FUNCS_4, REGISTER_SHMEM_OPS_FUNCS_3, REGISTER_SHMEM_OPS_FUNCS_2, REGISTER_SHMEM_OPS_FUNCS_1)(CLASS, ##__VA_ARGS__)

#define REGISTER_SHMEM_OPS_FUNCS_CHOOSER(_1, _2, _3, _4, _5, _6, FUNC, ...) FUNC

#define REGISTER_SHMEM_OPS_FUNCS_1(CLASS, FUNC1) \
.def(#FUNC1, &ShmemOps::CLASS::FUNC1)

#define REGISTER_SHMEM_OPS_FUNCS_2(CLASS, FUNC1, FUNC2) \
.def(#FUNC1, &ShmemOps::CLASS::FUNC1) \
.def(#FUNC2, &ShmemOps::CLASS::FUNC2)

#define REGISTER_SHMEM_OPS_FUNCS_3(CLASS, FUNC1, FUNC2, FUNC3) \
.def(#FUNC1, &ShmemOps::CLASS::FUNC1) \
.def(#FUNC2, &ShmemOps::CLASS::FUNC2) \
.def(#FUNC3, &ShmemOps::CLASS::FUNC3)

#define REGISTER_SHMEM_OPS_FUNCS_4(CLASS, FUNC1, FUNC2, FUNC3, FUNC4) \
.def(#FUNC1, &ShmemOps::CLASS::FUNC1) \
.def(#FUNC2, &ShmemOps::CLASS::FUNC2) \
.def(#FUNC3, &ShmemOps::CLASS::FUNC3) \
.def(#FUNC4, &ShmemOps::CLASS::FUNC4)

#define REGISTER_SHMEM_OPS_FUNCS_5(CLASS, FUNC1, FUNC2, FUNC3, FUNC4, FUNC5) \
.def(#FUNC1, &ShmemOps::CLASS::FUNC1) \
.def(#FUNC2, &ShmemOps::CLASS::FUNC2) \
.def(#FUNC3, &ShmemOps::CLASS::FUNC3) \
.def(#FUNC4, &ShmemOps::CLASS::FUNC4) \
.def(#FUNC5, &ShmemOps::CLASS::FUNC5)

#define REGISTER_SHMEM_OPS_FUNCS_6(CLASS, FUNC1, FUNC2, FUNC3, FUNC4, FUNC5, FUNC6) \
.def(#FUNC1, &ShmemOps::CLASS::FUNC1) \
.def(#FUNC2, &ShmemOps::CLASS::FUNC2) \
.def(#FUNC3, &ShmemOps::CLASS::FUNC3) \
.def(#FUNC4, &ShmemOps::CLASS::FUNC4) \
.def(#FUNC5, &ShmemOps::CLASS::FUNC5) \
.def(#FUNC6, &ShmemOps::CLASS::FUNC6)

#endif // ACLSHMEM_TORCH_REGISTER_H