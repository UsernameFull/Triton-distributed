# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

aclshmem_add_fusion_example(dispatch_gmm_combine main.cpp)

if(USE_EXAMPLES)
    target_compile_options(dispatch_gmm_combine PRIVATE --cce-auto-sync)
    target_include_directories(dispatch_gmm_combine PRIVATE ${PROJECT_SOURCE_DIR}/examples/dispatch_gmm_combine/include)
endif()
