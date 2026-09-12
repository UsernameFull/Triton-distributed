/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acc_common_util.h"
#include "acc_includes.h"
#include "acc_tcp_server_default.h"

namespace shm {
namespace acc {
AccTcpServerPtr AccTcpServer::Create()
{
    auto server = AccMakeRef<AccTcpServerDefault>();
    if (server.Get() == nullptr) {
        LOG_ERROR("Failed to create AccTcpserverDefault, probably out of memory");
        return nullptr;
    }
    return server.Get();
}
}  // namespace acc
}  // namespace shm