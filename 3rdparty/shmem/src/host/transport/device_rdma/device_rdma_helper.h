/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_RDMA_HELPER_H
#define MF_HYBRID_DEVICE_RDMA_HELPER_H

#include <netinet/in.h>

#include <cstdint>
#include <string>
#include "sotre_net.h"
#include "host/shmem_host_def.h"

namespace shm {
namespace transport {
namespace device {
Result ParseDeviceNic(const std::string &nic, uint16_t &port);
Result ParseDeviceNic(const std::string &nic, mf_sockaddr &address);
std::string GenerateDeviceNic(net_addr_t ip, uint16_t port);
}
}
}
#endif  // MF_HYBRID_DEVICE_RDMA_HELPER_H
