/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_DEVICE_SDMA_TRANSPORT_MANAGER_H
#define MF_HYBRID_DEVICE_SDMA_TRANSPORT_MANAGER_H

#include <map>
#include <mutex>
#include <memory>
#include <vector>

#include "host_device/shmem_common_types.h"
#include "transport_manager.h"
#include "aclnn/aclnn_base.h"

namespace shm {
namespace transport {
namespace device {
struct host_stream_info_t { // 与aicpu算子相同定义
    uint64_t stream_;
    uint64_t ctx_;
    int32_t stream_id;
    uint32_t sq_id;
    uint32_t cq_id;
    uint32_t logic_cq_id;
    uint64_t cqe_addr;
    int32_t dev_id;
    uint8_t reserved[20]; // 对齐到64字节
};

struct sdma_op_res_info_t {
    uint64_t size;
    uint64_t streams_addr;
    uint64_t workspace_addr;
    uint8_t reserved[40]; // 对齐到64字节
};

class SdmaTransportManager : public TransportManager {
public:
    SdmaTransportManager() = default;
    ~SdmaTransportManager() override = default;

    Result OpenDevice(const TransportOptions &options) override;
    Result CloseDevice() override;

    Result RegisterMemoryRegion(const TransportMemoryRegion &/*mr*/) override { return ACLSHMEM_SUCCESS; }
    Result UnregisterMemoryRegion(uint64_t /*addr*/) override { return ACLSHMEM_SUCCESS; }
    Result QueryMemoryKey(uint64_t /*addr*/, TransportMemoryKey &/*key*/) override { return ACLSHMEM_SUCCESS; }
    Result ParseMemoryKey(const TransportMemoryKey &/*key*/, uint64_t &/*addr*/,
                          uint64_t &/*size*/) override { return ACLSHMEM_SUCCESS; }
    Result Prepare(const HybmTransPrepareOptions &/*options*/) override { return ACLSHMEM_SUCCESS; }
    Result Connect() override { return ACLSHMEM_SUCCESS; }
    Result AsyncConnect() override { return ACLSHMEM_SUCCESS; }
    Result WaitForConnected(int64_t /*timeoutNs*/) override { return ACLSHMEM_SUCCESS; }
    Result UpdateRankOptions(const HybmTransPrepareOptions &/*options*/) override { return ACLSHMEM_SUCCESS; }
    const std::string &GetNic() const override
    {
        static const std::string empty_nic;
        return empty_nic;
    }

private:
    Result CreateNotifyIds();
    Result CreateStarsStreams(int32_t channel_num);
    Result MallocSdmaWorkspace(size_t workspace_size);
    Result CopyHostOpResToDevice();
    Result CreateAclTensor(const std::vector<uint64_t> &host_data, const std::vector<int64_t> &shape,
                           void **device_addr, aclDataType data_type, aclTensor **tensor);
    Result LaunchSdmaAicpuKernel(uint64_t streams_addr, uint64_t sdma_workspace_addr);

private:
    bool inited_{false};
    uint32_t mype_{0};
    uint32_t npes_{1};

    static sdma_op_res_info_t op_res_info_;
    static void *op_res_info_device_ptr_;
    static std::vector<host_stream_info_t> streams_;
};
} // namespace device
} // namespace transport
} // namespace shm

#endif  // MF_HYBRID_DEVICE_SDMA_TRANSPORT_MANAGER_H
