/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MF_HYBRID_HYBM_TRANSPORT_MANAGER_H
#define MF_HYBRID_HYBM_TRANSPORT_MANAGER_H

#include <memory>
#include "host/shmem_host_def.h"
#include "transport_def.h"

namespace shm {
namespace transport {

class TransportManager {
public:
    static std::shared_ptr<TransportManager> Create(TransportType type);

public:
    TransportManager()= default;

    virtual ~TransportManager() = default;

    /*
     * 1、本地IP（NIC、Device）
     * @return 0 if successful
     */
    virtual Result OpenDevice(const TransportOptions &options) = 0;

    virtual Result CloseDevice() = 0;

    virtual Result ConnectWithOptions(const HybmTransPrepareOptions &options);

    /*
     * 2、注册内存
     * @return 0 if successful
     */
    virtual Result RegisterMemoryRegion(const TransportMemoryRegion &mr) = 0;

    virtual Result UnregisterMemoryRegion(uint64_t addr) = 0;

    virtual Result QueryMemoryKey(uint64_t addr, TransportMemoryKey &key) = 0;

    virtual Result ParseMemoryKey(const TransportMemoryKey &key, uint64_t &addr, uint64_t &size) = 0;

    /*
     * 3、建链前的准备工作
     * @return 0 if successful
     */
    virtual Result Prepare(const HybmTransPrepareOptions &options) = 0;

    /*
     * 4、建链
     * @return 0 if successful
     */
    virtual Result Connect() = 0;

    /*
     * 异步建链
     * @return 0 if successful
     */
    virtual Result AsyncConnect() = 0;

    /*
     * 等待异步建链完成
     * @return 0 if successful
     */
    virtual Result WaitForConnected(int64_t timeoutNs) = 0;

    /*
     * 建链完成后，更新rank配置信息，可以新增rank或减少rank
     */
    virtual Result UpdateRankOptions(const HybmTransPrepareOptions &options) = 0;

    /**
     * 查询
     */
    virtual const std::string &GetNic() const = 0;  // X

    virtual const void *GetQpInfo() const;

protected:
    bool connected_{false};
};

using TransManagerPtr = std::shared_ptr<TransportManager>;
}
}

#endif  // MF_HYBRID_HYBM_TRANSPORT_MANAGER_H
