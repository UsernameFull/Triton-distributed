/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef MEM_FABRIC_HYBRID_HYBM_ENGINE_FACTORY_H
#define MEM_FABRIC_HYBRID_HYBM_ENGINE_FACTORY_H

#include "mem_entity_base.h"
#include "mem_entity_default.h"

namespace shm {
class MemEntityFactory {
public:
    static MemEntityFactory &Instance()
    {
        static MemEntityFactory INSTANCE;
        return INSTANCE;
    }

public:
    MemEntityFactory() = default;
    ~MemEntityFactory() = default;

    EngineImplPtr GetOrCreateEngine(uint16_t id, uint32_t flags);
    EngineImplPtr FindEngineByPtr(hybm_entity_t entity);
    bool RemoveEngine(hybm_entity_t entity);

public:
    std::map<uint16_t, EngineImplPtr> engines_;
    std::map<hybm_entity_t, uint16_t> enginesFromAddress_;
    std::mutex enginesMutex_;
};
}

#endif // MEM_FABRIC_HYBRID_HYBM_ENGINE_FACTORY_H
