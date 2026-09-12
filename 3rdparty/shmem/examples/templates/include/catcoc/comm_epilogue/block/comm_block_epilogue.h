/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_H
#define CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_H

#include "catcoc/catcoc.h"

namespace Catcoc::CommEpilogue::Block {

template <
    class DispatchPolicy,
    class... Args
>
class CommBlockEpilogue {
    static_assert(DEPENDENT_FALSE<DispatchPolicy>, "Could not find an epilogue specialization");
};

} // namespace Catcoc::CommEpilogue::Block

#include "catcoc/comm_epilogue/block/comm_block_epilogue_to_local_mem.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue_to_share_mem.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue_remote_copy.h"
#include "catcoc/comm_epilogue/block/comm_block_epilogue_local_copy.h"
#endif // CATCOC_COMM_EPILOGUE_BLOCK_EPILOGUE_H