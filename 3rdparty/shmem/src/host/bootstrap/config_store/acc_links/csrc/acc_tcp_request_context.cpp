/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "acc_includes.h"
#include "acc_tcp_request_context.h"

namespace shm {
namespace acc {
Result AccTcpRequestContext::Reply(int16_t result, const AccDataBufferPtr &d) const
{
    ASSERT_RETURN(d.Get() != nullptr, ACC_INVALID_PARAM);
    ASSERT_RETURN(link_.Get() != nullptr, ACC_LINK_ERROR);
    if (UNLIKELY(!link_->Established())) {
        LOG_ERROR("Failed to send reply message with message type " << header_.type << ", seqlo " << header_.seqNo
                                                                    << " as the link is broken");
        return ACC_LINK_ERROR;
    }
    AccMsgHeader replyHeader(header_.type, result, d->DataLen(), header_.seqNo);
    return link_->EnqueueAndModifyEpoll(replyHeader, d, nullptr);
}
}  // namespace acc
}  // namespace shm