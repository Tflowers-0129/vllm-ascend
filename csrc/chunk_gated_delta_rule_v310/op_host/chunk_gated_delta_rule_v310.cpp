/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_v310.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_v310.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_def.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {
OP_TYPE_REGISTER(ChunkGatedDeltaRuleV310);

const std::array<const aclTensor *, 2>
ChunkGatedDeltaRuleV310(const aclTensor *query, const aclTensor *key, const aclTensor *value, const aclTensor *g, const aclTensor *beta,
                        aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleV310, query, key, value, g, beta);

    DataType outType = value->GetDataType();
    Format format = Format::FORMAT_ND;
    auto out = executor->AllocTensor(outType, format, format);
    auto finalState = executor->AllocTensor(DataType::DT_FLOAT, format, format);

    auto ret =
        INFER_SHAPE(ChunkGatedDeltaRuleV310, OP_INPUT(query, key, value, g, beta),
                    OP_OUTPUT(out, finalState));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkGatedDeltaRuleV310 InferShape failed.");
        return {nullptr, nullptr};
    }

    ret = ADD_TO_LAUNCHER_LIST_AICORE(ChunkGatedDeltaRuleV310,
                                      OP_INPUT(query, key, value, g, beta),
                                      OP_OUTPUT(out, finalState));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ChunkGatedDeltaRuleV310 ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr};
    }

    return {out, finalState};
}
} // namespace l0op
