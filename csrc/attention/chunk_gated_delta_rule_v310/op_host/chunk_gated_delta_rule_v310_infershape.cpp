/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file chunk_gated_delta_rule_v310_infershape.cpp
 * \brief
 */

#include "register/op_impl_registry.h"

using namespace gert;
namespace ops {

const size_t KEY_INDEX = 1;
const size_t VALUE_INDEX = 2;
const size_t STATE_DIM_NUM = 4;

const size_t DIM_0 = 0;
const size_t DIM_1 = 1;
const size_t DIM_2 = 2;
const size_t DIM_3 = 3;

static ge::graphStatus InferShapeChunkGatedDeltaRuleV310(InferShapeContext *context) {
    const gert::Shape *query_shape = context->GetInputShape(0);
    uint32_t batchSize = query_shape->GetDim(0);
    uint32_t numHead = query_shape->GetDim(2);
    uint32_t headDimQK = query_shape->GetDim(3);
    const gert::Shape *value_shape = context->GetInputShape(2);
    uint32_t headDimV = value_shape->GetDim(3);
    gert::Shape *core_attn = context->GetOutputShape(0);
    *core_attn = *value_shape;
    Shape *last_recurrent_state = context->GetOutputShape(1);
    last_recurrent_state->SetDimNum(STATE_DIM_NUM);
    last_recurrent_state->SetDim(DIM_0, batchSize);
    last_recurrent_state->SetDim(DIM_1, numHead);
    last_recurrent_state->SetDim(DIM_2, headDimQK);
    last_recurrent_state->SetDim(DIM_3, headDimV);

    return ge::GRAPH_SUCCESS;
}
static ge::graphStatus InferDataTypeChunkGatedDeltaRuleV310(InferDataTypeContext *context) {
    const auto inputDataType = context->GetInputDataType(0);
    context->SetOutputDataType(0, inputDataType);
    context->SetOutputDataType(1, ge::DataType::DT_FLOAT);
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(ChunkGatedDeltaRuleV310)
    .InferShape(InferShapeChunkGatedDeltaRuleV310)
    .InferDataType(InferDataTypeChunkGatedDeltaRuleV310);
} // namespace ops
