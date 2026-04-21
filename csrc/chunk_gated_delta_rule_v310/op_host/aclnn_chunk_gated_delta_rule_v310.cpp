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
 * \file aclnn_chunk_gated_delta_rule_v310.cpp
 * \brief
 */

#include "aclnn_chunk_gated_delta_rule_v310.h"
#include "chunk_gated_delta_rule_v310.h"

#include "aclnn_kernels/common/op_error_check.h"
#include "opdev/common_types.h"
#include "opdev/op_dfx.h"
#include "opdev/op_executor.h"
#include "opdev/op_log.h"
#include "opdev/tensor_view_utils.h"

#include "aclnn_kernels/contiguous.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

namespace {
struct ChunkGatedDeltaRuleV310Params {
    const aclTensor *query{nullptr};
    const aclTensor *key{nullptr};
    const aclTensor *value{nullptr};
    const aclTensor *g{nullptr};
    const aclTensor *beta{nullptr};
    const aclTensor *out{nullptr};
    const aclTensor *finalState{nullptr};
};

static const std::initializer_list<op::DataType> QKV_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> G_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};
static const std::initializer_list<op::DataType> BETA_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> OUT_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT16};
static const std::initializer_list<op::DataType> STATE_TYPE_SUPPORT_LIST = {op::DataType::DT_FLOAT};

static inline bool CheckNotNull(const ChunkGatedDeltaRuleV310Params &params)
{
    OP_CHECK_NULL(params.query, return false);
    OP_CHECK_NULL(params.key, return false);
    OP_CHECK_NULL(params.value, return false);
    OP_CHECK_NULL(params.g, return false);
    OP_CHECK_NULL(params.beta, return false);
    OP_CHECK_NULL(params.out, return false);
    OP_CHECK_NULL(params.finalState, return false);

    return true;
}

static inline bool CheckDtypeValid(const ChunkGatedDeltaRuleV310Params &params)
{
    OP_CHECK_DTYPE_NOT_SUPPORT(params.query, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.key, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.value, QKV_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.g, G_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.beta, BETA_TYPE_SUPPORT_LIST, return false);

    OP_CHECK_DTYPE_NOT_SUPPORT(params.out, OUT_TYPE_SUPPORT_LIST, return false);
    OP_CHECK_DTYPE_NOT_SUPPORT(params.finalState, STATE_TYPE_SUPPORT_LIST, return false);

    return true;
}

static aclnnStatus CheckParams(ChunkGatedDeltaRuleV310Params &params)
{
    CHECK_RET(CheckDtypeValid(params), ACLNN_ERR_PARAM_INVALID);
    OP_LOGD("ChunkGatedDeltaRule check params success.");

    return ACLNN_SUCCESS;
}

static aclnnStatus PreProcess(ChunkGatedDeltaRuleV310Params &params)
{
    params.query->SetOriginalShape(params.query->GetViewShape());
    params.key->SetOriginalShape(params.key->GetViewShape());
    params.value->SetOriginalShape(params.value->GetViewShape());
    params.g->SetOriginalShape(params.g->GetViewShape());
    params.beta->SetOriginalShape(params.beta->GetViewShape());

    return ACLNN_SUCCESS;
}
} // namespace

aclnnStatus aclnnChunkGatedDeltaRuleV310GetWorkspaceSize(const aclTensor *query, const aclTensor *key, const aclTensor *value,
                                                         const aclTensor *g, const aclTensor *beta, const aclTensor *out, const aclTensor *finalState,
                                                         uint64_t *workspaceSize, aclOpExecutor **executor)
{
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleV310,
                   DFX_IN(query, key, value, g, beta),
                   DFX_OUT(out, finalState));

    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);

    ChunkGatedDeltaRuleV310Params params{query, key, value, g, beta, out, finalState};
    CHECK_RET(CheckNotNull(params), ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(CheckParams(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);

    auto ret = PreProcess(params);
    CHECK_RET(ret == ACLNN_SUCCESS, ret);
    if (out->IsEmpty() && finalState->IsEmpty()) {
        *workspaceSize = 0;
        uniqueExecutor.ReleaseTo(executor);
        return ACLNN_SUCCESS;
    }

    auto query_ = l0op::Contiguous(query, uniqueExecutor.get());
    auto key_ = l0op::Contiguous(key, uniqueExecutor.get());
    auto value_ = l0op::Contiguous(value, uniqueExecutor.get());
    auto g_ = l0op::Contiguous(g, uniqueExecutor.get());
    auto beta_ = l0op::Contiguous(beta, uniqueExecutor.get());

    auto out_ = l0op::Contiguous(out, uniqueExecutor.get());
    auto finalState_ = l0op::Contiguous(finalState, uniqueExecutor.get());

    auto outRet = l0op::ChunkGatedDeltaRuleV310(query_, key_, value_, g_, beta_, uniqueExecutor.get());
    if (outRet[0] == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (outRet[1] == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }

    auto outRet0 = outRet[0];
    auto outRet1 = outRet[1];
    auto ViewCopyOut = l0op::ViewCopy(outRet0, out_, uniqueExecutor.get());
    auto ViewCopyFinalState = l0op::ViewCopy(outRet1, finalState_, uniqueExecutor.get());
    if (ViewCopyOut == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    if (ViewCopyFinalState == nullptr) {
        return ACLNN_ERR_INNER_NULLPTR;
    }
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);

    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleV310(void *workspace, uint64_t workspaceSize, aclOpExecutor *executor,
                                         aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleV310);
    return CommonOpExecutorRun(workspace, workspaceSize, executor, stream);
}

#ifdef __cplusplus
}
#endif
