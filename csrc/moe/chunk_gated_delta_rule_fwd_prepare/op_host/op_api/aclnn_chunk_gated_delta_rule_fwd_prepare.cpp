#include "aclnn_chunk_gated_delta_rule_fwd_prepare.h"

#include "aclnn_kernels/contiguous.h"
#include "aclnn_kernels/common/op_error_check.h"
#include "chunk_gated_delta_rule_fwd_prepare.h"
#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"

using namespace op;

#ifdef __cplusplus
extern "C" {
#endif

struct ChunkGatedDeltaRuleFwdPrepareParams {
    const aclTensor *q = nullptr;
    const aclTensor *k = nullptr;
    const aclTensor *v = nullptr;
    const aclTensor *g = nullptr;
    const aclTensor *beta = nullptr;
    int64_t chunkSize = 64;
    bool useQkL2norm = false;
    const aclTensor *qOut = nullptr;
    const aclTensor *kOut = nullptr;
    const aclTensor *wOut = nullptr;
    const aclTensor *uOut = nullptr;
    const aclTensor *gOut = nullptr;
};

static aclnnStatus CheckNotNull(ChunkGatedDeltaRuleFwdPrepareParams params)
{
    CHECK_COND(params.q != nullptr, ACLNN_ERR_PARAM_NULLPTR, "q must not be nullptr.");
    CHECK_COND(params.k != nullptr, ACLNN_ERR_PARAM_NULLPTR, "k must not be nullptr.");
    CHECK_COND(params.v != nullptr, ACLNN_ERR_PARAM_NULLPTR, "v must not be nullptr.");
    CHECK_COND(params.g != nullptr, ACLNN_ERR_PARAM_NULLPTR, "g must not be nullptr.");
    CHECK_COND(params.beta != nullptr, ACLNN_ERR_PARAM_NULLPTR, "beta must not be nullptr.");
    CHECK_COND(params.qOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "qOut must not be nullptr.");
    CHECK_COND(params.kOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "kOut must not be nullptr.");
    CHECK_COND(params.wOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "wOut must not be nullptr.");
    CHECK_COND(params.uOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "uOut must not be nullptr.");
    CHECK_COND(params.gOut != nullptr, ACLNN_ERR_PARAM_NULLPTR, "gOut must not be nullptr.");
    return ACLNN_SUCCESS;
}

static aclnnStatus DataContiguous(const aclTensor *&tensor, aclOpExecutor *executor)
{
    tensor = l0op::Contiguous(tensor, executor);
    CHECK_RET(tensor != nullptr, ACLNN_ERR_INNER_NULLPTR);
    return ACLNN_SUCCESS;
}

static aclnnStatus ParamsDataContiguous(ChunkGatedDeltaRuleFwdPrepareParams &params, aclOpExecutor *executor)
{
    CHECK_RET(DataContiguous(params.q, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.k, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.v, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.g, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(DataContiguous(params.beta, executor) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepareGetWorkspaceSize(
    const aclTensor *q,
    const aclTensor *k,
    const aclTensor *v,
    const aclTensor *g,
    const aclTensor *beta,
    int64_t chunkSize,
    bool useQkL2norm,
    const aclTensor *qOut,
    const aclTensor *kOut,
    const aclTensor *wOut,
    const aclTensor *uOut,
    const aclTensor *gOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor)
{
    ChunkGatedDeltaRuleFwdPrepareParams params{q, k, v, g, beta, chunkSize, useQkL2norm, qOut, kOut, wOut, uOut, gOut};
    L2_DFX_PHASE_1(aclnnChunkGatedDeltaRuleFwdPrepare, DFX_IN(q, k, v, g, beta), DFX_OUT(qOut, kOut, wOut, uOut, gOut));
    auto uniqueExecutor = CREATE_EXECUTOR();
    CHECK_RET(uniqueExecutor.get() != nullptr, ACLNN_ERR_INNER_CREATE_EXECUTOR);
    auto executorPtr = uniqueExecutor.get();
    CHECK_RET(CheckNotNull(params) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    CHECK_RET(ParamsDataContiguous(params, executorPtr) == ACLNN_SUCCESS, ACLNN_ERR_PARAM_INVALID);
    auto result = l0op::ChunkGatedDeltaRuleFwdPrepare(
        params.q, params.k, params.v, params.g, params.beta, params.chunkSize, params.useQkL2norm,
        params.qOut, params.kOut, params.wOut, params.uOut, params.gOut, executorPtr);
    CHECK_RET(result[0] != nullptr, ACLNN_ERR_PARAM_NULLPTR);
    *workspaceSize = uniqueExecutor->GetWorkspaceSize();
    uniqueExecutor.ReleaseTo(executor);
    return ACLNN_SUCCESS;
}

aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepare(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream)
{
    L2_DFX_PHASE_2(aclnnChunkGatedDeltaRuleFwdPrepare);
    CHECK_COND(CommonOpExecutorRun(workspace, workspaceSize, executor, stream) == ACLNN_SUCCESS, ACLNN_ERR_INNER,
               "ChunkGatedDeltaRuleFwdPrepare launch failed.");
    return ACLNN_SUCCESS;
}

#ifdef __cplusplus
}
#endif
