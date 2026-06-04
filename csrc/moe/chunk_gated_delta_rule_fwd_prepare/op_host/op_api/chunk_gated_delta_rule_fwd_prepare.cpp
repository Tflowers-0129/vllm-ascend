#include "chunk_gated_delta_rule_fwd_prepare.h"

#include "opdev/make_op_executor.h"
#include "opdev/op_dfx.h"
#include "opdev/op_log.h"

using namespace op;

namespace l0op {

OP_TYPE_REGISTER(ChunkGatedDeltaRuleFwdPrepare);

const std::array<const aclTensor *, 5> ChunkGatedDeltaRuleFwdPrepare(
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
    aclOpExecutor *executor)
{
    L0_DFX(ChunkGatedDeltaRuleFwdPrepare, q, k, v, g, beta, chunkSize, useQkL2norm, qOut, kOut, wOut, uOut, gOut);
    auto ret = ADD_TO_LAUNCHER_LIST_AICORE(
        ChunkGatedDeltaRuleFwdPrepare,
        OP_INPUT(q, k, v, g, beta),
        OP_OUTPUT(qOut, kOut, wOut, uOut, gOut),
        OP_ATTR(chunkSize, useQkL2norm));
    if (ret != ACLNN_SUCCESS) {
        OP_LOGE(ACLNN_ERR_PARAM_INVALID, "ADD_TO_LAUNCHER_LIST_AICORE failed.");
        return {nullptr, nullptr, nullptr, nullptr, nullptr};
    }
    return {qOut, kOut, wOut, uOut, gOut};
}

} // namespace l0op
