#ifndef OP_API_INC_CHUNK_GATED_DELTA_RULE_FWD_PREPARE_H
#define OP_API_INC_CHUNK_GATED_DELTA_RULE_FWD_PREPARE_H

#include <array>

#include "aclnn/aclnn_base.h"
#include "opdev/op_executor.h"

namespace l0op {

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
    aclOpExecutor *executor);

} // namespace l0op

#endif
