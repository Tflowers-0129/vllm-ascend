#ifndef OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_FWD_PREPARE_H
#define OP_API_INC_ACLNN_CHUNK_GATED_DELTA_RULE_FWD_PREPARE_H

#include "aclnn/aclnn_base.h"

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepareGetWorkspaceSize(
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
    aclOpExecutor **executor);

__attribute__((visibility("default"))) aclnnStatus aclnnChunkGatedDeltaRuleFwdPrepare(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
