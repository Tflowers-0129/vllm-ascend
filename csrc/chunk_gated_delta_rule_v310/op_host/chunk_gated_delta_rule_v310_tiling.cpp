/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_v310_tiling.cpp
 * \brief
 */

#include "chunk_gated_delta_rule_v310_tiling.h"
#include "register/op_def_registry.h"
#include "../tiling_base/tiling_templates_registry.h"
#include "../tiling_base/tiling_util.h"
#include "../tiling_base/error_log.h"


namespace optiling {
REGISTER_OPS_TILING_TEMPLATE(ChunkGatedDeltaRuleV310, ChunkGatedDeltaRuleV310Tiling, 0);

void ChunkGatedDeltaRuleV310Tiling::InitCompileInfo()
{
    auto platformInfoPtr = context_->GetPlatformInfo();
    if (platformInfoPtr == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platformInfoPtr is null");
        return;
    }
    const auto &ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfoPtr);
    ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, compileInfo_.ubSize);
    compileInfo_.aivNum = ascendcPlatform.GetCoreNumAiv();

    if (compileInfo_.aivNum <= 0) {
        OP_LOGE(context_->GetNodeName(), "aivNum <= 0");
        return;
    }
}

ge::graphStatus ChunkGatedDeltaRuleV310Tiling::GetPlatformInfo()
{
    return ge::GRAPH_SUCCESS;
};

ge::graphStatus ChunkGatedDeltaRuleV310Tiling::GetShapeAttrsInfo()
{
    if (context_ == nullptr) {
        OP_LOGE(context_->GetNodeName() , "context_ is null\n");
        return ge::GRAPH_FAILED;
    }
    const gert::StorageShape *query_shape = context_->GetInputShape(0);
    if (query_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "query is null\n");
        return ge::GRAPH_FAILED;
    }
    if (query_shape->GetStorageShape().GetDimNum() != 4) {
        OP_LOGE(context_->GetNodeName() , "query dim is not 4\n");
        return ge::GRAPH_FAILED;
    }
    uint32_t batchSize = query_shape->GetStorageShape().GetDim(0);
    uint32_t seqLen = query_shape->GetStorageShape().GetDim(1);
    uint32_t numHead = query_shape->GetStorageShape().GetDim(2);
    uint32_t headDimQK = query_shape->GetStorageShape().GetDim(3);
    OP_LOGI(context_->GetNodeName() , "query shape is (%d, %d, %d, %d)\n", batchSize, seqLen, numHead, headDimQK);

    if (numHead % 16 != 0) {
        OP_LOGE(context_->GetNodeName() , "Number of head must be a multiple of 16.\n");
        return ge::GRAPH_FAILED;
    }

    if (headDimQK != 128){
        OP_LOGE(context_->GetNodeName() , "The head dimension of query and key must be 128. \n");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *key_shape = context_->GetInputShape(1);
    if (key_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "key is null\n");
        return ge::GRAPH_FAILED;
    }
    if (key_shape->GetStorageShape().GetDimNum() != 4) {
        OP_LOGE(context_->GetNodeName() , "key dim is not 4\n");
        return ge::GRAPH_FAILED;
    }
    if (batchSize != key_shape->GetStorageShape().GetDim(0) || seqLen != key_shape->GetStorageShape().GetDim(1) || numHead != key_shape->GetStorageShape().GetDim(2) || headDimQK != key_shape->GetStorageShape().GetDim(3)) {
        OP_LOGE(context_->GetNodeName() , "key shape is illegal\n");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *value_shape = context_->GetInputShape(2);
    if (value_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "value is null\n");
        return ge::GRAPH_FAILED;
    }
    if (value_shape->GetStorageShape().GetDimNum() != 4) {
        OP_LOGE(context_->GetNodeName() , "value dim is not 4\n");
        return ge::GRAPH_FAILED;
    }
    if (batchSize != value_shape->GetStorageShape().GetDim(0) || seqLen != value_shape->GetStorageShape().GetDim(1) || numHead != value_shape->GetStorageShape().GetDim(2)) {
        OP_LOGE(context_->GetNodeName() , "value shape is illegal\n");
        return ge::GRAPH_FAILED;
    }
    uint32_t headDimV = value_shape->GetStorageShape().GetDim(3);
    OP_LOGI(context_->GetNodeName() , "value shape is (%d, %d, %d, %d)\n", batchSize, seqLen, numHead, headDimV);

    const gert::StorageShape *g_shape = context_->GetInputShape(3);
    if (g_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "g is null\n");
        return ge::GRAPH_FAILED;
    }
    if (g_shape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(context_->GetNodeName() , "g dim is not 3\n");
        return ge::GRAPH_FAILED;
    }
    if (batchSize != g_shape->GetStorageShape().GetDim(0) || seqLen != g_shape->GetStorageShape().GetDim(1) || numHead != g_shape->GetStorageShape().GetDim(2)) {
        OP_LOGE(context_->GetNodeName() , "g shape is illegal\n");
        return ge::GRAPH_FAILED;
    }

    const gert::StorageShape *beta_shape = context_->GetInputShape(4);
    if (beta_shape == nullptr) {
        OP_LOGE(context_->GetNodeName() , "beta is null\n");
        return ge::GRAPH_FAILED;
    }
    if (beta_shape->GetStorageShape().GetDimNum() != 3) {
        OP_LOGE(context_->GetNodeName() , "beta dim is not 3\n");
        return ge::GRAPH_FAILED;
    }
    if (batchSize != beta_shape->GetStorageShape().GetDim(0) || seqLen != beta_shape->GetStorageShape().GetDim(1) || numHead != beta_shape->GetStorageShape().GetDim(2)) {
        OP_LOGE(context_->GetNodeName() , "beta shape is illegal\n");
        return ge::GRAPH_FAILED;
    }

    tilingData_.batchSize = batchSize;
    tilingData_.seqLen = seqLen;
    tilingData_.numHead = numHead;
    tilingData_.headDimQK = headDimQK;
    tilingData_.headDimV = headDimV;
    return ge::GRAPH_SUCCESS;
}

void ChunkGatedDeltaRuleV310Tiling::PrintTilingData()
{
    OP_LOGD(context_->GetNodeName(), "batchSize: [%d]", tilingData_.batchSize);
    OP_LOGD(context_->GetNodeName(), "seqLen: [%d]", tilingData_.seqLen);
    OP_LOGD(context_->GetNodeName(), "numHead: [%d]", tilingData_.numHead);
    OP_LOGD(context_->GetNodeName(), "headDimQK: [%d]", tilingData_.headDimQK);
    OP_LOGD(context_->GetNodeName(), "headDimV: [%d]", tilingData_.headDimV);
    OP_LOGD(context_->GetNodeName(), "chunkSize: [%d]", tilingData_.chunkSize);
    OP_LOGD(context_->GetNodeName(), "hasInitState: [%d]", tilingData_.hasInitState);
    OP_LOGD(context_->GetNodeName(), "outputFinalState: [%d]", tilingData_.outputFinalState);
    OP_LOGD(context_->GetNodeName(), "useQKL2normInKernel: [%d]", tilingData_.useQKL2normInKernel);
}

ge::graphStatus ChunkGatedDeltaRuleV310Tiling::DoOpTiling()
{
    PrintTilingData();
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleV310Tiling::UpdateMatMulTiling() {
    using namespace matmul_tiling;
    int M = 64;
    int N = 128;
    int K = 64;
    matmul_tiling::TPosition leftPosition = matmul_tiling::TPosition::VECIN;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    bool isTransA = false;

    TPosition rightPosition = TPosition::VECIN;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    bool isTransB = false;

    TPosition resultPosition = TPosition::GM;
    CubeFormat resultFormat = CubeFormat::ND;
    DataType resultDtype = DataType::DT_FLOAT;

    TPosition biasPosition = TPosition::VECOUT;
    CubeFormat biasFormat = CubeFormat::ND;
    DataType biasDtype = DataType::DT_FLOAT;
    bool isBias = false;

    int usedCoreNum = 1;
    int baseM = 64;
    int baseN = 128;

    matmul_tiling::MultiCoreMatmulTiling mm_;
    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context_->GetPlatformInfo());
    mm_.SetBufferSpace(-1, -1, -1);
    mm_.SetAType(leftPosition, leftFormat, leftDtype, isTransA);
    mm_.SetBType(rightPosition, rightFormat, rightDtype, isTransB);
    mm_.SetCType(resultPosition, resultFormat, resultDtype);
    mm_.SetBiasType(biasPosition, biasFormat, biasDtype);
    mm_.SetBias(isBias);
    mm_.SetDim(usedCoreNum);
    mm_.SetShape(M, N, K);
    mm_.SetOrgShape(M, N, K);
    mm_.SetTraverse(MatrixTraverse::FIRSTM);
    mm_.SetFixSplit(baseM, baseN, -1);
    if (mm_.GetTiling(tilingData_.matmulTiling) == -1) {
        OP_LOGE(context_->GetNodeName(), "CGDR: Get Tiling Failed!");
        return ge::GRAPH_FAILED;
    }
    tilingData_.matmulTiling.dbL0C = 1;
    tilingData_.matmulTiling.stepKa = 1;
    tilingData_.matmulTiling.stepKb = 1;
    tilingData_.matmulTiling.depthA1 = 1;
    tilingData_.matmulTiling.depthB1 = 1;
    tilingData_.matmulTiling.stepM = 1;
    tilingData_.matmulTiling.stepN = 1;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleV310Tiling::DoLibApiTiling()
{
    tilingKey_ = 0;

    if (UpdateMatMulTiling() != ge::GRAPH_SUCCESS) {
        OP_LOGE(context_->GetNodeName(), "DoMatmulTiling failed");
        return ge::GRAPH_FAILED;
    }

    return ge::GRAPH_SUCCESS;
}

uint64_t ChunkGatedDeltaRuleV310Tiling::GetTilingKey() const
{
    return tilingKey_;
}

ge::graphStatus ChunkGatedDeltaRuleV310Tiling::GetWorkspaceSize()
{
    // 固定系统 workspace 大小（16 MB）
    constexpr uint32_t sysWorkspaceSize = 16777216;
    uint32_t usrSize = 128 * 128 * 4 * 10;
    workspaceSize_ = sysWorkspaceSize + usrSize;
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus ChunkGatedDeltaRuleV310Tiling::PostTiling()
{
    context_->SetBlockDim(8);

    auto rawTilingData = context_->GetRawTilingData();
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData);
    OP_CHECK_NULL_WITH_CONTEXT(context_, rawTilingData->GetData());

    auto tilingDataSize = sizeof(ChunkGatedDeltaRuleV310::ChunkGatedDeltaRuleTilingData);
    OP_CHECK_IF(rawTilingData->GetCapacity() < tilingDataSize,
                OP_LOGE(context_->GetNodeName(), "raw tiling data capacity %zu is smaller than tiling data size %zu",
                        rawTilingData->GetCapacity(), tilingDataSize),
                return ge::GRAPH_FAILED);

    errno_t ret = memcpy_s(rawTilingData->GetData(), rawTilingData->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    rawTilingData->SetDataSize(tilingDataSize);

    size_t *workspaces = context_->GetWorkspaceSizes(1); // set workspace
    OP_CHECK_IF(workspaces == nullptr, OP_LOGE(context_->GetNodeName(), "workspaces is null"),
                return ge::GRAPH_FAILED);
    workspaces[0] = workspaceSize_;
    return ge::GRAPH_SUCCESS;
}

static ge::graphStatus ChunkGatedDeltaRuleV310TilingFunc(gert::TilingContext *context)
{
    OP_CHECK_IF(context == nullptr, OP_LOGE("ChunkGatedDeltaRule", "context is null"),
                return ge::GRAPH_FAILED);
    return Ops::Transformer::OpTiling::TilingRegistry::GetInstance().DoTilingImpl(context);
}

static ge::graphStatus TilingPrepareForChunkGatedDeltaRuleV310(gert::TilingParseContext *context)
{
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleV310)
    .Tiling(ChunkGatedDeltaRuleV310TilingFunc)
    .TilingParse<ChunkGatedDeltaRuleV310CompileInfo>(TilingPrepareForChunkGatedDeltaRuleV310);
} // namespace optiling

