#include "chunk_gated_delta_rule_fwd_prepare_tiling.h"

#include <register/op_impl_registry.h>

namespace optiling {

static constexpr size_t INPUT_Q_IDX = 0;
static constexpr size_t INPUT_V_IDX = 2;
static constexpr size_t ATTR_CHUNK_SIZE_IDX = 0;
static constexpr size_t ATTR_USE_QK_L2NORM_IDX = 1;

static constexpr size_t DIM_BATCH = 0;
static constexpr size_t DIM_SEQLEN = 1;
static constexpr size_t DIM_HEAD_NUM = 2;
static constexpr size_t DIM_HEAD_DIM = 3;

ge::graphStatus Tiling4ChunkGatedDeltaRuleFwdPrepare(gert::TilingContext *context)
{
    ChunkGatedDeltaRuleFwdPrepareTilingData tiling;
    gert::Shape qShape = context->GetInputShape(INPUT_Q_IDX)->GetStorageShape();
    gert::Shape vShape = context->GetInputShape(INPUT_V_IDX)->GetStorageShape();
    auto attrPtr = context->GetAttrs();
    const int64_t kNumHead = qShape.GetDim(DIM_HEAD_NUM);
    const int64_t vNumHead = vShape.GetDim(DIM_HEAD_NUM);
    const int64_t kHeadDim = qShape.GetDim(DIM_HEAD_DIM);
    const int64_t vHeadDim = vShape.GetDim(DIM_HEAD_DIM);
    const int64_t chunkSize = *(attrPtr->GetAttrPointer<int64_t>(ATTR_CHUNK_SIZE_IDX));

    if (chunkSize != 64 || kNumHead <= 0 || vNumHead % kNumHead != 0 || kHeadDim > 256 || vHeadDim > 256 ||
        vHeadDim < 128 || vHeadDim % 128 != 0) {
        return ge::GRAPH_FAILED;
    }

    tiling.set_batch(qShape.GetDim(DIM_BATCH));
    tiling.set_seqlen(qShape.GetDim(DIM_SEQLEN));
    tiling.set_kNumHead(kNumHead);
    tiling.set_vNumHead(vNumHead);
    tiling.set_kHeadDim(kHeadDim);
    tiling.set_vHeadDim(vHeadDim);
    tiling.set_chunkSize(chunkSize);
    tiling.set_useQkL2norm(*(attrPtr->GetAttrPointer<bool>(ATTR_USE_QK_L2NORM_IDX)) ? 1 : 0);

    const auto ascendcPlatform = platform_ascendc::PlatformAscendC(context->GetPlatformInfo());
    context->SetBlockDim(ascendcPlatform.GetCoreNumAic());
    size_t *workspace = context->GetWorkspaceSizes(1);
    workspace[0] = 0;

    tiling.SaveToBuffer(context->GetRawTilingData()->GetData(), context->GetRawTilingData()->GetCapacity());
    context->GetRawTilingData()->SetDataSize(tiling.GetDataSize());
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus TilingPrepareForChunkGatedDeltaRuleFwdPrepare(gert::TilingParseContext *context)
{
    (void)context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(ChunkGatedDeltaRuleFwdPrepare)
    .Tiling(Tiling4ChunkGatedDeltaRuleFwdPrepare)
    .TilingParse<ChunkGatedDeltaRuleFwdPrepareCompileInfo>(TilingPrepareForChunkGatedDeltaRuleFwdPrepare);

} // namespace optiling
