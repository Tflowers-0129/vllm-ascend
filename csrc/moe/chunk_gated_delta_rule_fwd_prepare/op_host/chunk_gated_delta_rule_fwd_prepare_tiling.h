#pragma once

#include <cstdint>
#include <register/tilingdata_base.h>
#include <tiling/tiling_api.h>

namespace optiling {

BEGIN_TILING_DATA_DEF(ChunkGatedDeltaRuleFwdPrepareTilingData)
TILING_DATA_FIELD_DEF(int64_t, batch);
TILING_DATA_FIELD_DEF(int64_t, seqlen);
TILING_DATA_FIELD_DEF(int64_t, kNumHead);
TILING_DATA_FIELD_DEF(int64_t, vNumHead);
TILING_DATA_FIELD_DEF(int64_t, kHeadDim);
TILING_DATA_FIELD_DEF(int64_t, vHeadDim);
TILING_DATA_FIELD_DEF(int64_t, chunkSize);
TILING_DATA_FIELD_DEF(int64_t, useQkL2norm);
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(ChunkGatedDeltaRuleFwdPrepare, ChunkGatedDeltaRuleFwdPrepareTilingData)

struct ChunkGatedDeltaRuleFwdPrepareCompileInfo {};

} // namespace optiling
