/**
 * This program is free software, you can redistribute it and/or modify it.
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_v310_tiling_data.h
 * \brief tiling data struct
 */

#ifndef CHUNK_GATED_DELTA_RULE_V310_TILING_DATA_H_
#define CHUNK_GATED_DELTA_RULE_V310_TILING_DATA_H_

#include <cstdint>

namespace ChunkGatedDeltaRuleV310 {
#pragma pack(push, 8)
struct alignas(8) ChunkGatedDeltaRuleTilingData {
    uint32_t batchSize;
    uint32_t seqLen;
    uint32_t numHead;
    uint32_t headDimQK;
    uint32_t headDimV;
    TCubeTiling matmulTiling;
};
#pragma pack(pop)
}  // ChunkGatedDeltaRuleV310
#endif  // CHUNK_GATED_DELTA_RULE_V310_TILING_DATA_H_
