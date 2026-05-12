/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file chunk_gated_delta_rule_v310_tiling.h
 * \brief
 */
#ifndef __OP_HOST_CHUNK_GATED_DELTA_RULE_V310_TILING_H__
#define __OP_HOST_CHUNK_GATED_DELTA_RULE_V310_TILING_H__

#include "register/op_def_registry.h"
#include "platform/platform_infos_def.h"
#include "tiling/tiling_api.h"
#include "tiling/platform/platform_ascendc.h"
#include "../tiling_base/tiling_base.h"
#include "../tiling_base/error_log.h"
#include "../op_kernel/chunk_gated_delta_rule_v310_tiling_data.h"

namespace optiling {
using namespace ChunkGatedDeltaRuleV310;
struct ChunkGatedDeltaRuleV310CompileInfo {
    uint64_t aivNum{0UL};
    uint64_t ubSize{0UL};
};

struct ChunkGatedDeltaRuleV310Info {
public:
    int64_t usedCoreNum = 0;
    const char *opName = "ChunkGatedDeltaRuleV310";
};

class ChunkGatedDeltaRuleV310Tiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit ChunkGatedDeltaRuleV310Tiling(gert::TilingContext *context)
        : Ops::Transformer::OpTiling::TilingBaseClass(context)
    {
      InitCompileInfo();
    };
    ~ChunkGatedDeltaRuleV310Tiling() override = default;

protected:
    bool IsCapable() override
    {
        return true;
    }
    // 1、获取平台信息比如CoreNum、UB/L1/L0C资源大小
    ge::graphStatus GetPlatformInfo() override;

    // 2、获取INPUT/OUTPUT/ATTR信息
    ge::graphStatus GetShapeAttrsInfo() override;

    // 3、计算数据切分TilingData
    ge::graphStatus DoOpTiling() override;

    // 4、计算高阶API的TilingData
    ge::graphStatus DoLibApiTiling() override;

    // 5、计算TilingKey
    uint64_t GetTilingKey() const override;

    // 6、计算Workspace 大小
    ge::graphStatus GetWorkspaceSize() override;

    // 7、保存Tiling数据
    ge::graphStatus PostTiling() override;

protected:
    void InitCompileInfo();
    void PrintTilingData();
    ge::graphStatus UpdateMatMulTiling();
    ChunkGatedDeltaRuleV310CompileInfo compileInfo_;
    ChunkGatedDeltaRuleTilingData tilingData_;
    ChunkGatedDeltaRuleV310Info inputParams_;
};
} // namespace optiling
#endif // __OP_HOST_CHUNK_GATED_DELTA_RULE_V310_TILING_H__
