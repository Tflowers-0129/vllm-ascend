# GLM-5.3-Flash：A5 LI C8 与 DSA-CP 代码适配

最初基于 17、20 的相同 GLM53 代码树，移植 PR
[#18534](https://github.com/vllm-project/vllm-ascend/pull/18534)
的 **INT8 Triton KPool** 路径。A5 当前首选下文的原生 CANN FP8 路径；INT8
作为独立选项保留。FP8 CP 的多卡精度和性能验收应与 TP=1 的 MTP 复测区分。

## 已整理的历史

共同基线为 `f769db916b53446de5779bced0b7d3d38e8a7540`。
两机原 HEAD 分别为 `f8fbccbb2` 和 `4998fa23a`，代码树均为
`cf44e05b22c4e7917d2d4fc6dcecc32c9e8856fb`，未提交补丁也相同。
整理保留每一步的代码树，合并提交改为顺序提交，并记录原始提交及 PR 来源。

| 顺序 | 功能 | 来源或依赖 |
| --- | --- | --- |
| 1 | 混合缓存 P/D 重算尾部使用 decode graph | #18136 |
| 2 | 按生产端 Mamba 布局拉取状态；校验 P/D 状态形状；跳过空 FP8 平面；区分 GLM 主缓存、indexer、tail 传输标识 | 两机 `new-fix`；Mamba 部分与 #18328 相关，另含本地 pooled-cache 修复 |
| 3 | CANN 池化索引所需的 Triton 元数据、缓存写入和索引打包适配器 | #18295，固定源版本 `fdf4100c9` |
| 4 | 普通 TP 使用 CANN KeyPool/PoolKeyIndexer；DSA-CP 保留 Triton；带硬件门控的 FP8 缓存 | #16730，固定源版本 `45daa4de7`；依赖上一项 |
| 5 | HcPre 的 A3 tiling 和大 batch 分核；保留 arch35 头文件兼容 | #18347；不据此宣称 A5 提速 |
| 6 | 对齐 tail 整行及拆分后的 FP32 scale 页 | 两机相同的未提交补丁 |
| 7 | INT8 LI C8 + DSA-CP 的 A5 分支适配，并修复通用描述及小块的对齐兼容 | #18534 的量化/评分内核和本地后端、布局适配 |

## 后端和缓存约定

| indexer_kv_dtype | 普通 TP | enable_dsa_cp | 物理布局 |
| --- | --- | --- | --- |
| auto | 原 CANN BF16 | 原 Triton BF16 | BF16 pooled keys |
| int8 | Triton INT8 | Triton INT8 | 每行 128 个 INT8 字节及 4 字节 FP32 scale |
| fp8 | 原有、硬件门控的 CANN FP8 | CANN FP8；全局 cache 更新、本地因果 query | 每个内核页独立 key 区与 FP32 scale 区 |
| mxfp4 | 明确拒绝 | 明确拒绝 | 未实现 |

INT8 在全局 K/gate gather **之后**池化和量化，评分使用各 rank 的本地 query
边界。tail 保留 FP32 的 K/gate 历史及 MTP 回滚语义。INT8 后端不要求 checkpoint
提供校准后的 indexer INT8 权重；FLOAT 标记不应把功能关闭。

没有把 INT8 字节交给 FP8 CANN 算子。LI dtype 与主 SFA 的 `--kv-cache-dtype`
分别控制：主缓存可继续使用当前 A5 的 FP8 配置。

现有 A5 分配器可能将 1152-token scheduler 页中的 9 个 kernel 页整体补齐。
INT8 视图采用等宽的 kernel 页步长，页内按 132 字节行取址；例如总页从 38016
补齐到 46080 字节时，每个 kernel 页跨 5120 字节。字节寻址和 padding 保留已有
CPU 回归，内核测试也包含 padded-page 变体。

## FP8 CP 的验证顺序

下面是多卡 FP8 CP 的完整验收要求。已经完成的纯 D MTP 复测见
[MTP metadata 与复测结果](glm53_mtp_graph_metadata.md)，不重复采集。

1. 在隔离工作树及现有镜像中运行新增内核测试，确认 A5 Triton 编译及写入/反量化、
   paged/packed 评分和 padding 正确。
2. 使用同一模型、环境、seed、并发与 MTP 参数，依次比较 BF16 LI + CP、FP8 LI
   无 CP、FP8 LI + CP；CP 主场景沿用 TP4 × DP2 / EP8。INT8 另作兼容性对照。
3. 对 P、D 同时设置一致的 indexer dtype，再验收真实 KV 拉取、短/长请求、混合
   长度、prefix hit、MTP=5 及图捕获/回放。不能只凭 `/health` 成功宣称跑通。
4. 精度和稳定性通过后比较缓存容量、TTFT、TPOT、吞吐、显存峰值；不把 A3 的
   容量或性能数字当作 A5 测量结果。

新增内核测试命令：

```bash
python -m pytest -sv --confcutdir=tests/e2e/nightly/single_node/ops/singlecard_ops/triton \
  tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_glm5next_kpool_li_c8_triton.py
```

在现有启动脚本上合并以下选项，保留当前模型路径和其余参数；已有
`--additional-config` 时只合并字段，不重复传参：

```bash
--attention-config '{"indexer_kv_dtype":"fp8"}' \
--additional-config '{"enable_dsa_cp":true}'
```

确认启动日志包含 `KPool indexer LI C8 enabled`、`dtype=torch.int8`、
`backend=Triton`、`dsa_cp=True`（INT8）或 `dtype=torch.float8_e4m3fn`、
`backend=CANN`、`dsa_cp=True`（FP8）。本适配不启用 PCP/DCP。FP8 LI +
DSA-CP 路径现在通过原生 CANN KeyPool/PoolKeyIndexer 执行：先汇聚全局 hidden
rows 进行池化与 FP8 缓存更新，再按各 rank 的 query 边界设置可见 pooled keys
和 pool_tail；top-k 与 causal tail 使用本地 positions，防止跨 rank 读入未来 token。
这会比汇聚投影后的 K/gate 多传输 hidden rows，尚未测量性能。
FP8 cache 与 FP32 scale 页、MTP rollback ring 和 Decode 图模式保持原契约。
不得与 INT8 布局混用；两端仍需使用相同 indexer dtype。

## FP8 DSA-CP 补充验证

新增 CPU 回归覆盖 rank-local 因果长度、跨 pool/request 切分、空请求和 padding、
MTP rollback ring，以及图模式 metadata 地址稳定性。FP8 后端继续调用原生 CANN，
不经过 INT8 Triton 压缩内核。已发布的 Triton FP8 内核实现及对应回归保留。
多卡 FP8 CP 的完整精度、吞吐及图回放验收仍需单独记录；不能用 D 端 TP=1 的
MTP profiling 结果代替该验收。

两端使用 `--attention_config.indexer_kv_dtype fp8`，P 端保留
`--additional-config` 内的 `enable_dsa_cp=true`；D 端保留现有
`FULL_DECODE_ONLY`；MTP eager 作为对照，MTP 入图适用范围见上述文档。
