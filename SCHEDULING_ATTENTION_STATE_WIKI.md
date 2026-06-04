# vLLM-Ascend 调度策略与 Attention State Wiki

本文是一份从 LLM serving 基本原理到 vLLM-Ascend attention-state 映射的学习型 wiki。它试图回答几个经常混在一起的问题：

- vLLM 的调度器到底在每一轮做什么。
- `prefill`、`decode`、`chunked prefill`、`split-fuse`、`prefix cache`、`MTP` 分别是什么。
- 一个 batch 为什么会被判成 `PrefillNoCache`、`DecodeOnly`、`ChunkedPrefill`、`PrefillCacheHit` 或 `SpecDecoding`。
- 这些状态在 310P 上分别走什么 attention 算子。
- 这些机制和 ACLGraph / graph mode 有什么关系。
- vLLM-Ascend 与其他 LLM serving 引擎在这些概念上有哪些共性和差异。

文中的 310P 说明主要基于以下代码路径：

- `vllm_ascend/worker/model_runner_v1.py`
- `vllm_ascend/_310p/model_runner_310p.py`
- `vllm_ascend/attention/attention_v1.py`
- `vllm_ascend/_310p/attention/attention_v1.py`
- `vllm_ascend/attention/utils.py`
- `vllm_ascend/compilation/acl_graph.py`

## 先建立术语地图

很多困惑来自术语边界不清。下面这些词听起来都和 prefill/decode 有关，但它们处在不同层。

| 术语 | 所在层 | 核心含义 |
| --- | --- | --- |
| `prefill` | 模型执行语义 | 处理 prompt token，建立初始 KV cache |
| `decode` | 模型执行语义 | 基于已有 KV cache，每轮生成后续 token |
| `continuous batching` | 调度策略 | 每个 step 重新组合正在运行和等待中的请求 |
| `FCFS` | 请求排队策略 | waiting 请求按到达顺序进入调度 |
| `chunked prefill` | 调度粒度 | 允许长 prompt 被拆成多个 step 调度 |
| `prefix cache` / APC | KV cache 复用 | 新请求的前缀命中已有 KV cache，只计算未命中的 suffix |
| `P/D 混推` | batch 组成 | 同一个 step 内同时有 decode 请求和 prefill 请求 |
| `P/D disaggregation` | 部署架构 | prefill 节点和 decode 节点分离，通过 KV transfer 协作 |
| `split-fuse` | attention 算子路径 | 用一个适配 KV cache 与变长 query 的 attention 路径处理 chunk / cache-hit / 混批 |
| `MTP` / speculative decoding | 解码优化 | 一轮提出多个候选 token，再验证或接受其中一部分 |
| `ACLGraph` / graph mode | 执行优化 | 捕获稳定 shape 的执行图，降低调度和 launch 开销 |

本文里的“P/D 混推”特指单实例 serving 内，一个 batch 同时包含 decode 请求和 prefill 请求。它不是 PD 分离部署。PD 分离部署通常需要 KV transfer；310P 当前代码中 `kv_transfer_config` 会被拒绝，因此不能把这两个概念混在一起。

## 从服务策略说起：FCFS 与 continuous batching

LLM serving 的请求不是同时到达的，输入长度、输出长度也不同。服务端每个 step 都要决定：

- 哪些 `RUNNING` 请求继续推进。
- 哪些 `WAITING` 请求可以进入当前 batch。
- 每个请求本轮计算多少 token。
- KV cache 是否足够容纳这些 token。
- 这一轮适合走 eager、piecewise graph，还是 full graph。

FCFS 的直觉是 first come first served，但在 vLLM V1 里它不是“一个请求完全跑完再跑下一个”。已经进入 `RUNNING` 的请求通常先被推进；如果本轮 token budget 还有剩余，调度器再从 `WAITING` 队列接纳新请求。

这就自然产生 continuous batching：

```text
Step N:
  RUNNING: A, B, C 正在 decode，每个请求本轮 q_len=1
  WAITING: D 是新来的 prompt

如果 token budget、KV cache、LoRA、encoder 等约束都允许：
  batch = decode(A, B, C) + prefill(D)
```

这个 batch 从业务语义看同时包含 decode 和 prefill；从 attention backend 看，它需要一种能同时处理“已有 KV cache 的 token”和“新 prefill token”的执行路径。

## 调度器最重要的三个长度

理解 attention-state 前，先把三个长度钉住：

```text
num_computed_tokens_cpu[i]
  第 i 个请求在本轮之前已经计算过多少 token。

num_scheduled_tokens[i]
  第 i 个请求本轮被调度了多少 token。
  这就是该请求本轮 attention 的 q_len。

seq_lens[i]
  第 i 个请求本轮 attention 可见的总上下文长度。
  通常等于 num_computed_tokens_cpu[i] + num_scheduled_tokens[i]。
  这就是该请求本轮 attention 的 kv_len / context_len。
```

310P runner 会用 `num_scheduled_tokens` 构造 `query_start_loc`：

```text
num_scheduled_tokens = [4, 100, 3]
query_start_loc      = [0, 4, 104, 107]

q_len[0] = 4   = query_start_loc[1] - query_start_loc[0]
q_len[1] = 100 = query_start_loc[2] - query_start_loc[1]
q_len[2] = 3   = query_start_loc[3] - query_start_loc[2]
```

在 310P split-fuse 路径中：

```python
qsl_cpu = attn_metadata.query_start_loc.cpu()
qlens = qsl_cpu[1:] - qsl_cpu[:-1]
context_lens = attn_metadata.seq_lens
```

`qlens` 传给 `_npu_paged_attention_splitfuse(..., seq_len=qlens, ...)`，`context_lens` 传给 `context_lens` 参数。因此：

```text
split-fuse 的 q_len = 本轮每个请求新算多少 token
split-fuse 的 kv_len = 本轮每个请求能看到的总上下文长度
```

## Token budget：2048 是上限，不是固定 q_len

`max_num_batched_tokens` 是一轮最多调度多少 token 的上限。OpenAI server 场景中常见默认值是 2048，但它不是 attention 算子的固定输入长度。

如果开启 chunked prefill，长 prompt 可以按剩余 token budget 被拆成多轮：

```text
max_num_batched_tokens = 2048
prompt_len = 5000

Step 1: q_len=2048
Step 2: q_len=2048
Step 3: q_len=904
```

如果当前 step 同时有 decode 请求，它们会占用同一个 token budget：

```text
max_num_batched_tokens = 2048
decode requests = 32
每个 decode q_len = 1

剩余 prefill budget = 2048 - 32 = 2016
```

因此，2048 表示“本轮总 q token 数的上限”，不是“每个请求 q_len 都是 2048”，也不是“FA 算子每次都输入 2048”。

## Chunked prefill 到底是什么

`chunked prefill` 是调度层概念。它解决的是长 prompt 占满整个 step 的问题。

如果没有 chunked prefill，一个长 prompt 要么完整进入某一轮 batch，要么等待。这样会带来两个问题：

- 长 prompt 可能因为 `max_num_batched_tokens` 不够而无法调度。
- 长 prefill 会长时间占用算力，decode 请求等待时间变长，影响 inter-token latency。

开启 chunked prefill 后，调度器允许只调度长 prompt 的一部分：

```text
prompt_len = 5000
max_num_batched_tokens = 2048

不启用 chunked prefill:
  必须一次放入 5000 token。
  如果 token budget 不足，就等待或需要调大 max_num_batched_tokens。

启用 chunked prefill:
  可以先放 2048，再放 2048，再放 904。
```

这不是 attention 算子本身的定义，而是 scheduler 给 attention backend 喂数据的方式发生变化。attention backend 看到的是：

```text
本轮 q_len = 这一块 chunk 的 token 数
本轮 kv_len = 历史已算 token + 本轮 q_len
```

## Split-fuse 到底是什么

`split-fuse` 这个名字容易让人困惑，因为它听起来既像“切块”，又像“融合”。在 vLLM-Ascend 310P 的语境里，可以从两个角度理解。

“Split” 描述的是输入形态不再是一个简单的纯 prefill：

- 长 prefill 被 split 成多个 chunk。
- batch 里可能 split 出 decode 段和 prefill 段。
- 某个请求可能只有 suffix 需要计算，因为 prefix cache 已经命中。
- 每个请求的 `q_len` 和 `kv_len` 可以不同。

“Fuse” 描述的是执行路径把这些复杂情况放进一个 attention 算子路径里处理：

- 既读已有 paged KV cache。
- 又处理本轮新 query token。
- 同时支持变长 `q_len` 和变长 `context_lens`。
- 用一套 mask/metadata 表达“本轮 query token 只能看见它应该看见的历史和当前 token”。

所以，split-fuse 不是简单等价于 chunked prefill。更准确地说：

```text
chunked prefill 是调度层是否允许切 prompt。
split-fuse 是 attention backend 为了处理 chunk/cache-hit/mixed-batch 而使用的算子路径。
```

在 310P 代码里，`ChunkedPrefill` 和 `PrefillCacheHit` 都会进入 `forward_chunked_prefill_310`，然后调用：

```python
torch_npu._npu_paged_attention_splitfuse(...)
```

这也解释了一个关键结论：

```text
关闭 chunked prefill 不等于关闭 split-fuse。
```

只要 batch 不是纯 no-cache prefill，也不是纯 decode，尤其是 P/D 混推或 prefix cache hit，就可能仍然需要 split-fuse。

## Split-fuse mask 在表达什么

310P 的 split-fuse mask 构造在 `vllm_ascend/_310p/attention/attention_mask.py`。

核心逻辑是：

```python
qsl = attn_metadata.query_start_loc.to("cpu", dtype=torch.int32)
qlens = qsl[1:] - qsl[:-1]
context_lens = attn_metadata.seq_lens.to("cpu", dtype=torch.int32)
pos_list = [p for ql, cl in zip(q_list, c_list) for p in range(cl - ql, cl)]
splitfuse_mask = causal_mask.index_select(0, position)
```

这里的 `range(cl - ql, cl)` 很关键。它表示本轮 query token 在完整上下文中的位置。

例如一个长 prompt 第 2 块：

```text
已算历史 tokens = 2048
本轮 q_len = 2048
本轮 kv_len = 4096

position = range(4096 - 2048, 4096)
         = range(2048, 4096)
```

这说明本轮 query 不是从位置 0 开始的；它们对应完整序列里的第 2048 到 4095 个 token。mask 必须按这些真实位置生成，否则 attention 可见范围会错。

这也是 split-fuse 和普通 FA 的差别之一：

- 纯 `PrefillNoCache`：`q_len == kv_len`，从位置 0 开始，普通 causal mask 足够。
- chunk/cache-hit/mixed：`q_len` 只是本轮增量，`kv_len` 是完整上下文，mask 要按真实上下文位置切出来。

## Attention state 的判定规则

vLLM-Ascend V1 的核心判定在 `NPUModelRunner._build_attn_state`，310P 继承并复用这套逻辑。简化后的判断顺序如下：

```python
if all(num_computed_tokens_cpu[:num_reqs] == 0):
    attn_state = PrefillNoCache
elif all(num_scheduled_tokens == 1):
    attn_state = DecodeOnly
    if speculative_config and speculative_config.method == "mtp":
        attn_state = SpecDecoding
elif all(num_valid_tokens == 1):
    if speculative_config:
        attn_state = SpecDecoding
    else:
        attn_state = ChunkedPrefill
elif scheduler_config.enable_chunked_prefill:
    attn_state = ChunkedPrefill
else:
    attn_state = PrefillCacheHit
```

这是 batch 级 state，不是单请求 state。一个混批 batch 中，decode 请求仍然是 q_len=1，新 prefill 请求可能是 q_len=1000，但整个 batch 的 `attn_state` 只有一个。

| 调度输入形态 | 典型条件 | Attention state | 含义 |
| --- | --- | --- | --- |
| 纯首次 prefill | 所有请求本轮前 `num_computed_tokens == 0` | `PrefillNoCache` | 没有历史 KV cache |
| 纯 decode | 所有请求本轮 `num_scheduled_tokens == 1`，且不是首次 prefill | `DecodeOnly` | 每个请求推进一个 token |
| MTP / speculative decode | `num_valid_tokens == 1`，本轮可能有 draft token | `SpecDecoding` | q_len 可能大于 1，但语义是 spec decode |
| 长 prompt 后续 chunk | 已经算过一部分 prompt，开启 chunked prefill | `ChunkedPrefill` | 基于已有 KV 继续算 chunk |
| P/D 混推，chunked 开启 | batch 同时有 decode 和 prefill | `ChunkedPrefill` | 混合 batch 走 chunk/cache-aware 路径 |
| P/D 混推，chunked 关闭 | batch 同时有 decode 和完整 prefill | `PrefillCacheHit` | 仍然不是纯 no-cache prefill |
| Prefix cache hit | prompt 前缀已命中，suffix 多于 1 token | `ChunkedPrefill` 或 `PrefillCacheHit` | 基于已有 KV 计算 suffix |

## 310P 上 attention state 到算子的映射

310P 的路由在 `vllm_ascend/_310p/attention/attention_v1.py::forward_impl`。

| Attention state | 310P 路径 | 算子 |
| --- | --- | --- |
| `PrefillNoCache` | `forward_prefill_310` | `torch_npu._npu_flash_attention` |
| `DecodeOnly` | `forward_paged_attention` | `torch_npu._npu_paged_attention` |
| `ChunkedPrefill` | `forward_chunked_prefill_310` | `torch_npu._npu_paged_attention_splitfuse` |
| `PrefillCacheHit` | `forward_chunked_prefill_310` | `torch_npu._npu_paged_attention_splitfuse` |
| `SpecDecoding` | 310P 当前 forward 路由未单独支持 | 需要看模型和额外 patch |

因此 310P 上最重要的心智模型是：

```text
PrefillNoCache  -> FA
DecodeOnly      -> paged attention
ChunkedPrefill  -> split-fuse
PrefillCacheHit -> split-fuse
```

如果你看到 `_npu_paged_attention_splitfuse`，它不一定意味着“scheduler 正在切 prompt”。它可能只是表示当前 batch 需要一种能处理已有 KV cache 的 prefill/extend/mixed attention 路径。

## q_len 与 kv_len 的通用公式

无论是哪种场景，都可以先用这个公式定位：

```text
q_len[i]  = num_scheduled_tokens[i]
kv_len[i] = seq_lens[i]
          = num_computed_tokens_before_step[i] + q_len[i]
```

区别在于状态不同，算子如何消费这些长度不同。

| 场景 | q_len | kv_len |
| --- | --- | --- |
| 纯首次 prefill | prompt 或 prompt chunk | 等于 q_len |
| 纯 decode | 1 | 历史上下文 + 1 |
| chunked prefill 后续块 | 本轮 chunk 大小 | 已算 prompt + 本轮 chunk |
| prefix cache hit | 未命中 suffix 大小 | 命中 prefix + suffix |
| MTP decode | `1 + draft tokens` | 历史上下文 + q_len |
| P/D 混推中的 decode 请求 | 1 | 历史上下文 + 1 |
| P/D 混推中的 prefill 请求 | 本轮 prefill token 数 | 已算 tokens + 本轮 q_len |

## Chunked prefill 开启时的典型执行

假设：

```text
max_num_batched_tokens = 2048
prompt_len = 5000
没有其他请求
```

执行通常是：

| Step | num_computed before step | q_len | kv_len | State | 310P 算子 |
| --- | ---: | ---: | ---: | --- | --- |
| 1 | 0 | 2048 | 2048 | `PrefillNoCache` | FA |
| 2 | 2048 | 2048 | 4096 | `ChunkedPrefill` | split-fuse |
| 3 | 4096 | 904 | 5000 | `ChunkedPrefill` | split-fuse |

第一块走 FA，因为所有请求本轮前都没有 computed tokens。后续块已经有历史 KV cache，所以 `kv_len > q_len`，进入 split-fuse。

如果混入 decode：

```text
max_num_batched_tokens = 2048
32 个 decode 请求，每个 q_len = 1
1 个新 prefill 请求
```

调度结果可能是：

```text
decode total q_len = 32
prefill q_len = 2016
batch total q_len = 2048
```

对于新 prefill 请求本身，`kv_len == q_len == 2016`。但整个 batch 不是纯 prefill，因为前面有 decode 请求，所以 310P 会走 `ChunkedPrefill -> split-fuse`。

## Chunked prefill 关闭后会发生什么

关闭 chunked prefill 后，调度层不再允许把一个 prefill 请求切成部分 chunk 去填剩余 budget。它必须完整放入当前剩余 token budget，否则等待。

例如：

```text
max_num_batched_tokens = 2048
已有 decode 占 32 tokens
新 prefill prompt_len = 3000
```

关闭 chunked prefill 后，调度器不会把这个 prefill 切成 2016 token 放进去。它会等待，因为完整 3000 放不进当前剩余 budget。

但“等待”不等于“一定走 FA”。下一轮取决于 batch 组成：

```text
下一轮只有这个 prefill:
  state = PrefillNoCache
  310P 算子 = FA

下一轮仍有 decode + 这个 prefill，且 budget 足够:
  state = PrefillCacheHit
  310P 算子 = split-fuse
```

如果显式把 `max_num_batched_tokens` 固定为 2048，同时关闭 chunked prefill，却允许 3000 token prompt，这通常不成立。vLLM 配置校验会要求 `max_num_batched_tokens >= max_model_len`，否则长 prompt 无法被完整调度。若不显式设置，关闭 chunked prefill 时默认逻辑会倾向把 `max_num_batched_tokens` 提到至少 `max_model_len`。

## Prefix cache 如何影响状态

Automatic Prefix Caching 的作用是复用已有 KV cache。它改变的是请求进入本轮时的 `num_computed_tokens` 起点。

例如：

```text
prompt_len = 3000
prefix cache hit = 2000
suffix to compute = 1000

num_computed_tokens_cpu = 2000
num_scheduled_tokens    = 1000
seq_lens                = 3000
```

这个请求虽然业务上仍然是 prefill 请求，但 attention 后端看到的是“已有 2000 token KV，继续计算 1000 token query”。它不是 `PrefillNoCache`。

常见状态如下：

| 场景 | chunked prefill | 典型 state |
| --- | --- | --- |
| prefix cache miss，纯首次 prefill | 开/关都可 | `PrefillNoCache` |
| prefix cache hit，suffix 多于 1 token | 开启 | `ChunkedPrefill` |
| prefix cache hit，suffix 多于 1 token | 关闭 | `PrefillCacheHit` |
| prefix cache hit 后只需要 decode 一个 token | 开/关都可 | `DecodeOnly` |

在 310P 上，`ChunkedPrefill` 和 `PrefillCacheHit` 都会走 split-fuse，所以 prefix cache hit 很容易让请求避开 FA 路径。

## MTP / speculative decoding 如何影响状态

普通 decode 每个请求一轮通常只调度一个 token：

```text
num_scheduled_tokens = [1, 1, 1]
state = DecodeOnly
```

MTP 会让一轮 decode 中出现多个 query token，例如一个真实 token 加若干 draft token：

```text
num_scheduled_tokens = [1 + num_spec_tokens, ...]
num_valid_tokens     = [1, ...]
```

这就是 `_build_attn_state` 中 `all(num_valid_tokens == 1)` 分支的意义：虽然本轮 q_len 可能大于 1，但有效推进的主 token 仍然是 1，其余 token 是 speculative draft 语义，因此不能简单当成 chunked prefill。

典型关系：

| 场景 | num_scheduled_tokens | num_valid_tokens | State |
| --- | --- | --- | --- |
| 普通 decode | 全部为 1 | 全部为 1 | `DecodeOnly` |
| MTP decode | 大于 1 | 全部为 1 | `SpecDecoding` |
| MTP method 为 `mtp` 且 q_len 为 1 | 全部为 1 | 全部为 1 | 也可能强制为 `SpecDecoding` |

在通用 Ascend 路径中，`SpecDecoding` 是专门语义。310P 的 `_310p/attention/attention_v1.py::forward_impl` 当前没有给 `SpecDecoding` 单独路由，遇到不支持的 state 会抛 `NotImplementedError`。因此在 310P 上使用 MTP 前，需要确认模型、worker、attention backend 是否有额外适配。

## 与 ACLGraph / graph mode 的关系

Graph mode 的核心目标是减少 Python 调度和算子 launch 开销。它通常要求输入 shape、执行路径、内存地址和部分元数据结构足够稳定。

这和调度/attention-state 强相关：

- `DecodeOnly` 最适合 graph capture。每个请求通常 q_len=1，batch shape 主要由并发数决定。
- `PrefillNoCache` 的 q_len 可能很大，且 prompt 长度差异明显，shape 更不稳定。
- `ChunkedPrefill` 和 `PrefillCacheHit` 的 `q_len`、`kv_len`、`query_start_loc`、`block_table` 都更动态，捕获和复用图更难。
- MTP decode 的 q_len 是 `1 + num_spec_tokens`，比普通 decode 更复杂，但仍可能比任意 prefill 更稳定。

在 vLLM-Ascend 里，常见 graph mode 包括：

| 模式 | 直觉 | 更适合的 batch |
| --- | --- | --- |
| eager | 不捕图，直接执行 | 动态 shape、调试、复杂混合路径 |
| piecewise graph | 对模型片段捕图 | 多种 batch size，但需要捕获多个 size |
| full graph | 整个 forward 捕图 | shape 更稳定的 decode |
| full decode only | 专门优化 decode | 纯 decode 高并发场景 |

310P 还有一个很实用的细节：在 `_310p/model_runner_310p.py` 中，如果 `attn_state` 是 `ChunkedPrefill` 或 `PrefillCacheHit`，会强制 `force_eager=True`。这意味着 310P 的 split-fuse 路径当前更偏向 eager 执行，而不是 graph replay。

这背后的原因不难理解：

```text
split-fuse 输入中有变长 q_len、变长 context_lens、动态 block_table、动态 mask。
这些信息随着每个 step 的请求组成变化而变化。
Graph replay 喜欢稳定形状和稳定执行路径。
```

Graph mode 里还会看到 `graph_pad_size`、`query_start_loc` padding、`BatchDescriptor` 之类的机制。它们的目标是把不稳定的 batch shape 映射到少数几个可捕获的 bucket：

```text
真实 batch:
  num_reqs = 27
  total_tokens = 27

图 bucket:
  num_reqs_padded = 32
  total_tokens_padded = 32
```

这种 padding 对纯 decode 很自然，因为每个请求 q_len=1。对 mixed/chunked prefill 更麻烦，因为一个请求可能 q_len=2016，另一个请求 q_len=1，`query_start_loc` 不能简单按请求数补齐，还要保证每段 token 的边界正确。

## 为什么 decode-only 更适合图模式

纯 decode 的形态非常规整：

```text
num_scheduled_tokens = [1, 1, 1, ..., 1]
query_start_loc      = [0, 1, 2, 3, ..., batch_size]
q_len per request    = 1
```

它的动态性主要是 batch size。只要捕获几个常见并发 bucket，比如 1、2、4、8、16、32、64、128，就能覆盖很多在线 serving 场景。

prefill/chunked/mixed 则有更多动态维度：

```text
num_scheduled_tokens = [1, 1, 2016, 7, 128]
query_start_loc      = [0, 1, 2, 2018, 2025, 2153]
seq_lens             = [历史+1, 历史+1, 2016, 历史+7, prefix+128]
```

不仅 total tokens 变，token 如何分布到请求上也变。attention mask、slot mapping、block table 的访问模式也随之变化。这就是为什么很多引擎都会优先把 decode-only 做成图模式或专门 kernel，而对 prefill/mixed 路径保留更多 eager 或动态 kernel 能力。

## 与其他 LLM serving 引擎的共性和差异

不同引擎名字不同，但核心问题很相似：

```text
如何在高吞吐和低延迟之间平衡？
如何复用 KV cache？
如何处理变长 prompt？
如何让 decode 稳定到可以捕图或使用高性能 kernel？
如何把 prefill 和 decode 的资源冲突降到最低？
```

### 与通用 vLLM 的关系

vLLM 的基础心智模型是：

- continuous batching。
- paged KV cache。
- prefix caching。
- chunked prefill。
- speculative decoding。
- CUDA graph / piecewise graph 等图优化。

vLLM-Ascend 继承了这些调度概念，但 attention backend 和图执行从 CUDA/NVIDIA 生态换成了 NPU/CANN/ACLGraph 生态。很多字段名字还保留 `cudagraph`，但在 Ascend 插件里实际对应 ACLGraph/NPU graph 相关能力。

### 与 TensorRT-LLM 类引擎的关系

TensorRT-LLM 类引擎也有类似 continuous batching 或 in-flight batching 的思想：请求可以在运行中加入 batch，decode 和 context/prompt 处理需要协调。它更强调 engine build、kernel 选择和 TensorRT 图优化，通常会把很多 shape、并发和模型结构约束提前固化到 engine 或 profile 中。

相同点：

- 都要处理 prefill/context 与 decode/generation 的资源竞争。
- 都会使用 paged KV cache 或类似 KV cache 管理机制。
- 都会为 decode 做高度优化。
- 都需要处理长 prompt 的切块或 context chunking。

差异点：

- vLLM 更偏动态调度和 Python 层灵活性，图模式是可选优化。
- TensorRT-LLM 更偏静态 engine/profile 与高度编译优化。
- vLLM-Ascend 则处在 vLLM 调度框架和 Ascend NPU backend 之间，需要把 vLLM 的动态 batch 语义映射到 CANN/NPU 算子和 ACLGraph 约束上。

### 与 SGLang 类引擎的关系

SGLang 类引擎也重视 prefix/radix cache、continuous batching、speculative decoding 等能力。它的高层抽象更偏面向程序化 prompt、结构化执行和 cache 命中管理。

相同点：

- prefix cache 命中后，本质上都会把请求变成“已有 KV + suffix compute”。
- 长 prompt 都会引入切块、抢占或调度公平性问题。
- decode-only 仍然最容易被优化成稳定高吞吐路径。

差异点：

- vLLM 的 attention-state 映射是硬件 backend 的一个明确接口。
- SGLang 更强调运行时请求图、radix cache 和前端执行语义。
- vLLM-Ascend 的 split-fuse 这类路径更直接暴露了 NPU attention 算子的约束。

### 与传统静态 batch 推理的区别

传统静态 batch 往往是：

```text
收集一批请求 -> pad 到相同长度 -> 一起 prefill -> 一起 decode -> 结束
```

在线 LLM serving 则更像：

```text
每个 step 动态接纳请求
每个请求处在不同阶段
每个请求 q_len 和 kv_len 都可能不同
KV cache 常驻并跨 step 复用
```

这就是为什么 `attention-state` 必须存在。它不是为了增加复杂度，而是为了告诉 backend：当前这轮 batch 到底是哪一种执行语义。

## 与上下文并行 PCP/DCP 的联系

Context Parallelism 会进一步改变 q_len/kv_len 的局部视角。全局上某个请求可能有 4096 token，但在某个 rank 上只处理其中一段。

对 chunked prefill 来说，PCP/DCP 会让 metadata 更复杂：

- 全局 `query_lens` 和本 rank 的 local `query_lens` 可能不同。
- `context_lens` 需要区分全局上下文和本地上下文。
- block table、slot mapping、mask 都要考虑 rank 内的 token 分布。

因此文档里讲的公式：

```text
q_len = num_scheduled_tokens
kv_len = num_computed_tokens + q_len
```

在非 CP 场景下最直观；在 CP 场景下要进一步拆成本地 rank 视角。vLLM-Ascend 代码中 `prefill_context_parallel_metadata`、`attn_chunk_seqlens`、`query_lens_pcp_full` 等字段就是为这类场景服务的。

## 调度、state、算子的完整链路

可以把整条链路压缩成下面这张图：

```text
请求到达
  |
  v
WAITING / RUNNING 队列
  |
  v
Scheduler 根据 FCFS、token budget、KV cache、prefix cache、MTP 等做本轮决策
  |
  v
num_scheduled_tokens
num_computed_tokens
seq_lens
query_start_loc
block_table
slot_mapping
  |
  v
_build_attn_state
  |
  +--> PrefillNoCache
  |       310P: _npu_flash_attention
  |
  +--> DecodeOnly
  |       310P: _npu_paged_attention
  |
  +--> ChunkedPrefill
  |       310P: _npu_paged_attention_splitfuse
  |
  +--> PrefillCacheHit
  |       310P: _npu_paged_attention_splitfuse
  |
  +--> SpecDecoding
          backend-specific support required
```

## 典型案例速查

### 短 prompt，单请求

```text
prompt_len = 512
num_computed = 0
num_scheduled = 512
seq_lens = 512
```

```text
state = PrefillNoCache
310P = FA
```

### 长 prompt，chunked prefill 开启

```text
prompt_len = 5000
max_num_batched_tokens = 2048
```

```text
Step 1: q=2048, kv=2048, PrefillNoCache, FA
Step 2: q=2048, kv=4096, ChunkedPrefill, split-fuse
Step 3: q=904,  kv=5000, ChunkedPrefill, split-fuse
```

### 长 prompt，chunked prefill 关闭

```text
prompt_len = 5000
max_num_batched_tokens >= 5000
没有 decode 混入
```

```text
Step 1: q=5000, kv=5000, PrefillNoCache, FA
```

如果仍然混入 decode，310P 可能进入 `PrefillCacheHit` 并走 split-fuse。

### Decode + 新 prefill 混推，chunked prefill 开启

```text
32 个 decode: q=1 each
新 prefill: prompt_len=3000
max_num_batched_tokens=2048
```

```text
decode q total = 32
prefill q = 2016
state = ChunkedPrefill
310P = split-fuse
```

### Decode + 新 prefill 混推，chunked prefill 关闭

```text
32 个 decode: q=1 each
新 prefill: prompt_len=1000
remaining budget = 2016
```

```text
prefill 可以完整放入，不会被切块
state = PrefillCacheHit
310P = split-fuse
```

如果新 prefill 是 3000 token，则放不进剩余 budget，会等待后续 step。

### Prefix cache hit

```text
prompt_len = 3000
prefix hit = 2000
suffix to compute = 1000
```

```text
q_len = 1000
kv_len = 3000
state = ChunkedPrefill 或 PrefillCacheHit
310P = split-fuse
```

### MTP decode

```text
num_spec_tokens = 2
每个请求本轮 q_len = 3
num_valid_tokens = 1
```

```text
state = SpecDecoding
```

310P 是否可用取决于当前模型和 backend 适配，不能只看调度层判断。

## 调试时建议打印的字段

如果要验证某一轮为什么走了某个 attention state，建议在 runner 或 attention backend 附近临时打印：

```python
logger.info(
    "attn_state=%s, num_scheduled_tokens=%s, "
    "num_computed_tokens=%s, seq_lens=%s, "
    "query_start_loc=%s, total_tokens=%s",
    attn_state,
    num_scheduled_tokens[:num_reqs],
    self.input_batch.num_computed_tokens_cpu[:num_reqs],
    self.seq_lens[:num_reqs],
    self.query_start_loc.cpu[:num_reqs + 1],
    total_num_scheduled_tokens,
)
```

310P attention 侧可以打印：

```python
qsl_cpu = attn_metadata.query_start_loc.cpu()
qlens = qsl_cpu[1:] - qsl_cpu[:-1]

logger.info(
    "310P attention: state=%s, qlens=%s, context_lens=%s, "
    "num_actual_tokens=%s, query_shape=%s",
    attn_metadata.attn_state,
    qlens.tolist(),
    attn_metadata.seq_lens.cpu().tolist(),
    attn_metadata.num_actual_tokens,
    tuple(query.shape),
)
```

注意不要在性能热路径长期保留大量 CPU/NPU 同步日志；这些字段适合临时定位调度和 state 映射问题。

## 常见误解

| 误解 | 更准确的说法 |
| --- | --- |
| 开启 chunked prefill 后 FA 的 q_len 一定是 2048 | 2048 是总 token budget 上限；FA 只在 `PrefillNoCache` 路径出现，实际 q_len 取决于本轮调度 |
| split-fuse 等于 chunked prefill | chunked prefill 是调度切块；split-fuse 是 attention 算子路径 |
| 关闭 chunked prefill 就不会走 split-fuse | P/D 混推和 prefix cache hit 仍可能走 split-fuse |
| P/D 混推等于 P/D 分离部署 | P/D 混推是单 batch 内混合执行；P/D 分离部署是多节点/多角色架构 |
| MTP q_len 大于 1，所以就是 chunked prefill | MTP 通过 `num_valid_tokens` 表达 spec decode 语义，不等同于 prefill chunk |
| graph mode 能自然覆盖所有 batch | graph replay 偏好稳定 shape；decode-only 最容易，mixed/chunked 更难 |

## 一句话总结

vLLM-Ascend 的 attention state 不是由“prefill 阶段”或“decode 阶段”静态决定的，而是由每个 step 的 batch 组成、每个请求的已计算 token 数、本轮调度 token 数，以及 prefix cache、chunked prefill、MTP、graph mode 等特性共同决定的。

对 310P 来说，最重要的实践结论是：

```text
PrefillNoCache -> FA
DecodeOnly -> paged attention
ChunkedPrefill / PrefillCacheHit -> split-fuse

chunked prefill 控制的是调度是否切 prompt；
split-fuse 是处理 chunk/cache-hit/mixed-batch 的 attention 算子路径；
关闭 chunked prefill 并不等于彻底禁用 split-fuse。
```
