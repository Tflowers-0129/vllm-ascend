# 压缩 Mask 算子功能：FA 与 PA 如何避免传入 max_seq_len * max_seq_len

这篇文档讨论的是 vLLM-Ascend 中 attention mask 的“压缩表达”问题：为什么推理引擎不希望每次都传入一个完整的 `max_seq_len * max_seq_len` dense mask，FA/FIA 和 PA 分别如何用长度、块表、稀疏模式和少量 mask 模板来表达同样的可见性规则，以及 310P 当前实现里哪些路径已经接近压缩表达，哪些路径仍然会保留 full mask 的内存开销。

这里先给出一个结论：

- 对通用 Ascend FIA 路径来说，真正重要的不是把完整二维 mask 每次喂给算子，而是用 `actual_seq_lengths_q`、`actual_seq_lengths_kv`、`block_table`、`sparse_mode`、`pre_tokens`、`next_tokens` 这些元数据描述 attention 的边界和因果关系。
- 对 PA decode 路径来说，通常根本不需要显式传入 causal mask。PA 通过 `block_table` 找到 KV cache 中属于每个请求的块，再通过 `context_lens` 限定每个请求可以看到的 KV 长度。
- 对 split-fuse / chunked prefill 来说，它处在 FA prefill 和 PA decode 中间：query 不是只有一个 token，但 KV 已经部分存在 paged cache 中，所以需要用 `q_len + context_len + block_table + mask rows` 一起表达可见性。
- 对 310P 当前代码来说，普通 prefill 的 `_npu_flash_attention` 和 split-fuse 的 `_npu_paged_attention_splitfuse` 仍然依赖显式 mask。尤其是 310P `AttentionMaskBuilder310` 会构造 `[max_model_len, max_model_len]` 的 full causal mask 并转成 FRACTAL_NZ。因此，严格意义上的“不分配 max_seq_len * max_seq_len mask”的压缩 mask 路径，在 310P 当前 prefill/chunked-prefill 路径里还没有完全实现。

这也是理解压缩 mask 时最容易踩的坑：压缩 mask 不是“没有 mask”，而是“不要把每个请求、每个 batch 的可见性都展开成一个巨大的二维矩阵”。它把一部分规则交给算子的稀疏模式和长度元数据，把另一部分规则交给 KV cache 的块表，把少量确实复杂的局部因果关系保留成小得多的 mask 片段。

## 为什么 full mask 不适合在线推理

在最朴素的实现里，causal attention 的 mask 是一个二维矩阵：

```text
mask[i, j] = 0      if j <= i
mask[i, j] = -inf   if j > i
```

如果最大上下文长度是 `max_seq_len`，这个 mask 的形状就是：

```text
[max_seq_len, max_seq_len]
```

这在训练或固定 shape 的离线场景里很直观，但在线推理会遇到几个问题。

首先是内存规模。`max_seq_len=16384` 时，元素个数是：

```text
16384 * 16384 = 268,435,456
```

如果是 fp16 additive mask，单个 dense mask 理论上就接近 512 MiB，还没有算上 NPU 特定格式转换、缓存、对齐和临时 tensor。对 310P 这类内存更敏感的设备来说，这个开销非常容易把 `--max-model-len` 变成 OOM 的直接原因。

其次是浪费。在线推理 batch 中的请求长度通常很不齐：

```text
req0: prompt 128
req1: prompt 2048
req2: decode 1 token, context 4096
req3: chunked prefill 512 tokens, context 8192
```

如果为了这些请求统一构造 `[max_seq_len, max_seq_len]`，绝大多数位置都是无效区域。真正需要告诉算子的只有几件事：

- 每个请求的 query 段在哪里。
- 每个请求的 KV 段到哪里为止。
- 当前 token 能不能看未来 token。
- KV 是连续 dense tensor，还是 paged KV cache。
- 是否有 sliding window、prefix cache、spec decode、MTP、PCP/DCP 这类改变可见性边界的特性。

所以压缩 mask 的核心目标不是改变 attention 的数学含义，而是把“二维布尔矩阵”改写成“算子可理解的结构化元数据”。

## Mask 真正表达的三类信息

在推理里，mask 不只是 causal 上三角那么简单。它通常混合表达三类信息。

**请求边界**

多个请求会被拼成一个 TND 形式的总 token 序列：

```text
q = [req0 tokens][req1 tokens][req2 tokens]...
```

不同请求之间不能互相 attend。这个边界可以用 cumulative sequence lengths 表达，例如：

```text
query_start_loc       = [0, 128, 640, 641]
actual_seq_lengths_q = [128, 640, 641]
```

这里的 `actual_seq_lengths_q` 通常是累计长度，而不是每个请求的单独长度。

**因果可见性**

同一个请求内部，位置 `i` 只能看 `<= i` 的 KV。传统 dense mask 会用下三角表达；压缩表达会用 `sparse_mode=3` 这样的 causal 模式，加上 `actual_seq_lengths_q/kv` 让算子知道每段序列的边界。

如果是 sliding window，则还要表达“只能看最近 W 个 token”。这时可以通过 `sparse_mode=4`、`pre_tokens=sliding_window`、`next_tokens=0` 来表达窗口范围。

**KV 存储位置**

prefill 时，K/V 往往就在本轮 dense tensor 中；decode 时，K/V 已经在 KV cache 中，并且按 block 分页存储。PA 不靠一个 `[seq, seq]` mask 找 KV，而是靠：

```text
block_table[request_id, logical_block_id] -> physical_block_id
context_lens[request_id]                 -> visible KV length
```

因此 PA decode 的“mask”实际上更多体现在 `block_table + context_lens` 上。

## Dense Mask 与压缩表达的区别

可以把两种方式理解成下面的对比。

**Dense mask**

```text
输入:
  query/key/value
  attn_mask: [max_seq_len, max_seq_len]

算子根据二维矩阵判断每个 q-k pair 是否可见。
```

优点是简单直观，缺点是 O(max_seq_len^2) 的内存和搬运成本。

**压缩表达**

```text
输入:
  query/key/value 或 query + paged kv cache
  actual_seq_lengths_q
  actual_seq_lengths_kv / seq_lens / context_lens
  block_table
  sparse_mode
  pre_tokens / next_tokens
  可选的小 mask 模板或局部 mask rows

算子根据序列边界、KV 长度、block table 和 sparse mode 在内部恢复可见性规则。
```

这种方式的内存复杂度不再被一个完整的 `max_seq_len * max_seq_len` mask 主导。它可能需要 `O(num_reqs)` 的长度数组、`O(num_reqs * max_blocks_per_req)` 的 block table，或者在 split-fuse 中需要 `O(sum(q_len) * max_seq_len)` 的局部 mask rows，但通常不会为每个 batch 都构造完整 dense mask。

## vLLM-Ascend 中与压缩 mask 相关的字段

压缩 mask 不是一个单独字段，而是一组元数据协同工作。

| 字段 | 作用 |
| --- | --- |
| `attn_mask` | 显式 mask tensor。可能是 full mask、固定模板、局部 mask，或者在某些路径为 `None`。 |
| `seq_lens` | 每个请求当前可见的 KV 长度。310P FA/PA 中常直接传给 NPU 算子。 |
| `seq_lens_list` | Python list 形式的 KV 长度，常传给 FIA 的 `actual_seq_lengths_kv`。 |
| `query_start_loc` | 每个请求 query token 在拼接后 TND tensor 中的起始位置。 |
| `actual_seq_lengths_q` | query 的累计长度，告诉 FIA 每个请求的 query 段边界。 |
| `actual_seq_lengths_kv` | KV 的累计长度或每请求长度，取决于具体算子接口。 |
| `block_tables` | paged KV cache 的逻辑块到物理块映射。PA/FIA paged 路径依赖它找 KV。 |
| `context_lens` | PA 中每个请求实际可见的 KV token 数。 |
| `sparse_mode` | 算子内部 mask 模式。常见语义是 `0` 表示不加 causal mask，`3` 表示 causal，`4` 表示 sliding window。 |
| `pre_tokens` / `next_tokens` | sliding window 或局部可见范围。 |
| `attn_mask_seqlens` | PCP/长序列并行中用于 masked attention 段的累计 query 长度。 |

所以当我们说“FA/PA 支持压缩 mask”，更准确的说法是：FA/FIA 和 PA 的算子接口支持用这些结构化元数据描述 attention 的稀疏可见性，而不是必须展开成完整 dense mask。

## FA / FIA 如何支持压缩 mask

FA 通常负责 prefill。prefill 的特点是 query 长度可能很长，而且 K/V 也来自本轮输入，所以每个请求内部需要完整的 causal 关系。

传统做法是传一个 full causal mask：

```text
query: [total_q, heads, dim]
key:   [total_k, kv_heads, dim]
value: [total_k, kv_heads, dim]
mask:  [max_seq_len, max_seq_len]
```

压缩做法则让算子知道每个请求的真实边界：

```text
actual_seq_lengths_q  = [q0, q0+q1, q0+q1+q2, ...]
actual_seq_lengths_kv = [kv0, kv0+kv1, kv0+kv1+kv2, ...]
sparse_mode           = 3
```

这样算子可以在内部按请求分段：

```text
req0: q[0:q0] attends kv[0:kv0] causally
req1: q[q0:q0+q1] attends kv[kv0:kv0+kv1] causally
...
```

它不需要一个巨大的二维矩阵来说明 req0 不能看 req1，也不需要把每个请求都 padding 到 `max_seq_len`。

在通用 `vllm_ascend/attention/attention_v1.py` 的 FIA 路径里，可以看到这种接口形态：

```python
torch_npu.npu_fused_infer_attention_score(
    query=query,
    key=key,
    value=value,
    atten_mask=attn_metadata.attn_mask,
    block_table=block_table,
    actual_seq_lengths=attn_metadata.actual_seq_lengths_q,
    actual_seq_lengths_kv=actual_seq_lengths_kv,
    sparse_mode=3,
)
```

这里 `atten_mask` 仍然可能存在，但 causal 的核心边界已经由 `actual_seq_lengths* + sparse_mode` 表达。对于 sliding window，路径会把 `sparse_mode` 改成 `4`，并传入 `pre_tokens` 和 `next_tokens`：

```text
sparse_mode = 4
pre_tokens  = sliding_window
next_tokens = 0
```

这表示每个 query token 只能看窗口内的历史 token，不需要展开一个完整的 window mask。

## 310P FA 当前为什么还不能算彻底压缩 mask

310P 的 `_npu_flash_attention` 路径和通用 FIA 路径不完全一样。

在 `vllm_ascend/_310p/attention/attention_mask.py` 中，`AttentionMaskBuilder310` 会生成 full causal additive mask：

```python
mask = torch.zeros((max_seq_len, max_seq_len), dtype=torch.float16, device=device)
mask.masked_fill_(upper, float("-inf"))
```

然后在 `_get_causal_mask` 中把它转换为 `ACL_FORMAT_FRACTAL_NZ`：

```python
self.causal_attn_mask_cache = torch_npu.npu_format_cast(
    nd_to_nz_2d(attn_mask),
    ACL_FORMAT_FRACTAL_NZ,
)
```

在 `vllm_ascend/_310p/attention/attention_v1.py` 的 `forward_prefill_310` 中，FA 调用是：

```python
torch_npu._npu_flash_attention(
    query=query,
    key=key,
    value=value,
    mask=attn_metadata.attn_mask,
    seq_len=seq_len,
    ...
)
```

这里 `seq_len` 可以告诉算子每个请求的真实长度，避免算子把 padding 区域当成有效 token；但 `attn_metadata.attn_mask` 本身仍然来自 max model length 的 full mask cache。因此，从“算子知道真实长度”这个角度看，它有压缩表达的一部分；从“是否不分配 `max_seq_len * max_seq_len` mask”这个严格标准看，310P 当前普通 prefill 还不是彻底的压缩 mask 路径。

这也是为什么 310P 文档里提醒不要依赖 `max-model-len` 自动探测：当前 310P attention path 会构造 `[max_model_len, max_model_len]` 的 mask，过大的 `max_model_len` 会直接放大 mask 内存。

## PA 如何支持压缩 mask

PA 主要服务 decode。decode 的典型形态是：

```text
每个请求本轮只产生 1 个 query token
历史 K/V 已经写入 KV cache
```

这时 causal mask 其实非常简单：当前 token 只能看已经存在的历史 token 和当前 token，不存在“未来 token”。如果 KV cache 中只暴露到 `context_len`，那么 PA 不需要再传一个二维 causal mask。

在 310P 和通用路径中，PA 调用都类似：

```python
torch_npu._npu_paged_attention(
    query=query,
    key_cache=self.key_cache,
    value_cache=self.value_cache,
    block_table=attn_metadata.block_tables,
    context_lens=attn_metadata.seq_lens,
    ...
)
```

这里的可见性由两部分决定：

- `block_table` 决定每个请求的第 N 个逻辑 KV block 在物理 cache 的哪个位置。
- `context_lens` 决定这个请求当前能看到多少个 KV token。

所以 PA decode 是压缩 mask 最典型的形态：没有显式 `attn_mask`，没有 `[max_seq_len, max_seq_len]`，只有 paged cache 元数据。

举个例子：

```text
req0 context_len = 1024
req1 context_len = 4096
block_size       = 128
```

PA 只需要知道：

```text
req0 使用 block_table[0, 0:8]
req1 使用 block_table[1, 0:32]
```

最后一个 block 中哪些 token 有效，由 `context_lens` 裁剪。这个信息足够表达 decode attention 的有效 KV 范围。

## Split-Fuse / Chunked Prefill 的 mask 为什么特殊

split-fuse 的关键点是：它不是纯 prefill，也不是纯 decode。

一个请求 prompt 很长时，chunked prefill 会把它分成多个 chunk。对某个正在处理的 chunk 来说：

```text
query 是当前 chunk 的 tokens
KV 包含已经写入 cache 的 prefix + 当前 chunk
```

所以它更像“多 token query 的 paged attention”：

```text
q_len      > 1
context_len = cached_prefix_len + q_len
KV         在 paged cache 中
```

这时不能像 decode PA 那样完全不需要 mask，因为当前 chunk 内部仍然有因果关系：

```text
chunk token 0 不能看 chunk token 1
chunk token 1 可以看 chunk token 0
chunk token 2 可以看 chunk token 0/1
...
```

310P 的 split-fuse 路径在 `AttentionMaskBuilder310.get_splitfuse_mask` 中做了一个“行选择”的压缩：

```python
qlens = query_start_loc[1:] - query_start_loc[:-1]
context_lens = attn_metadata.seq_lens
pos_list = [
    p
    for ql, cl in zip(q_list, c_list)
    for p in range(cl - ql, cl)
]
splitfuse_mask = full_causal_mask.index_select(0, position)
```

它的含义是：

```text
每个请求当前 chunk 有 q_len 个 query token
这些 query token 在完整上下文中的绝对位置是:
  context_len - q_len, ..., context_len - 1
从 full causal mask 中只取这些 query 位置对应的行
```

例如：

```text
cached prefix = 8
current chunk = 4
context_len   = 12
q_len         = 4

当前 chunk 的 query 绝对位置:
  8, 9, 10, 11

选择 full causal mask 的第 8, 9, 10, 11 行
```

这样得到的 mask 行数是 `sum(q_len)`，而不是 `max_seq_len`。从运行时传给 split-fuse 算子的 mask 形状看，它已经比 full square mask 更贴近实际 query token 数。

但是要注意 310P 当前实现仍然有一个关键限制：这些 rows 是从 `chunked_prefill_attn_mask` 这个 full causal mask 中 `index_select` 出来的，而这个 full mask 也是按 `max_seqlen` 生成的。因此它减少了 split-fuse 算子实际使用的 mask rows，但没有彻底消除 full mask cache 的分配。

## Split-Fuse 的 q_len 与 kv_len 关系

split-fuse 中最重要的两个长度是：

```text
q_len      = 本轮要计算的 query token 数
context_len = 当前请求已经可见的总 KV token 数
```

如果没有 prefix cache，也没有之前的 chunk：

```text
prompt_len = 2048
q_len      = 2048
context_len = 2048
```

这时它接近纯 prefill。

如果是长 prompt 的第二个 chunk：

```text
cached/chunked prefix = 2048
current chunk         = 1024
q_len                 = 1024
context_len            = 3072
```

第一个 query token 的绝对位置是 `2048`，它能看 `[0, 2048]`；最后一个 query token 的绝对位置是 `3071`，它能看 `[0, 3071]`。

如果 prefix cache 命中了一部分 prompt：

```text
prefix cache hit = 3000
new suffix       = 512
q_len            = 512
context_len       = 3512
```

mask rows 应该从完整 causal mask 的第 `3000` 到 `3511` 行选择。这正是 `range(context_len - q_len, context_len)` 的意义。

所以 chunked prefill、prefix cache 和 split-fuse 本质上都在改变同一个问题：当前 query token 在完整上下文中的绝对位置是多少，以及它能看到多少 KV。

## Prefix Cache 与压缩 mask

prefix cache 命中后，请求的 prompt 不一定需要完整重新 prefill。假设：

```text
prompt_len       = 4096
prefix_cache_hit = 3072
remaining_q_len  = 1024
context_len       = 4096
```

如果用 dense mask，仍然可能需要表达 `[4096, 4096]` 的完整因果矩阵。但对算子真正有用的是：

```text
query rows: 3072..4095
visible KV: 0..4095
```

在 FIA paged 路径中，这可以通过：

```text
actual_seq_lengths_q
actual_seq_lengths_kv / seq_lens_list
block_table
sparse_mode=3
```

来表达。

在 310P split-fuse 路径中，这会落到：

```text
q_len       = remaining_q_len
context_len = prefix_cache_hit + remaining_q_len
position    = range(context_len - q_len, context_len)
```

也就是只选择剩余 query 对应的 causal mask 行。

## Chunked Prefill 与压缩 mask

chunked prefill 的 `max_num_batched_tokens` 经常会被误解成“FA 算子输入一定是 2048”。更准确地说，它是调度器控制本轮最多安排多少 query token 的预算。实际每次 attention 的 query token 数是本轮所有请求被调度 token 的总和：

```text
total_q = sum(q_len_i for each scheduled request)
```

如果 `max_num_batched_tokens=2048`，那么一次调度通常不会超过 2048 个 query token，但不意味着每次都正好是 2048。请求到达时间、decode token、prefix cache 命中、剩余 prompt 长度、padding、图模式 shape bucket 都会影响实际 `total_q`。

对 mask 来说，chunked prefill 的价值在于：

- 不需要一次性为整个长 prompt 做 full prefill。
- 每次只需要表达当前 chunk 的 query rows。
- `context_len` 让算子知道当前 chunk 前面已有多少 prefix KV。
- 对 PA/split-fuse，KV 通过 `block_table` 从 cache 中取，而不是重新拼成一个大 dense KV。

但在 310P 当前实现里，chunked prefill 并不等价于“完全没有 full mask 内存”。它会在 split-fuse 阶段生成更小的 selected rows，但 selected rows 的来源仍是按 `max_seqlen` 缓存的 full mask。

## MTP / Spec Decode 与压缩 mask

MTP 或 speculative decoding 会让 decode 阶段不再是“每个请求只算一个 token”。例如每个请求一次产生：

```text
1 个 target token + 3 个 speculative tokens
```

这时单个请求本轮的 `q_len` 可能是 4。它看起来像小型 prefill，因为同一请求内这 4 个 token 之间也有因果关系：

```text
token0 不能看 token1/2/3
token1 可以看 token0
token2 可以看 token0/1
token3 可以看 token0/1/2
```

vLLM-Ascend 中有些路径会把多 token decode flatten 成多个逻辑 query 段，以避免不规则 mask shape；有些路径则依赖 FIA 的 `actual_seq_lengths_q` 和 `actual_seq_lengths_kv` 来表达每个 token 的可见范围。

可以把它理解成：

```text
普通 decode:
  q_len per req = 1
  PA + context_lens 足够

MTP/spec decode:
  q_len per req > 1
  需要额外表达同一请求内多个新 token 的 causal 关系
```

因此 MTP 会把“压缩 mask”问题重新带回来：虽然 KV 仍在 paged cache 里，但 query 侧已经不是单 token，需要通过 cumulative lengths、flatten 策略或局部 mask 表达 intra-step causal。

## PCP / DCP 长序列并行中的 mask 拆分

PCP/DCP 进一步说明了压缩 mask 的本质：它并不总是“构造一个更小的 mask tensor”，有时是把 attention 拆成几段不同语义的计算。

在 context parallel 路径里，长序列会被拆成 head/tail、mask/nomask 等部分。代码里会构造：

```text
q_head_idx
q_tail_idx
kv_with_q_head_nomask_idx
kv_with_q_head_mask_idx
kv_with_q_tail_nomask_idx
kv_with_q_tail_mask_idx
attn_mask_seqlens
head_attn_nomask_seqlens
tail_attn_nomask_seqlens
```

其中 nomask 部分可以用 `sparse_mode=0`，masked 部分才用 `sparse_mode=3` 和 `attn_mask`。这比给整段长序列构造一个巨大 mask 更细。

也就是说，PCP/DCP 的策略是：

- 对天然不需要 mask 的 KV 区间，不传 mask。
- 对需要 causal 约束的局部区间，传局部 mask 或使用 sparse mode。
- 最后把多段 attention output 和 LSE 合并。

这是压缩 mask 思路在长序列并行中的自然延伸。

## 图模式为什么喜欢压缩 mask

图模式，或者 ACLGraph / graph task，最怕两类东西频繁变化：

- tensor shape 变化。
- 需要重新分配的大临时 tensor。

如果每次请求组合变化都重新生成一个 dense mask，图捕获和 replay 会非常难稳定。压缩 mask 对图模式友好的地方在于：

- `attn_mask` 可以变成固定模板或固定 bucket。
- 每轮只更新 `actual_seq_lengths_q`、`actual_seq_lengths_kv`、`seq_lens`、`block_table`。
- PA decode 甚至不需要显式 mask，只需要更新 `context_lens` 和 block table。
- graph cache 可以按 `num_tokens` 或 bucket 复用，而不是按完整 mask shape 重建。

在通用 `attention_v1.py` 的 full graph FIA 中，可以看到图参数会保存：

```text
query/key/value
block_table
attn_mask
actual_seq_lengths_q
actual_seq_lengths_kv
sparse_mode
pre_tokens
next_tokens
```

这说明图模式中真正变化的 attention 语义主要被压缩进这些元数据里，而不是每次展开一个新的 dense mask。

## 与其他引擎的相同点

从思想上看，vLLM-Ascend 的压缩 mask 和其他推理引擎是一致的。

PagedAttention 的共性是：

- KV cache 分 block 管理。
- decode 时通过 block table 和 context length 找 KV。
- 不为每个 decode step 构造 full causal mask。

FlashAttention varlen 的共性是：

- Q/K/V 按 token 拼接。
- 用 cumulative sequence lengths 表达 batch 内每个请求的边界。
- kernel 内部根据 causal flag 或 window 参数做 mask。

Spec decode / MTP 的共性是：

- 一次 decode 多个 token 时，需要恢复小段 causal 关系。
- 可以 flatten 成多个小 query，也可以传更丰富的 sequence length 元数据。

差异主要在硬件和算子接口：

- CUDA/Triton 生态更常见 `cu_seqlens_q/kv + causal` 的 varlen kernel。
- Ascend CANN/FIA 接口会显式出现 `actual_seq_lengths`、`actual_seq_lengths_kv`、`sparse_mode`、`pre_tokens`、`next_tokens`、`block_table` 等参数。
- 310P 当前路径因为算子能力和适配状态，仍保留 full mask cache，和更完整的 FIA 压缩表达有差距。

## 310P 当前实现的关键事实

结合当前代码，310P 上可以这样总结。

**普通 prefill**

路径：

```text
forward_prefill_310 -> torch_npu._npu_flash_attention
```

输入：

```text
mask    = attn_metadata.attn_mask
seq_len = attn_metadata.seq_lens
```

特点：

- `seq_len` 表达每个请求真实长度。
- `mask` 来自 full causal mask cache，并转成 FRACTAL_NZ。
- 因此它不是彻底的 compressed-mask path。

**纯 decode PA**

路径：

```text
forward_paged_attention -> torch_npu._npu_paged_attention
```

输入：

```text
block_table
context_lens = seq_lens
```

特点：

- 不传显式 `attn_mask`。
- 可见性由 KV cache block table 和 context length 决定。
- 这是最接近“完全不需要 max_seq_len^2 mask”的路径。

**chunked prefill / split-fuse**

路径：

```text
forward_chunked_prefill_310 -> torch_npu._npu_paged_attention_splitfuse
```

输入：

```text
mask         = selected causal mask rows, FRACTAL_NZ
seq_len      = qlens
context_lens = seq_lens
block_table
```

特点：

- mask rows 按 `range(context_len - q_len, context_len)` 选择。
- 算子实际处理的 query rows 是 `sum(q_len)`。
- 但 selected rows 来源于 full causal mask cache，所以仍有 full mask 的基础内存开销。

## 如果要真正优化 310P 的 full mask 开销

如果目标是让 310P 侧也做到“不传入、不分配 `max_seq_len * max_seq_len`”，优化方向不是简单关掉 `attn_mask`，而是要让算子接口和上层 metadata 同时支持压缩语义。

可能的方向包括：

- 在 310P FA prefill 中引入支持 `actual_seq_lengths_q/kv + sparse_mode` 的 FIA 类路径，减少对 full mask 的依赖。
- 对 split-fuse mask 不再先构造 full causal mask 后 `index_select`，而是直接按 `position` 生成所需 rows，避免 class-level full mask cache。
- 如果底层 `_npu_paged_attention_splitfuse` 支持更强的 sparse mode，则用 `q_len + context_len` 表达 intra-chunk causal，而不是传 selected mask rows。
- 把 mask 生成做成按 bucket 的小模板，例如只生成 `[max_chunk_tokens, max_context_bucket]`，但这仍然不是最理想的 metadata-only 表达。
- 对 graph mode 预先固定少量 shape bucket，让 mask 模板、workspace、block table buffer 可复用。

这里最关键的是：只在 Python 层少传一个 `mask` 参数通常不够。算子必须知道 causal、request boundary、query offset、KV context length 这些信息，否则结果会错。

## 调试时应该看哪些量

如果怀疑压缩 mask 或 split-fuse mask 行为不符合预期，可以优先打印或断点检查这些字段：

```text
attn_metadata.attn_state
attn_metadata.query_start_loc
attn_metadata.actual_seq_lengths_q
attn_metadata.seq_lens
attn_metadata.seq_lens_list
attn_metadata.block_tables.shape
attn_metadata.attn_mask.shape
q_len = query_start_loc[1:] - query_start_loc[:-1]
context_len = seq_lens
position = range(context_len - q_len, context_len)
sparse_mode
pre_tokens / next_tokens
```

不同状态下重点不同：

| 场景 | 重点检查 |
| --- | --- |
| `PrefillNoCache` | `actual_seq_lengths_q`、`seq_lens`、`attn_mask` 是否和 prompt 长度一致。 |
| `PrefillCacheHit` | `q_len` 是否只包含未命中的 suffix，`context_len` 是否包含 cached prefix。 |
| `DecodeOnly` | `block_tables` 和 `context_lens` 是否正确，不应期待 full causal mask。 |
| `ChunkedPrefill` | `q_len` 是否是当前 chunk，`context_len - q_len` 是否等于已经处理的 prefix 长度。 |
| `SpecDecoding/MTP` | 每个请求是否有多个 query token，`actual_seq_lengths_q` 是否表达了 intra-step causal 边界。 |
| PCP/DCP | mask/nomask 的 index、`attn_mask_seqlens`、`actual_seq_lengths_kv` 是否对应同一段拆分。 |

尤其在 310P split-fuse 中，`position` 是判断 mask rows 是否正确的核心。如果 `context_len` 或 `q_len` 错了，selected rows 就会整体偏移，表现为 token 看到了不该看的未来 token，或者看不到应该可见的 prefix token。

## 常见误解

**压缩 mask 不是没有 mask**

PA decode 可以没有显式 `attn_mask`，但它仍然通过 `context_lens` 限制可见 KV。FIA 可以用 `sparse_mode` 表达 causal，但仍可能传一个模板 mask。split-fuse 可能传 selected rows。它们都属于“不要展开完整二维矩阵”的不同层级。

**`seq_len` 不等于 q_len**

在 prefill 中，`seq_len` 可能等于 prompt 长度；在 split-fuse 中，`q_len` 是当前 chunk 的 query token 数，而 `context_len/seq_len` 是当前可见的总 KV 长度。prefix cache 命中后，这两个值会明显不同。

**chunked prefill 不保证每次 FA 输入都是 2048**

`max_num_batched_tokens=2048` 是调度预算，不是每次算子输入的固定长度。真实 `total_q` 取决于本轮被调度请求、decode token、剩余 prompt、prefix cache 和 padding。

**310P split-fuse 的 selected rows 不等于彻底 compressed mask**

它减少了传给 split-fuse 算子的 mask 行数，但当前实现仍先缓存 full causal mask。因此它是“运行时局部化 mask”，不是“完全不分配 full mask”。

**PA 不传 mask 不代表没有 causal 语义**

decode PA 的 causal 语义来自“KV cache 里只暴露到当前 context length”。如果 `context_lens` 多算了，就可能看见不该看的 token；如果少算了，就会丢历史上下文。

## 一句话总结

FA/FIA 的压缩 mask 核心是把 full causal matrix 改写为 `actual_seq_lengths + sparse_mode + optional mask template`；PA 的压缩 mask 核心是把可见 KV 改写为 `block_table + context_lens`；split-fuse 则把两者混在一起，用 `q_len + context_len + block_table + selected mask rows` 表达当前 chunk 的因果可见性。

对 vLLM-Ascend 当前代码来说，通用 FIA/PA 路径已经体现了这种压缩表达；但 310P 普通 prefill 和 split-fuse 仍保留 full causal mask cache。因此，如果问题是“FA/PA 算子理论上如何避免传 `max_seq_len * max_seq_len`”，答案是依靠长度元数据、块表和 sparse mode；如果问题是“310P 当前实现是否已经彻底避免 full mask”，答案是还没有，纯 PA decode 路径做得最彻底，FA prefill 和 split-fuse 仍有继续优化空间。
