# GLM-5.3-Flash MTP 合并图的 metadata 与 token 容量

问题版 profiling 中，首个 draft Dispatch 的跨卡到达差中位数为 1.115 ms，
后续步骤为 158 us。另有 rank 4 的 draft 图晚提交，使其余七卡等待约 416.8 ms。
这份采集没有调用栈，不能将该长尾精确归因到某个 `.to()` 调用。

图回放减少了逐算子的 Host 提交工作，但每轮输入准备、首步 metadata 及不同
rank 的注意力工作量仍会不同。EP Dispatch 需要接收其他 rank 的路由输入，
先到的 rank 会在此等待，因此等待时间显示在 Dispatch 中。合并五步图不会
消除这五步之间的自回归和通信依赖，也不会自动均衡 DP 的有效上下文长度。

第 38 轮 rank 4 的 draft 图到其他卡进入 Dispatch 后约 416.16 ms 才提交，
到达 Dispatch 后自身只运行 16.16 us，其余七卡的 Dispatch 均约 416.8 ms。
同线程 Host `aten::to/copy_` 跨度约 321.6 ms，期间缺少对应 CANN 调用。
这证明 Host 路径拖延图提交；当前记录不能区分 CPU 拷贝、线程调度、分配或
采集开销，也没有相同负载 eager 对照，不能证明该停顿由入图新引入。

原路径每轮在图外准备全部五步 metadata，后四步仍按首步 verify token 桶运行。
例如 12-token 图的 LMHead 只处理 2 行，projection/routing/Dispatch 却处理 12 行。

## 代码修改

- 首步仍处理完整 verify 桶，保持 MTP 第一个预测的输入语义。
- 后续步骤使用请求容量：`min(max_num_seqs, max(verify_bucket // (K + 1), 1))`。
  相应更新 MC2 的 `padded_num_tokens`，避免 prepare 把小输入再补回 verify 容量。
- 在启动 warmup 时创建每步 metadata buffer；将后四步的 metadata 更新和模型
  forward 一起捕获，运行时图外只构建首步 metadata。
- 每轮写入一个稳定的 device 请求数标量。后续步骤的 device query boundaries、
  seq_lens 和 slot_mapping 都按该标量屏蔽 padding，包括空 DP rank。
- 直接读取 runner 的持久 block table，防止图引用一次性 clone 的旧地址；各步
  seq/query/slot buffer 独立。长度由选中 token 的 device position 推导，包含拒绝
  校正；越过最大长度的 token 禁止写 KV。
- profiler 增加 `mtp_set_inputs_first_pass`、`mtp_first_pass_metadata` 和
  `mtp_graph_execute` 区间，结合 runner 现有 `draft_token` 范围定位图前准备。

没有删除同步依赖，也没有添加跨卡 barrier。五个 draft 步骤仍按相同顺序执行
五次 EP 通信。

## 生效范围

当前只在 GLM multi-KV-group MTP、A5 NoPE、TP=1、PCP=DCP=1、FULL draft 图、
均匀 decode 布局、动态 `global_bs` 路径启用。DSA-CP、SP、TP>1、非均匀
prefill、KV offload、M-RoPE、压缩 attention、并行 drafting 和均匀 MC2 保留
原实现。这个范围对应问题版采集的 D 侧。非均匀 prefill 不能用 token 桶除以
`K+1` 推导请求数；启用 FULL 图本身不满足该优化的前提。

保持 `speculative_config.enforce_eager=true` 可运行 eager 对照。改变图内部结构后
必须重新启动并捕获图；不要在正在服务的进程中替换源码期待旧图更新。

## 验证

CPU 回归覆盖 6/12-token verify 桶到 1/2-row decode 桶、空 rank、请求数变化、
最大长度、持久 block table、每步 buffer 隔离，以及一次合并 forward 的五步依赖。
设备测试用例检查真实 ACL 图回放时请求数/拒绝边界变化后 metadata 是否正确。

上述 CPU 和设备测试分别验证 metadata 契约及真实设备行为，不能替代模型精度
验收。2026-10-14 的 A5 模型采集和 padding 设备测试结果如下。

## 纯 D 采集定位到的同步拷贝

固定 32 个 256K 请求，DP8 的请求分布为 `[5,2,2,4,2,5,7,5]`。预填充和 KV
接收完成后才开始采集；采集窗口起止的 prompt token 计数不变、等待请求为 0、
抢占计数不变，运行中请求数在八个域保持相同。主缓存和 indexer 均使用 FP8，
MTP 为五步。比较中固定输入集；5 秒对照各输出 1024 tokens，20 秒稳定性场景
将输出延长到 4096 tokens，不能把完整请求耗时直接放在同一张性能表中比较。

`_remove_spec_graph_padding_queries` 原来修正 CPU query boundaries 后，通过
`.to(device)` 同步上传。rank 0、5、7 的 5 个有效请求落入 6-request 图桶，
会执行该分支。拷贝的数据量很小，但同步语义会等待已经排队的设备工作；其他
rank 可以先提交下一轮 target 图，最终在首层 EP Dispatch 等待这些 rank。
因此 profiling 中长的是 Dispatch 等待，触发等待的位置却在图外的 metadata
准备阶段。相同负载 eager 对照也出现约 4 ms 的到达差，不能全部归因于 MTP 入图。

修复直接对已有 device query boundaries 做 `clamp`，阈值来自 CPU mirror。
CPU mirror 单独克隆修正，保留共享 MLA metadata 不变，不引入 device `.item()`、
跨卡 barrier 或新的 Host→NPU 上传。回归覆盖连续/非连续输入、不同 padding 桶、
不补齐路径、非法非后缀 padding；旧实现作为负对照会因同步上传而失败。

另外，FULL draft 桶可能超过 `max_num_seqs`。辅助 KV group 现在预分配每步独立
的 block table，按实际来源行数计算 slot mapping，再补齐剩余表行；避免 shape
不匹配，并保持 capture/replay 使用的地址稳定。

## A5 复测结果

以每轮 42 个 target Dispatch 和 5 个 draft Dispatch 对齐八张卡。只排除首两轮
及最后一轮采集边界，不按耗时剔除异常值。到达差是八卡同序号算子开始时间的
最大差；轮次间隔是相邻轮次最后一个 draft Dispatch 的最晚结束时间之差。
这些是 profiler 内的轮次指标，不能直接当作单 token TPOT 或无 profiler 吞吐。

| 场景 | 稳态轮次 | Target 首层到达差 P50/P99 (ms) | Draft 首步到达差 P99 (ms) | 轮次间隔 P50/P99 (ms) |
| --- | ---: | ---: | ---: | ---: |
| 修复前 graph，5 秒 | 101 | 4.067 / 4.699 | 0.266 | 52.120 / 52.743 |
| Eager，5 秒 | 100 | 4.015 / 7.913 | 0.288 | 52.222 / 56.034 |
| 修复后 graph，5 秒 | 108 | 0.639 / 1.141 | 0.257 | 48.472 / 49.150 |
| 修复后 graph，20 秒 | 411 | 0.671 / 1.794 | 0.340 | 48.697 / 49.725 |

5 秒修复前后，首层到达差中位数下降约 84%，轮次间隔中位数下降约 7%。
20 秒采集的 32 个请求全部生成 4096 tokens；padding 独立设备测试为 5/5 通过。
模型请求成功及 token 数正确不等同于完整的输出精度和接受率对照验收。

## 剩余等待与验收边界

典型轮次的首层计算输入仍有 12、24、36、42 行之分，首层图内计算约为
1.78–2.37 ms，形成部分可解释的到达差。20 秒数据最慢首层到达差为 3.098 ms，
发生在第 37 轮：rank 7 的 target 图晚提交，最后到达的 Dispatch 只执行约
23 us，而先到的 rank 2 等待约 3.108 ms。该窗口的 Host copy 没有原先 3–4 ms
的同步跨度；仍需区分 Host 调度、准备和采集开销，不能凭当前证据删除同步依赖。

原问题版的 416.8 ms 极端尾部没有在这段稳态采集中复现；有限采集不能证明
它在任意负载下彻底消失。CPU Event wait 中等待上一轮图完成的时间也不能全当作
可消除空泡。后续若出现新的明显尾部，应先检查图提交与到达差，再决定是否重采。

Eager worker 在退出阶段另有 glibc malloc 链表损坏和 SIGABRT；该问题尚未解决，
不能把推理成功当作退出生命周期已通过。无 profiler 的正式吞吐、模型输出精度、
接受率对照及退出稳定性仍需单独验收。
