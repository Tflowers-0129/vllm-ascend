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

待设备可用后，保持请求集、seed、上下文长度、并发和 K 一致，比较 eager、
原图和新图的输出、接受率、每轮延迟、Host 提交长尾及 Dispatch 到达差。
必须同时观察 Host 和 NPU；Dispatch 时长下降本身不能证明端到端提升。
当前未进行上板回放，不能宣称 416.8 ms 长尾已经在设备上消失。
