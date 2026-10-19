# 当前架构与测量口径

[文档索引](README.md) · [首页](../README.md)

当前入口是 Mac Local。FastAPI 控制进程通过 spawn 模型 Worker 编排 DiscussionService；A/B 共享一份模型权重和 Tokenizer，各自保存 Prompt、ledger、Cache、pending 与事务状态。Control Plane 通过 REST/WS 投影元数据，不承担 Tensor/packet 的数据传输。

Local 数据面为 `StatefulTokenSession → TKVT/TKVI/TKVK bytes → InMemoryByteTransport → 接收检查 → 增量 Prefill 或 KV 恢复/修正 → 后续 Decode`。双向交接只有接收方真实提交且无错误才计为已确认。失败保留原状态/错误，不静默回退 TOKEN。

| 层次 | 实际行为 | 不应混淆的口径 |
| --- | --- | --- |
| 模型 | 本地真实 Prefill / cached Decode | 模型输出正确性需另外评价 |
| TOKEN | uint32 Token IDs 协议与接收方增量 Prefill | 理论最小 logical bytes 不等于完整 wire bytes |
| 增量 KV V0–V3 | 真实消息 KV 导出/编码/恢复及所选修正 | 修正位置或部分内容不证明语义等价，wire 通常仍更大 |
| 模拟信道 | 每 handoff 冻结一次 R/L；由完整 bytes 计算网络耗时 | 无真实无线测量、默认不 sleep，不约束内存链路吞吐 |
| AUTO | 质量/合法性过滤后，比较 TOKEN 与指定一个 KV，仅执行选中路径 | 未执行路径只有预测；没有合法 Profile 时不能补造实际成本 |
| Dashboard | 展示模型、来源、receipt、模式/SNR轨迹、成本和原文 | UI 动画不是网络性能，null 不是 0 |

`observed_local_ms` 是本地正式路径 wall time；模拟网络时延、Profile 本地预测、Synthetic 假设、派生总成本和整场 wall time分别记录，不能相互冒充或重复相加。Canonical clean-Prefill KV / `--real-adaptive` 是另一条基准路径，不直接套入讨论增量 Profile。

运行时手动/轨迹网络及 Shannon AR(1) 都按交接冻结。网络 PATCH 仅作用于下一未冻结交接；GET、刷新、历史回放不会推进 RNG。相同条件/seed 的模拟信道可重现，不承诺多设备模型数值或墙钟耗时一致。

V0 raw delta；V1 额外修正 K 的 RoPE；V2 重算连续尾段；V3 修正首个合法 ASCII 数字槽位。完整语义、pending/termination/contract 及兼容约束见 [增量 KV](incremental-kv-transport.md)、[V0/V1 审计](kv-v0-v1-implementation-audit.md)、[V2](kv-v2-suffix-recompute.md)、[V3](kv-v3-feasibility-and-validation.md)。

Session 与最近三份历史元数据主要保存在服务生命周期内；前端刷新可由 REST/WS 恢复，服务重启不保证历史或 Cache 持久化。模型切换先停止旧 Worker；旧 Session 不能跨模型继续生成。

历史 SocketByteTransport、双 Relay、独立 Agent Worker / 分布式 Coordinator 实现及测试保留。它们的本地软件验收不等于服务器 CUDA、双 GPU 或无线现场验收；正式 UI 已转为 Local，不需要为 Quick Start 部署服务器。原 architecture.md（原研发资料，未随公开副本提供：`architecture.md`）、system-design.md（原研发资料，未随公开副本提供：`system-design.md`） 混有阶段累积记录，应通过 [文档索引](README.md) 按版本阅读。
