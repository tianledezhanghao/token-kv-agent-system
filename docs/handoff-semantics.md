# Token/KV Handoff Semantics

## Open Design Question — Working-Memory KV

当前 **Canonical KV Baseline** 比较 `Token(M)` 与 Sender clean `Prefill(M)` 构造的 `KV(M)`。优点是 same-semantic strict baseline；缺点是 Sender 需要重新 Prefill 消息 M。

正在研究 LatentMAS-style existing layer-wise KV working memory 经 `past_key_values` 继承的方案。它可能消除 Sender clean Prefill，但 existing working cache 可能包含 private prompt/history，Token/KV semantic boundary 因此仍需重新定义。它不是当前 baseline 的无成本替代。

7.1A 只准备模型/设备/独立 backend 适配，不实现 Existing Working KV、latent thought、full working-memory handoff 或 delta KV。Canonical Handoff Span、MESSAGE_READY、Sender clean Prefill 和 Selector cost semantics 保持不变。正式 CUDA Selector recalibration 暂缓至研究语义确认之后。

## Canonical Handoff Span

Phase 5/6 v1 将一次交付的 logical information boundary 定义为非空、有序、不可变的：

```text
handoff_token_ids: tuple[int, ...]
```

这个 Token sequence 是 TOKEN 和 KV 两种 representation 的共同 source of truth：

```text
Canonical Handoff Span
├── TOKEN: Token Protocol v1 bytes -> Receiver clean Prefill(span)
└── KV:    Sender clean Prefill(span) -> KV Protocol v1 bytes
           -> Receiver restore cache(span)
```

Span 可以来自真实 Sender generation，也可以由 tokenizer 生成确定性 scaling input。无论来源如何，同一次公平对照的 TOKEN/KV 必须使用完全相同的 ID tuple。不通过 text decode/re-tokenize 重建 span。

## Message-only representation

Canonical v1 是 message-only 语义。若 Sender 的私有 prompt/history 是 `P`，新生成消息是 `M`，公平对照是 `Token(M)` 与 `KV(M)`，而不是 `Token(M)` 与 `KV(P+M)`。后者交付了不同的信息集、位置语义和可复用计算，因此不是 representation-only comparison。

KV canonical preparation 对完整 `M` 从 clean cache 执行一次同步 Prefill。这使得 cache logical length 必须严格等于 `len(M)`，不存在 pending predicted Token，也不调用 materialization helper。Receiver KV path 不知道 prefix IDs，不重做 prefix Prefill；它只从 wire payload 恢复 cache 并 append Receiver-owned suffix。

## Cost boundaries

本项目同时保留两个成本边界，禁止用一个含义模糊的 total 覆盖它们：

1. **Representation-ready handoff**

   - TOKEN：Token IDs representation 已就绪，经 protocol/link/deserialize 后 Receiver Prefill。
   - KV：KV cache representation 已就绪，经 device-to-host/protocol/link/deserialize/restore 后 Receiver cache ready。

   ```text
   kv_handoff_cost_ms = total_serialize_ms
                      + simulated_link_latency_ms
                      + deserialize_ms
                      + total_restore_ms
   ```

2. **Message-ready end-to-end representation cost**

   ```text
   kv_end_to_end_representation_cost_ms
       = sender_kv_prepare_ms + kv_handoff_cost_ms
   ```

`total_serialize_ms` 已包含 device-to-host 和 protocol encode；`total_restore_ms` 已包含 CPU reconstruction、host-to-device 和 cache construction。Derived total 只使用 composite，不再加子阶段。`local_transfer_ms` 是单进程 software-call diagnostic，不与 simulated link 同时计入。

Sender clean Prefill 是 canonical KV(M) 的真实 representation preparation，不得隐藏。如果未来 workflow 天然持有一个语义恰好匹配 span 的可复用 KV，representation-ready 成本可能更合适；若需要为 message-only span 额外构造 KV，Phase 6 必须使用包含 Sender preparation 的 message-ready E2E。本轮不写死 Selector 策略。

## Full working-memory inheritance and LatentMAS

另一种合理但不同的协作语义是直接传递 Sender 当前完整 working cache。它可能包含 `P+M`，减少额外 Sender preparation，并更接近 LatentMAS 的 layer-wise working-memory inheritance。但它与 message-only Token mode 不是相同语义基线。

当前实现只复用 LatentMAS 可比的 layer-wise K/V、`past_key_values` inheritance 和 Receiver-owned input append 概念。它不实现 latent-thought generation、hidden-state recurrence、`Wa`、`inputs_embeds` latent reasoning 或 heterogeneous-model alignment。Full working-memory mode 留待未来独立语义/实验，不是 Phase 6 v1 adaptive selector 的默认公平基线。

## Correctness oracle

同一 span 与同一 Receiver suffix 同时运行：

```text
TOKEN path: Receiver clean Prefill(span) + suffix
KV path:    Sender clean Prefill(span) -> wire -> Receiver restore + suffix
Full:       Receiver clean Prefill(span + suffix)
```

三路 greedy next-token ID 必须一致。同时记录 TOKEN/KV 与 KV/Full 的 logits max/mean absolute difference；MPS/BF16 不要求 bitwise exact，但必须通过已明确的 numerical tolerance。Full Recompute 仅是 oracle，不是 KV Data Plane 的 Receiver prefix Prefill。

## Dashboard projection

Phase 5.4 的 Real KV Dashboard 固定使用上述 canonical message-span 语义，不提供 full working-memory 选项。Runtime 只把 `KVHandoffResult` 的 scalar/text/correctness metadata 投影到 Control Plane；packet、Neutral KV、tensor 与 DynamicCache 留在 Data Plane。UI 同时展示 representation-ready 与 message-ready E2E，且明确把 simulated link 标为仿真值。该固定 KV runtime 不运行 Selector，不能产生 `MODE_SELECTED`。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
