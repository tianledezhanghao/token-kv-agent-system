# 7.5A 独立 Agent 状态与 TOKEN 增量接收

7.5A 提供 Discussion Runtime 的状态/通信基础；7.5B 已在其上实现[简单算术讨论与总结](discussion.md)。7.5A 本身不包含讨论策略。旧 `TokenHandoffSession`、Canonical TOKEN/KV、`RealAdaptiveRuntime`、Selector、Cost Model、Dashboard 的行为保持原样。旧 Canonical TOKEN 仍 clean Prefill；本页介绍的增量 TOKEN 是独立新增路径。

## 状态与最小 API

`agent/state.py` 的 frozen `AgentState` 保存 agent_id、system_prompt、backend、ChatFraming、精确 token_ledger、cache、cached_token_count、pending_token_id、assistant_open、draft_message、sent_messages、received_messages、local_input_spans 和 version。Cache 属于 session，调用者不得原地改写它；frozen dataclass 不意味着 tensor 不可变。

`AgentMessage` 保存 message_id、每 Sender 的 sequence、source/target、payload_mode、原始 message_token_ids、termination_reason、Sender 的半开 Token span，以及接收后确定的 receiver_message_start/content_token_count。token_count 从原始 IDs 派生。消息记录与 IDs 是不可变值；draft 尚未成功发送，不能当成 sent 记录。

```python
from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.stateful_token import StatefulTokenSession
from token_kv_agent_system.transport import InMemoryByteTransport

session = StatefulTokenSession([
    LLMAgent("A", backend_a, system_prompt="You are Agent A. Solve briefly."),
    LLMAgent("B", backend_b, system_prompt="You are Agent B. Check Agent A briefly."),
], InMemoryByteTransport())
session.append_local_user("A", "A local input for a later task.")
message = session.generate_message("A", "B", max_new_tokens=8)
receipt = session.handoff("A", "B")
b_state = session.state("B")
```

backend_a/backend_b 可以是同一 HF backend（Mac 一份模型、两份 cache），也可以来自 `IndependentBackends.agent_a_backend/agent_b_backend`。后者复用同 checkpoint/结构/dtype/tokenizer 身份检查；不意味着服务器已部署或 CUDA 已验收。初始化角色 Prompt 通过实际 `apply_chat_template` 得到 system IDs 并分别 clean Prefill 一次，不依赖旧 `LLMAgent.generate_from_token_ids()` 的 system_prompt 行为。

`append_local_user()` 只追加本地 user turn，保存其 Token span；它可用于 7.5B 给 A/B 提供原题。7.5A 不设计原题分发、讨论轮次策略或总结选择；7.5B 已实现这些编排。`generate_message()` 仅为有界 raw-logit greedy 缓存生成，产生一个待发送 draft；先发送现有 draft 才能生成下一条。`materialize_message()` 可以显式补齐 draft 并关闭 turn，重复调用不再 forward。Receiver 收消息时若有自己的 pending draft，会先补齐/关闭该 draft，同时保留它以便随后发送。

## Ledger 与 Cache 不变量

- `ledger[:cached_token_count]` 与 KV 的有效位置一一对应；真实 get_seq_length 和全部层 K/V 的序列长度必须等于该 count，且层数匹配模型。
- 生成最后一个 accepted output ID 时，ledger 比 cache 多一个位置，pending_token_id 等于 ledger 最后 ID。此前仅用于 argmax 的未接受 prediction 不进入 ledger。
- pending 必须恰好 forward 一次才进 cache；关闭消息还要补充实际模板所需的结尾。该 forward 的下一预测不自动接受。
- 初始化后正常接收/生成都只用 `continue_from_cache()`，attention mask/position 复用 backend 已有行为。不 Prefill 历史，不 Prefill 完整 Sender 消息以制造 cache，也不 decode peer IDs 后重编码。
- 每个更新先复制 work cache。成功校验后替换有效状态；失败保留原 cache、ledger、draft、version 和有效消息序号。handoff 同时完成 Sender 补齐和 Receiver 更新后才提交双方。
- 当前是同步本地 session，锁串行执行；失败的 transport 尝试可能留下 diagnostic transfer record，重试可能再次传 bytes。它不是网络上的 exactly-once 协议。

复制使用当前设备上的 `K.detach().clone()/V.detach().clone()` 和公开 `DynamicCache(ddp_cache_data=..., config=...)`。它消耗 O(历史 KV 容量) 的复制时间及额外内存，handoff 期间同时有双方原 cache 和 work cache；没有历史模型 forward，成本不能称为零。旧 prefix 的 tensor storage 与新 work cache、A 与 B 均不共享。

当前 adapter 明确仅支持 batch=1、完整 dense full-attention DynamicCache/DynamicLayer。Sliding/static/offload/其他 Cache 需要另行验证，不能因 Token ledger 存在就宣称支持近似或截断 cache。

## 实际 Chat Template 与 framing

本地 Qwen checkpoint：`7ae557604adf67be50417f59c2c2f167def9a775`，tokenizer 的模板与 [Qwen 官方 tokenizer 配置](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/7ae557604adf67be50417f59c2c2f167def9a775/tokenizer_config.json) 一致。无 tools 模式：

```text
<|im_start|>system\n{自己的角色 Prompt}<|im_end|>\n
<|im_start|>user\nMessage from A to B:\n{A 的原始内容 IDs}<|im_end|>\n
<|im_start|>assistant\n{B 自己的新输出 IDs}...
```

peer 消息属于 Receiver 的 **user turn**，不冒充 Receiver 曾生成的 assistant 消息。身份 label 是 Receiver 本地产生的 framing IDs。接收结束后 user turn 已闭合；只有 `generate_message()` 才追加一次 assistant generation prefix。

`ChatFraming` 从真实 tokenizer 的 template 渲染中用测试占位符分离 prefix/closing，验证 text/tokenize 两种路径 IDs 一致、system context 不被重写、固定历史样本前后边界相同、assistant generation prefix 与模板相容。占位符不进入真实输入，真实 peer 内容从不转成文本。也验证普通角色/身份字段，拒绝生成的角色 opener 等控制 Token；不支持的模板明确失败。

Token 分片拼接保留 Sender 原始内容 IDs，即使全段文本的 BPE 重编码可能选择另一种 segmentation，也不会为追求重新 tokenization 的一致而改变通信 IDs。完整模板验证针对 framing 与固定测试样本；不是任意模型模板的通用证明。当前约束是 closing 由一个 special EOT 开头，尾部仅含普通 IDs。服务器 Llama 模板与 position/API 需另行实测。

Qwen `<|im_end|>` 为 151645（EOT，模型生成 stop）；151643 为 `<|endoftext|>`，是另一个生成 EOS，不能当成 user EOT；151644 为角色 opener。生成 stop 集合使用 backend 的实际 generation_config/config/tokenizer EOS，不猜测数字：

- 正常结束：wire 保留原始完整 IDs，包括最后 EOS。若它等于模板 EOT，复用该 ID 关闭 turn，只补 closing 的普通尾部，不再写第二个 EOT。
- 不同 EOS：Receiver 将最后 EOS 视为生成控制信息，内容仍逐 ID 原样复制，关闭 turn 用 Receiver 的实际 EOT。Sender 自己的历史保留实际生成 EOS 并补模板 EOT；两者不是同一个符号。
- 达到 max_new_tokens：没有生成 EOS，保留全部内容 IDs，补完整模板 closing。不会把截断假装成正常结束。
- 单个 EOS 的空内容消息可传递；中间 EOS、角色控制 ID 等不作为普通消息内容。

`receiver_message_start` 是 Receiver 关闭自己的 pending turn 后的 count + 本地 user prefix 长度，在执行消息内容 forward 前即可确定。`receiver_content_token_count` 排除末尾生成 stop；wire token_count 和 Sender span 保留该 stop。后续 V1 应明确选择要迁移的 content span/EOT 位置，不混淆两个计数。

## TOKEN bytes 与扩展边界

```text
Sender draft raw IDs
  -> unchanged Token Protocol v1 (TKVT, uint32)
  -> InMemoryByteTransport bytes
  -> strict deserialize / identity check
  -> local user framing + raw content IDs + local close
  -> Receiver old KV clone + incremental forward(new IDs only)
  -> validated AgentState
```

消息 metadata 目前是同步 session 的本地数据，不添加 v1 header 字段，也不称为完整远程消息协议。今后 KV 接收可以复用 AgentMessage/AgentState、work cache/提交边界及目标起点；V0/V1 尚未接入正式 Runtime，V2/V3 尚未实现，本轮没有新的 KV Protocol 或 KV 修正策略。

## 验证

```bash
HF_HUB_OFFLINE=1 .venv/bin/pytest
.venv/bin/python -m compileall -q src tests *.py
.venv/bin/python run_stateful_token_smoke.py --device mps --rounds 3 --max-new-tokens 4 \
  --output benchmark-results/stateful-token-smoke.json
git diff --check
```

2026-10-10 Mac/Qwen2.5-0.5B-Instruct/MPS 本地 cache-only smoke：一份模型实例、两份独立 KV，初始化 A/B=20/24 Token。6 次真实 Token Protocol bytes 经 InMemoryByteTransport，每包4 Token、26 B；每次 Receiver 新增15 Token（9 prefix + 4 content + 2 close），无 Receiver 旧历史 Prefill。三轮双向结束 A/B=59/48、83/72、107/96。每次生成保留最后一个 pending，成功 handoff 后 ledger/cache 长度相等，旧历史保留、storage 独立。全部本次模型输出为 max_new_tokens；真实文本与 IDs 保存在 gitignored JSON，无数学正确性声明。

测试还覆盖纯标准库 Mock 三轮/边界/bytes/失败重试与两 Backend，及随机 tiny Qwen2/CPU 的真实 DynamicCache 增量/完整历史 oracle、copy storage 与 forward 后失败隔离。全量 **346 passed**（新增41），2条既有上游警告；compileall、diff check通过，详细结果记录在 `.agent/STATE.md`。7.5B 已加入简单问题、轮流讨论与指定 Agent 总结；模型算错应如实记录，不使用计算器或预设答案补救。

## 7.5B 的最小扩展

AgentState 增加 local_messages，保存已经闭合、未发给 peer 的总结生成记录。StatefulTokenSession 增加可选 max_context_tokens、generate_message 的 local_prompt 参数与 generate_local_message；原 7.5A 调用默认不变。阶段提示和生成、总结和闭合分别在同一 work copy 中成功后才提交。消息 wire 与 Receiver framing/ID 语义不变；详见 [讨论设计](discussion.md)。7.5A 历史测试和真实 CPU oracle 继续保留。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
