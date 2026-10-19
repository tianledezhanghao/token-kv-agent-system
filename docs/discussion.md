# 双 Agent TOKEN / 增量 KV 通用讨论

## 7.7B Prompt 配置（当前）

A/B 通用角色、分离的五类阶段指令、后端权威解析、初始 system Cache、固定 Session 与可复现记录见 [Prompt 审计和规则](discussion-prompts.md)，完整真实输出见 7.7B 实验（原研发资料，未随公开副本提供：`discussion-prompt-validation.md`）。Web 已由 7.7 接入；下方通信生命周期继续适用。

## 7.6B 手动模式与可观测性

```bash
# 默认全 TOKEN
.venv/bin/python run_discussion_demo.py --device mps --mode token
# 全 V0 / 全 V1
.venv/bin/python run_discussion_demo.py --device mps --mode kv --kv-strategy v0
.venv/bin/python run_discussion_demo.py --device mps --mode kv --kv-strategy v1
# 两轮四次交付，plan 优先于默认模式
.venv/bin/python run_discussion_demo.py --device mps --rounds 2 \
  --handoff-plan "token,kv:v1,token,kv:v0" \
  --output benchmark-results/discussion-mixed.json
# 两题四模式 + 固定状态单步对照（本地 cache-only）
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_validation.py --device mps
```

Python 配置采用 `HandoffChoice(CommunicationMode.KV, KVStrategy.ROPE_ALIGNED_DELTA)` 或 `HandoffChoice.parse('kv:v1')`。plan 必须恰好 rounds×2 条，否则在模型加载前拒绝；非法策略/V2/V3 明确拒绝。手动路由不调用旧 Selector/Canonical Cost Model。

每次成功 handoff 保存 raw generated IDs、content count、Sender/Receiver span、before/after、requested/executed mode/strategy、M Prefill count、repair、RoPE 位移、真实 wire bytes、termination 与工程状态。`DiscussionResult.handoffs` 还记录失败尝试及其 transport bytes；失败 executed=null，不冒称成功，无法确认的阶段为 null，fallback_reason=null 表示无 fallback。成功 turns 不包含失败发言；失败停止轮次及总结。CLI 分别显示 TOKEN/KV 数量和含失败尝试的总 wire bytes。

TOKEN 对空内容保留旧合法行为；KV envelope v1 不支持空 span，明确失败，不伪造零长度 KV 或改走 TOKEN。最终总结仍从指定 Agent 自己保存的真实 Cache 生成。

`DiscussionRuntime` 是 `StatefulTokenSession` 上的最小轮次策略。支持手动 TOKEN、KV V0/V1 与混合 plan；不接 Selector、V2/V3、RelayCaching 或计算器；7.7 Dashboard 仅旁路投影，不改变旧 Canonical Runtime 的 clean-Prefill 语义。7.5A 已提交为 `216862b`，本轮在此基础上独立开发。

## CLI

```bash
.venv/bin/python run_discussion_demo.py \
  --model Qwen/Qwen2.5-0.5B-Instruct --device mps \
  --question "23 + 48 = ?" --rounds 2 \
  --max-new-tokens 64 --summarizer A \
  --max-context-tokens 4096 \
  --output benchmark-results/discussion-add.json
```

默认同上（`--output` 默认不写文件），`--rounds` 允许 1–3，`--summarizer` 可选 A/B。设备沿用 backend 的 auto/cpu/mps/cuda/cuda:N。默认仅用 HF 本地 cache；需要下载时显式 `--allow-download`。用户指定的 HF offline 环境仍优先。每次输出上限默认 64，必须为正整数；总预算是**每 Agent** 的 Token 上限，包括 system、原题、本地阶段提示、自己的发言、收到的消息与所有 framing，默认 4096，并取与两个 backend 配置窗口的最小值。

`--output` 保存真实模型元数据、全部 generated/received IDs、消息 span、接收新增 IDs、cache 长度、状态和失败信息。失败留下的 draft 单独列为 undelivered_drafts，不算成功交付。仅展示时 decode 为文本；文本从不回流到推理通信。程序不计算或替换模型答案，也不将正确性作为执行成功条件。

## 生命周期与轮次

```text
ready → running
  两个 system 角色分别初始化 Prefill
  A/B 分别追加同一个原始问题（local user turn）
  第 1～N 轮：A cached generation → A→B 手动路由 → B cached generation → B→A 手动路由
  指定 A 或 B：本地总结提示 → cached generation → pending 补齐/turn 闭合
→ completed / rounds_complete
任一阶段失败 → failed / error（终止，不继续或自动重试）
```

角色初始化在 Runtime 构造时进行；`run()` 追加原题、执行轮次和总结。一次完整轮次是 **A、B 各发言一次**，共两次 handoff。默认两轮共四条通信消息，随后独立生成一次总结；总结不占讨论轮数，不发送到另一端。Runtime 一次运行后不可重复调用 `run()`，failed 后必须创建新 Runtime 才能重新开始。

A 回答原题、给简短理由、处理 B 的实际反馈；B 独立检查相关性、事实、逻辑与计算，指出具体问题或核查依据，无法确认则表达不确定。默认文字与五个阶段指令集中在 `discussion_prompts.py`，详情见 [Prompt 定义](discussion-prompts.md)。不强迫反驳，不注入标准答案。

原题通过两次 `append_local_user()` 输入双方，属于共同任务，不出现在任何 A→B/B→A Token packet 中。阶段提示是本地 user turn，不是 System Prompt；系统初始化一次，阶段按轮增量追加；总结继续使用该端自己的原题和真实讨论 Cache，不重新 Prefill transcript。Prompt 只在新建 Session 时确定，不能换字符串后沿用旧 KV。

## 状态、Cache 与增量生成

唯一可变状态仍由 7.5A `AgentState` / `StatefulTokenSession` 管理。每端长期维护 system_prompt、framing、token_ledger、cache、cached_token_count、pending_token_id、draft_message、sent_messages、received_messages、local_input_spans、version；本轮最小增加 `local_messages` 保存不发送的总结记录。

Mac CLI 共享一个 HuggingFace backend/model/tokenizer，但 A/B 的 cache 对象和全层 K/V storage 独立。Python API 可分别传 `backend_a` / `backend_b`，通过已有 same-checkpoint/tokenizer 兼容检查；将来服务器可接两个独立 backend/model，不代表本轮加载了真实第二模型或完成 Llama/CUDA 验收。

初始化只有两次 system Prefill。之后原题、新的阶段指令、assistant prefix、新生成 Token、TOKEN 接收消息与所有本地 closing 经 `continue_from_cache()` 从对应 Agent 自己的长期 KV 增量 forward。`generate_message(..., local_prompt=...)` 将阶段提示和生成放在同一个 work cache 上，失败不提交提示或半条发言。原有未传 local_prompt 的接口语义保留。

每次先 clone 独立 work cache；生成最后 accepted Token 保留为 pending，交付时真实 materialize 一次；成功才替换 session 中的有效状态。TOKEN Receiver 只 forward `本地 user prefix + 新内容 IDs + 本地 closing`；KV Receiver 只 forward 本地 prefix/closing，内容直接追加 KV，每次新增量可由 receipt 精确核对。没有旧历史 Prefill，没有把 Sender 完整 Cache 交给对方。clone 仍有 O(历史 KV 容量) 时间/内存成本，不能解释为零成本。

`generate_local_message()` 与普通发言复用 `_generate_working()` 的同一 raw-logit greedy cached Decode 循环；然后在同一隔离 work cache 上补齐最后 Token、关闭 assistant turn、清空 draft、记录 local_messages。总结不会伪造 sent/received 消息数，target_agent 为自身，message_id 使用 `A:local:1` / `B:local:1`。local sequence 与 wire sequence 分别使用记录表识别，不是远程协议。

## Token 消息与边界

```text
本方新 generated IDs（含末尾生成 stop）
 → Token Protocol v1 / serialize_token_ids
 → InMemoryByteTransport.transfer(bytes)
 → deserialize_token_packet / exact identity check
 → 接收方本地 user framing + delivered 内容 IDs + closing
 → 接收方原 KV 的 work copy / 增量 Prefill
```

沿用 7.5A 的 EOS/EOT/max_new_tokens 规则，协议布局不变，wire = 10 + 4 × generated Token count。消息 metadata 为本地同步控制数据，不新增 wire header：

- `sender_message_span=[start,end)` 精确指向 Sender ledger 中所有本次 generated IDs；EOS 情形内容 span 为 `[start,end-1)`，截断情形内容 span 为 `[start,end)`。
- `receiver_message_start` / `receiver_content_token_count` 精确指向 Receiver ledger 中内容，不包含角色 prefix 或末尾结构 stop；wire count 包含 stop。
- 真实 Qwen EOT（151645）在 closing 起点相同时复用一次，再加普通尾部；不同 EOS（151643）不被当成普通内容，Receiver 使用自己的模板 EOT；Sender 保留实际 EOS 并补正确模板 closing。
- 达到 Token 上限时保留全部真实内容，termination_reason=max_new_tokens，materialize pending 后补模板 closing；半句内容可以继续作为反馈，但不会留下半开 chat turn 或冒充 EOS 完成。
- 模型生成 role opener 等不允许的结构 Token 时明确失败。单 EOS 的空内容消息合法。原题/本地指令中的 chat control token 被拒绝。

总结与通信发言有相同内容/结构边界处理。具体 ChatFraming 的兼容边界见 [7.5A](stateful-token.md)：只支持 append-compatible、以单个 special EOT 开头的 closing；不通用支持所有 chat template。

## 有界性与失败

所有增量操作在 model forward 前检查预算。生成保守预留完整 max_new_tokens 和 closing，所以剩余空间不足时会拒绝启动，即便模型可能提前 EOS；不裁剪历史 Cache。Receiver 在 transfer 前检查实际新增消息/framing长度。budget 不能代替 RAM 资源管理，dense cache clone 仍可能 OOM，错误应如实报告。

local prompt+生成、总结+闭合、双方 handoff 分别是事务边界。handoff 失败时保留已成功生成的 source draft 和 pending，但不修改有效 Sender sent/Receiver received 记录，也不把失败的 turn 放入成功列表。此前完成的讨论/原题保留；不做整个 session 的历史回滚。生成失败保留该阶段以前的有效 cache。transport 可能留下失败尝试的 diagnostic record，不能将其算作成功 handoff。

任何失败都报告原题初始化、round/Agent generation、TOKEN方向或 final generation/close 等阶段，停止讨论，不悄悄跳过发言，不复用污染 work cache。CLI 成功退出 0；模型加载、初始化或运行失败退出 1；非法参数由 argparse 退出 2。结果 status=completed 只说明执行链完成，不代表答案正确。

## 验证与后续

```bash
HF_HUB_OFFLINE=1 .venv/bin/pytest
.venv/bin/python -m compileall -q src tests *.py
.venv/bin/python run_discussion_demo.py --help
git diff --check
```

新增 Mock/真实随机 tiny Qwen2 CPU 测试覆盖原题双方输入、角色、独立 Cache、多轮顺序、bytes、新消息-only、增量输入、EOT/EOS/截断、真实历史可见、A/B 总结、model generation 入口、无算术求解替代、ledger/KV oracle、预算与失败隔离。旧 Canonical 和 7.5A 测试仍运行。两道完整 Mac/Qwen/MPS 讨论、逐次长度/wire数据及结论见 真实验收记录（原研发资料，未随公开副本提供：`discussion-validation.md`）。前端未改。

7.6B 已正式消费 7.6A Decode 增量 envelope，接入长期 Cache 和手动混合讨论；旧独立 7.6A unpack-only probe 保留。详见 [接收设计](incremental-kv-transport.md) 和 真实结果（原研发资料，未随公开副本提供：`discussion-kv-validation.md`）。


## 7.7 Web 接入

原 Runtime 增加可选 metadata-only `event_sink(kind, data)`，在实际 generation、handoff、finalizing 点发事件；不改变默认 CLI 或协议。KV trace 的 M Prefill 从 IncrementalKVReceipt 读取。Web Service 在单 worker 中构造并运行同一个 Runtime，最终释放 Session Cache。配置、REST/WS、恢复和限制见 [Discussion Dashboard](discussion-dashboard.md)，实际验收见 Mac Web 验收（原研发资料，未随公开副本提供：`discussion-dashboard-validation.md`）。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
