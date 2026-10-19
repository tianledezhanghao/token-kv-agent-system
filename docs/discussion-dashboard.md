# 7.7 / 7.7B 手动通信与双 Agent Discussion Dashboard

## 启动与使用

后端使用本地 HuggingFace cache，启动时只加载一次模型；当前验收环境是 Mac / Qwen2.5-0.5B-Instruct / MPS / BF16。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python run_dashboard.py --discussion --device mps
```

另一个终端：

```bash
cd frontend
npm run dev -- --host 127.0.0.1
```

打开 http://127.0.0.1:5173/，进入「双 Agent 讨论」Tab。输入问题，例如 `23 + 48 = ?` 或 `12 * 13 = ?`，设置 1–3 轮、1–128 每次生成 Tokens（默认 64）、总结 Agent A/B。每轮严格按 A 发言并交付给 B、B 发言并交付给 A 执行，最后指定 Agent 从自身讨论历史生成总结。总结不是 handoff。问题最多 512 字符，Runtime 的每端完整上下文预算仍为 4096，受模型窗口进一步限制。

统一 TOKEN、KV V0 或 KV V1 覆盖所有 handoff。Mixed 为每轮两个方向分别选择策略，默认两轮 `TOKEN → KV V1 → TOKEN → KV V0`。计划在点击开始前确定，运行期间配置和提交按钮锁定。V2/V3 禁用并标为「尚未实现」；Discussion AUTO 禁用，旧 Canonical Selector 尚未适配独立讨论语义。

实时聊天只显示 Runtime 实际生成的消息，每条保留 Agent、Round、生成 Token 数和 `eos` / `max_new_tokens`。生成完成和交付完成是不同阶段。传输行可展开 requested/executed mode、策略、内容 Tokens、实际 wire bytes、Sender/Receiver span、Receiver Cache before/after、M Prefill、位置位移、修复方式和错误。空输出与截断如实展示；没有算术求解器、评分器或输出修正。

最终结果独立展示模型原文、总结 Agent、状态、配置轮数、成功/尝试 handoff 数、TOKEN/V0/V1 次数、总 wire bytes 和 A/B 最终 Cache Token 计数。结束时释放 Session Cache，显示的最终计数是释放前记录的 metadata。

## Agent 提示词配置与预览（7.7B）

「问题与策略」内展开「Agent 提示词配置」，提供 A/B 独立多行编辑器，初始均留空。留空、null 或纯空白使用后端默认；非空自定义原样采用，每端最多 4000 字符。可只修改一方；各自恢复按钮只清空本方，“恢复默认”清空双方。运行中锁定配置，新 Prompt 只对新建 Session 生效。

展开“新 Session 实际生效 Prompt 预览”查看服务器解析的 A/B text/source；“阶段指令（只读）”单独显示五类阶段指令和版本。前端无默认 Prompt 文本副本，通过服务器解析 endpoint 取得结果，丢弃过期响应。错误明确展示，预览成功前不能启动。

讨论记录内“该 Session 实际 Prompt 配置（固定）”显示历史 Session 的实际 A/B Prompt、来源、阶段版本/文字和 model/device/dtype。刷新经 REST 恢复，不依赖当前编辑器草稿；修改草稿或恢复默认不会改历史配置。最近三个记录仅进程内保存，无数据库。

完整默认值、修改前真实来源、chat template/Cache 审计和规则见 [Prompt 文档](discussion-prompts.md)；工程与模型质量分别见 7.7B 完整实验（原研发资料，未随公开副本提供：`discussion-prompt-validation.md`）。

## REST API

所有 Discussion 路径并列于旧 `/api/state`、网络和 Canonical 通信接口。旧 API 响应字段没有改变。后端未用 `--discussion` 启动时，可查询 Discussion availability，但启动返回 503。

| 方法与路径 | 作用 |
| --- | --- |
| `POST /api/discussions/prompts` | 只读解析 A/B 覆盖，返回实际 text/source、阶段 version/stages，无 Prefill |
| `POST /api/discussions` | 验证配置、预留执行槽、返回 202 和 Session 快照；后台启动 |
| `GET /api/discussions/latest` | availability、模型元数据、idle/running/completed/failed、active_session_id 和最新 Session |
| `GET /api/discussions/{session_id}` | Session 当前状态、完整 transcript、handoff trace 与最终结果 |
| `GET /api/events?limit=100` | 复用已有最近事件接口（limit 1–200） |
| `GET /api/health` | 模型运行时仍可使用 |

统一模式请求：

```json
{
  "question": "23 + 48 = ?",
  "rounds": 2,
  "max_new_tokens": 64,
  "summarizer": "A",
  "mode": "kv:v1",
  "handoff_plan": [],
  "agent_a_system_prompt": null,
  "agent_b_system_prompt": ""
}
```

合法 mode 只有 `token`、`kv:v0`、`kv:v1`、`mixed`。前者三种不允许传非空 plan；Mixed 必须提供严格按轮次/方向排序的 rounds×2 项：

```json
{
  "question": "23 + 48 = ?",
  "rounds": 2,
  "max_new_tokens": 64,
  "summarizer": "B",
  "mode": "mixed",
  "handoff_plan": [
    {"round": 1, "source": "A", "target": "B", "mode": "token"},
    {"round": 1, "source": "B", "target": "A", "mode": "kv:v1"},
    {"round": 2, "source": "A", "target": "B", "mode": "token"},
    {"round": 2, "source": "B", "target": "A", "mode": "kv:v0"}
  ]
}
```

可选 `session_id` 为 8–64 位 ASCII 字母、数字、下划线或连字符；不提供则服务器生成。前端为每次新点击生成 UUID；重用同一 ID 返回 409，完成或快照被淘汰也不能重启该 Session（进程内保留 ID tombstone）。并行的新 Session 返回 409，不排队、不并行调用共享模型。非法请求（包括空问题、非整数轮数/预算、超限、V2/V3/AUTO、计划长度/顺序/方向错误、未知字段）返回 422。不存在或已淘汰的 Session 返回 404；模型未加载/服务关闭返回 503。

新增 Prompt 字段非法类型、超长、非法控制字符或当前 tokenizer 的特殊控制 token 返回 422，启动前拒绝。`POST /api/discussions/prompts` 只接受两个可选 Prompt 字段；共用 Runtime 解析来源。没有运行中修改 Prompt 的 PATCH，已启动 ID 不可复用。

Session 字段：`prompt_config`（实际 A/B text/source、版本与全部阶段文字）、`model`（该 Session 模型/设备/dtype），`session_id/status/config/revision/phase/round/agent/sequence/created_at/updated_at`，`turns`，`handoffs`，`final_answer`，`summary`，`initial_cache_lengths/cache_lengths/cache_released`，`error/failure_stage/termination_reason/transport`。`final_answer` 来自真实 `generate_local_message()`，没有后处理算术答案。`summary.handoff_count` 只计成功交付，`attempted_handoff_count/total_wire_bytes` 包含失败尝试的实际 transport 记录。

API 只投影配置、文本、计数、span 与状态，不暴露 DynamicCache、tensor、KV packet、neutral payload 或 Token ID 数组。`requested_mode` / `executed_mode` 用 Runtime 原始 `token/kv`；KV 策略为 `raw_delta/rope_aligned_delta`。未成功交付时 executed mode 为 null；失败 trace 保留请求策略、error、尝试 wire bytes。当前没有任何 fallback；`fallback_reason=null` 代表未发生 fallback，不能解读为交付成功。

## WebSocket

复用 `/ws/events` → `EventBroadcaster`，没有新 KV 数据通道。`CommunicationEvent` 保留原外层 `type/timestamp/source/target/mode/payload_bytes/.../metadata` 结构；Discussion 的关联信息放在 metadata，外层 mode/legacy latency 为空，不借用 Canonical 估算值。

| 事件 | 实际执行点与增量数据 |
| --- | --- |
| `DISCUSSION_STARTED` | 预留活动槽后，初始化角色/问题之前；包含配置、实际 prompt_config 与模型 |
| `AGENT_TURN_STARTED` | 调用该 Agent 的真实生成之前 |
| `AGENT_TURN_COMPLETED` | 成功生成、decode 后；包含一条 turn |
| `HANDOFF_STARTED` | 调用原 TOKEN/KV handoff 前；包含 requested mode/strategy |
| `HANDOFF_COMPLETED` | 原 handoff 返回有效 receipt 后；包含一条真实 handoff trace 与 cache lengths |
| `HANDOFF_FAILED` | 原 handoff 抛错；包含失败尝试 trace |
| `DISCUSSION_FINALIZING` | 指定 Agent 真实本地总结之前 |
| `DISCUSSION_COMPLETED` | Runtime 完成、Session Cache 引用释放后；包含最终答案/计数与实际 prompt_config |
| `DISCUSSION_FAILED` | 初始化/生成/接收/总结/worker 错误终止后；包含原因和有效结果计数 |

每个事件均有 `metadata.session_id/revision/status/round/agent/sequence` 与外层 UTC timestamp。`sequence` 是本讨论全局 turn/handoff 编号，总结为 rounds×2+1。初始化尚无当前 Agent 时为 null，Round 为 0。前端只在收到完成 trace 后标交付成功。事件不重复发送完整 transcript，不带 KV bytes 内容，不使用虚构延时或进度。

初次 REST 恢复、WS 连接/重连后 REST 校准；同 Session 通过 revision 拒绝旧事件/旧快照，检测 revision 缺口后取服务端快照。WS 断开或恢复请求失败时按 3 秒间隔查询 REST；重连绝不会自动 POST。两个 Tab 都打开后，各自订阅同一既有 WS pipeline。慢客户端仍按原 broadcaster 有界队列规则丢弃最旧事件，REST 是恢复依据，不承诺完整 WS replay。

## 执行与存储边界

`DiscussionService` 只是 Control Plane。配置转为原 `DiscussionConfig/HandoffChoice` 后，单个 `ThreadPoolExecutor(max_workers=1)` 执行原 Runtime 构造、角色初始化、所有轮次和总结。模型权重只在进程启动加载一次；同一 Runtime 中 AgentState/长期 KV 保留全部讨论阶段。初始化与同步推理都在 worker，FastAPI event loop 不执行 forward。RLock 保护活动槽/状态投影；worker 结束释放 Runtime/StatefulTokenSession 引用后再发终态事件和解除槽。

客户端断开不取消任务。服务关闭在后台线程等待已接受任务结束；不强行中断模型。进程内保留最近 3 个完整 metadata Session，以及已使用 Session ID 的轻量 tombstone；不保留结束的 Session Cache。进程重启后无历史/运行 Cache，REST 最新状态回到 idle，不报告旧任务 completed。没有数据库、Redis 或任务队列。

`--discussion` 同时保留旧 **Mock Canonical Dashboard** 网络/Selector/Timeline，以独立 Mock Runtime 运行；Discussion 共享的是事件 collector，不共享可变推理状态。旧 `--real-token/--real-kv/--real-adaptive` 入口与语义保持，不能和 `--discussion` 组合。工厂拒绝把 Discussion backend 与独立锁的真实 Canonical runtime 共用。

## 真实数据与实验性限制

TOKEN 正常增量 Prefill 新内容。V0 Raw Delta KV、V1 RoPE aligned Delta KV 严格复用 7.6B 原 envelope、接收和 append；V1 实际改动 post-RoPE K，V 不旋转。执行模式/策略和 KV M Prefill 从原 receipt 读取；没有前端常量 0。失败无静默 TOKEN fallback，不支持的 device/cache、恢复、RoPE 或 append 错误会进入 failed。

wire bytes 是服务器原协议/transport 的真实 packet 长度。Discussion 使用 InMemoryByteTransport，**没有物理网络、没有 simulated link 成本，也没有性能加速结论**。旧 Tab 的 bandwidth/base latency 和传输时间仍来自明确的 SimulatedNetwork，仅用于旧实验。新 Tab 不调用旧 Cost Model/Selector，不展示缺乏依据的端到端网络延迟。KV(M | Sender历史) 一般不等价于 Receiver TOKEN Prefill；同题答对不证明数值等价或一般质量。

V2/V3、AUTO、新 Selector、真实 Socket/网络、KV 压缩、Llama/CUDA、多用户调度、模型取消、Token 流式文本和持久化不在当前范围。当前为完整发言完成时逐条展示，不做 Token streaming。

## 检查与验收

```bash
HF_HUB_OFFLINE=1 .venv/bin/pytest
(cd frontend && npm test && npx tsc -b && npm run build)
.venv/bin/python -m compileall -q src tests run_discussion_web_validation.py run_dashboard.py
.venv/bin/python run_dashboard.py --help
.venv/bin/python run_discussion_demo.py --help
.venv/bin/python run_discussion_web_validation.py --help
git diff --check
# 后端已启动；客户端测试依赖 dev/httpx 与 websockets
.venv/bin/python run_discussion_web_validation.py
# 可另验收 B 总结
.venv/bin/python run_discussion_web_validation.py --summarizer B --output benchmark-results/discussion-77-api-b
```

验证脚本实际通过 REST 启动四种模式，WS 收集真实执行事件、运行中访问 health，核对 REST transcript/trace/summary/wire 与事件一致、策略一致、KV M Prefill=0、初始角色/问题 Cache 计数跨 Session 一致。答案原文记录但不评分。脚本不代替真实浏览器点击验收。浏览器需逐种选模式、点击开始、核对 A/B 顺序、展开传输、查看最终结果、刷新恢复并回归旧 Tab。

7.7 历史记录见 Mac Web 验收（原研发资料，未随公开副本提供：`discussion-dashboard-validation.md`）；当前通用 Prompt 的完整结果与 Web 验收见 7.7B 实验（原研发资料，未随公开副本提供：`discussion-prompt-validation.md`）。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
