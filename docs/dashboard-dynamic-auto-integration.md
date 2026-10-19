# 8.2B — Desktop Dashboard Integration for Dynamic Incremental AUTO

2026-10-14，起始 HEAD `6e0c351`。正式 React Dashboard 已完成配置 → 启动 → 真实讨论 → 决策/交付证据 → 历史/刷新恢复的浏览器闭环。复用 8.1 阅读优先桌面 UI，不重做通信算法或 Canonical Selector。以下测试数字均为本轮结果。

## 1. 原型审查与采用情况

| 资产 | 文件修改时间（本机） | 状态与本轮采用 |
|---|---|---|
| `docs/dashboard-redesign-preview.html`、desktop/narrow PNG | HTML 01:43:42 | 第一轮静态预览，未作为正式实现基准；用户未跟踪资产保留 |
| `docs/dashboard-design-exploration/` | index 02:23:53 | 第二轮 A/B/C 静态探索，index 明确标记未通过；全部保留，不暂存 |
| `docs/dashboard-reading-prototype/` | index 11:21:50，桌面规范 11:26:50 | 最新完整、用户已确认的第三轮；8.1 已迁入正式 React |
| `docs/dashboard-ui-implementation/` | README 12:44:30 | 8.1 正式前端与真实浏览器验收记录，本轮沿用组件/尺寸/交互 |
| `docs/token-communication-baseline_副本.md` | 用户既有副本 | 通信基线资料，非桌面原型；不改动、不暂存 |

候选判断依据是修改时间、明确的确认/否决记录和功能完整性，而非任意挑选。主面板、点阵 Agent 通道、连续阅读、原生配置 dialog、详情 drawer、阅读位置保持已经实现；原型中的固定回放和设计数据仍只是静态预览，不进入正式前端。无需重新迁移一套界面。

冲突仅在旧说明：V2/V3/AUTO 禁用、没有模拟网络成本、按前一次序号推断消息接收来源。这些是 8.1 时的能力记录，本轮以 8.2A 实际契约更新；不引入常驻技术侧栏挤压阅读。

## 2. 正式改动范围

复用 `DiscussionPanel`、`DiscussionConfigDialog`、`DiscussionDetails`、`useDiscussion`、`projectDiscussion/restoreDiscussion` 和 `Workspace` 组件。新增 `autoConfig.ts` 做配置解析/校验，`NetworkAutoSettings.tsx` 做表单，`HandoffInsights.tsx` 展示候选、网络与失败。

后端仅补观测/投影：`HandoffLog.message_id`、交接事件 message ID、发言 `received_message_id`、`HANDOFF_STARTED.selected_kv_strategy`、Session `active_handoff`、决策记录消息 ID/方向、latest 成本身份与最近三次会话摘要。仍调用原 `IncrementalSelector` 与 `StatefulTokenSession.handoff/handoff_kv`；没有新通信算法、模型后端或成本公式。

## 3. 桌面布局

沿用最大 1360px 主面板、900px 阅读列（含身份标记）、850px 正文、18px 字号和 32.4px 行距。html/body 固定可用视口，真正的长内容在 `.reader` 或 `.modal-body` 内滚动。网络仅增加一条紧凑状态行；候选成本、Cache、RoPE/重算/阶段计时均按需打开。

配置 footer 和详情返回按钮不参与内容滚动。有效高度≤550px的桌面放大场景另允许阅读面板内部滚动，保留160px正文区域，避免裁掉底部入口。新增字段沿用中性灰表面、现有字体和焦点规则，无图表/状态管理/UI 大型依赖。Canonical 页面继续独立存在，其 Mock/clean-KV 数据不参与增量讨论。

## 4. AUTO 操作流程

1. 启动 `HF_HUB_OFFLINE=1 .venv/bin/python run_dashboard.py --discussion --device mps`，另一终端运行 `npm --prefix frontend run dev`，打开 `http://127.0.0.1:5173/`。
2. 输入问题，在“配置 / 任务”设置 1–3 轮、1–128 Token 预算与总结 Agent，Agent 分区保持原服务器 Prompt 预览。
3. “通信”选 AUTO，默认 quality_first；需要性能实验时主动切换策略并配置候选/Profile。
4. “网络”选择 manual 或 trace、基础延迟，检查输入错误后开始讨论。
5. 正式 REST 返回 Session，WS 更新完整发言、决策、handoff 与终态。点击发言交付标签或实验详情查看证据。

手动 TOKEN、KV V0–V3、MIXED 逐次计划、V2 固定数/比例、secret_sum 与独立 Prompt 功能保留。没有假逐 Token streaming，也没有暂停/取消/实时网络控制按钮。

## 5. manual / trace

提交既有字段 `network_mode/bandwidth_mbps/base_latency_ms/bandwidth_trace`。manual 明确提交空 trace；trace 将逗号序列解析为数字数组，拒绝空项、非有限值、非正带宽与超过 1000 项。基础延迟为有限非负单向毫秒值。

trace 按 **Handoff 序号**推进，不按秒变化。网络详情展示当前序号、来自 manual/trace 的冻结快照、当前轨迹项、下一项配置、trace ID/来源以及 hold_last。耗尽后当前轨迹项保持最后一项，而不是虚构第 5/6 个配置项；实际 Handoff 5/6 仍独立列出快照。

活动 Session 配置只读，明确说明冻结和不支持实时 PATCH。旧 `/api/control/network` 仅属于 Canonical，不用于 Discussion。

## 6. Profile 输入与校验

支持粘贴 JSON，或浏览器选取 `.json` 文件，经 `File.text()` / `JSON.parse()` 后作为 **内联 `cost_profile` 对象**提交。文件导入上限 1 MiB。不提供服务器文件路径输入或文件读取接口，不内置 synthetic 默认成本/模型 preset。

校验增量字段、`decode_incremental_handoff_v1` semantics、version/notes、provenance、sample_count、sender/receiver 身份、有序整数适用域，以及受支持阶段和有限非负系数。measured 必须至少一个样本，synthetic 可以 0。latest 提供当前成本模型身份，完整对比模型 ID/revision/配置摘要/backend 摘要/dtype/device；不将 snapshot 路径的实测 Profile 改写为模型名身份。

消息/Cache/重算适用域、缺阶段/首次 startup 费用和执行能力仍由真实 Selector 逐次判定，UI 展示真实排除原因，不自行补成本。8.2A snapshot 实测 Profile 在本次按模型名加载的服务上被浏览器准确拒绝，未偷偷套用。本轮无新实测校准或性能优势结论。

## 7. 两种策略与来源

quality_first 默认只有 TOKEN 获得质量许可；界面说明 V0–V2 未认证、V3 已观察到退化，不暗示带宽变化会自动切 KV。可不提供 Profile，本地/总预测显示“未校准”。

performance_experiment 要求主动选择和显式 Profile JSON，可设置非空候选白名单。默认不包含 V3；加入 V3 后额外许可未勾选时不能提交。显示语义风险、无数字可能失败、实测尚无 KV 成本交叉点。

`measured_incremental` 与 `synthetic/test_only` 来源区别显示。Synthetic / Test Only 在配置 footer、会话主状态、讨论记录、候选详情与冻结配置中持续可见；预测数值不能当真实性能测量。历史性能策略无 Profile 时的安全 TOKEN 选择也不会错误解释为 quality_first 质量过滤。

## 8. 每条消息的真实通信标签

发言 `turn.message_id` 与同一 Session 的 `handoff.message_id` 严格匹配，并核对 source/target（可用时）；不按数组位置、相邻序号、正文或字符串猜测关联。

成功标签为“B 已接收 TOKEN”“A 已接收 KV V1”等，KV 版本读取实际 receipt 的 `executed_kv_strategy`：raw_delta → V0、rope_aligned_delta → V1、suffix_recompute_delta → V2、numeric_slot_delta → V3。必须同时有 executed_mode、append_succeeded 且无 error。实际接收版本不由 AUTO/MIXED 请求标签推断。

已生成尚未交接、已选择尚未交付、失败、缺少/歧义 ID 均有独立状态。最终答案标“本地总结，无额外通信交付”，不继承上一条标签。`received_message_id` 另保留明确的输入上下文来源；主界面显示本条发言自己的交付证据。

## 9. decision / receipt 关联

决策记录保留 `handoff_index/message_id/source/target`，receipt 保留 `sequence/message_id`，活动交接存入 `active_handoff`。在 AUTO_DECISION 时可看预测，HANDOFF_STARTED 仍未交付；只有 HANDOFF_COMPLETED receipt 才能确认成功。

详情分别呈现 requested、selected、executed，失败时实际交付为“未完成交付”，保留已选择的候选和失败阶段。AUTO_DECISION_FAILED 没有交付 receipt 时仍保存失败记录及网络，不伪造成功、fallback 或零成本。

## 10. 候选成本与排除原因

候选表读取真实 `auto_decision.candidate_estimates/excluded_candidates`，显示 TOKEN、V0–V3 的预测本地/模拟网络/预测总计和 Eligible / Selected / Excluded。null 不填 0，排除项不参与前端排名。

映射 not_whitelisted、quality_not_certified、known_quality_regression、model_profile_mismatch、profile_out_of_domain、profile_recompute_out_of_domain、missing_cost_profile、missing_startup_cost_profile、unsupported_empty_kv、unsupported_strategy。未知 reason 原样显示。失败 decision 的既有异常 JSON 仅用于读排除原因，绝不用正文关联消息。

主要理由显示质量许可/未校准或允许候选中最小预测总成本；同成本排序仍由后端确定。Profile 版本和不确定性 notes 保留，完整阶段预测仍可从交接原始元数据查看。

## 11. 预测、模拟与实测

- 预测本地处理/预测总计：Selector 执行前成本模型数字；无校准则未知。
- 模拟网络传输：冻结网络参数与真实 receipt 包长计算的理论值；不是物理网络耗时。
- 实测本地处理：当前设备实际 handoff 的 observed_local_ms；AUTO 为同步 instrumented handoff，手动计时按 observation_scope 原样说明。

额外组合命名 **“实测本地处理 + 模拟网络耗时”**，不叫真实 E2E。它不包含整个讨论、后续生成或物理分布式网络。Wire bytes、Cache before/after、span、RoPE 位移、V2/V3 重算 Token/层索引/forward 次数及工作量均来自 receipt。

## 12. REST / WS 字段来源

| 来源 | 前端用途 |
|---|---|
| POST `/api/discussions` | 原 strict 配置 schema，新字段均数字/布尔/内联 JSON，无路径 |
| GET `/api/discussions/latest` | 模型可用性、active_session_id、当前完整 Session、cost_model_identity、最近三次摘要 |
| GET `/api/discussions/{id}` | 显式历史 ID 与刷新恢复的完整投影 |
| POST `/api/discussions/prompts` | 服务器 Prompt 预览 |
| GET `/api/events?limit=100`、WS `/ws/events` | 有界事件日志与实时增量 |
| Session `auto_decisions/active_handoff/handoffs/turns/final_answer` | 决策历史、pending、实际交付与模型正文 |

事件使用实际既有名称：AGENT_TURN_STARTED/COMPLETED、AUTO_DECISION/FAILED、HANDOFF_STARTED/COMPLETED/FAILED、DISCUSSION_FINALIZING、DISCUSSION_COMPLETED/FAILED。

## 13. 恢复与异常

保留原 revision 防倒退、缺口 GET 修复、重连只 GET 的机制。实时 reducer 现在投影决策与活动交接，receipt 按序号去重；日志按 Session/revision 去重及排序。其他旧 Session 事件不能更新当前讨论；新的 DISCUSSION_STARTED 通过 REST 重新同步，历史固定 ID 不被切换。

详情“会话”提供最近三次元数据。显式选择将 ID 存入 `?discussion=<id>`，刷新仍 GET 此 ID，能查看较早的 AUTO 决策。同 ID 的旧 REST 不倒退 WS 状态。历史被逐出/进程重启时解释并恢复 latest；网络错误保留当前记录与重试入口。不宣称跨进程持久化。

另有活动讨论时不能重复启动。WS 终态清除活动标记；当前 Session 的终态不会被较旧 REST 的 active ID 再锁定按钮。所选历史期间通过终态事件及必要 GET 维护全局活动状态。错误、503/409/422 保留可理解反馈和草稿。

## 14. 真实联调证据

完整精简投影及 WS 序列（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/integration.json`） 来自本轮浏览器提交的真实 Qwen2.5-0.5B-Instruct / revision 7ae557… / MPS / BF16 / offline。单份权重、A/B 独立 Cache，21 次真实生成、16 次 handoff，全部 Session 工程 completed；不据此声称答案质量通过。

| 会话 | 浏览器配置 | 实际结果 |
|---|---|---|
| `4aae9032…` | quality_first，3轮/24 Tokens，trace 20/1000/50/500，2ms | 6 TOKEN；最后两次 hold_last；revision 33 |
| `4b064df1…` | synthetic 性能路由，2轮/16 Tokens，0.01/1000/0.01/1000 | TOKEN/V1/TOKEN/V1；两个 KV 均 199,943B；revision 23 |
| `ea4a4aa0…` | 最终契约，手动 TOKEN，manual 50Mbps/1ms，1轮/8 Tokens | 2 TOKEN；截断提示正确 |
| `2f912efa…` | 最终契约，quality_first，trace 20/1000，1轮/16 Tokens | 2 TOKEN；作为历史 ID 刷新与 WS 恢复验收 |
| `c1098efb…` | 最终契约，synthetic 性能路由，trace 0.01/1000 | TOKEN/V1；V1 199,943B；revision 13 |

所有 AUTO 预测 wire bytes 与 receipt 相等，所有发言/decision/receipt ID 相符。已真实收到 AUTO_DECISION、HANDOFF_STARTED/COMPLETED、发言、最终答案和终态。选择结果不替代 receipt。

新增历史列表投影需要重启本轮自建后端；此前两次 Session 完整快照已冻结保存，未覆盖历史 benchmark。随后仅做三次短预算浏览器复验，没有重跑大型校准。两次 synthetic 路由均永久标 test_only，只证明真实路由，人工成本不是速度测量。

本地测试代理只服务 production `frontend/dist` 并转发真实 8000 REST/WS。`/__test/drop_ws` 属于 `tests/dashboard_browser_harness.py`，不在生产 API 中。真实断开后 WS 连接数 7→8、三次 discussion POST 数不变；所选历史 ID 和两张 receipt 保持，见 integration/history records。初次讨论另验证了 Vite 代理中断恢复。

## 15. Chromium 浏览器与视口

使用 Codex 内置真实 Chromium 浏览器（Chrome 扩展控制本机不可用），不是 jsdom。以下是本轮实际**内容视口**；另额外扣除 87px 检查浏览器工具栏占用。没有将截图伪称为原生 Chrome 外窗实拍。

| 指定内容视口 | 额外较小内容视口 | 结果 |
|---|---|---|
| 1440×900 | 1440×813 | 讨论/答案/长配置全部通过 |
| 1366×768 | 1366×681 | 无遮挡、确认按钮可达、配置内部滚动 |
| 1920×1080 | 1920×993 | 主面板1360px，正文850px，无整页溢出 |
| 1280×800 | 1280×713 | 较窄桌面通过，正文不缩字号 |

24组最终布局记录（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/desktop-checks.json`）、5组详情与40px阅读位置保持（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/drawer-checks.json`）、5组 pending 标签边界（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/pending-checks.json`）、历史刷新记录（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/history-checks.json`）、720×406/640×356桌面放大检查（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/zoom-checks.json`）。document 始终等于内容视口且 scrollTop=0，正文 18px/32.4px/850px。桌面放大时阅读面板内部可滚动，底部入口实测仍在356/406px视口内，document仍不滚动。短视口配置内容约1205px，由375px内部阅读区滚动，footer仍可见；详情内容约2775px亦独立滚动。

罕见状态记录（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/state-checks.json`） 为**明确标记的 fixture，无模型推理**，覆盖生成、AUTO 决策/未交付、KV失败、决策失败、终态和空状态。TOKEN/KV成功、配置、trace、最终答案及恢复另有上述真实会话，不把 fixture 画面充当模型烟测。

必要 JPEG 共约0.4MiB，均正式 React 构建页面：配置1366（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/screenshots/config-1366.jpg`）、1920阅读（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/screenshots/desktop-1920.jpg`）、质量决策（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/screenshots/quality-decision-1440.jpg`）、真实 synthetic 路由（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/screenshots/synthetic-discussion-1440.jpg`）、明确标记的失败 fixture（原研发资料，未随公开副本提供：`dashboard-dynamic-auto-evidence/screenshots/fixture-kv-failed-1440.jpg`）。不提交大型 raw、权重或用户原型图片。

## 16. 本轮测试与发现

- Frontend **171 passed / 8 files**（历史121，新增50）：配置/REST数字映射、manual/trace、Profile/身份/provenance、浏览器文件解析、V3许可、候选/排除/未校准、实际KV版本、selected/executed、ID关联、重复/乱序/缺口、终态解锁与历史刷新。
- 受影响 Python/API **67 passed**（初轮；随后新增历史摘要回归已进入全量）。全量 Python **922 passed / 1原可选AutoGen skip / 2既有warning**；历史916不替代本轮结果。
- TypeScript `tsc -b`、Vite production build、`compileall src tests *.py`、`git diff --check` 通过。
- 浏览器 production bundle 完成布局、真实 REST/WS、Profile身份拒绝、V3守卫、局部滚动、历史选择、刷新与断线检查。

发现并修复：旧 fixture 无消息 ID；legacy/scripted backend 无成本身份不能导致 latest 500；空模型响应保持兼容；历史投影加入后旧 REST active ID 曾锁住已完成会话，已补 WS终态/当前Session守卫及回归；640×356桌面放大场景补面板滚动，secret_sum配置同样保留synthetic来源。旧测试未删除或降低质量/失败断言，仅按新增契约补 fixture ID 和真实新字段。

原始 REST/WS/log 在忽略目录 `benchmark-results/dashboard-82b/`；小型完整投影带 raw SHA。历史8.2A数据及用户设计文件均未改。Git提交仅本轮正式代码、测试、文档与 `.agent`，不 push。

## 17. 未实现能力

没有活动 Session 实时 PATCH、物理限速、跨机 Transport、取消/暂停、新通信算法、3B 专项、Prompt 优化或全量性能 benchmark。会话只有进程内最近三次，事件日志是100条有界窗口；普通讨论没有可靠自动质量评分，不显示伪造成功评分。

生成仍采用原1–128 Token cap；本轮显示真实 token_count、预算、termination_reason 与 max_new_tokens 提示。EOS 与达到上限分开，最终工程 completed 不掩盖输出截断。本轮已有可靠单条终止原因，不把缺值推断为 EOS。

## 18. 8.2C

系统处理正常讨论生成预算（含512 Token产品预算）、更完整的截断/上下文预算投影、活动网络 PATCH 的生效序号与更新事件、并发/失败/重试语义。物理网络与跨机传输需独立设计与验收；性能 Profile 的新环境校准和质量认证保持后端任务边界，不用界面规则制造 KV 优势。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
