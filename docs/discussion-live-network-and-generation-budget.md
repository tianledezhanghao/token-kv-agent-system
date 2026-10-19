# 8.2C — 实时模拟网络、生成预算与截断

2026-10-15（Asia/Shanghai），起始 HEAD `626da4e`。基于已确认展示/调试布局实现；没有改主面板、正文 CSS、点阵、页面滚动结构、历史 Prompt、评分或原始 benchmark。工程完成不代表模型语义质量通过。

## 1. 实时网络 API

Discussion 专属接口，与 Canonical `/api/control/network` 完全独立：

```http
GET /api/discussions/{session_id}/network
PATCH /api/discussions/{session_id}/network
Content-Type: application/json

{"bandwidth_mbps":50,"base_latency_ms":2,"manual_override":true}
```

PATCH 的带宽/延迟允许部分更新，至少提供一个值。带宽有限且 >0；单向延迟有限且 ≥0。strict schema 拒绝布尔代替数字、字符串、null、额外字段、空更新和非有限值。非法参数 422，不存在的 Session 404；终态 Session 或全部配置交接均已冻结时 409。后者即使仍在本地总结，也无法影响任何后续通信，故明确拒绝。

GET 返回 `config`、`config_revision`、`next_unfrozen_handoff_index`、`is_simulated=true`、`effective_rule=next_unfrozen_handoff`。终态也可查询。PATCH 返回 `updated=true`、Session ID、before/after、配置版本、接受时间、来源、模式变化、生效边界及 Session revision。

## 2. 生效边界

`HandoffNetwork.snapshot(index)` 首次冻结该序号；已冻结序号永远返回原不可变快照。`update(config)` 原子替换未来配置并增加配置版本。例：Handoff 2 已冻结为1000Mbps，随后 PATCH 50Mbps，返回 `effective_from_handoff_index=3`；Handoff 2继续1000，Handoff 3采用50并重新AUTO决策。

点击时间不是接受时间。服务器若先冻结 Handoff 2，之后才接受 PATCH，即使用户先点击，也只能影响 Handoff 3。不会改历史决策/receipt，不重生成 Sender，不中断已执行通信，不重启 Session，不改变 TOKEN/KV/AUTO策略。

## 3. 并发与生命周期

保留原 `ThreadPoolExecutor(max_workers=1)`。Runtime 构造、初始 Prefill、所有 Decode/交接均在 worker；FastAPI 同步路由由其线程池处理，HTTP事件循环不运行模型。

Service 在 dispatch 前创建活动 `HandoffNetwork`，注入 Discussion/SecretSum Runtime。初始化期间也可 PATCH，不等待模型构造。它使用 Service 同一个 `RLock`：状态检查、读旧配置、替换完整配置、增加版本、记录接受事件为同一线性化操作；snapshot冻结也使用该锁。PATCH 从不获取 Runtime.run 或 StatefulTokenSession 的推理/事务锁。

AUTO成本计算和实际通信不持有 Service 锁。允许更新事件先于某个旧冻结快照的 AUTO_DECISION/HANDOFF_STARTED 出现在流中，生效序号仍准确。每个交接只看到完整的一版带宽+延迟。终态释放 Runtime/Cache及网络对象，保留 metadata；历史仍只保留进程内最近三次，重启不持久化。

## 4. REST / WS 事件

复用原 collector、broadcaster、Session revision 与 REST恢复，不增加另一套状态管理。

`DISCUSSION_NETWORK_UPDATED` metadata 包含：Session ID、Session revision、`network`配置投影及 `network_update`。更新记录包含before/after完整配置、`config_revision`、source、`effective_from_handoff_index`、生效规则、模拟标志、`accepted_at`、mode_before/mode_after。

Session `config` 保持初始冻结配置；`network` 是当前配置投影，`network_updates` 保存所有接受记录。网络更新不覆盖正在生成/交接的 phase 或 active_handoff。HANDOFF_STARTED/COMPLETED/FAILED另携带 `network_state` 更新下一未冻结序号；handoff.network仍是本次实际冻结快照。WS按revision投影、重放去重、缺口GET修复；历史GET包含更新记录。

## 5. Trace / manual override

Trace仍按交接序号推进，hold_last，不按墙钟或物理网速。活动 trace 必须显式 `manual_override=true`，随后切到 manual、清空当前trace，停止其推进。前端需要勾选明确的覆盖许可。

部分字段未提供时，继承旧配置的manual值（trace中的 `bandwidth_mbps` 是初始manual配置值，不是最近轨迹项）。前端覆盖表单明确提交带宽和延迟两个值，避免猜测。原初始轨迹保留在 Session.config 和更新before中；旧交接trace来源/trace_id/revision均可查，界面将其称为“初始配置轨迹”，manual覆盖后不再把旧轨迹标作当前项。

## 6. AUTO 调用链和长消息域

```text
真实 cached Decode → 发布原始发言
→ 同一个 HandoffNetwork.snapshot(sequence)
→ IncrementalSelector（实际内容长度、双方Cache长度、冻结网络）
→ AUTO_DECISION → 唯一选中 handoff/handoff_kv → 实际receipt
```

每次交接重新计算，没有历史成本/决策回填。默认质量优先仍TOKEN。短消息Profile遇到长内容/Cache超域时，质量优先TOKEN保留安全路由，但 `cost_status=profile_out_of_domain`，本地/总预测为null，不外推校准成本；网络费用仍由真实协议长度计算。性能实验超域保持排除/无候选明确失败。错模型身份仍拒绝，不借域外安全规则绕过身份检查。

AUTO KV候选按实际包长、内容Token数及Sender位置排除 `protocol_capacity`，不会先执行不合格KV，也不因执行失败静默fallback。V2数量/比例、V3许可/数字槽位/无数字失败策略未改变。

## 7. 产品预算与冻结实验

| 入口 | 本轮规则 |
|---|---|
| DiscussionConfig / 普通讨论CLI | 默认512；正整数自定义 |
| Dashboard / POST Discussion（含Web秘密场景） | 默认512；接受256、512、1024及自定义小值；移除128硬上限 |
| SecretSumConfig / 冻结秘密验证CLI | 保留96默认；实验CLI原1–128范围保持 |
| 固定历史实验 | 显式64/96/128参数、Prompt、评分、原始结果均未修改 |

无全局替换128。API不使用任意固定新生成上限，Runtime依据真实模型窗口和已有每Agent上下文安全预算计算。本地CLI文本及`--output` JSON均能看到配置/生效预算和限制原因。

## 8. 上下文与资源保护

每Agent上下文预算仍为 `min(config.max_context_tokens, 两端实际模型context_window_tokens)`；产品原默认4096保留。不裁历史、不重建无历史Cache。预算计划按实际tokenizer计入：当前本地stage、assistant prefix/closing、Receiver incoming prefix/closing、双方真实ledger，以及全部后续stage/framing与每阶段至少一个输出Token（按非EOS内容保守计数）。

本次预算为Sender和Receiver剩余空间的最小值（本地总结仅需生成端）。若小于配置，保留原配置并记录：

- `configured_max_new_tokens`：用户配置值。
- `effective_max_new_tokens`：本次安全值。
- `context_remaining_tokens`：扣除当前/后续framing与最低后续输出后，可供本次输出的空间。
- `budget_limit_reason=context_limit`：缩减原因；未缩减为null。

若连最低后续阶段都容不下，在当前Decode/transfer前拒绝，不提交半状态。不保证这些最低预算使语义完整；即使之后只剩一个Token，也如实报告context_limit。

手动KV生成前，按加载模型层数/KV heads/head_dim/dtype和精确TKVK长度估计，连同TKVI metadata上界、header/digest及内容TKVT检查原64MiB、4096内容Tokens和Sender位置限额，超容量明确拒绝。不是所有模型一律1024。0.5B/BF16每内容Token12288B，1024 raw=12MiB；3B的36层/2KV heads/128维/BF16为36864B，1024 raw=36MiB，4096 raw=144MiB不能放进64MiB。相关shape/dtype边界是静态工程测试，不是3B实测。

两端独立Cache和事务work clone有额外内存成本；V2/V3还保留原repair私有工作态，协议限额不等同于总进程内存上限。本轮真实运行复用既有 ResourceMonitor/ResourcePolicy：swap仅warning、持续critical10秒、持续压力+快速swap30秒、压力下forward停滞60秒保护；未改MPS/Swap限额、无CPU fallback、无OOM重试。原生产Dashboard不新增后台watchdog；上下文/协议前置守卫和资源异常分类是生产改动，持续采样watchdog仍属于显式验收入口。未来3B正常产品部署需单独接入运行级资源监测，不能仅由schema可配置推断已完成该验收。

## 9. 终止定义

- `eos`：真实EOS/EOT ID主动结束；优先于任何预算原因，即使生效预算缩减后提前EOS也仍是eos。
- `max_new_tokens`：用尽配置生成预算，最后ID不是EOS；内容可能不完整。
- `context_limit`：用尽缩减的安全上下文预算，或无可用上下文而提前拒绝。前者消息可交付；后者Session failed、无伪造消息。
- `resource_abort`：ResourceStop或OOM导致Session失败。
- `error`：其他运行异常；保留失败阶段/异常链，原事务回滚，不隐式重试。

Session `rounds_complete/completed` 只说明配置工程阶段执行完毕，不说明每条消息EOS，更不代表语义完整或质量合格。

## 10. 截断 TOKEN / KV

非EOS截断在现有协议中安全：TOKEN传全部实际生成IDs，不删最后内容Token；KV内容span同样包含最后内容Token，pending在私有Cache补一次，再按原envelope bytes恢复/修正/追加。`context_limit`加入TKVI终止metadata的接受集合；二进制version和EOS字段语义不变。非EOS的termination_token_id保持null，绝不虚构EOS。

本地chat closing来自模板，属于framing，不追加到原消息生成IDs。不自动补写、不延长Benchmark、不改Sender内容。成功receipt的append_succeeded与generation_termination_reason并存，表示“交付成功，但内容不完整”。V0–V3长截断、pending、span、ledger与后续forward由专项tests覆盖；不兼容/空KV/无数字V3等仍明确失败。

## 11. UI 与视觉约束

新增实时控制只在运行中的既有配置浮层及网络详情，使用原 `fields-two`、`permission-check`、button、hint/error样式。显示接受成功、版本、生效序号、trace→manual变化；已冻结完所有交接则关闭更新。原config JSON仍为冻结初始值。

展示主阅读区：EOS无新技术标签；max_new_tokens/context_limit仅显示简洁异常提示。计数/配置预算/生效预算/上下文/原因在消息生成详情可查。主面板1360、阅读正文850px/18px/32.4px、点阵、展示/调试信息层级、页面固定高度与内部滚动均保持；styles.css无修改。

实际截断阅读（原研发资料，未随公开副本提供：`discussion-live-evidence/truncated-kv-top-1440.png`）、真实接受反馈（原研发资料，未随公开副本提供：`discussion-live-evidence/network-accepted-1440.png`）、实际网络历史（原研发资料，未随公开副本提供：`discussion-live-evidence/network-history-1440.png`）、最终1366控件fixture（原研发资料，未随公开副本提供：`discussion-live-evidence/active-controls-fixture-1366.png`）、尺寸与代理记录（原研发资料，未随公开副本提供：`discussion-live-evidence/browser-checks.json`）。最后一项明确fixture/no model，仅用于活动表单的小高度布局，不冒充真实PATCH。真实接受反馈截图在复用控件class修正前采集，最终控件以1366截图和源码为准。

## 12. 最小真实模型验收

一次实际模型加载，离线固定 Qwen2.5-0.5B-Instruct `7ae557604adf67be50417f59c2c2f167def9a775`，MPS/BF16、单权重、独立A/B Cache。另一次验收服务器初始化因未支持的构造参数在加载前失败，零生成，保留raw；没有重复加载/下载模型。

| 正式React启动会话 | 配置/目的 | 实际结果 |
|---|---|---|
| `fc663750…` | 2轮/512，原默认Prompt、长系统设计问题 | 5条13Token EOS拒答；保留失败目的证据，没有证明长生成或实时PATCH |
| `c0df89c7…` | 固定无害计数长度配置，2轮/512，AUTO trace | 512/47/47/47/47；首条真实达到512，4 TOKEN；浏览器等待首条时选择器超时和模型短后续导致错过修改窗口，未发送PATCH |
| `58b11518…` | 1轮/1024，专门补活动网络及真实1024预算；立即修改 | 3条均1024且max_new_tokens；实际PATCH50Mbps/2ms/revision1在第一条生成期间接受，从下一未冻结#1生效；2个AUTO及receipt均50Mbps/revision1，真实预测wire4106B=receipt |
| `bb69e340…` | 1轮/160（相对512的小自定义预算），合并长KV与截断 | 160 max_new_tokens → KV V1，46 EOS → TOKEN；本地总结160截断；KV内容span[119,279)，私有pending补1，wire1,970,185B |

**4场、16条真实生成、12次交接、4265次forward；最大单条1024。** 后两场只覆盖尚未得到证据的网络窗口/1024与长KV，不重复质量矩阵或成本校准。计数Prompt是固定长度压力配置，不是可靠协作/Prompt质量结论；默认Prompt拒答和后续读取失败原文全部保留。

完整轻量投影、实际IDs/receipt断言和revision事件（原研发资料，未随公开副本提供：`discussion-live-evidence/integration.json`）。原始文件在忽略目录`benchmark-results/discussion-82c-20261015/validated`，导出含raw SHA。每次实际交付检查生成/接收IDs一致、Sender/Receiver span一致、closed ledger及独立Cache；AUTO三场选中预测包长均与receipt一致。真实页面WS代理观察记录包含4次discussion POST及NETWORK_UPDATED/后续交接事件；刷新历史可看到同一更新记录。

连续资源采样（包含加载/空闲与工程检查时段）：MPS current峰约1.03GiB、driver约2.15GiB，系统Swap相对基线最多增加641.38MiB，压力normal/warning；无critical、OOM、保护中止或CPU推理替代。系统Swap增长不全部归因本模型，采样峰不保证捕获瞬时峰。

模型验收后仅补全“全部交接已冻结”拒绝、Sender位置前置守卫、协议header常量及CLI预算文字，并修正新增控件class；有效模型输入、Decode、通信和修正算法未变。专项/全量工程测试覆盖这些守卫，未为确定不触发的边界重复加载/生成。

## 13. 工程与浏览器测试

- 新专项：HTTP成功/非法/null/非有限/缺Session/终态，初始化及forward期间PATCH，冻结后更新、两次连续revision、12次并发整版配置、trace显式override、WS/REST顺序及恢复、全部已冻结拒绝。
- 预算专项：129/256/512/1024、EOS/生成上限/上下文缩减、最低后续framing拒绝、TOKEN实IDs、V0–V3长截断pending/ledger、context_limit KV、OOM/ResourceStop/普通error、模型shape/dtype/64MiB/位置、固定64/96/128及域外AUTO。最后相关组合73 passed；较广API/Runtime组合148 passed（随后追加2个守卫进入全量）。
- **全量Python一次：963 passed / 1原可选AutoGen skipped / 2既有依赖warning，15.45s。** 所有共享状态/API/Cache/协议功能已完成后执行。最后CLI文本改动另做真实scripted上下文缩减输出smoke及compileall，无新模型推理。
- **全量Frontend：180 passed / 8 files。** 最后新增控件class修正后做必要最终前端确认；此前全量结果同样180。
- TypeScript `tsc -b` + Vite build、`compileall src tests *.py`、Discussion/SecretSum/Dashboard/验收CLI help、diff检查通过。
- 最终正式React：1440×900展示/调试主阅读均无整页溢出、850px/18px/32.4px、常驻technicalClutter=0；真实网络历史恢复和截断详情可读，console无error/warn。1366×768明确fixture的活动表单44px应用按钮及footer可达，内容内部滚动；没有重复8.2B全视口矩阵。

## 14. 未实现事项

没有物理网络限速/跨机Transport、暂停/取消、Token streaming、自动续写、语义完成评分、历史持久化、在线成本学习或长消息全面校准。没有新3B模型实验、V0–V3秘密质量矩阵或Prompt搜索。域外profile不产生伪成本；KV仍为实验性质量路径。生产Dashboard持续资源watchdog及3B正常预算真实验收仍需单独推进。

## 15. 8.3 建议与复现

下一阶段先冻结有代表性的长消息/Cache长度与实际模型设备身份，做少量域覆盖校准，继续分别评价传输工程、模型语义质量与性能；勿根据更长KV包自行宣称高带宽收益。3B应先运行独立有界资源/协议兼容验收，再决定产品级监测接入；原秘密质量参照保持冻结。

```bash
# 正式服务与UI（缓存模型，无下载）
HF_HUB_OFFLINE=1 .venv/bin/python run_dashboard.py --discussion --device mps
npm --prefix frontend run dev

# 可选：带原资源监控与IDs审计的验收服务，必须用新目录
HF_HUB_OFFLINE=1 .venv/bin/python tests/discussion_live_validation_server.py \
  --output-dir benchmark-results/discussion-live-new --port 8010
.venv/bin/python tests/dashboard_browser_harness.py \
  --port 5178 --backend-url http://127.0.0.1:8010
```

验收后自动提交本轮源码、tests、必要文档/小型证据与.agent；用户原未跟踪设计/基线副本不提交，原服务和历史结果不覆盖，不push。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
