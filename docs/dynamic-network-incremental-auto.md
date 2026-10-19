# 8.2A 动态网络与真实增量 AUTO

2026-10-14；起始 `main / 32fc0a5eaed59a29d0f5e3dcb5bc13354b68e93b`。

**已经实现每次真实 handoff 前读取网络并重新决策，唯一选中的方案实际进入原 Session。** 默认质量优先仅允许 TOKEN。性能实验中的 TOKEN/V0/V1/V2/V3 均已通过真实 0.5B/MPS/BF16 AUTO 执行；原手动及 mixed 保留，算法没有修改。

**实测 profile 没有支持 TOKEN↔KV 切换。** TOKEN 本地成本更小、包也更小，任何正带宽下都胜出。另用永久标记 `synthetic/test_only` 的 profile，在真实 DiscussionRuntime 中完成 TOKEN→V1→TOKEN→V1；这是路由证据，不是 KV 速度优势。

## 1. Canonical 与增量路径

旧 `RealAdaptiveRuntime / RealCostBasedSelector / LatencyBasedSelector` 保持原行为：Canonical KV 对消息重新 clean Prefill，得到 KV(M)。它的 profile、网络 PATCH 和旧 Dashboard 不控制本轮 Discussion。

新 [incremental_auto.py](../src/token_kv_agent_system/incremental_auto.py) 比较的是真实 Decode 的 KV(M | Sender历史)。V0 原样追加；V1 修正 post-RoPE K；V2 再重算尾段；V3 再修复数字槽位。调用原 [StatefulTokenSession](../src/token_kv_agent_system/stateful_token.py)，没有调用 `perform_kv_handoff`，没有增加算法。

## 2. 网络状态与重放

[HandoffNetwork](../src/token_kv_agent_system/network/handoff_network.py) 保存不可变快照：handoff_index、Mbps、单向基础延迟、network_mode、trace_id、source、config_revision、simulation_step、trace_exhausted、is_simulated、advancement、latency_semantics。

以序号推进，**不是墙钟轨迹**。`simulation_step` 是当前配置的交接步数。trace_id 是配置内容摘要；相同配置和相同序号得到完全相同的快照，不依赖真实执行速度。默认 `hold_last`：20,1000,50,500 的六次交接得到 20,1000,50,500,500,500。

manual 使用会话级 bandwidth/base latency。Python 状态源 `update(config)` 在锁内替换配置，影响下一未冻结序号；manual→trace 从新轨迹第一项开始。既有序号永远返回同一快照。必须按连续序号申请，新序号不能跳跃。

带宽必须有限且大于零；延迟有限且非负。拒绝 bool、NaN、Infinity、负值、零带宽、空 trace 和手动携带 trace。浮点极端输入导致传输预测溢出时明确报错，不产生 Infinity 事件。

## 3. 更新时间、异常与基础延迟

Sender 完成真实生成和文本展示后，在尝试 handoff 前冻结快照，再预测、选择、执行。异常不会再推进网络；相同序号的显式重试复用快照。DiscussionRuntime 本身失败即终态，不隐式重试，不重新生成同一消息掩盖失败。

`simulated_transfer_ms = wire_bytes × 8 / (Mbps × 1000) + base_latency_ms`。base latency 是一次单向交接延迟，不是 RTT；每条实际 data-plane envelope 只收费一次。实际 Transport 仍是 InMemoryByteTransport，没有 sleep、物理限速或跨机网络。

8.2A REST 仅支持冻结会话配置/trace，没有新增实时网络 PATCH。旧 Canonical PATCH 仍属于旧 Runtime。单 worker 的活动 Session 不被外部异步修改；实时控制及更新事件留后续。状态源 update 的冻结/一致性由单元测试覆盖，不宣称已有 Dashboard 实时控制。

## 4. 统一成本与比较边界

候选 `estimated_total = predicted_local + simulated_transfer`。预测阶段不调用 handoff，不提取 KV，不进行任何候选 forward。只读取本次生成的 IDs、内容长度、Sender/Receiver Cache 长度和模型结构/身份；KV wire 只算一次，所有 KV 候选共用。

各 `StageCost` 支持 fixed + per_message_token×N + per_cache_token×(SenderCache+ReceiverCache) + per_recomputed_token×k。系数必须有限非负；候选总本地预测不得缺失或默认为 0。

|候选|预测本地阶段与对应真实工作|
|---|---|
|TOKEN|token_handoff：TKVT 编解码、内存传输、双端工作 Cache 克隆/关闭、Receiver 新消息及本地 framing 的实际增量 Prefill、验证/提交|
|V0|kv_base：pending补算及私有Cache、增量提取/设备到主机、encode、内存传输、decode、restore/设备上传、Receiver clone/prefix/append/closing、Sender close、验证/提交|
|V1|kv_base + rope：真实整消息 K 位置修正及相关处理；V不旋转|
|V2|kv_base + rope + suffix：按实际 k/ceil(N×ratio)/饱和裁剪预测尾段 forward；k=0 无内容重算|
|V3|kv_base + rope + selection + slot：逐Token数字识别、prefix crop、真实全部层单槽位 forward、scatter/Cache复制/验证|
|首次KV|若Session尚无合约，再加 kv_contract_startup：两端模型合约/权重摘要初始化；共享权重只检查一次。没有启动样本时排除KV，不能借暖机profile省掉此成本|

共用的角色初始化、公共问题、Sender Stage/Decode 发生在选择前，故从候选比较消去；实际 forward 日志保留费用。Receiver下一阶段/后续Decode在执行后使用真正的新Cache，并保留日志；未来回答长度不用于当前预测，不把不同回答长度当通信加速。Control Plane JSON/WS 展示开销、无诊断生产吞吐及完整会话网络端到端时间未校准。

## 5. 真实 wire 预测

TOKEN 使用当前 TKVT v1：10 B header + 4 B×原始生成Token数量，含终止EOS。不是每Token一字节。

KV 使用当前 TKVI：17 B envelope header + 32 B digest + 实际canonical JSON metadata长度 + 内容TKVT(10+4×N) + `estimate_kv_wire_size` 的完整 TKVK。dense batch1 的 raw tensor bytes 为 `2×layers×KV_heads×N×head_dim×dtype_bytes`。0.5B/BF16 为每内容Token 12288 B，另加真实协议元数据。

固定64字符摘要的占位长度与真摘要长度完全一致，不读取/复制权重估算wire。span adapter 的 neutral `source_device='cpu'` 是协议表示位置，不是CPU模型fallback。所有实际 restore/修正/forward 使用 MPS。

V0/V1/V2/V3 同Sender包的主要data-plane bytes完全相同。预测未选候选的包长只标 `predicted_*`。20次主验收+4次最终smoke的选中路径预测包长误差均为0；校准同Sender四KV版本也同包长。初次tiny测试暴露遗漏digest32 B，修正后才执行真实模型。

## 6. Profile、校准与不确定性

Profile 强制 `semantics=decode_incremental_handoff_v1`；拒绝 Canonical schema。保存 version、provenance、model_identity、stage coefficients、sample_count、消息/Cache/recompute域与notes。身份包含两端model路径、固定revision、tokenizer/backend/config摘要、dtype、device。错模型/设备/配置禁止复用。

独立校准：30条逐阶段样本（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/calibration.csv`）。两条真实Sender消息，内容2/25Token、两端Cache总计221/311Token，Receiver长度97/126；每策略每长度1次warmup+2次采样，20个暖机样本。使用同一已生成Sender/Receiver快照分别测候选，明确属于离线校准，**不是在线Selector先执行全部候选**；候选接收后不为选择获取答案。只有TOKEN基线继续生成第二条校准消息。

本机使用域内 pooled median，样本稀少，长度与Cache相关，未做可靠多元拟合/置信区间；不外推长历史。V2额外forward只校准k=1，recompute域[0,1]；大k需要新样本或明确synthetic模型。真实阶段采样含ResourceMonitor及同步诊断费用；冷加载、角色Prefill、Sender生成单列，不与暖机本地成本混合。

TOKEN基线43.4633ms，V0基础77.6907ms，RoPE53.0854ms，V2每重算Token15.5912ms，V3selection0.0170ms、slot17.1417ms。首次合约初始化另测976.4577ms（一个启动样本，不是稳态统计）。可直接给CLI的profile（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/measured-profile-with-startup.json`）保留它；已有合约的handoff不再收取。

profiles.json（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/profiles.json`）同时保留原暖机实测profile与原synthetic profile。synthetic TOKEN1000ms、KVbase1ms、RoPE1ms、suffix每Token1ms、selection/slot各1ms；最后smoke额外显式人工startup0ms。全部永久标记 `synthetic/test_only`，只能检验选择/路由，不宣称速度。配置/阶段有差异时以记录的完整profile与source SHA为准。

## 7. 质量优先与性能实验

`quality_first` 默认：TOKEN是当前任务可靠性参照，不是普遍正确保证。未经认证的V0/V1/V2排除 `quality_not_certified`；显式列入V3也排除 `known_quality_regression`。不根据V版本号推断质量。此版没有开放新的KV质量认证入口。

无profile时可选择唯一安全TOKEN，cost_status=uncalibrated、predicted_local/total=null、profile_provenance=not_measured；这是安全路由，不是将TOKEN计算成本当0。预测网络费用仍可计算。给了错profile不静默忽略，而按过滤规则失败/选择剩余合格候选。

`performance_experiment`：显式启用，默认白名单TOKEN/V0/V1/V2；V3需白名单和allow_experimental_v3=true，否则配置拒绝。保存quality_risk，选V3额外v3_semantic_risk。性能模式不改变人工质量标准、不反向改写选择、不伪造回复。

## 8. 过滤、错误与tie-break

候选过滤原因包含 not_whitelisted、quality_not_certified、known_quality_regression、model_profile_mismatch、profile_out_of_domain、profile_recompute_out_of_domain、missing_cost_profile、missing_startup_cost_profile、unsupported_empty_kv、unsupported_strategy。没有合法候选发出 AUTO_DECISION_FAILED，保存原因/冻结网络，讨论终止，Transport不执行。

可比较候选按预测总成本升序；相同成本固定 TOKEN→V0→V1→V2→V3，与白名单顺序无关。保存次优候选及差值。V3无数字仍沿用现有执行失败约束，不改为TOKEN。协议/合同/forward/ledger错误保持双端事务回滚，HANDOFF_FAILED与actual_execution_status=failed可见。

## 9. 正式调用链与手动兼容

```text
DiscussionRuntime / SecretSumRuntime
  Sender generate_message（真实cached Decode）
  → network.snapshot(handoff_index)
  → IncrementalSelector（质量过滤→预测→唯一选择）
  → AUTO_DECISION（selected，尚未交付）
  → HANDOFF_STARTED
  → TOKEN: StatefulTokenSession.handoff
  → KV: StatefulTokenSession.handoff_kv(strategy, k/ratio)
       prepare_incremental_message → TKVI/TKVK bytes → InMemoryByteTransport
       receive_incremental_kv → restore/repair/append → closing forward
  → 双端验证及原子commit → receipt
  → HANDOFF_COMPLETED / HANDOFF_FAILED
  → Receiver从真实Cache继续Stage/Decode
```

全局AUTO与手动KV/mixed plan冲突明确拒绝。原手动TOKEN、V0/V1/V2/V3、mixed计划、V2数量/比例、V3无数字失败均保留；HandoffChoice仍表示实际执行方案，不把AUTO伪装成一种wire协议。

## 10. CLI与REST

```bash
# 安全默认；无profile不捏造本地成本
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_demo.py --mode auto \
  --rounds 3 --network-mode trace --bandwidth-trace 20,1000,50,500 \
  --base-latency-ms 2 --output benchmark-results/auto-new.json

# 性能实验必须提供与实际backend identity相同的增量profile
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_demo.py --mode auto \
  --auto-policy performance_experiment --auto-candidates token,kv:v0,kv:v1,kv:v2 \
  --cost-profile /path/to/compatible-incremental-profile.json \
  --network-mode manual --bandwidth-mbps 1000

# V3另需 --auto-candidates kv:v3 --allow-experimental-v3
# 校准验收严格使用snapshot路径；CLI按repo名加载时identity可能不同，明确拒绝错用。
```

`POST /api/discussions` 在既有strict schema增加：mode=auto、auto_policy、auto_candidates、allow_experimental_v3、cost_profile（内联JSON，无任意服务器文件读取）、network_mode、bandwidth_mbps、base_latency_ms、bandwidth_trace；保留kv_recompute_tokens/ratio和secret_sum场景。

示例：`{"question":"23 + 48 = ?","mode":"auto","rounds":3,"network_mode":"trace","bandwidth_trace":[20,1000,50,500],"base_latency_ms":2}`。额外字段/无效数值/混合冲突在启动模型工作前拒绝。手动模式不能携带性能AUTO配置。

## 11. Event/REST/WS数据契约

原Service单worker、最近三Session、revision恢复机制保持。新增AUTO_DECISION/AUTO_DECISION_FAILED；REST session.auto_decisions保留选择前记录，handoffs保存执行后记录。新的事件在前端投影中推进revision，避免被当丢包。

|字段位置|含义|
|---|---|
|handoff_index/source/target|序号与方向；WS兼容sequence/wire_bytes|
|network.*|冻结Mbps、单向延迟、轨迹/配置来源、模拟标志、步数、版本|
|requested_mode|auto、token或kv；AUTO不证明已执行|
|selected_mode/selected_kv_strategy|选择TOKEN/KV及算法；仍不证明交付|
|executed_mode/executed_kv_strategy/append_succeeded|由真实receipt确认；失败为null/false|
|auto_decision.*|policy、候选预测、排除、reason、次优/差值、profile版本/来源/notes、risk|
|predicted_wire_bytes/local_ms/total_ms|选择前模型预测；未校准成本null|
|observed_local_ms/observed_stage_ms/observation_scope|此次执行本地wall/同步阶段；不回填历史样本|
|simulated_transfer_ms|用此次actual_wire_bytes和冻结网络推导；不是物理传输实测|
|actual_wire_bytes/actual_execution_status|实际byte receipt/尝试状态；失败已有发送量仍可见|
|fallback_reason/error|无silent fallback；失败原因保留|

未执行候选没有actual elapsed或actual packet字段。AUTO_DECISION的状态selected，handoff.auto_decision的状态executed/execution_failed。决策本身失败通过专用事件和Session终态呈现，不伪造一个成功receipt。

## 12. 真实验收与带宽结果

模型：Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`；单份权重、A/B独立Cache、MPS/BF16、offline/cache-only、greedy。冻结生产Discussion/SecretSum Prompt，没有搜索、下载或CPU替代。

冻结配置（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/frozen-spec.json`）、逐次候选决策（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/decisions.json`）、实际成本（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/handoffs.csv`）、完整模型原文/IDs（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/outputs.json`）、审计（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/audit.json`）。原始结果保存在忽略目录 `benchmark-results/dynamic-auto-82a/`，不覆盖历史benchmark；原始SHA索引（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/raw-index.csv`）包含冻结源码副本。

|正式运行|带宽Mbps|实际执行|
|---|---|---|
|quality-discussion，3轮|20,1000,50,500,500,500|6次TOKEN；每次独立决策|
|measured-secret，1轮|20,1000|TOKEN,TOKEN；真正实测profile|
|synthetic-dynamic，2轮|0.001,1000,0.001,1000|TOKEN,V1,TOKEN,V1；实际字节54/765558/74/863894|
|synthetic V0/V1/V2/V3，各1轮|各20,1000|每策略2次真实KV，共8次|
|最终源码startup-smoke，2轮|0.001,1000,0.001,1000|TOKEN,V1,TOKEN,V1；首KV收费startup、后续不重复|

主验收29条真实生成（含2条独立校准生成）、20次AUTO交接（10TOKEN/10KV）、937次forward；另30次离线校准handoff+2次TOKEN基线推进。启动补测5条生成、4次AUTO（2TOKEN/2KV）、236次forward。**本轮合计34条真实生成、24次正式AUTO交接、1173次forward**；不是把离线校准当在线选择。

## 13. 实际增量执行与质量分开审计

tiny回归中，唯一选中的Session方法被跟踪，禁止初始化后clean Prefill；五种方案均从独立Cache继续生成。真实MPS审计匹配KV Receiver closing与之后已有Cache的forward；V0/V1内容Prefill=0，V2/V3实际1Token×24层重算。预测/receipt模式、wire、cache ledger一致；没有旧Canonical替代、CPU推理fallback或silent fallback。

measured-secret最终：`A's number is 17. B's number is 26. The sum of their numbers is 43.`，本条严格私有交换通过，不能推广为所有TOKEN任务合格。

V0最终`26 + 17 = 43`，缺明确来源/双方归属，不算严格交换成功；原scorer的task_success只反映算式，本轮另存strict_private_exchange_pass。V1/V2最终`17 + 17 = 34`；V3最终`Agent A's number is 17. Agent B's number is unknown. 17 + unknown = 17`。均不合格。保留原scorer和历史秘密结论，不因数值/某一次输出放宽质量门槛。

## 14. 预测误差与资源

主验收所有20条选中包预测误差0B，最终smoke4条也0B。非选中KV bytes仍为估算，校准同Sender包长相同仅作协议交叉核验。

实测profile的两次TOKEN local预测均43.4633ms，实际47.5799/48.4622ms，误差+4.1165/+4.9988ms（约+9.47%/+11.50%）。这只是两条instrumented样本，不是普遍误差保证。synthetic本地预测1–1000ms与真实83–162ms等不符是人工配置预期；CSV完整保存差异，绝不能解释成真实收益。

20Mbps首条真实profile预测TOKEN43.4721ms、V0 88.9051ms；1000Mbps反向TOKEN43.4642ms、V0 80.1769ms。高带宽减小KV网络惩罚，却无法弥补V0较高的本地处理；V1/V2额外费用进一步增大。首次KV合约计费也不会创造收益。

主验收采样峰MPS current约0.955GiB、driver约1.049GiB；压力normal/warning，Swap新增0MiB，无critical/OOM/保护中止。最终smoke也无资源中止。采样每2秒，峰值不保证捕获瞬时最大；资源记录/forward原始日志保持。真实物理网络时延、跨机吞吐、无诊断steady-state性能：not_measured。

## 15. 最终源码与失败记录

主验收后审计补上首次KV的合约成本，变更限于incremental_auto、Discussion传入合约是否已有、独立验收入口；原Session/KV算法与bytes未变。保存主验收完整冻结源码副本及SHA，原结果不改写。最终源码再做4handoff/5生成的startup-smoke，冻结SHA完全一致。startup-smoke（原研发资料，未随公开副本提供：`dynamic-auto-82a/evidence/startup-smoke.json`）保留最终配置、决策、receipt投影及输出。审计显式列出三文件漂移，没有将主验收冒充零漂移的最终代码执行。

工程过程：wire估计初测少32B已修复；旧完整Discussion对象相等测试因新增wall计时产生差异，改为只移除observed_local_ms再比较协议/输出/状态，其余断言保持。没有模型运行失败，也没有据模型回答改成本或Prompt。启动费用缺失时现在明确排除KV；synthetic startup=0是显式人工实验输入。

## 16. 分层测试与局限

L1覆盖manual/trace重放、六方向、hold_last、非法数值、序号/重试、模式切换和并发冻结读；L2/L3手算低/高带宽、相等成本、TKVT/KV格式、大消息、缺profile/错身份、Cache/消息系数、V2参数域、V3排除及cold/warm费用；这些synthetic单测不是模型性能证据。

L4覆盖五条真实tiny Session路径、bytes/合同、独立Cache/事务失败、无fallback、多轮/mixed、CLI正负配置、REST schema/profile/V3风险、真实WS revision/Event/REST一致；L5为上面的固定MPS双Agent运行。

最终工程结果见下方验收记录。本机profile只有两个短消息长度和小Cache；未认证KV质量、未做3B/服务器/跨机/实时限速/新算法。V3没有数字时仍失败。暂无在线成本自学习、质量认证管理或控制面网络费用模型；后续不可把本机profile外推服务器。

## 17. Dashboard与8.2B

本轮只启用质量优先AUTO选项、AUTO事件revision兼容、请求/选择/真实执行及预测/模拟/实测详情标签；布局不变。页面新建AUTO默认20Mbps，性能实验/轨迹用CLI/REST。恢复的性能Session可查看真实receipt及完整metadata；页面下一次创建的AUTO仍按安全默认发起。

8.2B建议在现有配置浮层增加manual/trace、白名单/profile来源和性能风险提示；详情用handoff_index对齐网络/候选预测/排除/receipt结果，区分selected未交付与executed，未执行候选保持estimated。可加候选成本随带宽的图，但不要求制造交叉点。实时PATCH应作为独立并发语义任务，先定义生效序号/更新事件，再复用这一状态源。

## 18. 本轮最终工程验收

最终Python **916 passed / 1既有可选AutoGen skip / 2既有上游warning**；新增网络/Selector/Runtime测试 **59 passed**。Frontend **121 passed / 7 files**，TypeScript与Vite build通过。compileall（src/tests/全部根Python入口）、Discussion/Dynamic验收/export/Dashboard/SecretSum CLI help、git diff --check通过。原TOKEN、V0–V3、mixed、事务/fallback、CLI/REST/WS及旧Canonical回归包含在全量pytest内。

验收17项已完成，没有关键工程阻塞。按用户固定授权自动task-scope commit，不push；hash以Git历史为准。用户未跟踪设计文件、基线副本、历史benchmark均保留，不暂存；Git仅纳入本轮代码/测试/报告、小型证据和必要.agent状态。

复现必须使用新目录，原benchmark不覆盖：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python run_dynamic_auto_validation.py --output-dir benchmark-results/dynamic-auto-new
HF_HUB_OFFLINE=1 .venv/bin/python run_dynamic_auto_validation.py --phase startup-smoke --output-dir benchmark-results/dynamic-auto-new/startup-smoke
.venv/bin/python export_dynamic_auto_validation.py --results-dir benchmark-results/dynamic-auto-new --output-dir /tmp/dynamic-auto-new-evidence
```

新的full运行会直接记录最终startup stage；本轮首次full的暖机结果和其冻结源码保持原样，后续startup-smoke单独存储。所有预测/模拟/实测及synthetic来源标签均进入可核查输出。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
