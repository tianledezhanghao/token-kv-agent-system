# 8.5 — 泛化 AUTO 成本模型与理想化切换

2026-10-16（Asia/Shanghai），起始 HEAD `65e1c5b3bde209908720d4d4a31045b7bb235fa0`。四项功能已实现；本地证据来自离线 Qwen2.5-0.5B / MPS / BF16。CUDA 入口完成工程验证，**没有访问服务器或执行 CUDA 实测**。质量优先仍不放行未经认证的 KV。

## 配置与兼容

AUTO 固定比较 `['token', 'kv:vX']`，其中 X 由用户选择。Python `IncrementalAutoConfig`、REST 和 CLI 均拒绝多个 KV 或缺少 TOKEN 的候选；前端改为一个 KV 下拉，不再使用多选白名单。枚举中的其他版本仅可出现在排除记录，不能参与排名或执行。历史 Session 的原配置仍可阅读；从旧五候选历史恢复新草稿时取第一个 KV，新请求仍必须满足双候选约束。

REST 新字段：`auto_kv_version: v0|v1|v2|v3`、`idealized_simulation: boolean`（默认 false）。保留 `auto_candidates` 作为严格双候选兼容输入，显式 KV 版本与其冲突时拒绝。V2 默认落成显式 `kv_recompute_tokens=1`，也支持原数量/比例配置。V3 实验需额外许可；无合法 ASCII 数字内容槽位时，在执行前明确排除 `v3_no_numeric_slot`，TOKEN 可按明确的合法候选规则参与决策。这不是选中 V3 失败后换成其他 KV。手动 V3 及底层异常、事务回滚保持原样。

三种使用方式：

|方式|成本来源|质量许可|
|---|---|---|
|quality_first（默认）|可无 Profile；有域内校准才显示本地预测|当前只允许 TOKEN；带宽不能撤销质量限制|
|performance_experiment|独立泛化实测 Profile，或原精确回放 Profile|显式允许指定 KV，有语义风险；性能不代表语义等价|
|理想化场景仿真|独立 `synthetic` 公式，不接受同时带入 Profile|显式性能实验；V3 仍需许可及合法槽位|

## 旧精确 Profile 保留

`IncrementalCostProfile / decode_incremental_handoff_v1` 及其研究数据没有改写。8.3C 的单端历史、源位置、termination、pending、ledger、消息 IDs、framing SHA 仍用于精确 replay；Runtime 继续强制旧实测 Profile 的 workload domain。旧 metadata-only 离线分析默认语义保持。旧审计脚本的变更范围比较现固定为8.3C正式提交 `dc084d1`，避免后续已授权的模型菜单/本轮改动被误判为历史研究漂移；原始数据、AST与SHA检查保持。没有通过删除 SHA 守卫把旧实验包装成泛化证据。

旧路由研究的 `synthetic/test_only` 兼容输入仍明确标为测试假设，绝不升级成 measured；新教学开关使用独立成本源。

新入口 `load_cost_profile` 以独立 `schema: generalized_incremental_cost_v1` 分派。旧类不接受新 schema；新类不接受 synthetic 或不支持的 schema/单位。Canonical Profile 不能进入增量预测器。

## 新 Profile 结构与身份

见 [实际 Profile](generalized-auto-85/evidence/measured-profile.json)。包含：

- schema、version、`provenance=measured`、`units=ms`、校准时间（带时区）、说明、`reliability=experimental`。
- Sender/Receiver model identity：实际本地模型路径、revision、config SHA、Tokenizer/结构身份摘要、dtype、device；复用既有 `backend_cost_identity`，不每次扫描权重。
- 各端 environment：MPS 的实际 SoC/机器/OS；CUDA 的名称、capability、显存容量、CUDA runtime；Torch/Transformers 版本及实际算法/协议/adapter 源码摘要。设备编号也在模型身份内。
- points：候选、工作负载特征、完整本地时间 median、波动、有效样本数、实际 wire bytes。
- `sample_count` 必须等于所有点有效样本之和，points 的 canonical JSON SHA 必须等于 `evidence_sha256`。
- 独立 `startup_cost_ms` / `startup_uncertainty_ms`：本轮真实模型合约初始化 n=1，未伪装稳态多样本。

源码身份覆盖 Session、Agent state/framing、TOKEN/TKVI/TKVK 协议、导出/接收、adapter、修正及 backend。它防止不同实现静默复用，不是对任意用户上传 JSON 的真实性签名。Profile 的 measured 声明须由保留样本与 SHA 证据支持。

## 工作负载与域

三个连续坐标：内容 Token 数 **N**、Receiver 已缓存历史 **H**、Sender 消息起点即生成前上下文 **S**。N 与 S 共同确定 Sender 内容 Cache 的规模，N/H 联合表达增量 Prefill 的注意力工作量。

分类条件逐项精确相等，不在条件之间插值：EOS/max_new_tokens/context_limit、两端 pending、两端关闭 draft 所需实际输入长度、incoming prefix/closing 长度、真实 adapter 分派、合约 ready 状态、V2 实际 k、V3 数字槽位索引。V3 实际层数受模型身份限制；当前算法固定全部层 × 一个合法槽位。

读取这些特征不导出 KV、不执行候选 forward。具体消息 IDs 和双方 ledger SHA 不是新 Profile 的域坐标；新内容只要落在同条件、同身份、有完整支持单元的长度域内，就能预测。历史内容改变导致的数值差异由实验边界与误差包络承担，不能据此认证所有文本分布。

## 有界联合插值

对指定候选与分类条件找到 N/H/S 各自的上下邻点，要求它们的笛卡尔积角点全部有真实校准点。用非负且和为 1 的多线性权重对完整本地 median 插值。完整三维单元最多八个点，可表达 N/H 的相互作用；长度轴可以只有一个值，此时只支持该精确坐标。

缺角点 `profile_sparse_cell`、超范围 `profile_out_of_domain`、未校准条件 `profile_condition_unverified`；不对稀疏网格做边界盒猜测、不外推、不假设时间严格线性或单调，不产生负时间。一个 Profile 可以包含多个条件组；当前 CLI 每次校准一个 termination/Receiver pending 条件。V2 比例/饱和 k 和 V3 不同槽位是独立条件，未覆盖时明确拒绝，不能假称全 k/全槽位泛化。

每个预测给出来源、方法（校准点或有界联合插值）、训练数量、支持点数量、条件、可靠性与经验不确定性。新 Profile 不改变语义质量许可。

## 不确定性和在线选择

每点 spread 为所有有效样本相对 median 的最大偏差，不删除慢样本。插值包络使用加权 spread 与 `relative_error_floor × prediction` 的较大者，再按单元内部距离扩大。本轮预设 floor=25%；若独立 holdout 超出，工具只会上调并保留原始误差，不用调整后的包络重新宣称独立验证成功。

这是保守的**经验包络，不是统计置信区间**。KV 总成本较低但与 TOKEN 的两个误差包络重叠时保守选 TOKEN。完全相同成本也选 TOKEN。性能模式缺少TOKEN基准预测时，即使KV有预测也明确拒绝；无可用预测同样拒绝；无 Profile 的旧兼容 API 可保留唯一安全 TOKEN，并明确本地成本未知。quality_first 域外可保留 TOKEN，但不能给出域外本地/综合成本；身份不符仍拒绝。

在线顺序仍是：真实 Sender 生成 → 冻结网络快照 → 许可和适用性 → 两路本地预测 → 网络成本 → 比较及保守处理 → **只执行选中一路** → 原事务与 receipt → REST/WS。未执行路径从不产生“实际时间”。

## 计时边界与启动成本

直接复用 `HandoffLog.observed_local_ms`：Runtime 在 Session 调用前启动原有 wall timer，在返回/异常后构造交接记录时读取；包括完整 Session 调用、锁、真实通信、必要 pending、Cache 操作、修正、双端验证和提交，以及很少量返回后元数据处理。未新增同步、forward 或生产计时包装。

旧 receipt 内部 `handoff_total` 边界较短：TOKEN 未包括最终提交，KV 还未包括末尾 record/验证；因此消息标签采用现有外部完整 wall，不能把内部总计当完整 wall。分段字段仅作诊断，不把父段和子段重复求和。

校准使用相同的 `measure_timing=True` Session 通路，在调用前后用外部 wall 包围整个返回；包括既有分段同步和资源 forward 记录，称 `runtime_instrumented_full_session_call_v1`，不冒充零仪器吞吐。Runtime 返回后元数据开销与离线校准略有差异，经验包络没有将其假装成零。Sender 回答 Decode、角色/历史 setup、基础快照重置、SHA 审计和后续回答生成均不计入本地交接成本。

校准点的完整 ready handoff 已包含所有本地实际工作；V1/V2/V3 的修正包含在该策略实际总计，不再叠加旧阶段拟合值。模型合约 hashing 在 channel setup 单独测量；当新 Session 冷 KV 需要它时，预测加一次该独立成本和保守的不确定性；TOKEN 不承担不存在的 hashing。合约 ready 后不再收费。没有把生成 Decode 当成某条通信路径的独有成本。

## Wire 与网络

沿用经审计公式：`T_network = wire_bytes × 8 / (Mbps × 1000) + base_latency_ms`。一次有向 handoff、一个完整 envelope，基础单向延迟只加一次，非 RTT。网络没有 sleep、物理限速或跨机流量。

TOKEN 精确为 `10 + 4 × raw_generated_token_count` B（含实际终止 ID）。KV 使用真实模型层数/KV heads/head_dim/dtype、真实内容 N 及 TKVI/TKVT/TKVK metadata 长度精算，EOS 不进入内容 Tensor；不假设跨模型每 Token KV 大小相同。在线不做 Tensor 导出，校准和真实 Discussion 逐条核查预测 bytes 等于实际 receipt。

完整本地 wall 包含原 InMemoryTransport 的微小本地调用，网络项是另外明确标记的假想链路代价；不是把本地 transport 当成第二次物理网络测量。没有把 Host/设备搬运的组合时间叫纯 DMA。

## 校准 CLI 与资源

[入口](../run_generalized_calibration.py)；本地完整命令记录于 evidence/spec.json。参数支持本地模型/缓存、显式设备/dtype、KV 版本、N/H/S 列表、EOS或预算 pending、Receiver pending、V2 k/比例、warmup/有效数/holdout 数、输出、handoff/时间/权重/设备内存预算。输出目录必须全新，已有目录拒绝覆盖。

只加载一份真实预训练权重，A/B 独立 Cache。真实生成后调用既有 `real_prefix` 保留真实 Decode K/V 的合法前缀与最后 pending；历史 filler 仅用于建立受控背景历史，不对消息 IDs 重新 Prefill 伪造 Sender KV。每一路从同一原始双端状态开始，基础 Cache 前后 SHA 不变、实际 received IDs、bytes、提交状态核查。warmup、慢样本与失败均保留。

进程先检查指定本地模型服务是否 active/loading，再用独占校准锁防止两个校准 CLI 并行；其他端口服务须通过 `--service-url` 指定，不能声称能检测系统中任意第三方模型进程。权重仅本地完整 snapshot，分片索引缺失即拒绝，不下载、不 CPU 回退。Mac 保留既有 ResourceMonitor；Linux 记录进程 RSS/swap 和 CUDA 分配/保留量，不冒称采集了 macOS 内存压力。内存预算在工作边界检查，时间 watchdog 可终止自己的进程；没有修改系统限制。

`--verify` 只读核验 SHA、样本数、median/spread、holdout 误差、决策实测最优、消息 SHA 独立性和冻结源码。大日志仍在忽略目录。`export_generalized_evidence.py` 导出小型证据与独立统计，拒绝覆盖。

## 本地真实结果

单次加载固定 revision `7ae557604adf67be50417f59c2c2f167def9a775` 的 Qwen2.5-0.5B、MPS/BF16。N={4,12}、H={64,256}、S={95,256}，V0、预算 pending、Receiver closed、原自动 packed 通路。

8 个联合组合 × 2 路径 ×（1 warmup + 2 有效）=48 次交接，其中 **32 有效训练样本、16 warmup**。另一条独立内容（从 31 开始的数字列表，而训练从 1 开始）在两个 S 分别真实生成；holdout N={8,12}、H=160，共 **4 对工作负载、16 有效交接样本**。完整源生成共4次、两种内容，重复计时不是独立任务样本。

|路径|holdout 预测数|平均绝对误差|平均相对误差|最大相对误差|
|---|---:|---:|---:|---:|
|TOKEN|4|1.295 ms|3.069%|6.679%|
|KV V0|4|2.249 ms|2.931%|8.147%|

八个路径预测均落在原预设25%经验包络内，无失败样本、无删样本、未上调 floor。20/10000 Mbps 两个条件共8次决策与实际配对最优一致，全部 TOKEN；8个训练组合均无正带宽交叉点。独立两轮数值不能认证所有域内消息的误差上界，故仍为 experimental，尤其未验证长文本、自然复杂历史、EOS组、V1–V3实测拟合或 CUDA。

单独合约初始化 **962.514 ms，n=1**。后续两个 Discussion 共8次生成、6次成功交接；连同校准共 **70 次成功交接、12次完整生成、364次受保护 forward**。CLI 除240秒临时浏览器窗口外，加载、校准、讨论及工具工作共约13.52秒；整个 CLI 253.52秒。资源124次采样 pressure=1、swap增长0；采样 MPS allocated 最大约0.953 GiB、driver约2.041 GiB，不称瞬时峰值。没有1.5B/3B性能矩阵或追加样本。

## 理想化场景的数学假设

开关默认关闭，只有显式性能实验可以启用；Profile 必须为 null。假设优化计算环境的完整本地代价：

`TOKEN_ms = 20 + 2N + 0.002NH`

`KV_ms = 5 + 0.05N + 0.2k`

N为内容数，H为Receiver历史；k为V2实际重算数，V3为1，V0/V1为0。单位ms，固定项、随消息的计算复用收益、N/H注意力项和修正代价均是教学参数，不是本机拟合。各KV算法本身不变，公式只是共同的假设环境，不认证其实际计算收益。

在合法消息、KV包更大的条件下，正交叉点自然来自 `B* = 8(KV_bytes − TOKEN_bytes) / [1000(TOKEN_local − KV_local)]`；没有 `if bandwidth > threshold`。本次首消息 N8/H86、42B vs101796B，对应假设约25.46Mbps交叉；常见20与1000Mbps也可用于教学。固定延迟两路相同故在差值中抵消。

真实动态烟测为10000Mbps开始，首条实际V0；HTTP PATCH为0.001Mbps（明确的演示低带宽），后续三条实际TOKEN。第一条实际本地wall约1021.95ms，包含冷合约 hashing，远高于假设5.4ms；这个差异保留，恰好说明本模式不证明实际KV更快。

## REST/WS 与真实讨论证据

实测模式 Session（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/measured-discussion.json`） `17c71e52967e43088c0c80590ee35586`：一轮、每条预算8，两个新消息均域内预测并实际TOKEN，完整总结和Cache释放。实际本地43.77/45.12ms，预测39.94/40.52ms；没有为了切换强迫KV。

理想化 Session（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/idealized-discussion.json`） `401ea9bf0b854b03b5ac8e5eaba531eb`：两轮、预算8，实际 V0→TOKEN→TOKEN→TOKEN，Receiver继续真实Decode，最终总结与Cache释放成功。所有8预算内容可能未完成，工程完成不代表算术/交流语义正确。

活动 PATCH（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/dynamic-patch.json`） 从下一未冻结#2生效，配置版本0→1；首#1保持10000Mbps/rev0，后三条0.001Mbps/rev1。patch由真正WS接收线程看到首receipt后发送HTTP，不在事件锁内注入状态。完整 measured WS（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/measured-ws.json`） / synthetic WS（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/idealized-ws.json`） 与最终REST receipt逐条相等，wire预测=实际，没有执行后改决策或伪造未选路径实测。

## Dashboard 与数据来源

消息按钮仅在“实验详情→调试”后显示 `B 已接收 KV V0 · 本地交接 ... ms`，由 message_id 与对应发送方/目标关联，缺历史字段为“不可用”，不填0。展示模式不显示新增计时。

详情显示指定两候选本地/模拟网络/综合成本、经验不确定性、排除原因、选择原因、版本/校准说明、实际版本与wire bytes、本次 observed wall、带宽及配置revision。新预测为 `predicted`，校准点为 `measured`，理想化为 `synthetic`，本次实际时间为 `observed`。synthetic的不确定性显示“不适用（假设）”，不显示0误差；假设场景始终有可见标记。

1440×900 与1920×1080真实浏览器检查：主面板约1019.5/1359.4px（1360px最大），正文850px/18px/32.4px，无页面横纵溢出，固定视口和内部滚动、中间Agent区、抽屉保持。实际点击V2后显示唯一候选与k1，刷新恢复原V0冻结Session，未再次POST。没有修改CSS或模型切换worker；原菜单/加载/历史隔离自动测试继续覆盖。浏览器烟测使用单加载的临时service，模型详情正确；Mac正式菜单另在原worker入口检查，CUDA固定模型入口不声称支持菜单切换。

截图：调试（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/screenshots/debug-final-1440.png`）、预测与观察（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/screenshots/receipt-final-1440.png`）、配置（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/screenshots/config-v2.png`）、展示（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/screenshots/presentation-1440.png`）、宽桌面（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/screenshots/presentation-wide.png`）。

## 迁移与边界

公开副本省略历史服务器部署操作说明。MPS Profile 不能直接用于 CUDA；不同设备必须独立校准。

## 工程验收与复核

最终 Python **1110 passed / 1既有skip / 2既有warning**；Frontend **190 passed**，TypeScript、Vite构建、compileall、四个CLI help和diff检查通过。新Profile/导出证据SHA、holdout误差/决策统计复算、8.3C历史审计、12个核心路径未改与26个用户文件SHA检查通过。详见 acceptance.json（原研发资料，未随公开副本提供：`generalized-auto-85/evidence/acceptance.json`）。

原8000/5173入口已恢复0.5B，菜单三个选项及真实当前标记、同身份Profile导入在浏览器检查，无追加生成。校准/讨论共享一次0.5B加载；恢复常规服务另加载一次0.5B。进程重启仍按原架构清空内存Session，此前3份元数据已保存到忽略目录 `benchmark-results/generalized-auto-85/previous-session-*.json`；本次两个真实讨论的完整记录随证据保留。临时8015服务及验收标签页已关闭。可复核命令：

```bash
.venv/bin/python run_generalized_calibration.py --verify --output-dir benchmark-results/generalized-auto-85/local
.venv/bin/python run_generalized_calibration.py --verify --output-dir docs/generalized-auto-85/evidence
.venv/bin/python verify_system_quality_evidence.py
```

实测冻结源码保留在 evidence/measured-sources；测量后只增强了解析/审计、独立服务入口与来源显示，详见 source-drift.json，没有修改V0–V3、handoff计时、Prompt或旧benchmark。用户原26个未跟踪文件保留且SHA不变；不reset/clean/push。完整验收后只提交本任务范围。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
