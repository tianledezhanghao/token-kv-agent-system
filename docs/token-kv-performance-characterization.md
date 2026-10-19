# 8.3A — 增量 TOKEN / KV 成本、长度扩展与带宽交叉点

2026-10-15（Asia/Shanghai）。**研究完成；当前实测范围没有可信 KV 速度优势或正带宽交叉点。** 这不是所有模型、设备或实现的性能定律。起始 HEAD `45286565ae3231a1df687d4b2e5be562ab73b990`。

2 内容 Token、EOS 的正式秘密场景：完整本地 handoff 的 steady median 为 TOKEN **39.40ms**、V0 **79.15ms**，V0 多 **39.75ms / 约101%**。512 内容 Token 的真实生成前缀：TOKEN **104.58ms**、V0 **117.09ms**，两条中位数差 **12.51ms**；同轮差值全部为正，但波动明显。最主要的 V0 额外费用是 Receiver 分开的 framing forward、Restore、Host 导出；截断消息还有额外 pending forward。没有改通信算法、正式 Prompt、评分、Selector 或 Dashboard。

## 1. 研究目标与证据入口

研究实际增量 handoff 的计算、Tensor 搬运、Cache 操作及真实协议字节，分别控制消息长度和 Receiver 历史长度，并测量 V1–V3 附加修正。目标是解释成本并界定 Profile 可用范围，允许结果始终选择 TOKEN。

- 冻结配置和源码 SHA（原研发资料，未随公开副本提供：`performance-83a/evidence/frozen-spec.json`）、模型/Tokenizer/权重文件与合约身份（原研发资料，未随公开副本提供：`performance-83a/evidence/settings.json`）。
- 每次计时、forward 形状、阶段、MPS 内存（原研发资料，未随公开副本提供：`performance-83a/evidence/samples.csv`）、分组统计与输入（原研发资料，未随公开副本提供：`performance-83a/evidence/groups.json`）。
- 两条实际 Sender 输出和 IDs（原研发资料，未随公开副本提供：`performance-83a/evidence/generations.json`）、成对差值（原研发资料，未随公开副本提供：`performance-83a/evidence/paired-deltas.json`）。
- 模拟网络与交叉点（原研发资料，未随公开副本提供：`performance-83a/evidence/network.json`）、研究 Profile（原研发资料，未随公开副本提供：`performance-83a/evidence/profiles.json`）、50 次离线 Selector 预测（原研发资料，未随公开副本提供：`performance-83a/evidence/selector-decisions.json`）。
- 真实性/事务/资源/源码漂移审计（原研发资料，未随公开副本提供：`performance-83a/evidence/audit.json`）、54 个原始产物 SHA（原研发资料，未随公开副本提供：`performance-83a/evidence/raw-index.json`）、Git 证据 SHA（原研发资料，未随公开副本提供：`performance-83a/evidence/evidence-sha256.json`）。

大型原始 JSONL、完整实际 Cache 的 `.kv` 和资源记录仅保留在忽略目录 `benchmark-results/performance-83a/`，没有覆盖旧 benchmark。Git 证据约1.3MiB，包含全部标量样本、输入 IDs 和两个发生分析代码漂移的实测源码副本（原研发资料，未随公开副本提供：`performance-83a/evidence/measured-sources/performance_validation.py`）。

## 2. 当前真实语义

|策略|实际交接工作|
|---|---|
|TOKEN|A 发送 TKVT Token IDs；B 保留自己的长期 Cache，将 incoming prefix、内容、closing 合并为一次新增输入 forward；不重新 Prefill 全历史|
|V0|A 实际 Decode 内容增量 KV → 必要 pending 私有补算 → span clone → Host bytes → TKVI/TKVK 编码/checksum → InMemoryTransport → 解码 → Restore → B 工作 Cache clone → 本地 prefix forward → 外来 KV cat/reconstruct → closing forward → Sender close → 双端验证/提交|
|V1|V0 加全消息 K 的实际 RoPE 位置修正；V 不旋转|
|V2|V1 加 Receiver 连续尾段 k 个内容 Token 的一次真实批量 forward；完整 KV 仍传输、恢复及旋转，k=N 是全消息重算参照|
|V3|V1 加首个合法 ASCII 数字 Token 的选择、prefix crop、单槽位真实 forward、scatter/reconstruct/验证；保留其他槽位|

KV 来自 `KV(M | H_A)`，一般不等于 TOKEN 在 B 生成的 `KV(M | H_B)`。本轮只研究成本，没有撤销 V3 的历史质量负面结论，也没有用性能结果认证 V0–V2 的质量。

## 3. 本机与模型

Apple M5、16GiB、arm64、macOS 26.7.1；PyTorch 2.14.0、Transformers 5.17.0。Qwen2.5-0.5B-Instruct，固定 revision `7ae557604adf67be50417f59c2c2f167def9a775`，BF16、MPS、batch=1、greedy、offline/local_files_only。

24 层、2 KV heads、head_dim=64，每内容 Token 的 K/V raw 总量 **12,288B**。本实验进程只加载一次该模型，A/B 共享权重并保持独立 Cache。此前用户服务保持空闲，不终止；没有加载3B、下载新模型、访问服务器或更改 Swap/MPS 限制。CPU 用于协议 bytes/重建是实际流程的一部分，模型 forward/恢复后的 Cache 均为 MPS，没有 CPU 推理替代。

首次模型合约/权重指纹初始化另测 **1092.69ms，n=1**，单列为启动诊断；不是稳态统计，也不视为免费。所有正式分支预先拥有同一真实合约。首次角色 Prefill、模型加载、身份文件 SHA、Cache 证据保存均不进 handoff 排名。

## 4. 先审计旧计时，再定义完整边界

旧 `StatefulTokenSession.handoff` / `handoff_kv(measure_timing=True)` 的内部 `handoff_total` 均未覆盖最终双方状态提交；KV 还在最终 record/验证之前结束计时。TOKEN 的 `sender_clone_close`、KV 的 `export`、`restore` 以及修正阶段有嵌套子费用，不能直接累加所有字段。Backend `_timed_forward` 只覆盖同步模型 forward，不包含它外面的输入张量/attention mask 构造、argmax、状态检查等。

本轮统一核心边界：**Sender draft 已生成完成、双方基础快照已准备好 → 调用原正式 handoff → 双端事务验证及 `_states.update` 已完成 → MPS 同步结束。** 外部 `perf_counter` 覆盖 lock、入口检查、实际通信、Cache/forward、最终提交；没有将内部旧 `handoff_total` 当完整成本。恢复基础快照字典和安装/撤销研究包装在计时之外，正式算法自己的所有工作 clone 都在计时之内。

Sender 真正生成消息是公共费用，单独记录：短消息生成 wall **80.64ms**（3 raw IDs，含EOS）；长消息 **8381.68ms**（512 raw IDs，截断）。其初始化、stage/assistant prefix、Decode 不加到某一个候选。Receiver 后续 stage/回答 Decode、完整会话 E2E wall 为 `not_measured`。KV 的实际 closing 已在新 Cache 上完成真实 forward，属于 handoff 就绪状态；不以不同策略回复长度比较通信速度。

## 5. 同步、仪器和统计口径

研究代码在独占 CLI 的 context 中临时包装既有函数，默认生产路径没有任何新钩子。原算法调用、输入、Tensor、协议及 guards 保持；失败时恢复所有包装。

|口径|同步与诊断|
|---|---|
|steady|完整边界前/后 `torch.mps.synchronize()`；保留 Backend、export、Restore 自带同步；不安装逐阶段包装，不做 Tensor/位置/mask CPU trace，不逐 forward 写磁盘|
|instrumented|同一完整边界，额外函数阶段前/后同步；保存父/子 inclusive 和 exclusive；Adapter 自带子指标另列|
|网络|按实际 wire bytes 离线计算；不 sleep、不限速；不是物理网络实测|
|资源|原 ResourceMonitor 每2秒采样；forward 仅记录 input count、前 Cache 长、实际 device 和 Backend 时间，均无 Tensor CPU 导出；watchdog 保留|

steady 仍有低量级的 forward 标量记录和后台资源采样，不能称零仪器生产吞吐。阶段包装自身同步也在 instrumented 时间内。两种模式分块测量，系统负载、采样时机及顺序会影响差值，不能将两块中位数之差全部认定为仪器净开销。

每个条件/候选/模式 **2 warmup + 5 有效样本**，候选顺序逐轮反转，不每次重生成 Sender。11组、27个候选分支、**378 次成功 handoff = 108 warmup + 270 有效样本**；steady/instrumented 各135有效样本。2条实际 Sender 生成。记录索引显示共1817次实际 forward，其中1288次发生在 handoff，其他为生成/初始化/受控历史 setup；完整 handoff 的1288条形状/device/forward_ms可复核，生成窗口亦单列。

表格除特别说明均为 ms 的 median；每项 n=5；min/max/MAD 和每次数据完整保存，不构造置信区间、不删除慢样本。这些是冻结任务的重复成本样本，不是270个独立任务成功率样本。每次从原始不可变基础状态重新出发，经正式私有 work clone 执行；分组前/后逐Tensor SHA相同。

## 6. 输入来源、span 和快照控制

短消息使用生产 SecretSum 原 System/PUBLIC_TASK/STAGES，真实 A 生成 `17`，内容 IDs `[16,22]`、末EOS；A Cache124、源内容起点122，B Cache97。与历史8.2A第一条的小Cache条件一致，但不汇总为同一数据集。

长消息复用8.2C原1024场景的冻结 System/question/默认 first stage，将本轮一次真实 Decode 上限设512。**旧 JSON 只有 IDs/元数据，没有可恢复的旧 KV Tensor；本轮没有声称从 JSON 恢复当时 KV。** 新实际输出从 `1. The first number is 1...` 开始，512内容Token、无EOS、max_new_tokens。

N=2/8/32/128/256/512 分组均截取这一条新真实生成的合法前缀；保留原绝对位置的实际 K/V，私有 crop 到 source_start+N−1，让第N个已接受 Token 为 pending，等价于 Decode 在该位置停止。未对消息重新 Prefill、未用随机 Tensor。源历史起点 **119固定**，Sender cached count=119+N−1；内容长度增长时源Cache增长是实际 Decode 的必要伴随变量，不能把 Sender clone 成本变化从长度效应中隐藏。

固定长度组 B Cache=128，目标内容起点=128+9=137、RoPE shift=18；所有候选使用同一framing。历史组固定N=8、Sender Cache126/源起点119，B为128/512/1024/2048，目标起点=B+9。B历史由原独立 System/问题保留，加一次**受控本地 user filler**真实 forward 建立；这是受控历史 workload，不是自然长对话。内容 KV仍来自真实 Decode，非固定输入 Prefill KV。本轮没有 C类 synthetic/fixed-token 消息 KV 校准。

冻结实际完整 Cache bytes及ledger可见长源状态（原研发资料，未随公开副本提供：`performance-83a/evidence/long-source-A.json`）、短A（原研发资料，未随公开副本提供：`performance-83a/evidence/short-A.json`）、短B（原研发资料，未随公开副本提供：`performance-83a/evidence/short-B.json`）、B2048（原研发资料，未随公开副本提供：`performance-83a/evidence/history-2048-B.json`）。大 `.kv` 在 raw目录；Git保存其SHA，不包含Tensor本体。

## 7. 最重要的短消息 TOKEN / V0 分解

下面是 instrumented n=5。父项和缩进含义用“其中”明确标识；**不能将父项与子项再相加，也不能期待独立阶段中位数相加恰等于总中位数。** 导出器验证每次 exclusive 阶段和恰等于该次根 span，外部完整边界另有不到1ms的包装差。

|实际阶段|TOKEN|V0|
|---|---:|---:|
|入口双方状态检查|0.192|0.186|
|Token 编码 / KV编码及checksum|0.006|0.116|
|InMemoryTransport|0.003|0.003|
|Token 解码 / KV解码及checksum|0.006|0.173|
|Sender出口完整 composite|不适用|9.689|
|其中内容span Tensor clone|不适用|0.355|
|其中Host导出（搬运/contiguous/bytes/逐层接口）|不适用|9.135|
|出口pending补算|不适用|0，EOS样本内容均已缓存|
|Restore完整 composite|不适用|9.994|
|其中CPU Tensor重建|不适用|0.101|
|其中Host→MPS上传|不适用|9.416|
|其中Cache对象构造|不适用|0.359|
|Receiver工作态clone（含验证）|0.925|0.892|
|Receiver framing + 新消息Prefill合并|18.957|未执行此合并路径|
|Receiver单独incoming prefix|合并在上项|17.546|
|内容消息Prefill|合并在18.957中，无法单拆|0，确实没有内容forward|
|KV cat + 私有reconstruct|不适用|0.861|
|Receiver单独closing|合并在18.957中|17.235|
|Sender工作态clone（含验证）|1.295|1.052|
|Sender关闭消息forward及状态操作|16.787|17.027|
|未进一步拆分的framing派生/参数检查/核验/提交残差（exclusive）|1.706|6.709|
|**完整外部handoff**|**39.702**|**81.201**|

纯物理 device→Host copy、checksum单项、TOKEN内容/framing单项、最终事务提交单项均为 **not_measured**；对应 composite 全部已在总成本收费，没有凭估计拆分。Adapter所谓device_to_host约9.100ms也包含Python、48个Tensor的CPU/bytes转换，不能把它等同于纯链路搬运时间。

steady本地结果为39.405 [38.733,46.375] vs79.153 [77.062,80.670]。instrumented约+0.30/+2.05ms；这个短条件比较稳定。V0与TOKEN相同Sender关闭的约17ms是双方共有费用，不解释V0相对差距。

## 8. V0额外费用排名及pending影响

短EOS条件的三项最明显额外费用：

1. Receiver prefix+closing约34.781ms，而TOKEN把framing+内容一起做约18.957ms，阶段中位数差约 **15.82ms**。
2. 真实 Restore约 **9.99ms**，其中上传约9.42ms；CPU重建仅0.10ms，Cache对象构造0.36ms。
3. Sender Host出口约 **9.69ms**，其中Host导出9.14ms、span clone0.36ms。

KV cat0.86ms、codec合计0.29ms、内存Transport0.003ms均不是这条短消息的主要费用。未进一步拆分的残差在V0约6.71ms，包含重复incoming framing推导等接口/检查，不全部叫“事务成本”。首次Contract1.09s没有混进这里。

同为N=2的长消息前缀被合法截断，V0 steady升至96.08ms。该消息的最后内容Token pending，出口需clone完整Sender Cache并单Tokenforward，随后正式Sender close又在另一私有work中消费该pending。N=32分解：pending forward **15.38ms**、pending clone **0.98ms**。这是实际收费的第二次Sender处理，不是伪造KV或再次生成消息；EOS短消息没有此出口补算，不能只按N池化两种终止条件。

## 9. 消息长度扩展

B固定128；源历史119；真实Decode前缀、max_new_tokens语义。每格steady median [min,max]、n=5。

|N|Sender cached|TOKEN|V0|V0−TOKEN中位数差|TOKEN wire B|KV wire B|
|---:|---:|---:|---:|---:|---:|---:|
|2|120|40.80 [40.42,40.99]|96.08 [94.80,97.26]|55.28|18|28,045|
|8|126|41.40 [40.85,41.53]|94.23 [92.70,97.22]|52.82|42|101,797|
|32|150|45.56 [42.71,46.65]|98.56 [93.55,101.10]|53.00|138|396,807|
|128|246|50.48 [49.64,54.12]|96.61 [95.85,99.89]|46.13|522|1,576,841|
|256|374|70.47 [68.65,74.36]|103.15 [100.79,106.05]|32.68|1,034|3,150,217|
|512|630|104.58 [103.03,114.34]|117.09 [107.84,146.52]|12.51|2,058|6,296,969|

TOKEN新增Prefill随N显著增加；instrumented合并消息阶段由N8的20.69ms增至N512的80.34ms。V0没有内容forward，长度增长主要增加bytes/codec、搬运、Sender克隆/close及后续attention成本，不能称恒定时间：N512 export30.10ms、Restore11.77ms、encode2.73ms、decode2.72ms。

Host导出由2Token的约9ms到512Token的9.82ms，raw bytes增加256倍而时间增加较小；在这个范围固定接口/逐Tensor/sync费用显著，**没有证据说纯物理字节复制是唯一最大瓶颈**。具体host copy、contiguous与Python各自占比仍未单测。

N512的V0 MAD7.12ms；同轮steady差值 `[41.94,13.14,2.75,4.80,5.77]`ms，median5.77ms，全部正。这个“成对差值中位数”不同于两策略median之差12.51ms。不同统计量均保留，不挑有利口径。未测1024/更长消息；没有根据接近趋势外推真实优势。

## 10. Receiver历史Cache扩展

固定N=8和同一Sender真实KV/Cache126；B历史为受控本地输入，内容span/pending不变。128取length-8已有组，不重复执行同条件。

|B cached|双方cached合计|TOKEN steady [min,max]|V0 steady [min,max]|TOKEN B clone|V0 B clone|V0 cat+reconstruct|TOKEN合并Prefill|
|---:|---:|---:|---:|---:|---:|---:|---:|
|128|254|41.40 [40.85,41.53]|94.23 [92.70,97.22]|0.916|0.940|0.999|20.694|
|512|638|46.88 [42.14,51.62]|106.64 [95.70,111.58]|1.228|1.240|1.332|21.864|
|1024|1150|42.60 [42.00,63.66]|99.82 [98.79,100.90]|1.472|1.578|1.673|21.051|
|2048|2174|44.76 [44.06,45.70]|106.39 [104.64,125.34]|2.345|2.275|2.469|23.758|

后三列为instrumented composite median、各n=5。clone/cat随历史增长有清晰趋势，完整steady总时间则**没有单调增长**；512条件的扰动较大，不作严格拟合。V0 Receiver prefix/closing从B128的17.59/16.91ms到B2048的23.50/22.86ms，attention工作也增长。

源码clone会复制所有48个历史Tensor；cat会为历史+delta分配，再由reconstruct clone一次。固定层数/heads/dtype下，复制字节量与已有历史长度成比例；这是源码层面的线性字节复杂度，**不是根据这四个wall样本宣布总handoff严格O(N)**。模型attention、Python固定费用、allocator及系统噪声同时存在。

历史KV raw分别1.5/6/12/24MiB。本地已分配MPS内存的steady采样median（TOKEN/V0相同）约977.40/1004.30/1016.30/1040.30MiB，保存每次数据。这是整个实验进程，含权重、冻结源和工作Cache；length-8较早时尚未持有全部其他长度快照，不能把第一段差额都归于B增长。512→1024→2048额外占用与原/工作两份B历史并存相符，但不声称采样捕获瞬时cat峰。wire恒定42/101,797B，与已有B历史不一起发送。

## 11. V1真实RoPE附加费用

代表条件N32、A150/B128、源119/目标137。V0 steady98.56ms；V1 **149.87 [145.84,154.87]ms**，两策略median差 **51.31ms**。V0与V1不在同一交替块，因此差值是代表性额外总成本观察，不能看成高精度配对估计。

V1 instrumented total155.29 [154.12,163.27]ms。实际 tables两次合计 **0.84ms**；24层 K rephase合计 **55.72ms**；RoPE合约检查0.42ms。额外tensor临时分配/FP32转换包含在keys内，独立copy成本 `not_measured`。V保持原值。

keys helper每层重复5个finite reduction+`.item()`、4个table pairing `torch.equal`、1个denominator reduction+`.item()`，随后多次FP32算术及cast/cat。24层共240个这类Host布尔判定接口，是合理的同步/逐层优化候选；**没有单独量化guards与算术的占比**，不能把55.72ms全称纯RoPE数学运算或纯sync。

## 12. V2尾段重算附加费用

同N32/source/target快照，所有KV策略wire **396,807B**。每项steady n=5；delta相对该块V1 median149.87ms，不是只收费forward。

|V2配置|实际k|steady完整 [min,max]|相对V1 median|instrumented完整|实际suffix composite|
|---|---:|---:|---:|---:|---:|
|k=0|0|155.95 [146.94,157.18]|+6.08|161.24|0，未执行|
|k=1|1|162.50 [160.24,180.27]|+12.63|196.84|17.91|
|k=2|2|172.59 [161.53,182.92]|+22.72|179.13|17.41|
|ratio25%|8|169.26 [163.79,198.21]|+19.39|199.33|19.85|
|k=N|32|175.39 [169.34,183.15]|+25.52|188.25|21.55|

k0算法消息处理等同V1，观察到6.08ms差异主要只能作为波动证据，不宣称固有“k0修正费”。k1/k2都是partial，ratio25%实际ceil(32×.25)=8；k32明确full。k32仍付整包/Restore/整段RoPE，省掉外来KV append，因此不能简单用V1+k32 forward逐项预测。

suffix composite包含backend输入/forward/结果校验/ledger变更，prefix视图、整消息RoPE、cat与其他准备在完整总计中另已收费。重算k个Token是**一个批量forward**，17.91/17.41/19.85/21.55不支持“每Token固定15ms”线性外推。instrumented k1/ratio块波动明显，RoPE自身中位数也随该块负载改变；不能用这几次总时间声称k2比k1快。未拟合V2长N比例Profile。

## 13. V3真实选择和单槽位修正

同N32消息第0个内容Token为ASCII数字`1`，所有有效样本确实选择index0，24层×1Token真实修正，无fallback。

V3 steady **167.66 [163.56,173.14]ms**，相对V1 median多 **17.79ms**；instrumented total189.42 [171.23,205.42]ms。数字选择 **0.016ms**；完整slot_repair **19.47ms**，其中真实单槽位 `continue_from_cache` **17.05ms**、crop和最后Cache重建两次合计 **1.80ms**。slot自身exclusive残差包含全Cacheclone、scatter和验证；这些没有被忽略，但scatter纯写回/各次crop独立值为 `not_measured`。

只计forward17.05会漏掉约2ms左右的实际slot准备/写回（阶段median差仅为描述性近似）。与V2一样整条wire不减、V1整消息修正仍收费。没有更改数字规则、重新搜索Prompt或重新评价质量。

## 14. Wire增长与网络模型

TOKEN：`10+4×raw_generated_count`；短EOS样本raw3，所以22B。KV：raw `12288×content_N` + 真实TKVT/TKVK/TKVI headers、metadata及32B SHA。KV包也携带content IDs作为ledger证据，不代表Receiver做TOKEN内容Prefill。V0–V3同消息bytes相同；N512 KV约6.01MiB，TOKEN2058B，wire比约3060倍。少量metadata整数位数增长导致外层常量并非完全恒定。

单向模拟模型：

```text
Network(B) = actual_wire_bytes × 8 / (B_Mbps × 1000) + 2ms
Total_TOKEN(B) = measured steady median_TOKEN + simulated Network_TOKEN(B)
Total_KV(B) = measured steady median_KV + simulated Network_KV(B)
```

本地InMemoryTransport是真实操作；上述网络是独立替代链路项，按用户指定本地+network模型计算，保留本地Transport微小固定费。共同2ms基础延迟会从交叉点方程消去；改同一个base latency只会平移曲线。

## 15. 1–10000Mbps的模拟总成本

下表为**实测本地median + 模拟网络**，不是物理测量；每个本地median n=5，单位ms。其他8条件全部数据在network.json。

|Mbps|短EOS TOKEN|短EOS V0|N512 TOKEN|N512 V0|
|---:|---:|---:|---:|---:|
|1|41.58|305.44|123.04|50,494.84|
|20|41.41|92.37|107.40|2,637.88|
|100|41.41|83.40|106.74|622.85|
|1000|41.41|81.38|106.59|169.46|
|10000|41.40|81.18|106.58|124.13|

20Mbps时N512仅KV传输项约2518.79ms，1Gbps约50.38ms，10Gbps约5.04ms（不含共同base）；高带宽减少惩罚，不能反转仍较高的实测本地成本。

## 16. Break-even数学与不确定性

令 `dL = Local_KV−Local_TOKEN`，`dW=Wire_KV−Wire_TOKEN`：

```text
Total_KV−Total_TOKEN = dL + 8*dW/(1000*B)
B_cross = -8*dW/(1000*dL)   （仅有限且正时有效）
```

10组中dL和dW的实测median差都为正，故**在该组实测成本条件和同单向基础延迟、正带宽网络模型下，没有正带宽交叉点**。B趋于无穷仍保留正本地差；这不推广到未测更大N/模型/CUDA或改实现。

N512独立min/max有交叠。如果人为组合TOKEN最慢114.339与KV最快107.839，可算出约 **7747Mbps** 的“乐观极值交叉点”。二者来自不同轮，**不是实测优势或可信crossing估计**；对应同轮5次均KV较慢，保留极值敏感性而不宣称10Gbps已反超。当前证据没有要求有界真实AUTO交叉验收的可信新成本区间，因此未为了制造切换再跑AUTO模型场景。

## 17. AUTO研究Profile的适用域

导出10个point-domain Profile：真实backend/tokenizer/config/revision/dtype/device身份，`measured_incremental`、`decode_incremental_handoff_v1`、每点sample_count=10（TOKEN/V0各5次steady）。分别保存real_decode和real_decode_prefix来源、确切N/双方cache总和及各端实际长度说明，未用Canonical或synthetic性能参数。

现有Selector只检查N范围、双方cache总和范围，不能区分相同总和的不同A/B长度，也不能强制原消息/framing/终止条件一致。因此这些是**研究离线Profile，不能作为完整部署域保护**。调用者还必须核对各端长度、source/target位置、pending/EOS、输入/framing；不同模型路径（repo名 vs snapshot绝对路径）也不能复用身份。

每点只含token_handoff/kv_base，没有startup或V1–V3拟合；未准备合约的KV会按现有Selector拒绝。V1–V3虽实测但覆盖单N、方差较大，不强行池化；不把批量重算计成线性每Token成本。没有合并长消息点为一个稀疏多元回归或矩形适用域。

现有正式Selector仅离线预测：10点×5带宽=50次，均TOKEN，预测的两候选wire与正式receipt完全一致。它们是**offline_prediction**，没有在线执行两条候选或回填真实选择。当前数据支持后续域设计和查表Profile研究，尚不支持跨消息内容/历史分布的长期泛化性能承诺。

## 18. 工程瓶颈与KV固有费用

|性质|证据与含义|
|---|---|
|实现/编排固定费用|Receiver两次framing forward；截断消息pending私有补算后Sender close再消费同Token；每Tensor跨设备/bytes接口；重复framing派生/检查|
|实现Cache费用|事务全历史clone；cat后再次reconstruct clone；历史B128→2048的clone/cat确实增加|
|实现RoPE费用|24层重复tables guards/Host布尔判定及多kernel处理；tables本身不到1ms，keys约56ms|
|KV表示固有数据量|当前未压缩BF16需要12288B/内容Token，TOKEN为4B/raw Token；即使零编码开销wire数量级差仍在|
|必须计入的实际语义费用|KV提取/接收/独立所有权、兼容性与事务保障、V1变换、V2/V3真实重算不能当免费；具体次数和拷贝方式有工程优化空间|
|未被性能研究解决的问题|跨不同历史的KV语义偏差和历史V3质量退化；提高带宽不修复语义|

“算法是否值得传KV”与“此实现有没有额外固定费用”是不同结论。本轮证明后者有多个可调查方向，没有证明前者已划算，也没有把本机MPS速度当KV通信理论上限。

## 19. 有证据支持的8.3B优化顺序

1. **减少重复模型处理。** 对截断消息研究同一个私有Sender工作态完成pending/closing后合法提取内容，避免约15–18ms重复pendingforward；Receiver framing是否能降低启动/派生成本，需要保持真实目标位置与模型输入语义，不能直接把两个forward合并跳过KV间隙。
2. **批量化Host导出与Restore数据通路。** 短消息导出/上传各约9ms，48张量固定接口显著；优先测聚合布局/减少同步与Tensor接口，保持真实bytes、BF16 bits和独立Cache。CPU重建0.10ms/短codec0.29ms不应先成为优化主项。
3. **检查V1重复guards与逐层处理。** 单消息keys约56ms远大于tables0.84ms，候选是共用tables validation一次、减少Host判定及批量算子。须保持finite/合法RoPE拒绝和真实数值验证，不能为benchmark关安全检查。
4. **减少全历史clone/cat重建。** 实测B2048 clone/cat各约2.3–2.5ms；优先评估事务安全下的所有权/追加设计，而不是共享可变Tensor。需要长历史/失败回滚测试，当前未实现优化。
5. **改善成本域。** 分开A/B历史、pending/EOS、消息长度与重算k/ratio；先查表/分段和holdout，不把当前点域替换成无证据线性模型。

前两项对应V0最大的额外费用；第三项仅对需修正策略有效。N512 codec已各约2.7ms，可以作为长消息后续项，仍远小于模型/修正费用。排序依据本轮代表条件，不保证所有设备相同。

## 20. 理想化下界：不能当优化实测

hypothetical_lower_bound / not_measured（原研发资料，未随公开副本提供：`performance-83a/evidence/hypothetical-lower-bounds.json`）：对每个instrumented V0有效样本，减去指定已测阶段的inclusive时长，再取median；其余成本和wire保持。不是删阶段后实际跑出的结果，也把阶段sync/validation一起理想化删除。

|条件/假设|假想KV本地median|同条件TOKEN instrumented|理论交叉点|
|---|---:|---:|---|
|短EOS，Host出口归零|72.07|39.70|无正带宽|
|短EOS，Host出口+Restore归零|62.28|39.70|无正带宽|
|N512，Host出口归零|107.25|107.05|无正带宽|
|N512，Host出口+Restore归零|95.91|107.05|约4518Mbps，仅数学假设|

所以只解决Host出口还不足以支持本轮任何实测点反超；极端同时消掉出口及Restore才使N512的理想本地值可能划算，而1Gbps仍被约50msKV网络费压过。它说明优化研究有意义，但不承诺这些阶段可变为零、不把约4.5Gbps当真实硬件成果。

## 21. 与8.2A历史Profile的差异

历史TOKEN43.46ms/V0base77.69ms来自2/25Token、A+B221/311、小B97/126、每点2有效instrumented样本的池化，含逐forward Tensor/位置/持久化等诊断。当前短EOS同样A+B221、B97，但改为完整提交后边界、各5有效、较低开销记录，steady39.40/79.15、instrumented39.70/81.20。

边界延长和诊断减少方向相反，不能把TOKEN下降约4ms叫算法优化、不能直接用新旧差值估算某一监控费用。旧记录保持不改。历史RoPE53.09ms与当前tables+keys约56.56ms大致同量级；历史V2k1约15.59ms与当前suffix17.91ms接近，但N/负载/仪器不同。V3本轮代表性slot19.47ms包含crop/forward/writeback，与历史18–27ms一致范围。历史未覆盖长消息，本轮扩展不能回填为旧AUTO当时已知的profile。

## 22. 资源、波动和失败记录

38次资源采样；压力normal/warning（level1/2），无critical、OOM、保护中止或CPU fallback。MPS current采样峰约1.016GiB、driver约2.174GiB、RSS约727.84MiB；系统Swap相对加载前最多增长 **1129.31MiB**，未隐藏。Swap单独不触发原保护；系统范围增长不全归因该进程，2秒采样也不保证捕获瞬时峰。没有调整系统限制或停止其他用户服务。

长消息V0和instrumented修正有慢样本，全部保留min/max/MAD；mode分块、2秒资源命令及系统内存状态都可能扰动。没有CPU利用率/thermal逐样本关联，具体慢样本原因是 `not_measured`。没有通过无限加样本或删outlier制造排名；结论限制在当前方向一致的对照。

模型侧11组/378handoff全部完成，SHA/Cache验证通过。其后原CLI的离线Selector分析把 `(choice,decision)` 元组误作dict，产生TypeError，原failure.json和原始SHA索引保留；不是通信/模型失败。修正分析函数、补实际Selector无模型测试，从已有标量/身份重新离线验证，不重跑模型。另一次离线metadata适配遗漏vocab_size在正式导出前由实际wire一致性断言发现并修正，没有任何新模型执行或原始样本变更。

原测量模块与最终模块的差异仅为离线分析提取/适配；两份原源码以冻结SHA保存在证据，审计AST核对所有计时/快照/正式handoff测量函数不变，其余生产src冻结SHA均一致。原JSON不回写成“首次全流程成功”；最终audit明确是 `verified_measurements_and_offline_analysis`。

## 23. 工程回归与可复核性

新工具12项测试覆盖nested不重计、同步边界、same snapshot真Tensor一致、TOKEN/V0–V3原算法结果不变、prefix/pending/EOS合法、双端失败不提交/包装恢复、网络单位/交叉点符号、身份/来源/点域、实际Selector离线metadata预测、SHA损坏/路径拒绝和CLI help无模型。首轮针对性143 passed；追加离线/SHA测试后新工具12项通过，全量 **975 passed / 1既有可选AutoGen skipped / 2既有依赖warning**。

最终compileall、两个新增CLI help、diff check、54个raw SHA及Git证据SHA、源身份、样本统计重算、每次exclusive和、实际wire/IDs/MPS/no-content-forward检查通过。全量pytest包含原Runtime、Selector、协议、Cache、事务和V2/V3回归。首个数学测试精确比较浮点等式导致失败，改为数值近似断言，不改交叉点计算。

只新增独立研究Python模块、测试、报告、标量证据及.agent；没有frontend/API/共享生产行为修改，因此按任务许可跳过前端test/typecheck/build及浏览器矩阵。真实MPS样本是性能证据，pytest中tiny随机CPU只验证工程契约。

## 24. 换模型或CUDA后必须重验

必须重新固定checkpoint/revision/tokenizer/config/actual weights、dtype、batch、attention实现、RoPE类型/位置语义、Cache API/layout/heads/head_dim与bytes/token；不能复用本机常数或身份。

在实际目标GPU上检查同步到确切device、冷/暖编译、单/双独立backend、相同Sender/Receiver快照、pending/framing事务、Host/PCIe或其他实际链路、逐层复制/restore/cat/修正、消息与A/B历史长度范围、内存峰/allocator/资源保护。重新测V2批量k增长和V3完整crop/writeback，不仅量forward。CUDA异步/stream、物理网络和跨进程两端分布需要重新定义完整可测边界；物理吞吐/延迟/并行重叠不自动等于本轮模拟公式。

更大模型可能增加Prefill和每TokenKV字节，KV heads/layers不同会同时改本地与网络账。仍需独立验证质量/语义；性能优势不证明跨历史KV可靠。3B/CUDA/RTX5090/跨机/物理网络/1024+消息的性能及修正成本本轮均未测。

## 25. 复现与8.3B建议

所有运行使用新目录，已存在目录（含失败/partial）拒绝覆盖：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python run_performance_characterization.py \
  --output-dir benchmark-results/performance-83a-new --warmup 2 --runs 5
.venv/bin/python export_performance_characterization.py \
  --results-dir benchmark-results/performance-83a-new --output-dir /tmp/performance-83a-new-evidence
# 只读复核本轮原样本及修复后的离线分析，无模型加载：
.venv/bin/python export_performance_characterization.py \
  --results-dir benchmark-results/performance-83a --output-dir /tmp/performance-83a-recheck
HF_HUB_OFFLINE=1 .venv/bin/pytest
.venv/bin/python -m compileall -q src tests run_performance_characterization.py export_performance_characterization.py
```

默认historical-input指向本轮已哈希的8.2C原JSON；其仅提供冻结输入。没有该本地raw时需指定等价保留输入文件，不能自动下载或声称从Git中的IDs恢复旧KV。Git中的frozen-spec保留逐字配置用于审核。新运行修复后的CLI可以直接完成offline阶段，无需原失败恢复目录；若原源码与当时源码不同，导出审计会拒绝没有相应冻结源证据的漂移。

8.3B建议只选一个优化点做同快照before/after：先研究截断Sender重复pending或Host/Restore批量通路，保留完整计时与协议/事务测试；分别评价固定费用、长N、长B和质量。然后为Profile加各端历史/位置/pending域及holdout。不先做新的通信算法、不先制造AUTO切换、不把未测1024/更大模型趋势当性能收益。全部工程与研究验收通过后按用户授权仅任务范围commit，不push。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
