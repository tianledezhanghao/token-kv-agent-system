# 7.9C — KV V3 可行性、误差诊断与最小验证

2026-10-14，起始/验收前 HEAD `94384cf`，main。**决策 A：V3 最小实现成功，可作为新增实验策略；质量结论为负面，不能作为可靠修复或自动选择候选。** 本轮没有实现 RelayCaching，也没有证明分层修复不可能。

新增 `numeric_slot_delta` / `kv:v3` 是真实单槽位重算：选首个 ASCII 数字内容 Token，裁出其前缀，真实执行全部 Decoder 层，再将新 K/V 写回该槽位，保留后续槽位。两组秘密均丢失 Receiver 自己的数字，暗号丢失中文部分。**工程可执行与语义有效完全分开；结束本轮研究，下一阶段直接推进动态网络和增量 Selector。**

## 1. 基础与边界

读取 `.agent` 工作流、README、system-design/architecture、[V0/V1审计](kv-v0-v1-implementation-audit.md)、[V2报告](kv-v2-suffix-recompute.md)、[TOKEN冻结基线](token-communication-baseline.md)、私有交换验收（原研发资料，未随公开副本提供：`private-information-exchange-validation.md`）、[早期增量/RoPE与RelayCaching笔记](incremental-kv-experiment.md)、相关生产源码/测试/Harness。以代码为准，既有文档中旧阶段的“V2未实现”不是当前事实。

TOKEN 用内容 IDs 在 Receiver 自己的历史上增量 Prefill；V0 用真实 Sender delta；V1 修正 post-RoPE K 的位置而不旋转 V；V2 用 V1 前段 + 连续尾段 k 个 Token 的真实 forward。均已有正式 Session、Discussion、SecretSum 与 bytes 路径，未重复开发。V2 历史43生成/40交接只作旧证据，本轮不计入新运行。

初始工作区只有用户未跟踪 `docs/token-communication-baseline_副本.md`、`docs/dashboard-design-exploration/` 和三份 redesign 预览，全部保留，不暂存。没有 reset/clean、覆盖旧实验、访问服务器/SSH、模型下载、3B实验、CPU冒充MPS、修改Swap/MPS限额或终止其他进程。旧8000/5175服务保留；新能力需要正常重启加载新代码。

## 2. 研究依据：核实到的RelayCaching

重新读取[论文原文 §3–4](https://arxiv.org/html/2603.13289v1)。论文研究相同生成内容在不同前缀下 Decode→Prefill 的复用，报告中间层/部分Token偏差；使用离线层范围校准、缓存hidden states、检测层的真实计算及偏差/注意力影响/尾段集合选择。检测层之前存在密集计算，不能把最后选中的槽位数当作全部成本。论文测试模型、任务和H20条件与本机Qwen2.5-0.5B/MPS不同，其收益不能转移为本项目结论。

固定官方revision `6b0e9d37b74d89cfb0fbad276790150a0d9c0b49`，再次获取并保存以下原文SHA（源索引（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/audit.json`））：

- [fixed_qwen2_hijack.py](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/RelayCaching/utils/fixed_qwen2_hijack.py)：自定义Qwen2 forward，逐层load/save hidden states，处理不同query位置/causal mask与token drop。
- [relay_utils.py](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/RelayCaching/utils/relay_utils.py)：RelayCache将选中KV写回原位置，并维护中间hidden状态；计算新V与旧V的cosine偏差。新V必须通过实际Transformer操作获取。
- [requirements.txt](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/requirements.txt)：Transformers4.52.4/torch2.6.0。本机为5.17.0/2.14.0，旧源码依赖key_cache/value_cache和旧forward/mask接口，不能直接patch。

没有clone/运行官方框架或声称复现论文。没有将论文图中存储描述代替本机源码事实。新的V3是本项目小范围槽位算法，**不是RelayCaching移植**。

## 3. 上下文、层和位置的实际计算依赖

本地检查 `transformers/models/qwen2/modeling_qwen2.py` 与 `cache_utils.py`：

```text
x_l = input_layernorm(h_l)
Q_l = Wq_l x_l; K_l = RoPE(Wk_l x_l, position); V_l = Wv_l x_l
cache.update(K_l, V_l, layer)      # DynamicLayer 沿sequence维cat
h_(l+1) = residual + Attention(Q_l, cached K/V, causal mask)
         + 本层残差/归一化/MLP计算
```

DynamicCache存 `layers[l].keys/values`，BHSD，序列维2；本机24层、2 KV heads、head_dim64、BF16。第一层投影输入是同一Token的embedding，因此V基本相同；后续层hidden已混入不同的历史/本地framing。V1只解决K的旋转坐标，既不会改变V，也不能逆转Attention、残差和MLP造成的上下文差异。

V2可以用标准 `continue_from_cache`：Cache长度=start+N−k，尾段IDs连续，模型推导连续position_ids，attention mask覆盖历史和新增尾段，K/V直接append，无未来槽位冲突。它看到的前段仍是Sender条件化的混合Cache，并非完整Receiver oracle。

任意中间Token不能拿完整长度Cache直接append：位置会成为消息末尾之后，旧槽位重复存在，因果mask也会错误。必须裁剪至目标前一位置、执行forward、写回原槽位，或者定制稀疏query/mask/scatter更新。修复j不会自动重算j之后已经缓存的hidden/KV；之前复用的错误槽位仍条件化所选Token。

单独修复第l层需正确h_l。KV不存h_l，GQA投影也不是可逆的完整hidden-state保存；需要从embedding执行下层计算，或另传hidden states并接受其Sender上下文偏差。只拷贝oracle某层KV是**反事实干预**，没有计算该层输入、Attention或上下层同步。`output_hidden_states=True`能通过真实完整forward得到hidden，但不是免费缓存；Backend公开continuation目前只返回最后logits/cache。当前接口没有原生任意层/任意中间Token写回。

## 4. 冻结诊断与资源估算

[Harness](../src/token_kv_agent_system/kv_v3_diagnostics.py)、[入口](../run_kv_v3_diagnostics.py)、冻结spec与源码SHA（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/frozen-specs.json`）。历史没有可恢复Tensor快照，沿用旧Harness的精确Prompt、IDs检查与配置，**所有本轮Tensor/生成均为新实验**。没有Prompt、评分、k或层范围结果后搜索。

Qwen2.5-0.5B-Instruct revision `7ae557604adf67be50417f59c2c2f167def9a775`；缓存路径、offline/local_files_only、MPS、BF16、batch1、greedy、seed77、96预算。三个冻结用例：17/26、17/31、紫色袋鼠5729。每组诊断Sender只Decode一次，各Receiver从同一独立快照分支：TOKEN、V0、V1、V2 k1、V2 k=N全消息重算、首数字槽位候选、中间层[8,16)Oracle干预，以及独立完整Prefill oracle的继续Decode。

Receiver oracle真实Prefill **相同完整Receiver历史IDs+incoming framing prefix+内容IDs**，再追加相同closing/Stage/assistant prefix；不是Sender历史，也不是仅裸M。oracle包含108/108/119个输入Token，共335×24=8040层×Token工作量（不含后续closing/Decode）。TOKEN和全消息增量重算沿用已经计算的Receiver历史，可能与独立全Prefill有BF16分段/批量误差。所有分支的Receiver ledger逐ID一致。

两个秘密内容均 `[16,22]`（17），source[122,124)、target[106,108)；暗号7 IDs `[111413,100509,101547,20,22,17,24]`，source[144,151)、target[112,119)。对齐单位是相同ID/相同Receiver绝对槽位/相同层/head，不对齐Sender私有历史或不同framing中的语义对象。

单个256Token全KV Cache约3MiB；最大的7Token消息oracle仅84KiB。顺序分支、一份模型、只持久保存CPU BF16消息参照及标量，不保存全部多份层级激活或大型Tensor快照；多个工作Cache和oracle临时同时存在但都很小。每层统计时才转FP32 CPU；不同K/V分开。raw目录 `benchmark-results/kv-v3-79c/` 拒绝已存在输出目录；正式验证在两个新的子目录，raw全部忽略。

## 5. 数值与logits结果

统计定义：每Token拼接KV heads/head_dim，记录mean/max absolute difference、`||actual−oracle||₂ / max(||oracle||₂,1e−12)`、cosine。cosine采用拼接head向量，**不是论文逐head cosine均值**。表中再对层与Token均值；不是全Tensor的单个归一化范数。逐层/逐Token以及完整标量数据见summary（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/summary.csv`）、layers（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/layers.csv`）、tokens（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/tokens.csv`）、layer-token metrics（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/layer-token-metrics.csv`）。

|用例/策略|K relative L2均值|V relative L2均值|V cosine均值|首Decode logits relative L2|
|---|---:|---:|---:|---:|
|17/26 V0|0.488|0.600|0.796|0.142|
|17/26 V1|0.268|0.600|0.796|0.129|
|17/26 V2 k1|0.179|0.354|0.902|0.097|
|17/26 槽位候选|0.129|0.321|0.887|0.075|
|17/31 V0|0.490|0.600|0.796|0.139|
|17/31 V1|0.270|0.600|0.796|0.125|
|17/31 V2 k1|0.179|0.350|0.903|0.096|
|17/31 槽位候选|0.131|0.325|0.885|0.072|
|暗号 V0|0.565|0.636|0.762|0.843|
|暗号 V1|0.352|0.636|0.762|0.315|
|暗号 V2 k1|0.331|0.602|0.789|0.291|
|暗号 槽位候选|0.324|0.590|0.785|0.309|

V1降低K的位置距离，V与V0逐项相同。V2只改变末槽位，其余内容K/V保持V1；候选只改变索引0/0/3，其余槽位保持V1。TOKEN/全消息重算的均值K relative L2为约0–0.006、V约0–0.012，首logits约0.015–0.021，三组输出IDs与独立oracle一致。独立oracle自身距离0仅是定义，没有收益意义。

分层并非所有高层都恢复：秘密V1第14层V relative L2最大1.178/1.196、cosine最低0.473/0.466；暗号第12层最大1.227、cosine最低0.406。第一层V relative L2=0；第23层仍为0.583/0.559/0.671。不能把8B论文层范围硬编码迁移为本模型校准结论。

|V1内容索引|17/26 V relative L2|17/31 V relative L2|暗号 V relative L2|
|---:|---:|---:|---:|
|0|0.570|0.563|0.556|
|1|0.629|0.637|0.562|
|2|—|—|0.650|
|3（首数字5）|—|—|0.666|
|4|—|—|0.667|
|5|—|—|0.650|
|6|—|—|0.702|

没有“首数字Token拥有最高偏差”的规律，秘密两个Token都显著偏离，暗号残差也分布于全部内容。V3数字类别是低成本**任务启发式**，没有证明其重要性预测有效。数字槽位的误差和首logits距离降低，却使Receiver丢失自己的数字，说明数值距离与语义可靠性不等价。没有用oracle误差排名在线选位置。

首次输出分叉索引按完整生成IDs、0-based：V0三组均0；V1三组均0（暗号V1仅多“暗号是”仍复述正确）；V2 k1为0/5/0；槽位候选为0/11/0。首logits Top1是否相同、max/mean距离和所有完整IDs见outputs（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/outputs.json`）。分叉位置或BF16误差不能直接替代质量评分。

## 6. 两种候选与实现门槛

**候选A：首数字Token槽位修复。** 在线扫描已有content IDs，单ID decode、strip，选首个 `isascii() and isdigit()` 的Token；固定修复一个，−号/小数或Unicode数字不视作该规则的数字，无匹配则明确失败。不根据答案/Full Prefill挑位置，不发送额外hidden状态。由于所选位置0/0/3不是V2尾端1/1/6，可验证实质差异；任意索引的底层裁剪/写回逻辑仅由此固定规则调用，不提供任意用户索引或参数搜索。

**候选B：中间层[8,16)修复。** 用离线完整Receiver oracle的该层消息K/V覆盖V1，再重算closing并Decode。范围在实验前固定，不是事后选择最高误差层；这是作用探针，不是正式算法。新计算状态若不用oracle，就需计算下层hidden或另传Sender hidden、适配检测层/query mask/write-back。相同整个消息的跨层hidden也可能携带Sender历史偏差。本轮Oracle干预尚且未恢复两秘密，暗号反而输出虚构数字；不启动大规模forward重写。

|门槛|候选A|候选B|
|---|---|---|
|明确非空算法|固定数字类别选择与单槽位真实修复|只有反事实Oracle干预；在线生成正确h_l未定义|
|区别于V2|保留尾段、修复较早槽位|层级覆盖区别存在，但不等于在线重算|
|真实计算操作|embedding、24层Q/K/V/Attention/MLP、最终norm/head；forward1Token|需下层dense计算、选定层计算及上下层状态管理|
|依赖|正确本地历史+此前混合KV；后续槽位保持近似|需要缺失hidden states/特定forward或昂贵重建|
|位置/mask|私有cache裁至start+j，标准连续单Tokencausal mask；scatter原槽位|普通append不能执行非连续query+逐层写回|
|可控接口|现有continue_from_cache与DynamicCache私有重建|旧官方patch与5.17不兼容；本轮不port|
|A/B独立状态|全部修复在工作副本，双端validate后提交|探针隔离，未接生产|
|无正式Full oracle|通过|未通过：当前探针必须Full Prefill|
|成本记录|forward1Token×24层、完整wire和槽位阶段时间|全Prefill335Tokens成本显式收费，不能只报8层copy|
|验证对照|同Sender/Receiver快照多策略、tiny数值/事务、正式真实bytes|已完成Oracle作用对照，不足以正式实现|

因此**A通过10项技术执行门槛，进入一个最小实验实现；B不通过在线状态/接口/oracle门槛，不实现**。A质量失败不被改写成技术不可执行；按用户允许的“质量未改善也可完成工程研究”交付。没有第二个正式V3或无限调参。

## 7. 正式V3调用链与事务

[communication](../src/token_kv_agent_system/communication.py) 新增 `NUMERIC_SLOT_DELTA`；V2参数仍仅属于V2。

```text
CLI --kv-strategy v3 / REST mode=kv:v3 / UI手动V3 / mixed plan
 -> DiscussionRuntime或SecretSumRuntime
 -> StatefulTokenSession.handoff_kv
 -> 原prepare_incremental_message / ModelContract / TKVI bytes / Transport
 -> receive_incremental_kv: 验证、数值Token选择、restore、工作副本、Receiver旧draft关闭
 -> 本地incoming prefix真实forward -> V1整消息RoPE K修正 -> append完整delta
 -> token_slot_repair.repair_slot:
      clone/crop前缀至start+j (不含所选位置或未来)
      continue_from_cache((content_id[j],), length=start+j)
      每层复制新K/V的slot[start+j]到完整工作Cache私有副本
      保留其他内容槽位，长度保持start+N
 -> Receiver closing真实forward -> validate / storage disjoint
 -> Sender关闭与双端validate成功 -> 原子提交
 -> Receiver Stage/assistant prefix/真实Decode使用新Cache
```

[修复核心](../src/token_kv_agent_system/model/token_slot_repair.py)、[Receiver](../src/token_kv_agent_system/incremental_kv_receive.py)、[Session](../src/token_kv_agent_system/stateful_token.py)。没有global monkey patch或重写模型；模型权重仍共享，长期Cache/ledger不同。内部裁出的prefix只供局部计算，**不是提交Receiver被截断的状态**。

选择索引为消息内0-based，目标位置=start+j；原sender span只用于整消息V1位置修正。mask是全1、长度start+j+1，内部causal规则只允许过去/当前位置。修复Token的下层hidden现场重建；未缓存/未传输辅助hidden。后续M槽位不重算，属于明确的近似不一致，而不是完整语义修复。所有层原始非选槽位精确保持，closing在修复后真实执行。

仍发送完整TKVI/TKVT/TKVK v1包，contract/checksum/source/target/sequence/非空/上下文预算守卫不变，无TOKEN/V1/V2 fallback。无数字时在restore和本地forward前拒绝，已发生的bytes传输记为attempt，不能宣称没有通信成本。任何repair/closing/Sender close异常都不提交两端工作状态；draft可保留供显式重试。

新增receipt/log：`repaired_token_indices`、`repaired_layer_indices`、`repair_layer_token_count`；V2同样补充实际尾段索引和全部层，仅增加观测，旧算法不变。V3 `receiver_message_prefill_count=1`、`recompute_forward_count=1`、24层×1Token。`token_selection`和`slot_repair`阶段独立计时，slot阶段含crop、真实forward、copy/reconstruct/validation，完整handoff包含这些费用。

REST/计划支持 `kv:v3`、拒绝V2数量/比例用于纯V3；mixed仅对V2项应用这些字段。UI仅增加手动选项/实际策略名称/receipt工作量，不重做布局；说明已观察质量退化、无数字会失败。Discussion AUTO仍禁用。

## 8. 真实工程与质量验证

先完成17/26正式V3，再检查真实forward/shape/位置/生成，随后新子目录完成17/31及暗号。正式初始Tensor快照hash、Sender IDs、KV packet SHA与诊断逐项相同；正式B输出完整IDs也逐项等于离线槽位候选。这是实现一致性复核，**不是新的统计成功率样本**。反向自然使用各分支实际生成的B消息，不能声称不同策略反向输入相同。

诊断27次真实生成、21次bytes交接；正式8次生成、5次KV bytes交接；合计35生成、26交接。ResourceMonitor共641次真实forward（497+60+84），含初始/本地framing/Oracle/Decode。正式V3修复5个内容Token×24层=120层×Token工作；不将Decode或初始化forward混为5Token成本。离线候选另有3个槽位forward以及完整oracle成本，分别记录。

|策略|17/26 B|17/31 B|暗号B|
|---|---|---|---|
|TOKEN / 全消息重算 / 独立Full|A17、自己26、43|A17、自己31、48|完整紫色袋鼠5729|
|V0|只输出26，无完整归属/求和|31+31=62，错误|1234，错误|
|V1|26+17=43，缺明确归属|31+17=48，缺明确归属|完整暗号（多引导词）|
|V2 k1|同V1缺归属|A31/自己17颠倒，和48不代表成功|保持V1已有正确|
|V3槽位修复|A17、自己unknown，失败|A17、自己unknown，失败|仅5729，丢中文|
|中间层Oracle|26+17=43，仍缺归属|31+17=48，仍缺归属|“1234567890”，虚构暗号|

两组正式V3最终原文：

- 17/26：`Agent A's number is 17. Agent B's number is unknown. 17 + unknown = 17`。A保留17并读到B的unknown，但丢失B26，最后把未知数求和为17，错误且矛盾。
- 17/31：`A's number is 17. My number is unknown. 17 + B's number = unknown`。B31缺失；A声明自己的数字unknown又声明A17，归属/自身保留矛盾。

B两例能读到A17，却没保留自己的System数字；没有重复询问已收到数字，没有虚构第三Agent消息来源，但这些局部合格项不能抵消归属/求和失败。暗号数字子串正确不等于完整秘密正确。完整原文/IDs/逐维人工判读见outputs（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/outputs.json`）、assessments（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/assessments.csv`）。旧V2最终交流结果沿用旧报告，不假装本轮重跑了完整旧矩阵。

工程方面：24层/[1,2,S,64]/torch.bfloat16/MPS:0、长短span/长度mask、contract bytes、实际Receiver Decode、A/B存储独立与工作快照未改变均经后验审计。tiny CPU是独立工程测试，不充当本机真实MPS证据。新正式run没有使用oracle，没有静默fallback。

## 9. 实际成本、资源及局限

A→B/反向明细（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/costs.csv`）及逐阶段receipts（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/outputs.json`）。单位ms，单次instrumented测量，不是steady-state benchmark；后两例正式运行与tiny CPU回归/前端检查有短暂重叠；不同正式run之间有系统负载变化，不据小差值排名速度。

|用例/正式V3方向|wire B|槽位修复|完整handoff|Receiver Stage+prefix+Decode forward总时长|
|---|---:|---:|---:|---:|
|17/26 A→B|28036|18.30|172.52|248.71|
|17/26 B→A|175542|17.85|161.15|447.12|
|17/31 A→B|28036|18.71|166.00|252.25|
|17/31 B→A|163250|26.87|220.25|456.28|
|暗号 A→B|89496|18.34|164.10|180.70|

同Sender包，V0/V1/V2/V3完整wire SHA相同；秘密TOKEN22B、暗号TOKEN42B。**V3新增wire metadata=0B**：规则固定、选择在本地、receipt索引走Control Plane；既有envelope/IDs/header字节仍包含在总wire内，没有省略。receipt新增三个修复字段的紧凑JSON表示增量为148B/交接（五次均相同），本轮Harness仅本地保存；Control Plane实际网络成本未测量，不能把data-plane新增0B推广为全系统无额外metadata开销。反向bytes增大来自B实际14/13Token内容，不是相同消息策略对比。

同轮诊断V2 k1修复约15.92–18.59ms；正式V3约17.85–26.87ms，都会多执行1Token×24层。V3额外需要crop+槽位scatter+完整Cache拷贝，不能称其分层省算；K/V encoding、restore、RoPE、append、closing、Sender close、selection、slot repair及总handoff在receipt均单列。selection采用逐ID decode，O(N)，未运行注意力评分/Full oracle，新增传输hidden=0B，辅助hidden专门forward=0；所选Token全部层的内部hidden计算已经收费。由于标准CausalLM forward也执行最终norm/lm_head，这些操作未人为排除。

Receiver列从持久化ResourceMonitor日志按**顺序匹配**ForwardTrace，汇总真实同步backend_forward_ms，包含Stage/prefix/Decode；CSV另给first forward开始至last forward结束的instrumented窗口（270.27/487.71/276.33/506.76/201.37ms）。不把窗口称为纯Decode吞吐，也不把生成长度改变当作handoff加速。未单独测量无观测生产吞吐、物理网络与完整会话端到端wall，明确 `not_measured`。模型合约首次权重hash为预先准备的启动费用；不冒充免费操作。

Oracle干预的完整prefill时间/335输入Token成本在source/export中显式保留；诊断分支copy计时不含前面oracle时间，`oracle_cost_excluded_from_branch_ms=true`明确标注，不能用它宣称8层稀疏重算。Oracle槽位正确不构成全输出质量或计算性能的数学上界。分层真正在线实现还需检测密集层与辅助hidden/协议成本，本轮未测量。

资源采样峰值：诊断MPS current0.985GiB/driver1.033GiB/RSS564.5MiB；首次正式0.941/1.024GiB/RSS539.2MiB，后两例0.948/1.033GiB/RSS558.8MiB。pressure均1/2；诊断Swap采样新增11.69MiB，正式两次0新增；无critical、OOM、保护中止或CPU fallback。资源每2秒采样可能遗漏瞬时峰；系统压缩/Swap与进程/MPS不是同一作用域。没有为了降低采样数字杀掉其他进程。

本轮仅三条A消息、短内容、单模型、单机、greedy冻结运行；未证明重要性规律、可重复质量优势或性能收益。层Oracle失败不证明所有层范围/稀疏算法不可行；不继续改变层范围/数字规则/Prompt寻找有利结果。V3技术执行成功，**质量相比V1/V2退化，不接近TOKEN，成本合理性和可靠收益均未成立**。

## 10. 证据审计、测试与失败记录

[只读exporter](../export_kv_v3_diagnostics.py)核查真实日志位置mask/shape、相同输入/packet/hash、正式源码冻结hash（两个生产spec均无漂移）。原诊断spec对应修改前源码；抽出生产helper和接入Runtime后的漂移逐文件显式列出，不重写旧spec。小型Git证据约0.57MB，raw日志/源码副本在忽略目录；原始SHA索引（原研发资料，未随公开副本提供：`kv-v3-79c/evidence/raw-index.csv`）覆盖所有raw与失败记录。

保留过程中的非模型错误：新增测试最初把tiny tokenizer数字display当字母、oracle测试传入整历史而非消息span、CLI测试patch错符号、export测试未通过文件路径导入根脚本；修正测试fixture后通过。首次人工审计要求tensor.device恰等于`mps`，实际为`mps:0`，已改成同设备族核查；首次export错误要求相同生成trace只出现一次，实际有TOKEN/Full/Oracle重复，改为按forward日志时间顺序匹配。两者均没有重跑模型或改模型输出。新CSV默认CRLF在暂存diff核查时被判尾部空白，已统一LF并修正export writer，标量/评分/原始SHA值不变。测试旧“V3必非法”更新为仍未实现的V4，其余旧无fallback/数值/事务断言保持。

测试覆盖在线规则/无数字拒绝、对齐指标/KV分离/finite、真实tiny crop/位置/mask/所选slot数值oracle/其他槽位不动、k参数互斥、EOS/EOT/截断/pending、失败后双端Cache不提交、contract/损坏bytes拒绝、双向/mixed/后续Decode、CLI dispatch、REST receipt/summary和前端提交/标签。最终Python **857 passed / 1原可选AutoGen skip / 2既有上游warning**；相关KV/诊断回归 **217 passed**；Frontend **118 passed / 7 files**，TypeScript与Vite build通过。compileall（src/tests/根CLI）、诊断/export/Discussion/SecretSum/Dashboard CLI help、文档/源码/raw SHA核查与git diff --check通过。

复现（必须新目录，示例不会下载模型）：

```bash
HF_HUB_OFFLINE=1 .venv/bin/python run_kv_v3_diagnostics.py --output-dir benchmark-results/kv-v3-79c-new
HF_HUB_OFFLINE=1 .venv/bin/python run_kv_v3_diagnostics.py --phase production --case secret-17-26 --output-dir benchmark-results/kv-v3-79c-new/production-first
HF_HUB_OFFLINE=1 .venv/bin/python run_kv_v3_diagnostics.py --phase production --case remaining --output-dir benchmark-results/kv-v3-79c-new/production-remaining
# 新目录复用已核实的固定revision源码快照（不含模型）
cp -R benchmark-results/kv-v3-79c/sources benchmark-results/kv-v3-79c-new/sources
# export要求所列raw sources/index.json一并保留（官方固定revision源哈希在audit.json）
.venv/bin/python export_kv_v3_diagnostics.py --results-dir benchmark-results/kv-v3-79c --output-dir /tmp/kv-v3-new-evidence
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_demo.py --mode kv --kv-strategy v3 --rounds 1
# 若生成消息没有ASCII数字Token，V3明确失败；不是任意普通讨论的可靠策略。
```

## 11. 最小下一阶段系统集成计划

1. 为增量 `StatefulTokenSession` handoff统一TOKEN与V0/V1/V2/V3的实测ledger：tokenization、pending/export、encode/decode、restore、RoPE、repair、完整handoff、Receiver continuation，记录样本长度/设备/扰动。TOKEN需要补齐同口径阶段时间；V3默认质量风险拒绝。
2. 增加可重放的带宽时间序列和固定latency模型，区分模拟值与物理实测；每次handoff前读取当前网络状态，而不是一整轮复用初始化带宽。
3. 增量候选成本 = 本次真实packet估计字节/当前带宽 + 本地handoff成本 + continuation成本；先应用质量风险门槛，再比较成本。质量优先可始终选择TOKEN；V3未证实可靠，默认不自动选择。
4. 将决策真正接入 `DiscussionRuntime` 每次消息handoff，复用独立A/B状态/原子失败规则。现有Canonical Selector选择的是clean KV(M)，**不能将旧Canonical KV参数当增量KV(M|H_A)成本**。明确分开的profile/候选类型，完成增量适配后再桥接，禁止只改UI标签。
5. Dashboard事件投影实际带宽、候选成本、质量排除项、决策原因、请求/执行策略和wire bytes；沿用当前布局和REST/WS，不开发第二套Runtime。
6. 正常讨论预算与冻结benchmark96分开配置，解决128Token过早截断；冻结实验Prompt/预算/评分不随产品预算修改。

**最终决策：A — V3最小实现成功，可作为新增实验策略。其质量和性能价值未成立，保留手动实验能力并冻结V3研究，直接推进动态网络模拟与增量自动Selector；不得让V3的负面质量结果阻塞系统主线。**


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
