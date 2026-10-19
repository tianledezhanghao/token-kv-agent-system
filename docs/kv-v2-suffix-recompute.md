# 7.9B — Bounded V2 contiguous suffix recompute

2026-10-14，起点 `bfafe7788de46b0308aa0c2faa74c4951830bb55`（已含8.1前端），不是回退到7.9A审计提交。本轮完成真实可执行 V2；**部分尾段重算没有获得稳定语义改善，也没有证明性能收益**。V0/V1既有协议、旋转公式、Prompt与模型Backend行为保持。

|结论层次|本轮证据|
|---|---|
|已实现|正式 Session / Discussion / SecretSum / CLI / REST / UI 手动 `kv:v2`，真实尾段 forward|
|工程测试通过|新增V2测试，旧V0/V1无消息forward/协议/事务断言保留，全量回归见末节|
|真实模型验证通过|固定0.5B/MPS/BF16，3个真实Sender、24条接收分支、40次bytes交接、43次生成|
|语义质量改善|部分V2未稳定改善；k=N全消息参照恢复本轮TOKEN正确结果，不能归为稀疏收益|
|性能收益|未证明；完整KV通信量不变，非零尾段多一次真实forward|

原始结果在新目录 `benchmark-results/kv-v2-79b/validated/`，历史benchmark零覆盖。Git保留冻结配置（原研发资料，未随公开副本提供：`kv-v2-79b-frozen-spec.json`）、原始文件SHA索引（原研发资料，未随公开副本提供：`kv-v2-79b-index.csv`）、完整输出及核心IDs（原研发资料，未随公开副本提供：`kv-v2-79b-raw-outputs.md`）、质量人工表（原研发资料，未随公开副本提供：`kv-v2-79b-assessments.csv`）、成本明细（原研发资料，未随公开副本提供：`kv-v2-79b-costs.csv`）、工程后验审计（原研发资料，未随公开副本提供：`kv-v2-79b-audit.json`）。大型原始JSON留本机忽略目录，无权重入Git。

## 1. 算法与范围

V0原样复用 Sender 内容delta。V1将每层post-RoPE K从Sender绝对位置修正到Receiver目标位置，V不旋转。本轮V2先对**整条收到的delta**执行相同V1修正，再保留前N−k个位置，以Receiver真实模型forward重新计算连续末尾k个内容Token的K/V：

```text
H_B = Receiver independent history + actual incoming source prefix
retained = append(H_B.cache, rephase(Sender_delta)[:N-k])
V2 = continue_from_cache(M[N-k:N], retained)  # real causal-LM forward
then local closing -> Stage -> assistant prefix -> greedy Decode
```

这是 **contiguous suffix recompute / 连续尾段选择性重算**。前段KV仍受H_A条件化，尾段forward看到的是混合Cache；它通常不等价于Receiver完整Prefill(M)，不能修复所有上下文差异。不支持任意Token、任意层、非连续scatter write-back或hidden-state修复。选择连续尾段，是因为当前Backend已可靠支持连续append，能够在最小生产变更内得到真实计算和事务隔离。

## 2. 最小源码调用链

[communication.py](../src/token_kv_agent_system/communication.py) 新增 `KVStrategy.SUFFIX_RECOMPUTE_DELTA='suffix_recompute_delta'` 与 `HandoffChoice.parse('kv:v2')`。

```text
CLI / DiscussionStartRequest.runtime_config
 -> DiscussionRuntime / SecretSumRuntime
 -> StatefulTokenSession.generate_message (actual Sender Decode)
 -> handoff_kv(strategy, recompute_tokens / recompute_ratio)
 -> prepare_incremental_message (ledger span, EOS exclusion, pending completion)
 -> encode_incremental_message_with_metrics -> InMemoryByteTransport(bytes)
 -> receive_incremental_kv -> decode -> restore DynamicCache
 -> work clone -> close Receiver previous draft -> local incoming prefix forward
 -> rotary_tables + rephase_key for complete delta
 -> delta_prefix(N-k) -> append_delta (if retained > 0)
 -> append_working_ids(content suffix) -> backend.continue_from_cache (if k > 0)
 -> local closing forward -> validate
 -> Sender private close -> validate both -> atomic state commit
 -> Receiver Stage / Decode -> reverse handoff -> local final generation
```

核心改动在[接收入口](../src/token_kv_agent_system/incremental_kv_receive.py)、[Cache适配](../src/token_kv_agent_system/model/delta_repair.py)、[Session](../src/token_kv_agent_system/stateful_token.py)。复用[既有真实forward](../src/token_kv_agent_system/model/huggingface.py)及[工作态append](../src/token_kv_agent_system/agent/state.py)，未重写Backend、生成循环、协议或自动Selector。

## 3. k / ratio配置和边界

`HandoffChoice` 的 `recompute_tokens` 与 `recompute_ratio` 互斥，仅V2可设置。两者均为None时默认k=1。

- 数量：`k=min(requested_tokens,N)`，非负整数；**明确饱和裁剪**，不是静默不一致。bool、负数、非整数拒绝。
- 比例：`k=min(ceil(N*ratio),N)`，有限ratio∈[0,1]；NaN/Inf、bool、越界拒绝。
- N=0：既有非空TKVI协议明确拒绝，无TOKEN/V1 fallback。
- N=1：默认/正比例重算1；k=0保留V1分支。
- k=0：完整V1 append，无内容forward，receipt重算0/forward0。单测K/V精确等价，真实三组V1与k0输出IDs一致。
- k=N：不append任何外来内容KV，用Receiver工作Cache forward全部N内容IDs；仍传输/解包/恢复/旋转整包KV，因此不是带宽优化或稀疏计算。

```bash
# 正式CLI默认仅缓存模型；示例不会下载
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_demo.py \
  --mode kv --kv-strategy v2 --kv-recompute-tokens 2 --rounds 1
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_demo.py \
  --rounds 2 --handoff-plan token,kv:v2,kv:v1,kv:v2 --kv-recompute-ratio 0.25
```

REST字段遵循既有KV前缀：`mode='kv:v2'`、`kv_recompute_tokens=1` 或 `kv_recompute_ratio=0.25`。Mixed计划中的所有V2项共用这两个请求级字段，其他模式保持自身路径；没有V2项却给重算字段时拒绝。Python每个HandoffChoice可有不同策略。SecretSumConfig直接复用相同choice/plan，不改变原System/Public/Stage。

```json
{"scenario":"secret_sum","secret_a":17,"secret_b":26,"question":"exchange","rounds":1,"summarizer":"A","max_new_tokens":96,"mode":"kv:v2","kv_recompute_tokens":1}
```

UI启用实验性V2及数量/比例输入，mixed也可选；显示实际 `suffix_recompute_delta` 为KV V2，receipt/detail显示实际重算数与forward数。V3/AUTO仍禁用，无Dashboard视觉重设计。部署中原8000服务仍是旧进程，使用新能力需正常重启加载新代码；本轮没有终止用户保留的服务。

## 4. Cache、位置、mask与ledger

DynamicCache层为 `layers[*].keys/values`，K/V布局BHSD，sequence维=2。`delta_prefix`验证所有层shape/dtype/device一致与count边界，返回前段view；`append_delta`沿dim2拼接并以公开 `DynamicCache(ddp_cache_data=..., config=...)` clone重建私有存储。没有原地crop已提交Cache。k=N直接复用已隔离的Receiver工作Cache，不构造零长度外来Cache追加。

本地prefix真实forward后才确定目标start，`shift=start-sender_span_start`。复用原V1静态Qwen2/Llama/full-attention守卫和FP32旋转计算，最后cast为原dtype，V不旋转。当前0.5B为24层、2 KV heads、head_dim64、BF16/MPS。

保留前段后工作ledger只追加对应前段IDs，cached_count严格=start+N−k；尾段 `append_working_ids`输入恰好末k个内容IDs，执行一次multi-token真实forward，返回count=start+N与追加全部内容的ledger。现Backend按Cache长度推导连续position_ids，attention mask为全1、长度=(历史+前段+k)。k0不调用此forward。closing在内容end之后，再由Stage/Decode正常继续。

EOS不进消息内容；Sender截断最多一个accepted pending内容Token在Sender私有副本补forward，不计入Receiver重算。Sender EOS/pending/span的既有检查不变，不能以metadata冒充计算。单测覆盖EOS/EOT、截断、N=1、不同历史长度、Receiver旧pending、三轮双向/mixed与后续Decode；真实本轮43条生成均EOS，截断真实验收沿用旧证据，V2截断新验证属于tiny CPU工程测试。

A/B权重可共享，但长期Cache、ledger、System和工作态存储独立。快照hash后验确认每条分支都未修改基准A/B Cache。Receiver闭旧draft、incoming prefix、保留/重算/closing均在work中；末尾Sender关闭成功、两端validate与storage-disjoint后才同时提交。suffix forward即使已原地修改work后抛错，旧两端state/sequence/draft均保留，不发成功receipt、不自动TOKEN或V1重试。bytes已传输仍可记失败attempt，不能假造未发送或成功接收。

## 5. Wire成本和观测字段

TKVI envelope v1、TKVT v1、TKVK v1完全不变。包仍含**全部N内容IDs及所有层完整N位置K/V**，并沿用checksum/model contract/source/span验证。V2只在Receiver选择k；相同Sender同包的TOKEN/V0/V1/V2分支经正式bytes路径，没有传Python Tensor引用绕过Transport。KV分支packet SHA逐一相同，包含k=N也相同。

Receipt `receiver_message_prefill_count`现在是实际消息内容重算数量（V0/V1为0，V2为k），`recompute_forward_count`为0或1；HandoffLog额外投影 `receiver_recomputed_token_count`，事件/REST summary报告V2交接数与实际数量。默认不打开阶段计时；Python `handoff_kv(...,measure_timing=True)`返回 `timings_ms`。未计时为None，不填0；k0无重算、V0无RoPE、k=N无append等**实际跳过**阶段才标0。

计时拆分：export（含pending/extract）、encode、local_transport、decode、restore、work_clone_and_previous_close、receiver_prefix、rope_rephase、cache_append_copy、suffix_recompute、receiver_closing、sender_close、handoff_total。tensor_extraction与sender_pending是export子项，不能重复相加。handoff_total边界为已生成Sender draft后的Session执行，至Sender close与Receiver验证后的记录准备；最终metadata/原子赋值末尾有极小未单独分解开销，另有Harness外层完整handoff wall time。模型首次合约hash在本实验已预先准备，不纳入稳态handoff表；它是另有成本，非免费。

## 6. 实验冻结与公平性

[Harness](../src/token_kv_agent_system/kv_v2_validation.py)与[启动入口](../run_kv_v2_validation.py)在加载模型前保存完整frozen-spec：算法、固定8条路径、model revision、3用例、评分、96预算、输出目录、资源阈值及生产源码/测试hash。成功spec SHA256为 `f45f36d34239a2d1a09a0b5b1f52cd133ca708c3a5aafdd784f0318382c1397e`。

Qwen2.5-0.5B-Instruct，revision `7ae557604adf67be50417f59c2c2f167def9a775`；本地缓存路径加载，offline/no download，MPS、torch.bfloat16、batch1、seed77、greedy、96 max_new_tokens、每端4096上下文。Torch2.14.0、Transformers5.17.0，一份本轮权重，A/B独立Cache，未SSH/访问服务器/加载3B/改Swap或MPS限额。

两秘密用例17/26、17/31沿用完全相同的原SecretSum System/Public/Stage。每组A仅真实Decode一次，content IDs `[16,22]`（“17”），EOS排除；所有Receiver从同初始state快照分支，generation完整ledger逐ID相同，独立模板校验通过。source内容[122,124)，target[106,108)，位移−16，B Cache97→110。k2=N，**必须归为全消息重算**。

长消息复用7.7C冻结的私有紫色袋鼠5729用例与当前兼容Prompt，因历史磁盘JSON没有可恢复Tensor快照，进行一次新的真实Sender Decode，不用fixture。内容7IDs `[111413,100509,101547,20,22,17,24]`，source[144,151)，target[112,119)，位移−32，B Cache103→121。k1/k2/25%分别1/2/2，真正0<k<N；其余内容仍来自Sender上下文。

每分支继续真实B生成；两秘密组再真实B→A及A Final。**只有A→B完全同输入**；反向B消息允许按策略自然变化，不能把多轮不同B内容当单步same-input比较。k=N参照只全算当次消息，双方此前仍各有本地历史；本轮full分支每向都全算全部实际消息。TOKEN保持既有prefix+content+closing合并forward，V2分段forward可有BF16误差，不要求逐bit，也不声明普遍与TOKEN相同。

首次启动保存在父目录，因Harness误将metadata `torch.bfloat16`与`bfloat16`比较而生成前断言停止，0 forward，无资源异常。只修正启动检查，成功矩阵在新的 `validated` 子目录重新冻结；没有模型质量结果后调Prompt/k/预算/用例。两次证据都在SHA索引，不能删失败冒称首次成功。

## 7. 真实质量结果

每条完整原文/IDs在原始输出（原研发资料，未随公开副本提供：`kv-v2-79b-raw-outputs.md`），按冻结归属、数字、求和、重复索要、来源、最终整合、矛盾逐项人工评估，保守旧parser原样保留，数字正确不替代信息成功。

|策略|17/26 B / A最终|17/31 B / A最终|严格完整交流|
|---|---|---|---|
|TOKEN|明确A17/自己26/43；最终归属正确43|明确A17/自己31/48；最终归属正确48|两组通过|
|V0|B仅26；A无归属26+17=43|B31+31=62；A裸62|两组失败/证据不足|
|V1、V2 k0|B无归属26+17=43；A17+17=34|B无归属31+17=48；最终归属正确48|两组未完整通过|
|V2 k1|与V1同失败34|B称A31/自己17（颠倒）；最终归属正确48|两组失败|
|V2 k2|B全消息重算归属正确；反向只重算2后A颠倒归属，和43|同理A颠倒归属，和48|两组失败；不能归为partial修复收益|
|V2 25%|A→B实际k1，B无归属；反向k3后A仍34|A→B实际k1，B归属颠倒；反向k7后A最终正确48|两组失败|
|V2 full（两向k=N）|B/最终归属和43均正确|B/最终归属和48均正确|两组通过，仅全算参照|

本轮所有B均未重复索要数字，但部分缺乏归属/错误信息整合；不重复索要本身不构成交流成功。V2k1的17/31错误归属与自己的System冲突，最终正确数字也不能覆盖。k2两组A Final交换归属，和在交换律下碰巧正确。未观察虚构第三方消息来源，但错误所有权仍属来源绑定失败，分别记录。

暗号：TOKEN输出“紫色袋鼠5729”；V0“1234”；V1/k0/k1/k2/25%均“暗号是：紫色袋鼠5729”；full“紫色袋鼠5729”。部分V2保持V1已有复述能力，不能称新增改善。k0三组输出逐ID等于V1；first logits Top5、argmax、共同输出前缀与首分叉均保存在raw。所有生成EOS终止；无silent fallback。

## 8. 实际计算与通信成本

共43次真实生成、40次handoff（35 KV / 5 TOKEN）；V2累计重算 **90内容Tokens / 20次suffix forward**，包括full和短消息k=N边界；不能将90都称部分选择性修复。25个V2交接中5个k0没有消息forward。922次资源守护forward；909次带完整position/mask的审计trace（另13次为初始化/本地input）；全部已审计forward输入长度、实际rotary位置、全mask、返回Cache长度正确，无错误receipt。

同一秘密消息：TOKEN22 B，所有KV分支 **28,036 B**。长消息：TOKEN42 B，所有KV分支 **89,496 B**。与历史KV包数字差异来自冻结本地checkpoint路径作为model_id的metadata长度，本文只比较本轮同包，不称为带宽变化。V2没有减少任何本轮KV字节。

以下为A→B单次同步测量，单位ms；不是中位数/长期Benchmark。Receiver生成列含Stage、assistant prefix、Decode及诊断trace。详细所有阶段见costs.csv（原研发资料，未随公开副本提供：`kv-v2-79b-costs.csv`）。

|用例/策略|实际k|V1旋转|V2重算|handoff_total|Receiver Stage+Decode|
|---|---:|---:|---:|---:|---:|
|17/26 V1|0|51.65|0|154.31|202.83|
|17/26 V2 k1|1|50.55|16.11|147.74|203.11|
|17/26 V2 k2=N|2|52.34|22.98|160.26|445.12|
|17/31 V1|0|50.57|0|133.46|205.65|
|17/31 V2 k1|1|50.48|16.07|152.85|448.96|
|17/31 V2 k2=N|2|64.02|19.94|164.88|449.10|
|7Token V1|0|52.76|0|165.26|222.87|
|7Token V2 k1|1|50.37|16.05|146.91|218.36|
|7Token V2 k2|2|51.24|20.11|152.89|218.50|
|7Token V2 25%|2|51.11|17.79|153.03|238.98|
|7Token V2 full|7|51.09|20.37|157.26|156.13|

k1实际额外模型重算约16ms；k2约18–23ms，本轮所有V2重算阶段见CSV，不把k=N成本记为不足N。V2虽有更多计算，单次total偶尔小于V1，来自export/prefix等波动（例如长消息V1 prefix43.34ms）；不能宣称V2更快。Full输出更短/更长也改变Decode总成本，不能据此归算法计算加速。

本地阶段与模拟网络分别报告：实际InMemory交付；模拟20/1000Mbps加5msbase latency由真实bytes推导，不sleep；物理网络 `not_measured`。KV秘密模拟16.2144/5.224288ms；暗号40.7984/5.715968ms。算法纯无诊断耗时、真实网络及长期steady-state统计均 `not_measured`，不以0替代。

阶段计时有显式MPS同步；ForwardTrace拷贝IDs/mask/实际positions至CPU，ResourceMonitor对每forward持久化，PacketEvidence对bytes做SHA，因此local_transport也含哈希证据开销。外层handoff wall含这些观测，重算列含工作态validation；无法单独剥离测量扰动，不能作为生产吞吐结论。export已包含host copy，restore已包含upload，不能遗漏CPU/设备边界。

资源监控每2秒采样，所列为采样峰值，可能漏掉瞬态峰；沿用7.8A-R分级：压力1/2（normal/warning），无critical、OOM、保护中止或CPU fallback；swap全程3578MiB无新增，MPS current峰约0.949GiB、driver峰1.033GiB，进程RSS峰约597MiB。系统压缩/Swap与进程/MPS不同作用域，不能相加。旧Dashboard模型进程仍保留；没有终止其他用户进程，本轮权重顺序单份加载。

## 9. 局限与V3待解决事项

只验证本机0.5B密集full-attention；3B KV、Llama/CUDA、跨进程/GPU/物理网络未验收。N=7是有界非全算对照，不是长上下文泛化；两秘密不代表成功率样本。未做新Prompt搜索、额外k搜索、层稀疏修复或自动Selector。

V2保留的V及高层K仍含Sender历史语义，重算末Token并不消除此偏差；后续生成可能利用不一致信息。Full目前仍付完整KV transport/restore/rephase成本，仅作为计算参照，不是合理高效协议。未来若只传前段需要新协议设计，本轮未做。

V3需要先定义修复目标/质量门槛、层/Token依赖、hidden-state或重建信息、position/mask/write-back、带宽账与事务；自动选择前需统一质量约束与完整端到端成本、更多独立上下文/长度/模型验收。不要因为本轮forward和Cache长度通过，就推定语义或稀疏最优性。

## 10. 复现与验收

```bash
# 必须新目录；已有目录包括失败/partial都拒绝覆盖
HF_HUB_OFFLINE=1 .venv/bin/python run_kv_v2_validation.py \
  --output-dir benchmark-results/kv-v2-79b-new
# 只读验证冻结source hash、bytes、位置mask、kforward与生成数量，输出小型Git证据
.venv/bin/python export_kv_v2_validation.py \
  --results-dir benchmark-results/kv-v2-79b/validated --docs-dir /tmp/kv-v2-evidence
HF_HUB_OFFLINE=1 .venv/bin/pytest
cd frontend && npm test -- --run && npm run build
```

冻结hash校验要求对应实现checkout；未来代码变化应显式报告漂移，不能重新写旧spec绕过。ResourceConfig已写入spec，默认监控不允许OOM后重试/下载/CPU替代/超配。

本轮Python **824 passed、1原可选AutoGen skip、2既有上游warning**。生产源码冻结后只新增Receiver边界工程测试，没有改变算法或实验条件。V2新工程测试62项（主文件52、Receiver旧pending补充10）（含quantity/ratio、真实tiny CPU数值oracle、k0/V1、kN、prefix shape/dtype、位置mask、pending/EOS、失败回滚、三轮mixed、API/CLI与same-Sender harness）；旧测试只将V0/V1无内容forward范围明确限定两旧策略，并移除过时“kv:v2一定非法”的假设，不放宽历史KV/质量断言。Frontend **116 passed / 7 files**，TypeScript与Vite build通过。KV针对性338 passed，另Receiver旧pending/EOS与非法k在传输前拒绝10 passed；compileall（src/tests/所有根CLI）、Discussion/新V2/SecretSum/Dashboard CLI help、git diff --check通过。

全部验收完成才执行任务范围commit，不push；指定副本及已有未跟踪设计文件均不修改/暂存，历史大型JSON与权重不提交。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
