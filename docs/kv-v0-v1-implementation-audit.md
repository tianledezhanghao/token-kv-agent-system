# 7.9A KV V0/V1实现与历史证据审计

状态：Completed（只读审计与工程检查）。审计起点`main / 99918bc6c626ece76a71ec6f1b682d67e649bb35`。工作区有来源不明的未跟踪`docs/token-communication-baseline_副本.md`，已报告、保留并排除提交。旧worktree与历史JSON不改。本轮真实预训练模型生成、下载、服务器/SSH、V2/V3实现均为`not_run`；没有改变V0/V1生产行为。

**主要决策A：V0/V1确已实现并在0.5B/MPS上真实提取、传输、导入、继续forward及双向多轮，足以作为现有工程基线，可进入有界V2研究，不需要重写或重复证明基本KV复用。** 次要问题E：增量Discussion尚未接自动Selector/完整性能面板；3B专门KV验收未执行，不能据静态通用代码宣布通过。质量不可靠与工程Bug必须区分。

证据附件：源码符号/行号/hash索引（原研发资料，未随公开副本提供：`kv-v0-v1-code-index.csv`）、63份历史原始文件SHA及用途索引（原研发资料，未随公开副本提供：`kv-v0-v1-evidence-index.csv`）。链接指当前仓库，行号对应审计HEAD；raw文件在Git忽略的benchmark-results，CSV不包含权重/大Tensor。文件数和嵌套projection记录数不是独立样本或handoff数。CSV文本内部换行表示为字面量`\n`；完整原文仍在对应raw文件。完整文本来自原报告，不重新生成或改写历史结果。

## 1. 三条不同的数据路径

| 路径 | 实际语义/状态 | 当前入口 | 不能混淆的部分 |
|---|---|---|---|
| Mock估算 | MockAgent/estimated KVPayload/模拟网络 | [Orchestrator](../src/token_kv_agent_system/orchestrator.py) L43；[KVTransport](../src/token_kv_agent_system/transport/kv_transport.py) L8 | 没有真实Tensor导入；UI选KV不证明真实KV |
| Canonical KV（Phase5/6） | 对M重新clean Prefill得到KV(M)，bytes→restore→suffix；不保留B独立长期历史 | [perform_kv_handoff](../src/token_kv_agent_system/kv_handoff.py) L142；[RealAdaptiveRuntime](../src/token_kv_agent_system/real_adaptive_runtime.py) L63 | 真实KV，但不是V0/V1的Decode增量KV(M\|H_A)；Selector成本仅覆盖此语义 |
| 正式增量V0/V1（7.6B起） | 从A实际Decode取新消息内容KV(M\|H_A)，bytes→加入B历史→B继续Decode | [DiscussionRuntime.run](../src/token_kv_agent_system/discussion.py) L196→[handoff_kv](../src/token_kv_agent_system/stateful_token.py) L269 | 已接正式Runtime/API/讨论Dashboard；手动模式，不是只存在研究Harness |

7.2A的RAW/RoPE是更早的直接Tensor研究，没有wire；7.6A的IncrementalKVReceiver/Channel是出口+bytes+解包旁路，本身不导入B Cache。7.6B是完整接收实现。不能拿7.6A“仅解包”限制否定当前V0/V1，也不能拿Canonical时延当V0/V1性能。

## 2. 策略与实际调用链

用户目标V0=朴素增量、V1=增量+RoPE，与现有语义一致，但“朴素”仍包含安全边界、协议、Cache事务与本地framing；不是任意复制整个Cache。

- V0：`KVStrategy.RAW_DELTA = raw_delta`。
- V1：`KVStrategy.ROPE_ALIGNED_DELTA = rope_aligned_delta`。
- `HandoffChoice.parse('kv:v0'/'kv:v1')`在[communication.py](../src/token_kv_agent_system/communication.py) L11–35映射到真正执行分支。只允许TOKEN或带V0/V1策略的KV，无V2/V3/AUTO占位执行。
- `DiscussionConfig.choice_for`（[discussion.py](../src/token_kv_agent_system/discussion.py) L50）支持统一选择及每方向手动mixed计划；rounds1–3、计划长度=2×rounds。普通讨论先对A/B分别提供原问题，然后A/B轮流实际生成，最后指定A或B总结。
- CLI [discussion_demo.main](../src/token_kv_agent_system/discussion_demo.py) L57：`--mode kv --kv-strategy v0/v1`，或`--handoff-plan token,kv:v1,...`；没有自动Selector调用。

完整生产链：

```text
CLI / REST DiscussionStartRequest.runtime_config
 -> DiscussionRuntime / SecretSumRuntime
 -> 两LLMAgent -> StatefulTokenSession独立AgentState初始化
 -> generate_message -> _generate_working -> backend.continue_from_cache (greedy Decode)
 -> config.choice_for -> session.handoff_kv(strategy)
 -> inspect_model_contract (首次缓存指纹)
 -> prepare_incremental_message -> complete_message(仅必要pending) -> extract_huggingface_kv_span
 -> encode_incremental_message_with_metrics
 -> InMemoryByteTransport.transfer(bytes)
 -> receive_incremental_kv -> decode_incremental_message -> restore_huggingface_kv_cache
 -> B work clone -> 本地incoming prefix forward
 -> [V0保留delta / V1 rephase_key]
 -> append_delta -> ledger仅记录内容IDs -> 本地closing forward
 -> validate + Sender work close -> 同时提交A/B状态
 -> 下一B Stage/assistant prefix及Decode -> 反向交接 -> 最终总结
```

| 责任 | 符号与当前源码位置 | 完成度/可核实事实 |
|---|---|---|
| 初始化与独立Cache | [agent/state.py](../src/token_kv_agent_system/agent/state.py) `initialize_agent_state` L82、`working_state` L101 | System真实Prefill；工作态clone；长期ledger/cache校验 |
| 增量生成 | [stateful_token.py](../src/token_kv_agent_system/stateful_token.py) `_generate_working` L127 | 只追加本地Stage/assistant prefix及逐生成Token，不全历史Prefill |
| TOKEN参照 | 同文件`handoff` L212 | TKVT真实IDs，B增量输入内容及本地frame |
| KV事务入口 | 同文件`handoff_kv` L269–325 | encode→bytes→receive→Sender close→双端提交；失败不转TOKEN |
| Sender出口 | [incremental_kv.py](../src/token_kv_agent_system/incremental_kv.py) `prepare_incremental_message` L55–116 | 验证accepted draft/span/ledger，取content delta |
| 协议 | [incremental_kv_protocol.py](../src/token_kv_agent_system/incremental_kv_protocol.py) `encode…` L174、`decode…` L204 | 真KV Tensor body、内容IDs、contract和checksum |
| Restore | [model/kv_adapter.py](../src/token_kv_agent_system/model/kv_adapter.py) `restore_huggingface_kv_cache` L232–310 | CPU张量重建→target device→公开DynamicCache构造 |
| Receiver | [incremental_kv_receive.py](../src/token_kv_agent_system/incremental_kv_receive.py) `receive_incremental_kv` L22–81 | 本地prefix→实际delta拼接→本地closing；M本身不forward |
| V0拼接 | [model/delta_repair.py](../src/token_kv_agent_system/model/delta_repair.py) `append_delta` L30–51 | 每层K/V沿seq dim2 cat，clone重建Cache，不旋转 |
| V1转换 | 同文件`rephase_key` L60–87 / `rotary_tables` L91 / `inspect_rope` L102 | 真K数值修正，V保留；不是仅改配置名或position metadata |
| 接收后真实计算 | [model/huggingface.py](../src/token_kv_agent_system/model/huggingface.py) `continue_from_cache` L342–399 | past_key_values=导入后的Cache；完整mask，实际模型forward |

## 3. Sender：发的是什么

各Agent以自己的System独立Prefill（agent/state.py L92），原题与Stage本地增量进入自己的KV。A实际生成每个Token；`sender_message_span`起点在assistant header消费完成之后。原raw span含所有生成IDs（末预测可pending），出口逐ID核对ledger，不从文本重tokenize推断边界。

`prepare_incremental_message` L65–75把EOS/EOT排除为内容，记录termination token，截取`[start,start+len(content))`。System、原题、Stage、历史、assistant prefix、closing **不在传输Tensor的索引范围内**。但传出的KV已经受Sender历史条件化，排除前缀行不等于向量不含前缀影响/隐私，也不等于KV(M)。

如果最后accepted内容Token尚未forward，只允许缺一个位置，复制Cache后调用[decode_kv.complete_message](../src/token_kv_agent_system/model/decode_kv.py) L42–63 / `materialize_token_into_cache`一次，忽略新prediction。EOS不是内容时不为出口补EOS；超过一个缺失位置、伪造span、空内容、控制Token等明确失败。`extract_span` L34复制每层BHSD对应区间，Adapter L211创建只用于序列化的view；不是伪造模拟值或clean Prefill(M)。真实7.6A pending样本记录仅补一个Token，原双方Cache/pending不变。

## 4. Wire与实际Transport

两种算法都使用相同wire格式：**KV Wire Protocol v1不是算法V1**；增量Envelope version1也不是算法V1。

```text
TKVI + version1 + metadata/TKVT/TKVK长度（17B固定头）
 + canonical JSON（消息序号、source/target、content count、终止信息、源绝对span、ModelContract）
 + TKVT v1（content IDs）
 + TKVK v1（每层真实K/V Tensor descriptors与raw bytes）
 + SHA256（32B，对此前整包）
```

[KVProtocol](../src/token_kv_agent_system/kv_protocol.py) L179/L198编码解析真实K/V，metadata大端，BF16/FP16/FP32 Tensor body规范小端，保持原dtype bits；不使用pickle/torch.save。Standalone TKVK没有checksum（它结构校验，不保证每种等长bit损坏都被发现）；增量TKVI L191/L216**有SHA256**，还验证完整长度、层序/shape/dtype/body size、词表/控制IDs、方向/序号及contract。不能把旧Canonical的checksum限制错误套给当前Envelope。

[InMemoryByteTransport.transfer](../src/token_kv_agent_system/transport/byte_transport.py) L36只接受bytes并记录`len(data)`，当前同步返回相同不可变bytes对象；不传Cache/Tensor的Python引用，不要求bytes地址另复制才算序列化边界。Receiver由bytes重建独立Tensor。**单进程内存交付不等于跨机网络**，无Socket/SSH/CUDA跨机执行证据。

KV包也带内容Token IDs，供ledger/边界审计；Receiver没有用它们Prefill M，不能据包内有TKVT就称“实际退化TOKEN”。实际packet包括Tensor、外层metadata、checksum、嵌套Token及KV headers，不能以raw KV bytes替代wire bytes。示例7.6A15内容Tokens：raw184320B、TKVK186601B、TKVI187641B；相应原TOKEN包74B。

## 5. Receiver、继续推理与事务

Receiver维持自己的历史，不能直接替换成A完整工作Cache。正式接收先decode/校验、restore中立payload，再`working_state`clone B；需要时关闭B旧assistant turn，本地来源user prefix实际forward后才确定target start。随后将delta K/V加入所有层，`cached_token_count`增长M，ledger写相同content IDs，再只forward本地closing。B随后Stage和assistant prefix的真实forward使用这个joined cache。

`append_delta`检查层数、rank、KV heads/head dim、dtype/device、各层长度，沿seq维cat；`reconstruct`以公开`DynamicCache(ddp_cache_data=..., config=...)`构造，Tensor clone确保无可变storage共享。库原地更新只发生在私有工作副本。Receiver `require_disjoint_caches`、Session双端状态validate之后才提交。

KV M Prefill=0指**消息内容M**不重算；prefix/closing、B Stage及Decode仍需要真实模型计算，不能声称Receiver完全零forward。Sender截断可能做一个pending补算，也不等于M全段重算。

`handoff_kv` L304核对decoded envelope与Sender原包内容，关闭Sender work后L324双端一次提交；unpack/repair/append/closing/Sender-close任何失败保留旧Cache、ledger、序号及已生成draft。传输记录可已存在，不能把“发过bytes”算交付成功。`DiscussionRuntime.run` L248/L290标failed、记录attempt/error，不继续剩余轮次或总结；fallback_reason固定None。没有静默TOKEN或V0重试。

HF `device=auto`加载失败可显式记录CPU fallback，属于模型后端另一路政策（huggingface.py L142），不是通信模式fallback；历史主KV证据实际MPS、device_fallback_reason=None。cached `continue_from_cache`没有静默TOKEN/CPU替代执行。

可运行A→B→A→B及TOKEN/KV mixed共享方向序号。复制/拼接仍有O(历史KV)时间与内存成本，不是zero-copy。状态一致仅证明结构/事务，不证明Cache等价于B完整重算或回答正确。`message_prefill_count=0`是现有实现的设计常量，不是自动硬件计数器；本结论同时根据源码没有M forward、历史forward trace及当前无M重算单测佐证，不能仅凭字段0打勾。

## 6. RoPE：V0与V1的真实差别

V0不修改传来的K/V数值，照样更新目标Cache的长度/位置索引；记录位置差值不代表做了旋转。V1只改变delta **post-RoPE K**，不旋转V、不改旧B历史K、不重新计算Sender hidden states。

源区间为sender span `[s,s+M)`；目标起点`t`取自B真实prefix forward之后的cached count（receive L56），位移`t-s`。使用实际模型`rotary_emb`/inv_freq/缩放，分别生成源/目标各位置cos/sin，不硬编码theta或只修改position_ids。配对为split-half（前/后半维度），FP32计算逆旧旋转再施加新旋转，最后一次回到原cache dtype：

```text
J(x) = concat(-x_second_half, x_first_half)
x = (K_old*cos_old - J(K_old)*sin_old) / (cos_old² + sin_old²)
K_target = x*cos_target + J(x)*sin_target
```

本机Transformers5.17 Qwen2源码`modeling_qwen2.py` L211先对Q/K apply_rotary、L214才cache.update(K,V)；L362–364从cache.get_seq_length()推导新position_ids，V未旋转，与本实现假设一致。本轮只静态读取安装源码，没有加载模型。

生产V1守卫（delta_repair L102–123）仅允许qwen2/llama、static default/llama3、full-attention、完整head rotary维；拒绝dynamic/yarn/longrope、滑窗/混合、未审计partial维等。数学helper测试可以partial dim，不代表正式V1接受partial模型。Llama分支代码存在，不代表真实Llama/CUDA已验收。

源K不在发送时归零，Receiver从原绝对源位置转目标位置；新的真实Decode消息再按其本地source span发送。重复序号拒绝，合法流没有对同一delta重复旋转。外层声明`sender_absolute_dense_cache_indices_unmodified`；模型/Tokenizer/权重必须在Session期间不可变。contract使用state_dict，非持久buffer不在权重哈希中；实际rotary inv_freq/配置还需在独立模型适配时核实，不能把该摘要称为所有运行时状态的完整证明。

离线数学标量oracle覆盖正负/零位移、逆变换、非单位缩放、尾维不变；实际Qwen7.2A BF16 round-trip max abs0.0625–0.25，不能说bitwise可逆。位置变换正确→导入成功→继续forward→语义正确是四层不同结论。V1不修正由H_A而非H_B造成的深层hidden/V差异，质量改善没有普遍保证。

## 7. Git与历史实验盘点

源码/执行版本必须分开：八核心文件`stateful_token.py / incremental_kv.py / incremental_kv_protocol.py / incremental_kv_receive.py / model/{decode_kv,delta_repair,kv_adapter,huggingface}.py`与`706745c`逐blob相同；7.2A旧研究中的七个append/旋转函数与现共用delta_repair的AST相同。本轮核验，不是仅凭“已实现”文字。Discussion/Prompt在6630d12后来变化，所以旧问题71/156不能直接保证当前默认Prompt仍生成旧全文；7.7F已诊断此变化。

已扫描18个结果目录的现有JSON（含嵌套目录）及root实验文件，63份KV/Canonical/出口相关原文件进入SHA索引。7.8A-R/B/C没有新V0/V1接收质量实验，3B实际TOKEN不被计为KV证据。源码实现提交不一定是每次采样时的timestamped checkout；CSV的stage_commit是文档/历史对应，无法还原精确调用commit时注明unknown。

证据标签：`mock_verified`=工程Mock/随机tiny CPU测试；`real_model_verified`=历史已执行预训练Qwen/MPS并有报告或raw，细分raw-present/report-only；只有函数且未核验执行才`implemented_test / execution_not_verified`；缺资料`unknown`。

| 阶段/实现提交 | 模型/后端/模式与来源 | 实际测试与结果 | 当前证据及能/不能证明 |
|---|---|---|---|
| 5.1 `188cb46` | Qwen0.5B/MPS BF16，真实Prefill KV；revision unknown | 9/30Token Cache，24层[1,2,N,64]，raw110592/368640B，全部K/V raw bits round-trip | [KV协议](kv-protocol.md)、.agent STATE旧5.1记录；real_model_verified/report-only，独立原大JSON未找到，精确Prompt unknown。证明编码，未证明B导入 |
| 5.2 `8b208aa` | Qwen0.5B/MPS，真实restore；revision unknown | 原/restored cached suffix与多步相同；Full有BF16近并列分叉；pending补齐 | .agent STATE旧5.2、[kv_restore_demo](../src/token_kv_agent_system/kv_restore_demo.py)、当前tiny CPU单测；report-only，原raw未找到；非独立历史V0/V1 |
| 5.3 `e445934`、5.4 `665f0db` | Canonical0.5B/MPS，真实/固定IDs clean Prefill(M) | 8Token wire100587B，TOKEN/KV next576；KV/Full max diff0.21875；双向restore/continuation及独立RealKV页面 | [语义](handoff-semantics.md)、.agent5.3/5.4；report-only为主。证明Canonical，不能证明Decode delta/长期B历史 |
| 6.1 `46ed296`–6.4 `dee5601` | Canonical0.5B/MPS，8/32/128固定scaling IDs，真实Tensor | warmup1+3次median；实际与估计wire一致；MPS MESSAGE_READY无正带宽crossover，真实Adaptive主记录均选TOKEN | root `mac-mps-selector-{profile,validation}.json`、[selection-policy](selection-policy.md)、.agent；真实KV calibration，非V0/V1成本。强制KV分支单测为mock_verified，真实自动选KV/CUDA未验证 |
| 7.2A `7440167` | 0.5B revision7ae557…、MPS BF16、真实A Decode；RAW/RoPE tensor-only | access7392/locker17/cup blue/forbidden elm；TOKEN4/4、RAW2/4、RoPE4/4；shared-prefix各4/4；三进程输出稳定 | root `incremental-kv-probe{,-cold,-repeat}.json`、[报告](incremental-kv-experiment.md)；real_model_verified/raw。M含EOS、B放assistant report，与正式user来源framing不同；不证明wire/通用交流/加速 |
| 7.6A `117930a` | 0.5B/MPS BF16，真实Decode内容delta，TKVI bytes | 23+48四消息15/7/11/7Tokens，全层位级一致；截断仅pending1；Sender/Receiver不被旁路污染 | root transport-add/final、pending-mps及[报告](incremental-kv-transport-validation.md)；real_model_verified/raw。IncrementalKVReceiver为unpack-only，不证明KV导入B |
| 7.6B `706745c` | 0.5B/MPS BF16、64 greedy两轮、真实Decode，TOKEN/V0/V1/Mixed | 23+48、12*13八主场景完成32交接（20KV/12TOKEN），71/156；另2截断控制、1固定三路 | `discussion-76b/{add,multiply}-{token,v0,v1,mixed}.json`，fixed-step、truncated及完整报告（原研发资料，未随公开副本提供：`discussion-kv-validation.md`）；KV M Prefill0，无fallback。证明正式复用/多轮，不证明依赖性或质量泛化 |
| 7.7 `3f6d320` | 0.5B/MPS，真实Discussion Web/API/WS | 23+48浏览器四模式/A总结，API A/B总结，各模式actual wire/cache与回收；KV同样真复用 | `discussion-77`、77-api-a/b，Web验收（原研发资料，未随公开副本提供：`discussion-dashboard-validation.md`）；real_model_verified/raw。旧Canonical Tab的1000Mbps“选KV”记录是Mock，不能拿来证明真实V1自动选择 |
| 7.7B `6630d12` | 0.5B/MPS，真实自定义System及V1/Mixed | math/fact等不同Prompt、A/B总结；接受KV且后续生成，模型质量局限保留 | `discussion-77b/{math-custom-v1-A,fact-custom-v1-B,web-modes/*,web-custom-B/*}`、报告（原研发资料，未随公开副本提供：`discussion-prompt-validation.md`）；真实接入，不是质量合格证书 |
| 7.7C `9ce6c0c` | 0.5B/MPS，同一真实A暗号Decode，TOKEN/V0/V1 | A“紫色袋鼠5729”；TOKEN复述、V0答1234、V1答“暗号是：紫色袋鼠5729”；各KV M Prefill0 | `message-understanding-77c{,-repeat,-final}/fixed-decode-token-v0-v1.json`、报告（原研发资料，未随公开副本提供：`message-understanding-diagnostics.md`）；real_model_verified/raw，示范同Sender质量差异；另Full/CPU oracle不能外推所有KV |
| 7.7D `2692ab0` | 0.5B/MPS，真实Sender私有暗号，8读取/审查/来源变体 | TOKEN/V0/V1同快照接收，各有来源/审查失败；与人工TOKEN fixtures分开 | `source-binding-77d/decode-*.json`八份；报告（原研发资料，未随公开副本提供：`source-binding-reviewer-study.md`）。人工token-private-secret系列不算真实Sender/KV主实验；kv-summary只有compared字段不单独证明成功 |
| 7.7E `d048970` | 0.5B/MPS BF16，原secret_sum96，真A→B→A | 17/26 V0最终43但B仅26无归属；V1最终34；17/31 V0给62，V1最终48但B无归属；Mixed和43但A/B颠倒；TOKEN43/48/61明确归属 | `secret-sum-77e{,-final}`、fixed-decode、Web及设计/完整原文（原研发资料，未随公开副本提供：`secret-sum.md`）；real_model_verified/raw。工程通过而语义不可靠，不能说V1统一更好 |
| 7.7F `201fc5c` | 0.5B/MPS，当前代码+历史Prompt/真实Decode对照 | TOKEN/V0/V1单步执行；secret V0/V1已有结果复现，包括34 | `discussion-77f/continuation/{decode-token-v0-v1,secret-kv-v0,secret-kv-v1}.json`及报告（原研发资料，未随公开副本提供：`discussion-regression-investigation.md`）；真实回归、非新可靠率 |
| 7.7G `81cc8aa` | 0.5B/MPS，真实正式TOKEN/V0/V1回归，官方AgentChat另列 | secret V0/V1输出IDs与7.7F相同，包括34；固定Sender三路；不以官方框架名证明KV | `autogen-77g/verified/{decode-token-v0-v1,regression-secret-kv-v0,regression-secret-kv-v1}.json`、报告（原研发资料，未随公开副本提供：`autogen-style-agent-comparison.md`） |
| 7.8A/R/B/C | 3B revisionaa8e725…、MPS BF16；0.5B匹配TOKEN历史 | 3B TOKEN普通算术改善，但秘密P0/P2失败；P1部分。0.5B原TOKEN17/26、17/31逐input/output与77E全等 | [TOKEN冻结基线](token-communication-baseline.md)、严格验收（原研发资料，未随公开副本提供：`private-information-exchange-validation.md`）；V0/V1真实3B=not_run，不能升级其Level2–7 |

7.7E settings明确保存loaded权重SHA `eed572287f948e34640c5274dd0c006ed074e3bf0243c91d9671f5514a6dbd7b`、config/identity SHA、Torch/TF/MPS/BF16；该settings没有明确checkpoint revision字段，不能仅凭模型名推定revision。7.2A主JSON明确7ae557604adf67be50417f59c2c2f167def9a775。后续TOKEN冻结记录有明确revision并与77E Token输入/三段输出逐IDs全等，是可比性补证，不代替最初缺失的revision日志。

旧worktree`discussion-75b-repro`仍fe888ee，`discussion-boundary-repro`仍6630d12，两者没有独立benchmark-results目录；相关输出在主仓库77F目录。原raw/doc未改。当前KV核心与706相同不表示所有历史框架/Prompt/模型配置仍完全相同。早期仅找到测试函数的扩展（如新增模型/CUDA/滑动cache）不写real_model_verified。

## 8. 验证矩阵

主列限定0.5B/MPS、当前核心；3B状态另列。Verified指对应工程事实有证据，不是生产可靠性认证。

| 层次/能力 | V0实现 | V0验证 | V1实现 | V1验证 | 证据/边界 |
|---|---|---|---|---|---|
| L0 策略/API/Runtime | 已实现 | Verified | 已实现 | Verified | communication / discussion / CLI / Web |
| L1 Mock与单测 | 已实现 | Verified | 已实现 | Verified | 当前217针对性工程通过；含随机tiny Qwen2真实CPU运算，但不是预训练能力 |
| L2 真实增量提取/Tensor序列化 | 已实现 | Verified | 已实现共用 | Verified | 7.6A全层bytes原始位级一致、pending实测 |
| L3 真实KV导入B并forward | 已实现 | Verified | 已实现 | Verified | 7.6B/7.7E M Prefill0，prefix/closing及B Decode用joined KV |
| L4 单轮真实A→B | 已实现 | Verified | 已实现 | Verified | 固定step、暗号、secret同Sender；回答质量另评 |
| L5 多轮双向/混合 | 已实现 | Verified | 已实现 | Verified | 两题20KV交接，Web A/B总结与截断；长期稳定性有限 |
| L6 TOKEN/Full质量对照 | 有对照入口 | Partially verified | 有对照入口 | Partially verified | 7.2/7.6固定TOKEN、7.7暗号/秘密；有差异/失败。完整相同逻辑输入Full在跨历史KV中的广泛语义/分层数值覆盖不足，不要求强行等价 |
| L7 相同质量E2E性能 | 基础bytes/局部计时存在 | Partially verified | 基础bytes/局部repair计时存在 | Partially verified | 非完整受控V0/V1性能suite；不能用Canonical calibration补勾 |
| RoPE位置修正 | 按设计不实施 | 不适用（Raw verified） | 真K变换已实现 | Verified数学/接入，语义部分 | 7.2独立数学oracle与实际Qwen，BF16不位级可逆 |
| 实际跨机网络 | 未实现具体网络Transport | Not implemented | 同左 | Not implemented | BaseByteTransport接口不是网络执行证据 |
| 3B提取/restore/V1真实forward | 通用结构代码存在 | Implemented but unverified | 通用结构/静态guard存在 | Implemented but unverified | 3B TOKEN Cache不能替代KV路径验收 |
| 稀疏/边界重算V2、层级V3 | Not implemented | 不适用 | Not implemented | 不适用 | enum/Receiver只有两策略；无稀疏forward执行 |

## 9. 质量与性能覆盖细分

**有实际质量对照**，不能说过去没测：7.2 RAW2/4 vsRoPE4/4；7.6两模式两题都正确但双方本来知道题；7.7C V0暗号错误/V1正确；7.7E V1(26)比V0更差、V1(31)局部更好、归属未完整。V1没有统一质量优越性结论。相同Sender/Receiver快照比较比独立多轮更强控制；后续轮次真实A/B消息因模式改变，不能强行当同输入。

**有局部真实计时，但没有完整V0/V1 E2E性能验收**：

| 指标 | 当前证据 | 结论 |
|---|---|---|
| actual wire bytes | 正式receipt、HandoffLog、Web真实len包 | Verified，包含TKVI/TKVT/TKVK/checksum |
| 本地transport时间 | Canonical单次诊断；正式Discussion只记录byte_count | V0/V1独立分解unknown；不能当真实网络RTT |
| serialize/deserialize | 7.6A export-side诊断实际ms，KV adapter有结构化计时 | Partially verified；正式handoff未投影完整阶段计时 |
| V1修正成本 | 7.2主四题约56.946–60.975ms，RAW append约0.753–1.165ms | 真实局部诊断，修正列不含全部tables/history clone/wire流程 |
| TOKEN M Prefill | 7.2稳态约17.452–23.741ms，冷416.750ms保留 | 不能据此宣布RoPE更快；冷/稳态分开 |
| Receiver/生成耗时 | backend.forward_ms、独立profiling与7.2suffix/decode | V0/V1正式完整E2E/per-stage的持续记录不足 |
| 网络带宽影响 | SimulatedNetwork公式、Canonical20/1000Mbps/calibration | 真计算+模拟link，非物理传输；没有增量策略网络自适应实测 |
| 内存/资源 | Inspector读真实bytes、3B TOKEN资源采样 | 正式V0/V1峰值/随历史增长资源未系统测量，unknown |
| 同质量标准下收益 | 有限短任务，不同输入/历史/质量结果 | Partially verified；无统一quality-constrained速度结论 |

正式`receive_incremental_kv`取restore结果的`.cache`，没有向receipt传restore细分时间；`ReceiverAppendInfo`仅span/repair/delta/M Prefill。`HandoffLog`也无repair_ms等字段。事件timestamp可粗看持续时间，但不代替同步、边界明确的benchmark。权重fingerprint首次扫描、clone/cat O(N)、pending、prefix/closing与后续生成都应在未来明确是否计入；7.2计时和Canonical MESSAGE_READY不是同一边界。

## 10. 模型兼容性与具体风险

| 检查 | 源码结论 | 当前证据/缺口 |
|---|---|---|
| 24层/64维是否硬编码 | 核心没有；[inspect_kv_model_config](../src/token_kv_agent_system/model/cache_inspection.py) L68从config读取，缺head_dim才hidden/attention derive | 0.5B24/2/64；3B真实TOKEN36/2/128已观察，但KV未跑 |
| Cache形状 | 实际每层BHSD，动态head count/dim/seq长度；incremental仅dense batch1 | 同头/层均匀、DynamicLayer限制；sliding/offloaded/static需另适配 |
| RoPE | 来自模型rotary_emb，静态qwen2/llama guard，full head | 3B config表面匹配Qwen2/default/full attention；实际3B KV guard与数学/continuation仍unverified |
| 同模型contract | [inspect_model_contract](../src/token_kv_agent_system/incremental_kv_protocol.py) L56哈希实际state_dict参数/持久buffer、完整config、BackendIdentity，首次每Session缓存 | 强于只看shape/model名；要求不可变且model_id一致；不是异构模型桥接 |
| Tokenizer/Template | BackendIdentity比vocab hash/class/length/special/EOS等；[ChatFraming](../src/token_kv_agent_system/agent/chat_framing.py) L27/59/80验证每端append-compatible | **没有完整tokenizer JSON/merges或chat_template hash/equality字段**。共享backend自然同对象；独立backend未完整证明模板一致，未来冻结文件/模板身份需补检查 |
| 不同M/N | 源span与目标cached长度动态，允许不同Agent历史及消息长；序号/metadata/shape严格 | 7.6主/截断、7.7秘密变化已有例子；不等于任意超长上下文稳定 |
| 长度保护 | Discussion预算min(配置4096,两模型context window)；Envelope默认4096 message、source位置2^20、packet64MiB | 3B raw每Token36864B，长M可能先撞64MiB包限（约1820Token量级，含header更低），不能按0.5B上限预算；96/128短消息远低于此 |
| Selector旧参数 | RealAdaptiveRuntime L108校验model/层/head_dim/dtype/device等，旧profile是Canonical0.5B | 不能给3B或增量V0/V1直接复用；profile也不是完整revision/tokenizer身份记录 |
| NaN/非有限诊断 | rephase检查K/tables有限；7.2harness显式诊断logits | 正式continue_from_cache的argmax前没有全logits finite断言，V0也没有全部K/V finite检查；记录为异常诊断缺口，未有历史证据证明当前正常KV路径被此阻塞 |

同构独立backend API存在（DiscussionRuntime可传backend_a/backend_b），incremental合约有actual weights fingerprint；本机主要真实证据共享单权重+独立Cache，未证明不同进程/不同GPU/不同模型可通信。不能因代码接受cuda:N就宣称CUDA验收。

## 11. Runtime、Dashboard、Selector与课程目标差距

REST `/api/discussions`→[DiscussionStartRequest.runtime_config](../src/token_kv_agent_system/dashboard/discussion_service.py) L77→`DiscussionService._run` L200，创建原Discussion/SecretSumRuntime、使用单worker/单活动slot；完成仅保留metadata、释放Cache。WS/REST投影实际turn/handoff/result，不把Tensor送前端。原默认secret_sum评分parser并非7.8C严格B1–B7质量认证。

[DiscussionPanel](../frontend/src/discussion/DiscussionPanel.tsx) L8–12手动V0/V1，L23–39展示真实wire/span/cache count/M Prefill/repair/delta/error/fallback，L163明示InMemory及不使用旧network/Selector。真实Web记录支持这些字段，并非仅Mock显示；UI单测仍属于mock/tiny工程测试。讨论页没有网络带宽控制的成本决策、完整repair/restore/E2E ms或实际缓存GiB面板，缓存主要展示Token数。

自动路径是另一条：`RealAdaptiveRuntime.communicate_next` L178先`RealCostBasedSelector.select(MESSAGE_READY)`，L208 TOKEN走TokenHandoffSession clean Prefill，L225 KV走Canonical.perform_kv_handoff（L143封装）；没有调用StatefulTokenSession.handoff_kv，也不选择raw/rope策略。MPS真实profile主记录20/1000Mbps均选TOKEN；unit forced-KV验证分支是mock_verified。Mock Orchestrator的LatencyBasedSelector+estimated KVPayload另列，不可称真实KV性能。

| 课程目标 | 当前状态 | 说明 |
|---|---|---|
| 用户问题/两个真实Agent轮流/最终汇总 | 已实现并验证 | Discussion/SecretSum，1–3轮上限，有限质量 |
| TOKEN/KV及V0/V1手动切换 | 已实现并验证 | 正式CLI/API/UI与mixed，非only harness |
| 网络变化自动选增量TOKEN/V0/V1 | 未实现 | 自动Selector只接Canonical；现讨论模式与旧network解耦 |
| Dashboard观察实际交接 | 已实现并验证 | mode/bytes/span/length/repair/error来自实际结果 |
| 完整实际KV size/repair time/资源/E2E | 已实现但证据有限 | 核心Inspector/局部harness有，讨论展示与统一计时不足 |
| 预计TOKEN/KV时延、选中执行时延 | 已实现并验证（Canonical边界） | 模拟link+真实计算；不是V0/V1质量匹配速度 |
| 质量/正确性评分 | 已实现但证据有限 | secret事后保守parser/人工各研究；普通讨论非通用准确率体系 |
| 可复现实验/原始失败 | 已实现并验证 | config/IDs/hash/报告；早期原raw/revision存在unknown |
| V2/V3 | 未实现 | 可复用receive/repair/export基底，不存在隐式稀疏修复 |
| 真实跨机网络/CUDA部署 | 未实现或本轮not_run | 接口/预检不是远程运行；本轮不访问服务器 |

陈旧文档需按时间理解：[incremental-kv-transport.md](incremental-kv-transport.md)既有后来7.6B正式接收章节，也保留7.6A“只发送”阶段段落；.agent旧7.5A“V0/V1未接入”也是历史。当下source和706745c起的真实结果优先。审计不为统一措辞改旧历史记录。

## 12. 下一阶段决策、最小准备与复现

**A为主，E为次；不是“V0/V1未开发”或“需要先重做整套验收”。** 0.5B传来的KV确实进入B，并已真实多轮。没有发现使这条路径无法运行的阻塞性源码Bug；语义失败不归为工程C。旧原raw缺失局限不足以让整个项目变成D，因为后续raw真实证据充分。

最小后续工作：

1. 冻结已有0.5B原secret_sum17/26、17/31的System/Public/Stage/96 greedy（见[7.8C基线](token-communication-baseline.md)），用已保存TOKEN成功与V0/V1失败作为起点。二者TOKEN历史与78AR冻结条件input/三段output IDs相同，本轮核对；不做新Prompt搜索、不无意义重跑176/90/40生成。
2. 可以进入独立V2**受控研究**：先明确重算边界及k/M比例、是否连续尾段、必须重算的prefix/closing分界、临时Cache与回滚、新的M Prefill实际计数。2Token消息若重算全部2个，不能冒称“稀疏修复且节省计算”。先保留V0/V1不动并对同一真实Sender与Receiver快照比较。
3. 当前standard continuation只追加连续Token、position来自cached长度；它不是任意位置/任意层稀疏重算接口。连续边界/尾段候选可以设计受控临时prefix Cache+少量forward；非连续token/层级V3需要另行设计position/mask/cache write-back/hidden-state信息，不能只增加enum名称。
4. 为V2补实际计时与payload账：pending、extract/host copy、checksum/encode、transport（本地/模拟/物理分开）、decode/restore、table/rephase、recompute、clone/cat、closing、后续Decode及峰值内存。明确哪些是MESSAGE_READY哪些完整讨论E2E，不借旧Canonical calibration证明更快。
5. 先固定0.5B同revision/context/数值前缀做真实KV对照；若改用3B，才追加一次有界兼容门槛（提取/bytes bit round-trip/restore、V1 table与position、independent cache、M/k计数、continue、资源），复用分级保护。3B单独需验收不等于0.5B核心未验收。
6. V3前重新核对所选方法/官方实现与当前Transformers接口，定义层/Token误差、选择目标、必需额外IDs/索引/hidden状态等字节。历史7.2A只记部分RelayCaching源码阅读、不同依赖版本、未运行官方方法；本轮未新增外部论文验证，不能把那份阅读当现有V3。
7. 独立后续任务再决定如何把incremental成本与质量约束接Selector/UI；不直接复用Canonical profile，也不把新策略在前端自动当V1（当前displayMode非raw就显示V1，只针对现有两枚举）。

可复核入口（本轮仅help检查，以下真实运行均**not_run**；必须另选新目录）：

```bash
# 当前正式增量KV，可在下一明确任务运行：
HF_HUB_OFFLINE=1 .venv/bin/python run_discussion_demo.py \
  --question '23 + 48 = ?' --mode kv --kv-strategy v1 --rounds 2 \
  --output benchmark-results/new-kv-check.json
# 现有完整secret suite（不是仅两case过滤器）：
HF_HUB_OFFLINE=1 .venv/bin/python run_secret_sum_validation.py --device mps \
  --max-new-tokens 96 --output-dir benchmark-results/new-secret-kv-check
# 独立数学/短信息研究、出口-only、正式四模式、Web分别见：
.venv/bin/python run_incremental_kv_probe.py --help
.venv/bin/python run_incremental_kv_transport_probe.py --help
.venv/bin/python run_discussion_validation.py --help
.venv/bin/python run_discussion_web_validation.py --help
```

当前default一般Prompt已不同于7.6B；若要求逐全文复现当时结果，使用706745c隔离checkout/精确冻结旧System+Stage，而不是reset主分支或假设最新CLI输出必然等于旧报告。已保存两个历史worktree不修改。原secret suite原System/Stage已固定，可用于既有0.5B基线；只选两个case需后续独立harness，不声称已存在该CLI选项。

## 13. 本轮检查与交付验收

- 现有KV针对性测试：**217 passed、0 skipped、2既有warning**。覆盖wire/BF16 bits、wrong-contract/corruption、pending、Cache storage与事务、无M forward/无fallback、RoPE标量数学、双向mixed、API/WS。随机tiny CPU/Mock明确不替代预训练真实模型。
- 全量Python：**763 passed、1 skipped、2既有warning**；skip仅可选隔离AutoGen。
- Frontend：**69 passed / 4 files**；未改前端，TS/build `not_run`。
- 15相关CLI help、compileall（src/tests）通过；源码符号行号/hash、历史raw SHA及Markdown本地路径逐项核查，git diff --check通过。CLI help没有模型加载；本轮没有模型生成或benchmark。
- 生产KV/通信/Selector/模型文件及历史结果零改动；只提交本审计报告、两个索引与必要.agent记录，保留用户副本未暂存。Git hash/最终状态以提交回复为准，不push。

审计验收完整；下一任务已有可靠工程起点，未把“不可靠质量/不完整性能”包装成完成，也不把真实已实现的V0/V1倒退为待开发。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
