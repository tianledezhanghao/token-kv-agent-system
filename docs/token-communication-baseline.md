# TOKEN通信研究基线（冻结于7.8C）

**TOKEN是可运行的工程与功能参照；3B私有信息交换仍未达到严格标准。** 7.8C结论C，结束当前私有数字Prompt研究。后续进入独立KV对照，不修改生产默认，不把不同模型/Prompt的成功合并为一个统一合格系统。KV质量与性能没有因此获得保证。

本基线由[7.8C冻结规范](token-baseline-78c.json)、分开冻结的参考配置（原研发资料，未随公开副本提供：`token-baseline-reference-configs.json`）、历史原始文件哈希索引（原研发资料，未随公开副本提供：`token-baseline-reference-index.csv`）、7.8C结果/评分索引（原研发资料，未随公开副本提供：`private-exchange-78c-index.csv`）组成。阅读顺序：本文件→验收报告（原研发资料，未随公开副本提供：`private-information-exchange-validation.md`）→对应完整原文/IDs/JSON。规范与源码测试的固定SHA可发现Prompt/用例漂移。

## A. 工程基线

实机M5 MacBook Air/16GiB，HF Transformers5.17.0+PyTorch2.14.0，MPS BF16、batch1、greedy，模型仅一份权重共享。Agent A/B各有独立System、ledger及KV，权重共享不会自动共享对话。每次新Session重新初始化私有信息；不拼接双方秘密到公共任务。

正式路径是StatefulTokenSession实际Decode→Token Protocol v1 binary bytes→InMemoryByteTransport→strict deserialize/IDs核验→Receiver自己的来源user framing及新增内容增量Prefill→继续Decode。最终A从长期缓存总结；不重跑完整历史Prefill来冒充增量接收，不用程序算答案回填模型。

7.8A-R保存48次真实Decode TOKEN双向handoff。7.8B原始秘密失败核实B System26/31、A17来源包装、EOS边界/pending及173完整生成输入IDs；相同输入标准Full计算生成与T逐输出IDs相同，但logits有BF16误差。该结论只适用于已测原P0两失败，不证明普遍计算逐bit相等。

7.8C新增24次实际handoff（22真实Decode发言、2固定无数字fixture）；IDs完整、每次Receiver只追加receipt.receiver_appended_ids，实际forward前Cache非0。初始私有上下文隔离，独立Cache/共享backend逐阶段核实，Receiver mask/absolute positions与cached长度一致。14组全部逻辑模板与实际generation ledger IDs一致；P2未运行Full数值oracle，不把“模板一致”当成已做计算等价性实验。

| Cache配置 | 0.5B | 3B |
|---|---:|---:|
| 层数 | 24 | 36 |
| Attention heads | 14 | 16 |
| KV heads | 2 | 2 |
| head dim | 64 | 128 |
| 每层K/V形状 | [1,2,N,64] | [1,2,N,128] |
| dtype | BF16 | BF16 |
| 两张量全层每Token bytes | 12288 | 36864 |

数字来自真实配置及Cache，不通用假定同系列相同形状/Token IDs。具体case的实际N、层metadata/字节和pending见JSON。3B7.8C最大A331/B240 cached Tokens，无OOM/critical/资源中止，MPS driver采样峰6.743GiB；这不是满4096双端或长期运行保证。继续沿用7.8A-R持续critical、压力+Swap速率、压力下forward停滞与OOM保护，Swap绝对量/新增1GiB不能直接判运行失败。

TOKEN能力指当前单进程实际bytes与Cache流程，不宣称物理跨机网络已测量。每条模型记录含input/output IDs、EOS/预算原因、实际roles/文本来源、模型revision、forward和内存；大型原JSON在Git忽略目录，索引/精简完整原文保存在Git。

## B. 分开的质量基线

| 家族 | 冻结条件 | 真实证据与适用范围 |
|---|---|---|
| secret-sum-05b-P0 | 0.5B revision 7ae557604adf67be50417f59c2c2f167def9a775；原System/Public/STAGES；96 greedy，单轮双向+A总结 | 17/26与17/31正条件明确A17、B26/31及43/48，历史逐IDs复验。只证明这些正条件；双方无handoff时编造及错属仍存在 |
| discussion-3b-general-v1 | 3B revision aa8e72537993ba99e69dfaafa59ed015b17504d1；原Solver/Critic、原问题、128 greedy，两轮+A总结 | 10+10与23+48真实讨论相关反馈/最终20/71；另方程、乘法有历史算术改善。不能代替非对称秘密交换或普遍Reviewer可靠性 |
| secret-sum-3b-P0 | 3B原私有System/Public/Stage，96 | 17/26、17/31读到17却自己的数字unknown；Full同样失败。工程/计算参照，质量失败参照 |
| secret-sum-3b-P1 | 只把B-first改7.8B S4，96 | 两例真实最终43/48、B保留自己，但B未明确A17并索要已经收到的数字。部分信息整合，严格失败 |
| secret-sum-3b-P2 | 7.8C固定P2，仅B-first改变，96 | 六正条件严格0/6；六负控制均未完整通过；已知两组新进程input/output逐IDs复现失败。不作为合格私有交换质量基线 |

P2消除重复请求，却丢失一端身份事实；13/29与42/57只有数值正确但A被标unknown；37/18错和25并最终截断；不能算通用自然交流成功。完整P0/P1/P2原文与B1–B7在7.8C报告及raw，不挑最好例子。

保留其他限制：3B无A消息的普通Reviewer可虚构A回答；0.5B缺handoff秘密可编造；3B原Stage/自己的秘密读取不稳定；指定中文的真实双Agent讨论仍输出英文；来源、反馈归属、地理审查仍有失败。不要把单模型中文或算术正确替代整个交流验收。

P0/P1/P2始终有不同B Stage，0.5B/3B也有不同A真实输出；所有跨条件比较必须逐条标注模型/输入差异。冻结文件明确保存两参考家族全部System/Stage、问题、budget/rounds，禁止悄悄拼接成功条件。

## C. 后续KV的可比条件

先以**secret-sum-05b-P0，17/26**开始真实Sender固定消息对照，再17/31验证B秘密变化。该家族存在真实正条件、旧7.7E/7.8A-R复现和精确IDs依据；不能从“缺消息时A也算出和”认定通信成功。随后以3B10+10作为独立模型形状/普通讨论smoke，23+48作第二个算术条件。P2保留失败家族，用于验证KV没有掩盖模型问题，不能拿来宣称修复后的交流基准。

两个参照目的不同：

- **TOKEN语义/功能参照**：同一Agent私有初始化、相同真实Sender消息及角色包装、相同Receiver任务/Stage，评价来源、信息利用、正确性、缺消息/重复请求/最终反馈。KV若产生不同IDs，也按同冻结语义标准评价，不能只按数字正确。
- **完整Prefill模型计算参照**：先验证最终完整input IDs、模板边界、positions/mask、EOS/pending相同，再从同一模型计算logits/Top-K/argmax、共同前缀首分叉、Cache形状/长度。BF16不要求逐bit；不同输入不能用输出差异证明底层计算Bug。

在受控单步实验里，真实Sender只Decode一次，保存同一消息/内容IDs和源Cache span；Receiver初始化快照独立clone给TOKEN与KV路径。Sender收到的历史/System/Stage、输出停止及截断条件一致，Receiver本地framing/Stage亦一致；不将A发言伪装成B自己的assistant历史，不将固定fixture冒称真实Sender KV。

固定与记录：model/revision、实际tokenizer/template哈希、dtype/device、原System及每端秘密、公共问题、每条Stage、greedy及96/128预算、轮次、完整Sender IDs、EOS如何排除为内容、本地Receiver prefix/closing、源/目标span与实际缓存位置、KV heads/head_dim/层数、每种修正策略。TOKEN实际wire全消息IDs含生成EOS；KV内容span按既有规则排除EOS并由Receiver补本地closing，两者必须明确对应相同逻辑内容，不能比较不同消息。

现有KV V0/V1代码保持；V2/V3是未来单独任务，不暗示已实现。首次3B KV前核查模型contract和tokenizer身份、36层/[1,2,N,128]/BF16、源Decode span/pending补齐、wire还原、Receiver position/RoPE条件、独立Cache事务、KV内容不重复Prefill、异常无静默fallback及资源峰值。0.5B Selector/成本校准不适用于3B；本轮未更新这些模块，也没有3B KV质量或速度结论。

全程保留正负失败，冻结B1–B7评价；若KV结果正确但错所有权/虚构信息/重复请求，也不合格。对普通算术家族使用其已有阅读/来源/反馈/语言/总结维度，不能套秘密数字评分后拼接成统一总分。缺消息控制与上下文泄露检查同样需要，不因TOKEN正条件成功而省略。

## D. 证据、重现入口与终点

历史证据：`benchmark-results/qwen3b-78ar/{05b-secret,3b-token-core,3b-token-extra}/`；归因计算证据：`benchmark-results/secret-sum-78b/3b-pair/`；新P2完整正负/复验：`benchmark-results/secret-sum-78c/{core,repeat}/`。各文件SHA在Git索引可复核；0.5B/3B旧全量输出（原研发资料，未随公开副本提供：`local-qwen3b-raw-outputs.md`）、7.8B完整IDs（原研发资料，未随公开副本提供：`private-information-exchange-raw-outputs.md`）、7.8C完整IDs（原研发资料，未随公开副本提供：`private-exchange-78c-raw-outputs.md`）保持。

```bash
# 历史参考家族重跑必须使用新目录，顺序运行，不并行加载模型。
.venv/bin/python run_local_qwen_comparison.py --model 05b --phase secret \
  --output-dir benchmark-results/token-baseline-new/05b-secret
.venv/bin/python run_local_qwen_comparison.py --model 3b --phase token-core \
  --output-dir benchmark-results/token-baseline-new/3b-token-core
# P2失败参照：
.venv/bin/python run_private_exchange_validation.py --phase core \
  --output-dir benchmark-results/token-baseline-new/3b-P2
```

原秘密入口包括其既有双方无handoff条件，区别于7.8C分别阻断一边；不要强行逐条件比较。当前任务仅复用历史0.5B/算术，不宣称新跑了它们。完整Full oracle入口在7.8B：`run_private_exchange_probe.py --phase pair`（3B原P0真实Sender与Full），适用于该任务配置，不能直接冒称针对P2或未来KV跑了oracle。

本轮冻结了**可运行研究基线及明确质量失败**，没有得到合格的3B私有交换配置。正式默认/Prompt/TOKEN/KV算法均保持；下一阶段聚焦TOKEN与KV机制及修正策略，不继续优化本任务Stage。更多秘密、长轮次、满上下文、3B KV、CPU/FP32和服务器条件未验证；有限已测TOKEN正例不能保证KV或未来所有对话可靠。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
