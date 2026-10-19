# 7.2A Decode 增量 KV 复用与位置修正实验

本轮为独立研究，**不属于正式生产 KV 能力**。2026-10-09（Asia/Shanghai）在当前 Mac 上完成；起始 Git HEAD 为 `3c4a8c1`，工作树干净。没有连接服务器、加载 Llama 或执行 CUDA。Token/KV Protocol v1、Canonical KV、RealCostBasedSelector、calibration、生产 RealAdaptiveRuntime、Dashboard 均未修改。

## 环境与复现

- 模型：`Qwen/Qwen2.5-0.5B-Instruct`，revision `7ae557604adf67be50417f59c2c2f167def9a775`。
- Python 3.14.6、PyTorch 2.14.0、Transformers 5.17.0；macOS 26.7.1 arm64，MPS 可用，CUDA 不可用。
- 同一模型实例、batch=1、BF16、eval/inference mode；A/B 有各自独立历史 cache。只读取本地已缓存模型，CLI `local_files_only=True`。
- raw-logit greedy argmax，不采样、不应用 repetition penalty；Sender 上限64、Receiver上限16。真实 EOS 集合 `[151645,151643]`。

```bash
HF_HUB_OFFLINE=1 .venv/bin/python run_incremental_kv_probe.py \
  --device mps --output benchmark-results/incremental-kv-probe.json
HF_HUB_OFFLINE=1 .venv/bin/pytest
.venv/bin/python -m compileall -q src tests *.py
git diff --check
```

完整 JSON 位于 gitignored `benchmark-results/incremental-kv-probe.json`，初次冷运行与第二次分别保存在 `incremental-kv-probe-cold.json`、`incremental-kv-probe-repeat.json`。三次运行的全部消息与答案 Token IDs 相同，logits 差异指标相同；timing 波动。这是同一环境内的复现证据，不是跨硬件保证。

## 启动审计与最小实现

现有 HuggingFace backend 已支持按精确 IDs Prefill、单 Token Decode、多 Token cached continuation、完整 attention mask、pending-token materialization。DynamicCache inspector/adapter 已验证 `[B,Hkv,S,D]`、公开 `ddp_cache_data/config` 构造与独立 storage；现有 Canonical handoff 的 KV 必须由 clean Prefill(M) 构造。`real_cost.py` 和 RealAdaptiveRuntime 的生产语义仍是 Canonical MESSAGE_READY，本实验不接入它们。

缺口是 Decode 消息区间截取、保留 B 历史的追加、post-RoPE K 的实际重定位和客观信息评估。新增 `experiments/incremental_kv.py` 与独立 CLI 填补研究缺口，不创建协议、selector 或大型适配框架。实验直接在本地 tensors 上运行，无 byte transport、serialization、真实网络或 wire 性能声明。

## 实验协议与边界

1. 固定四个 `TASKS` 后才采样。A 的独立天气笔记历史，要求正常生成一条精确短报告；没有 teacher forcing、固定输出 IDs 或 Clean Prefill(M)。实际生成内容四题均符合报告要求。
2. A_history 包括请求与 assistant 起始边界。Prefill 仅生成首个 prediction；后续每次输入前一个预测 ID Decode。输出最后一个 ID 尚未消费，补齐仅使用 `materialize_token_into_cache(last_id)`。
3. M 定义为全部新生成 IDs，**包含 EOS**。截取范围为 `[len(A_history), len(A_history)+len(M))`，只复制全部24层该区间，不传 A 历史。
4. B 有音乐主题的独立历史，结束在 incoming assistant report 的起始边界。缓存一次 B_history 并保留不可变 tensor snapshot；每种路径通过公开 DynamicCache 构造重新复制 storage，断言 B snapshot 与 A cache 逐元素不变。
5. TOKEN_BASELINE 在 B_history 之后正常 cached Prefill M，再正常 Prefill B 的问题；RAW 直接追加原始 delta；ROPE 先修正 K 再追加。两种 KV 路径的 `receiver_message_prefill_tokens` 均为0，receiver `forward_inputs` 只含自己的 suffix，后续 Decode 只消费预测答案。
6. 通过 tokenizer 的实际 chat template 生成边界。M 内 EOS 已关闭 report assistant turn，B 的 suffix 删除同一个首个结束符，避免重复。B 问题不含 M 文本/答案，没有 decode→re-tokenize(M)。接收的整体 IDs 定义为已分段的 `B_history + M + suffix`；基线也使用相同拼接，不依赖整串文本重新 tokenization。
7. Attention mask 全为1，覆盖已有 cache+新增输入；每个 cached forward 校验长度。异常/非有限 logits、提前 EOS、重复输出与长度错误记录；运行完成和答对是不同指标。
8. 共享前缀正向对照使用与 A 完全相同的 history IDs，仍独立创建 B cache，随后运行同样三路径。它验证正常前缀复用；不算独立历史主任务得分。

## RoPE 实现与验证

实际配置为 default RoPE、theta=1,000,000；24层、14 query heads、2 KV heads、head_dim=rotary_dim=64，split-half pairing。实际 Qwen2Attention 在 cache.update 前已经对 K 应用 RoPE，V 不旋转。Transformers 5.17 Qwen2Model.forward 未声明 cache_position；position_ids 省略时从 `cache.get_seq_length()` 生成后续 arange。只改变 position_ids 无法修复已有 K。

本实现使用模型 rotary_emb 生成原始/目标位置的模型 dtype cos/sin，计算时升至 FP32：

```text
J(x) = concat(-x_second_half, x_first_half)
x = (K_old * cos_old - J(K_old) * sin_old) / (cos_old² + sin_old²)
K_new = x * cos_new + J(x) * sin_new
```

分母保留真实缩放和 BF16 table 舍入；最后一次转回 cache dtype。V 只 clone，不改变数值。使用模型实际 inv_freq，而不是硬编码默认 theta 去处理未来 Llama3 scaling。实验入口仅允许已审计的 Qwen2/Llama split-half、default/llama3 静态频率和 dense full-attention；dynamic/yarn/longrope、滑窗、混合 cache、其他 rotary dimensions 明确拒绝。**Llama3 分支尚未做真实权重验收**。

离线数学测试用独立标量2×2旋转公式验证目标结果、正负位移、逆变换、恒等位移、非单位 scaling、partial dimension 与不变尾部；另用 Qwen theta 和64维在86→66位置验证。FP32数学 oracle 在指定容差内通过；实际 BF16跨上下文 K round-trip max abs 为0.0625–0.25，不能宣称位级可逆。

## Sender 实际消息与 cache

三个11-Token消息及一个15-Token消息均包含最后EOS=151645。四次生成都恰好差一个最后 prediction；仅补齐这一个位置。

| 任务 | A_history | B_history | M tokens | Decode后/补齐后cache | Sender区间 | Target区间 | raw KV bytes |
| --- | ---: | ---: | ---: | --- | --- | --- | ---: |
| identifier | 86 | 66 | 11 | 96/97 | [86, 97] | [66, 77] | 135168 |
| fact | 86 | 66 | 11 | 96/97 | [86, 97] | [66, 77] | 135168 |
| color | 86 | 66 | 11 | 96/97 | [86, 97] | [66, 77] | 135168 |
| constraint | 90 | 66 | 15 | 104/105 | [90, 105] | [66, 81] | 184320 |

区间均为左闭右开。11 Token=132 KiB；15 Token=180 KiB；每 Token为12,288 B，按24×2(K/V)×2(KV heads)×64×2(BF16 bytes)计算。

- identifier: `The access code is 7392.<|im_end|>`
  IDs: `[785, 2615, 2038, 374, 220, 22, 18, 24, 17, 13, 151645]`
- fact: `The parcel is stored in locker 17.<|im_end|>`
  IDs: `[785, 29309, 374, 9768, 304, 34679, 220, 16, 22, 13, 151645]`
- color: `The cup is blue. The plate is red.<|im_end|>`
  IDs: `[785, 10525, 374, 6303, 13, 576, 11968, 374, 2518, 13, 151645]`
- constraint: `The allowed labels are oak and pine. The forbidden label is elm.<|im_end|>`
  IDs: `[785, 5420, 9201, 525, 37871, 323, 33597, 13, 576, 36813, 2383, 374, 42205, 13, 151645]`

## 实际语义质量与答案

评分在大小写、空白及末尾简单标点归一化后，要求答案严格等于固定期望；不是流畅度或关键词包含评分。

| 问题 | 期望 | TOKEN | RAW KV | RoPE KV |
| --- | --- | --- | --- | --- |
| access code | 7392 | 7392 | 7392 | 7392 |
| parcel locker | 17 | 17 | 空字符串，ID=151643，直接EOS | 17 |
| cup color | blue | Blue | Blue | Blue |
| forbidden label | elm | elm | Oak | elm |
| **主任务答对** | 4 | **4/4** | **2/4** | **4/4** |
| **主任务完成推理** | 4 | **4/4** | **4/4** | **4/4** |

没有异常、NaN/Inf logits、重复输出诊断或不合理长度；RAW柜号虽然正常终止，却没有传达信息，算错误。shared-prefix三路径均4/4且答案IDs相同，RAW与修正路径的logits相同；相对TOKEN仍有数值误差。共享前缀包含A原有的报告请求，所以其4/4不是独立信息传递证据，只用作同前缀cache/logits控制。

主任务输出IDs（TOKEN与RoPE相同）：编号 `[22,18,24,17,151645]`，柜号 `[16,22,151645]`，颜色 `[10331,151645]`，约束 `[23162,151645]`。RAW差异为柜号 `[151643]` 与约束 `[85839,151645]`。全部 shared-prefix 输出IDs及每步cache长度保存在JSON。

## Next-token logits 与概率分布

比较位置为 **B 自己的问题 Prefill 完成之后、预测答案第一个Token之前**；均对完整词表比较。KL=KL(TOKEN||path)，TV=0.5×Σ|p_token−p_path|，使用FP32 CPU概率。基线自身误差为0。

| 任务 | RAW max/mean abs | RoPE max/mean abs | RAW KL / TV | RoPE KL / TV |
| --- | --- | --- | --- | --- |
| identifier | 7.62500 / 1.38424 | 4.45703 / 0.68347 | 1.457212 / 0.768287 | 0.014222 / 0.017213 |
| fact | 31.46875 / 5.10659 | 7.61719 / 1.32341 | 30.887011 / 0.999999 | 0.003214 / 0.002380 |
| color | 6.67578 / 1.07423 | 3.93750 / 0.55502 | 0.628847 / 0.507965 | 0.025807 / 0.048563 |
| constraint | 11.46875 / 1.71804 | 5.51562 / 0.92263 | 3.969210 / 0.967604 | 0.072248 / 0.073213 |

shared-prefix的RAW/RoPE相对TOKEN max abs为0.21875–0.296875、mean为0.03882–0.04435；全题top1相同。主任务RoPE后top1也全部相同，但完整分布仍不等价。

RAW→RoPE只改变K的几何位置，因此本轮答案与分布改善来自位置修正干预。修正不能重做A消息的深层hidden states/V；相对B重新Prefill的残差仍可能来自不同历史、角色/请求语境、BF16 table与KV量化、cached Decode vs multi-token Prefill数值差异。shared control证实即使历史相同仍有小数值偏差。本轮没有相同长度但不同内容历史、position-only真实模型对照或FP32真实模型实验，**无法精确分解每一种残差的贡献**。

## 轻量成本诊断

下表为最终第三次运行的主任务同步耗时，单位ms。KV修正包含逐层校验、FP32转换、旋转和V clone；append包含cat、storage复制与公开cache重建。history clone和B suffix Prefill也分别保存在JSON。rotary table构造、数学/一致性诊断、logits比较不计入此列，所以不是完整handoff总成本。

| 任务 | pending补齐 | delta提取 | TOKEN M Prefill | RAW追加 | RoPE修正 | RoPE追加 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| identifier | 14.469 | 0.392 | 23.741 | 0.844 | 56.946 | 0.869 |
| fact | 14.269 | 0.355 | 17.452 | 0.753 | 60.975 | 0.773 |
| color | 14.931 | 0.353 | 18.083 | 1.165 | 57.242 | 1.047 |
| constraint | 14.197 | 0.384 | 19.747 | 0.762 | 58.207 | 0.813 |

初次冷运行保留真实TOKEN Prefill 416.750ms/165.961ms及首次RoPE修正129.719ms，没有删除异常高值。第二次独立运行TOKEN Prefill 17.803–24.729ms、RoPE修正53.822–58.682ms。当前逐层Python校验和 `.item()` 同步开销明显，不能声称修正更快；单次小消息任务不用于Selector重新标定。

额外数据预算记录在JSON：可选uint32 M IDs为44/60 B；两个uint64位置起点+一个count为24 B；hidden states为0 B。位置预算包含接收端本可自行知道的目标起点，是显式候选descriptor口径，**不是已实现的wire格式**；若只发送源起点可缩减，但不可在未来成本中忽略。实验harness持有IDs用于统一计数/EOS边界检查和基线，KV路径不以这些IDs重算M。完整选择性修复若传hidden states、索引、influence scores或IDs，须另计真实字节数。

## RelayCaching 论文与官方源码检查

参考 [PMLR论文入口](https://proceedings.mlr.press/v306/geng26a.html)、[论文HTML §4与附录](https://arxiv.org/html/2603.13289v1)、[官方仓库](https://github.com/YingshengGeng/RelayCaching)。论文主张从Decode KV中定位层/Token偏差并少量修复；这不能直接成为本项目小模型的质量或性能结论。PMLR PDF链接未能解析，阅读的是论文HTML。

本机初次clone因DNS失败；后续clone仍长时间无法完成并停止。以下文件通过网页/raw读取，检查时官方main tree revision为 `6b0e9d37b74d89cfb0fbad276790150a0d9c0b49`；下方源码链接固定到该revision。**没有声称完整clone、运行官方实验或检查所有文件**。

| 已阅读源码 | 检查到的机制 | 本项目适配边界 |
| --- | --- | --- |
| [gpt_chat.py](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/RelayCaching/llm/gpt_chat.py) | 生成后截取实际seen response区间；response_tokens去掉最后prediction；保存IDs、hidden和influence；按RELAY/BLANK拼装不同上下文 | span/边界和IDs审计可借鉴；本轮选择真实补齐最后ID而非丢弃。不能搬自动trim到最短长度的容错策略 |
| [anchor_engine.py](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/RelayCaching/llm/anchor_engine.py) | apply_rotary_pos_emb复制cache并旋转K的offset；发送归零、接收shift、V保留 | 静态RoPE可用现有rotary_emb做数学转换；不能只改索引；Anchor pool不适配本轮 |
| [relay_utils.py](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/RelayCaching/utils/relay_utils.py) | RelayCache.relay_select、_fused_deviation_select、update_cache；hidden-state load/save、pre/post_drop、GQA influence | 下一轮可研究value cosine偏差、少量尾部/选中位置写回；稀疏forward仍需另外实现和验证 |
| [fixed_llama_hijack.py](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/RelayCaching/utils/fixed_llama_hijack.py) | patch prepare_inputs_for_generation、LlamaModel.forward、LlamaAttention.forward及causal mask；逐层隐藏状态relay、不同query/recompute位置 | 必须改forward/mask才能跳层和稀疏重算，不能靠标准append实现。禁止直接全局patch本项目生产模型 |
| [fixed_qwen2_hijack.py](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/RelayCaching/utils/fixed_qwen2_hijack.py) / [profiler_utils.py](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/RelayCaching/utils/profiler_utils.py) | Qwen2也有专门patch；model-name dispatch启用各模型patch，profiler辅助层校准 | 框架假设需按5.17接口重审，不能将8B层范围硬编码给1B |
| [requirements.txt](https://github.com/YingshengGeng/RelayCaching/blob/6b0e9d37b74d89cfb0fbad276790150a0d9c0b49/requirements.txt) | Transformers4.52.4、torch2.6.0 | 本地5.17.0/2.14.0不兼容旧cache.key_cache/slice_/concat_/seen接口假设 |

[官方README](https://github.com/YingshengGeng/RelayCaching#baselines) 的 Direct Reuse/ZERO 用EPIC prefix=0，**包含RoPE修正**；对应本轮RoPE路径，不对应RAW诊断。本轮RAW是额外未经位置修正的诊断。

论文图5描述pre-RoPE存储；实际读取的fixed_llama_hijack先apply_rotary再update cache，gpt_chat生成后通过负prefix offset调整response key，anchor_engine接收再做正offset。故本轮以**实际post-RoPE DynamicCache**为输入，并明确旧位置逆变换；没有把论文描述当作源码事实。

Direct append/position correction无需Token embedding或hidden states，标准Transformers cached suffix接口足够。官方选择性路径则使用完整input IDs、recompute indices、缓存的中间hidden states及influence候选；输入准备从指定位置取IDs、mask使用不同query绝对位置，hidden-state relay需逐层改forward。哪些hidden信息必须传输取决于7.2B选定算法，当前0B不代表未来免费。

## 尚未解决与7.2B判断

建议进入7.2B作为**受控研究**：先在更长消息和更有差异的独立历史上定位对齐后残差，考虑value偏差/末尾少量位置修复，再验证真实稀疏计算。当前4个短任务RoPE全对，无法证明修复的必要性，也没有加速收益。

Llama-3.2-1B尚未加载或验收；官方8B/Coder7B的层范围、阈值、10-Token suffix比例与GPU优化不应直接迁移。需先确认其实际RoPE scaling、cache/forward语义和基础任务质量，再采集自己的层/Token偏差。FP32对照、position-only control、同长度不同历史、长序列、多轮/更大固定任务集合、真实额外payload大小与优化计时仍待后续。

本轮不实现选择性重算、KVCOMM、latent reasoning、working-memory全量handoff或网络；不以Full Prefill替代KV修复。完成7.2A后停止，7.2B由用户决定。

## 验收

新增38项不加载真实权重的测试，覆盖截取边界、末Token补齐/重复补齐、层序、长度、公开DynamicCache构造、三组storage隔离、数学旋转/逆变换、EOS与上限终止、边界EOS去重、receiver KV绝不Prefill消息、NaN失败与严格评分。最终 `HF_HUB_OFFLINE=1 .venv/bin/pytest`：305 passed，2条既有FastAPI/Starlette上游弃用警告；`compileall -q src tests *.py`、CLI help、`git diff --check`通过。Frontend未改。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
