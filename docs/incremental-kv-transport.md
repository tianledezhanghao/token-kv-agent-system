# 7.6A Incremental Message KV：Decode 出口与解包

## 7.6B Receiver Append（当前正式入口）

`StatefulTokenSession.handoff_kv(source, target, strategy=KVStrategy...)` 串联原 `prepare_incremental_message`、KV Protocol v1、Incremental Envelope v1、`InMemoryByteTransport`、strict decode、原 device restore adapter 与 `receive_incremental_kv`。无对象/tensor/Cache 传输捷径，无 Clean Prefill(M)。Envelope 所有必要字段已齐全，**格式仍为 v1，无字段或二进制语义修改**。下方 7.6A 章节描述历史旁路，仍可独立运行。

接收入口读取 Receiver 有效 AgentState，检查合约与共同发送序号，解包后恢复独立 device delta。工作副本先补齐 Receiver 自己的 pending/close，再正常 forward 本地 incoming user header；此时真实 cache length 才是目标起点。接下来：

- V0 `raw_delta`：原 dtype/device 的 Delta K/V 在 dim=2 直接拼接，不修正位置、不重算内容。
- V1 `rope_aligned_delta`：用 envelope 的 Sender start 和实际 Receiver start，连续生成两组真实模型 rotary tables，FP32 中反转旧 split-half RoPE 再施加新 RoPE，最后转回缓存 dtype；V 不旋转。位移逐条计算，不改 cache_position 冒充转换。

`model/delta_repair.py` 是 7.2A 原 append/rephase/table 算法的共同实现；研究模块导入同一函数，测试和旧实验接口保留。每次 V1 检查实际 model_type、rope_type、full_attention、inv_freq/rotary_dim、有限值与 split-half pairing。仅接受已审计 Qwen2/Llama 静态 default/llama3、完整 head rotary dimension；动态/未经审计/部分 rotary 模型明确拒绝。Llama 代码兼容检查不等于真实 CUDA 验收。长期 Cache 中没有 padding/sliding/位置重编号；位置依 backend 按真实 Cache 长度和完整 attention mask 延续。

仅将精确 content IDs 写入 ledger，不 forward M；再 forward 本地 closing。生成 EOT/EOS 只是控制 metadata，不追加其 Sender KV。匹配 EOT 在逻辑上仅出现一次；异种 EOS 由 Receiver 本地 EOT 闭合；截断保留全部内容并补 closing。每层 K/V shape/length/dtype/device、ledger 和独立 storage 必须通过检查。

入口返回 work state 而不提交。Session 验证解包与 Sender 原始 envelope 完全相同，关闭独立 Sender work state，写双方 sent/received 记录后才同时替换状态；任一解包/repair/append/closing/Sender-close 失败，旧双方 Cache、ledger、版本与序号均保留。已成功生成的 source draft 保留，不算已交付。TOKEN/KV 共用每方向消息序号，混合切换不会被独立旁路的 KV-only unpack counter 干扰。

首次 KV handoff 扫描两端实际权重/配置/tokenizer 指纹；共享 backend 只扫描一次。合约缓存于 session，两端模型/词表/配置须保持不可变（同 7.6A 限制）。Receiver framing、local 后续 prompt 和 Decode 可正常 forward；不把 M 的文字写回任何 prompt。clone、cat、构造 cache 仍有 O(历史 KV) 复制/内存成本；本轮未优化或宣称加速。

结果与 TOKEN 一般不等价：复用的是 KV(M | H_sender)，旋转修正不消除跨历史 hidden/value 差异。没有质量判定驱动的自动 TOKEN 重试。

本阶段只完成发送、二进制传输、解包和校验。`DiscussionRuntime` / `StatefulTokenSession` 的默认 TOKEN 路径完全不变；没有 Receiver KV 长期追加、RoPE 修正、V0/V1 正式策略或多轮 KV 讨论。V0/V1 接收与混合讨论留给 7.6B，V2/V3 尚未实现。

## 与 Canonical KV 的区别

Canonical 模式仍对同一 handoff IDs 从空 Cache 执行 Clean Prefill，发送 `KV(M)`，原协议和成本边界不变。增量消息发送 Sender 在自己的历史下正常 Decode 已计算的 `KV(M | H_A)`；不把 M 单独 Clean Prefill，不发送 `H_A` 的 KV。这里的历史包括本地阶段指令和 assistant header。两种 KV 的数值语义不同，不能把本阶段结果用于旧 Canonical Selector calibration。

## API 与范围证明

`model/decode_kv.py` 提取 7.2A 已验证的 `cache_layers`、`extract_span` 和 `complete_message` 通用操作；研究模块继续导入/导出相同函数，原研究算法和测试保留。实验的 append、旋转、任务编排没有进入正式出口。

`prepare_incremental_message(state, message, contract=...)` 接受实际 Sender draft 或已发送记录，验证 AgentState、原始 span 和精确 ledger IDs。原始 `AgentMessage.sender_message_span=[start,raw_end)` 覆盖全部生成 IDs：

| 范围 | 增量消息 KV |
| --- | --- |
| 原历史、阶段指令 | 排除 |
| 本地 assistant header | 排除；start 在 header 后 |
| 真实生成内容 M | 保留 `[start,start+len(content))` |
| 末尾 EOS/EOT | 作为 termination metadata，排除 |
| 本地 closing / 普通尾部 | 排除 |

7.5B TOKEN wire **仍包含生成 EOS**。其 Receiver 将最后 EOS 作为控制信息处理，内容为 `raw[:-1]`；匹配本地 closing 的 EOT 可以复用为 closing，其他 EOS 用本地 EOT 替换。增量 envelope 的嵌套 TOKEN v1 只编码 `content`，终止 Token 另存 `termination_token_id`，可恢复 raw generated IDs。`max_new_tokens` 则保留所有生成 IDs，不凭文本或 special-token decode 猜边界。

所有层 K/V `shape[2] == message_token_count == len(message_token_ids) == sender_span_length`。旁路还核对 TOKEN Receiver 的实际 ledger 内容 slice，证明两条表示对应同一内容；Receiver 自己的 user prefix/identity label/closing 是其本地 framing，不属于 M。若边界不能证明就报错，不标成功。

如果内容已缓存，不执行 forward。若只有最后一个 accepted 内容 Token pending，则先 clone Sender Cache 到隔离 work storage，只调用一次 `materialize_token_into_cache(content[-1])`，确认增长恰好 1，再提取。新 logits prediction 不接受、不写 ledger。原有效 Sender Cache、pending 和消息记录保持不变；forward 失败也只污染副本。EOS 不属于内容时不为增量出口补算 EOS。超过一个缺失位置直接拒绝。

`extract_huggingface_kv_span` 在每层 `[1,Hkv,S,D]` 上截取并 clone 精确 span，构造只用于序列化的 view，复用原 HF KV Adapter 的 dtype/shape 校验和无损 CPU byte extraction。中立载荷的 `source_device=cpu` 表示 CPU 表示，不硬编码原 GPU placement；原设备只进入本地诊断。

## Envelope v1

```text
TKVI[4] | version=1[1] | metadata_bytes[uint32]
        | token_packet_bytes[uint32] | kv_packet_bytes[uint32]
        | UTF-8 canonical JSON metadata
        | unchanged Token Protocol v1 (content IDs)
        | unchanged KV Protocol v1 (only Delta KV)
        | SHA-256[32] of all preceding envelope bytes
```

固定头 17 B，所有整数字段使用 big-endian。KV v1 的 tensor body 继续采用原 canonical little-endian raw BF16/FP16/FP32，不重新实现 Tensor 编码、不转 dtype。KV 的 logical_cached_token_count 在这个 envelope 中明确指 Delta 的长度，而非 Sender 全历史长度；这是外层语义，没有改动 TKVK 格式或 Canonical 用法。

Metadata 必须完整且无额外/重复 key：message_id（`source:sequence`）、sequence（每方向单独从 1 递增）、source_agent、target_agent、message_token_count、termination_reason、termination_token_id、sender_span_start、sender_span_length、payload_mode=`incremental_kv`、payload_format_version=1、model_contract。message_token_ids 位于嵌套 TOKEN v1 中，与全部 metadata/终止信息一起计入通信 bytes。

model_contract 包含 model ID、词表/模型身份摘要、**实际加载参数与 buffer 的 SHA-256**、完整配置 SHA-256（包括 RoPE/位置配置标识）、vocab size、层数、KV heads、head_dim、dtype、EOS/special IDs、dense batch-one BHSD layout 和 Sender 原始绝对位置解释。Receiver 用本地完整配置验证配置标识，不需要 Sender 知道其历史或目标位置。两个独立 backend 必须全部匹配；仅 shape/model name 相同不够，异构模型转换不支持。摘要在 channel 初始化时计算，模型权重与配置须在 channel 生命周期内保持不变；设备 placement 不进入合约。此初始化扫描的成本不属于逐消息序列化时间。

默认上限：总包 64 MiB、metadata 64 KiB、消息 4096 Tokens、Sender span end 不超过 2^20。限制可通过 `IncrementalLimits` 配置；在 clone/提取前检查消息资源界限，在 parse 前检查实际 bytes、各 section 长度和 checksum；KV v1 再做原有 tensor rank/dimension/body/order 校验。SHA-256 用来检出载荷损坏，不定义网络认证。

## Sender / Receiver 与失败隔离

```text
Sender Decode Cache → ledger-proven content span → isolated minimal pending completion
 → Delta tensors → existing HF Adapter → neutral CPU bytes
 → existing KV v1 + TOKEN v1 → envelope bytes
 → InMemoryByteTransport → strict envelope + nested codec parse
 → independent immutable neutral Delta KV
```

`IncrementalKVChannel` 是独立 A/B 旁路，注册同构端点，不替换讨论 session。`IncrementalKVReceiver.unpack` 不持有 AgentState，不上传 tensor、不构建/追加长期 DynamicCache、不调用模型 forward。只在完整验证（以及旁路 bitwise oracle）通过后推进本方向接收序号。

检查来源/目标/message_id/序号、数量/span、vocabulary 与 control Tokens、termination、合约、层数/顺序、每层 K/V dtype/shape/head_dim/seq_len/raw byte size。拒绝重复/乱序、损坏、截断、尾随、版本错误、不兼容和过大载荷。Channel 在提取前拒绝重复序号，transport identity 校验与全部 K/V 原始字节 equality 在接受前完成；失败不改变任何 Agent 的有效 cache/ledger/version/pending/message 记录或接收序号。Transport 可能保留失败尝试的 byte-count 诊断。

纯 parser 可还原合法 bytes；channel 同时执行 Sender/decoded 全载荷原始 byte oracle，不能以 allclose 代替。成功只代表 unpack，不代表 KV 能安全用于 Receiver 自身历史。

## 独立 CLI 与诊断

```bash
HF_HUB_OFFLINE=1 .venv/bin/python run_incremental_kv_transport_probe.py \
  --device mps --question "23 + 48 = ?" --rounds 2 --max-new-tokens 64 \
  --output benchmark-results/incremental-kv-transport-add.json
```

CLI 在单个 probe 实例临时包装原 `handoff`：先正常 TOKEN handoff（原流程已补齐并 closed），随后从 Sender 已接受的 sent record 提取 Delta。结束后恢复原方法；生产 DiscussionRuntime 文件没有改动。因此旁路不重复补算已经在 TOKEN close 中缓存的内容位置，也不为验证独立 Clean Prefill(M)。Receiver 的历史增长来自原 TOKEN handoff，KV 接收增长始终为 0。KV 旁路失败会令 probe 报告失败，但不会撤销之前已成功的 TOKEN 阶段；它不声称整个 TOKEN+KV probe 是一个跨通道事务。

显示原讨论文本、raw/content IDs、termination、生成时与提取时 Sender Cache 长度、Delta span/seq_len、模型/设备/dtype/层数、原 TOKEN wire、envelope 内 content TOKEN wire、Delta raw、KV v1 wire、envelope overhead/total、tensor 提取（含 device→CPU）、pending forward、KV serialize、envelope serialize 和完整 deserialize 时间。JSON 保存相同证据与 transport records。pending forward 时间是模型同步 forward，不包含副本 clone；clone 有 O(历史 KV) 时间/内存成本。

单 EOS 的空内容在 TOKEN 讨论中合法，但本出口明确报错，因为已有 KV v1 要求非空 span；不伪造零长度 KV。默认只使用本地模型 cache，可显式允许下载。支持 CPU codec/mock 和 tiny Qwen2；实测 Mac/Qwen/MPS，未验收 Llama/CUDA、真实第二独立 checkpoint 实例或真实网络。测试与测量见 [7.6A 验收](incremental-kv-transport-validation.md)。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
