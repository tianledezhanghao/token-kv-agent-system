# Token Protocol v1

## 目标与边界

Token Protocol v1 是 `token-kv-agent-system` Token mode 的第一个真实二进制 Data Plane 序列化格式。它将一个有序 `tuple[int, ...]` 编码为可跨进程、跨机器解析的完整 byte stream，并能不依赖 tokenizer、HuggingFace 对象、PyTorch 或模型类还原 Token IDs。

本阶段只定义 codec 和 wire size，不实现 Socket、Agent-to-Agent transport、协议协商、KV serialization 或接收端模型执行。`TokenPayload` 仍表示领域层的逻辑 Token IDs；`TokenWirePacket` 和 codec 属于 Data Plane serialization layer。

## 三种字节口径

- Token logical payload：`ceil(token_count × ceil(log2(vocab_size)) / 8)`，是理想 bit packing 的理论下界。
- Token wire payload：`len(serialize_token_ids(...))`，是本协议真正生成的 header 加 Token body byte stream。
- Python / torch memory：Python `int` 或 `torch.int64` 在运行时的内存占用，不是 network wire size。

## Packet layout

所有多字节整数都使用 network byte order（big-endian）。Header 固定为 10 bytes：

| Offset | Size | Field | v1 value / meaning |
| ---: | ---: | --- | --- |
| 0 | 4 B | `magic` | ASCII `TKVT` (`54 4b 56 54`) |
| 4 | 1 B | `version` | `1` |
| 5 | 1 B | `token_width_bits` | `32` |
| 6 | 4 B | `token_count` | unsigned big-endian Token 数量 |
| 10 | `4 × token_count` B | `tokens` | 按原顺序排列的 unsigned 32-bit big-endian Token IDs |

因此：

```text
wire_header_bytes     = 10
wire_token_data_bytes = token_count × 4
total_wire_bytes      = 10 + token_count × 4
wire_overhead_bytes   = total_wire_bytes - theoretical_minimum_bytes
```

Magic 和 version 是独立字段：`TKVT` 用于识别 Token/KV 系统的 Token packet，不把版本号隐藏在 magic 中。

## 为什么 v1 选择 uint32

`Qwen/Qwen2.5-0.5B-Instruct` 的 model config `vocab_size` 为 151936，理论上需要 18 bits/token。`uint16` 最大只能表示 65535，不能覆盖该模型的完整 Token ID 域。

v1 选择 4 B/token 的 uint32，以换取简单、稳定、易审计和确定性解析。18-bit bit packing 可以更接近 theoretical minimum，但会引入跨 byte 边界、padding 和更复杂的 malformed stream 处理，不适合作为第一版正确性基线。

## 序列化 API

- `serialize_token_ids(token_ids) -> bytes`
- `serialize_token_payload(payload) -> bytes`
- `deserialize_token_packet(data) -> TokenWirePacket`
- `measure_token_wire_size(payload, packet) -> TokenWireSize`

Serializer 只处理 Token IDs 和 framing，不调用模型/tokenizer，不了解 Agent、Network、bandwidth、Selector 或 Dashboard。数据面不使用 pickle、marshal、`torch.save` 或 JSON Token 数组。

## Validation rules

v1 允许空 Token 序列；它会产生只含 10-byte header 且 count 为 0 的合法 packet。

Serializer 拒绝：

- 非整数、负 Token ID 和大于 `4294967295` 的 Token ID；
- 超过 `1,000,000` 个 Token 的 packet。

Deserializer 拒绝：

- 少于 10 bytes 的 header；
- 错误 magic；
- 未知 version；
- v1 不支持的 token width；
- 超过 `1,000,000` 的声明 token count；
- 实际长度小于 `10 + count × 4` 的 truncated packet；
- 实际长度大于 `10 + count × 4` 的 trailing garbage。

先对声明 count 做上限和精确长度校验，再解析 body，避免信任未来网络输入中的任意大 count。

## Future evolution

新 framing 或 Token representation 必须使用新 version，当前 codec 明确拒绝未知版本，不做复杂 negotiation。后续可在独立任务中评估 packed Token-ID encoding、压缩或更大 framing；KV protocol 不复用本 Token body 定义。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
