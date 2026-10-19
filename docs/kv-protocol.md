# KV Wire Protocol v1

## Scope

KV Wire Protocol v1 serializes every layer's key/value cache from a real causal
language-model Prefill into deterministic binary bytes, then parses those bytes
into a framework-neutral payload. It provides this boundary:

```text
HuggingFace DynamicCache
  -> HuggingFace KV Adapter
  -> NeutralKVPayload
  -> KV Wire Protocol v1 bytes
  -> strict parser
  -> NeutralKVPayload
```

Phase 5.2 reconstructs a Receiver `DynamicCache` from the parsed neutral
payload and verifies standard `past_key_values` continuation. Phase 5.3 uses
the resulting packet as opaque bytes across `BaseByteTransport`; the protocol
itself still does not run inference, choose endpoints, compress/quantize KV, or
implement adaptive TOKEN/KV selection.

Token Protocol v1 carries ordered Token IDs and requires Receiver Prefill. KV
Protocol v1 instead carries all layer-wise attention K/V values already computed
by the Sender. LatentMAS is a semantic reference for treating layer-wise KV as
transferable Agent working memory and propagating it through `past_key_values`.
This project currently implements standard causal-LM KV handoff data, not
LatentMAS latent thoughts, hidden-state autoregression, `Wa` alignment, or
`inputs_embeds` latent generation.

## Compatibility contract

Version 1 assumes homogeneous Agents: the Sender and future Receiver use the
same model family/checkpoint weights, Transformer architecture, layer count,
KV-head layout, head dimension, dtype interpretation, cache layout, and
position/cache semantics. The packet stores:

- UTF-8 `model_id`;
- `num_hidden_layers`;
- `num_key_value_heads`;
- `head_dim`.

These fields permit structural rejection but do not prove identical weights.
Version 1 deliberately does not hash approximately 1 GB of model weights.
Distributed use will need a model revision/fingerprint or compatibility
handshake. Heterogeneous layer mapping, projection, and automatic adaptation are
not supported.

`source_device` is UTF-8 diagnostic metadata only. A packet produced on `mps:0`
may later be restored to CUDA or CPU; device selection is Receiver policy and is
not a wire compatibility requirement.

## Packet layout

All metadata integers are unsigned and encoded in network byte order
(big-endian). Reserved fields must be zero. K always precedes V within a layer.
Layer records must appear exactly as layer 0, layer 1, ..., layer `L-1`.

### Fixed header (44 bytes)

| Offset | Size | Field | Encoding |
| ---: | ---: | --- | --- |
| 0 | 4 | magic | ASCII `TKVK` |
| 4 | 1 | version | `1` |
| 5 | 1 | flags | `0` |
| 6 | 2 | reserved | `0` |
| 8 | 4 | layer count | uint32, big-endian |
| 12 | 8 | logical cached token count | uint64, big-endian |
| 20 | 8 | total packet bytes | uint64, big-endian |
| 28 | 4 | model hidden-layer count | uint32, big-endian |
| 32 | 4 | model KV-head count | uint32, big-endian |
| 36 | 4 | model head dimension | uint32, big-endian |
| 40 | 2 | model ID byte length | uint16, big-endian |
| 42 | 2 | source device byte length | uint16, big-endian |

The header is followed by exactly `model_id_length` UTF-8 bytes and then exactly
`source_device_length` UTF-8 bytes.

### Layer and tensor records

Each layer starts with a 4-byte big-endian uint32 layer index. It is followed by
one K tensor record and one V tensor record.

Each tensor descriptor is:

| Size | Field | Encoding |
| ---: | --- | --- |
| 1 | dtype code | enum below |
| 1 | rank | uint8 |
| 2 | reserved | zero |
| 8 | tensor body byte length | uint64, big-endian |
| `rank * 8` | dimensions | positive uint64 values, big-endian |
| declared byte length | tensor data body | canonical little-endian elements |

Ranks and dimensions are encoded rather than assuming Qwen's current rank-4
shape. The HuggingFace adapter separately validates the current homogeneous Qwen
layout `[batch, num_key_value_heads, cached_tokens, head_dim]`.

## Tensor dtype and byte order

Version 1 codes are:

| Code | Logical dtype | Bytes per element |
| ---: | --- | ---: |
| 1 | bfloat16 | 2 |
| 2 | float16 | 2 |
| 3 | float32 | 4 |

Unknown codes are rejected. Tensor elements use canonical little-endian byte
order, independent of host byte order. Current little-endian hosts use the raw
contiguous byte sequence directly; a big-endian host reverses bytes within each
element. Strides are not wire semantics: a non-contiguous source may be made
contiguous and is then encoded in canonical logical row-major order.

Bfloat16 is viewed as bytes after device-to-CPU copy. It is never converted to
float32 or float16, so NaNs, signed zero, and every other raw bit pattern survive
round-trip exactly.

## Cache length and completeness

`logical_cached_token_count` describes positions actually present in the cache
at serialization time. Every layer's K and V sequence dimension must equal this
value or the adapter rejects the cache.

A Prefill over a prompt caches the prompt and uses its logits to predict `y1`;
`y1` is not cached until a later Decode forward consumes it. Likewise, stopping
immediately after predicting a final output Token may leave that Token outside
the cache. The serializer never executes an extra forward, adds a Token,
materializes a prediction, or infers semantic generated-output length. Phase
5.2 supplies an explicit helper; whether a real handoff invokes it belongs to
the Phase 5.3 semantic boundary.

The packet contains no prompt or generated Token IDs and does not recompute or
modify RoPE/position-derived K values. Phase 5.2 now provides an explicit
pre-serialization materialization helper as a Sender policy candidate; this
does not change packet v1 semantics.

For the Phase 5.3 canonical benchmark, the controlling handoff session defines
one `handoff_token_ids` tuple outside the packet. Sender clean-Prefills exactly
that complete tuple, so the packet's logical cache length must equal the tuple
length and no pending prediction exists. The same tuple is used by the TOKEN
comparison path. A full Sender private/history cache is valid working memory,
but is not a same-semantic comparison against message-only Token transfer.

The real Agent handoff must pass `KVSerializationResult.packet` itself through
the byte-only transport and parse the delivered bytes. Passing a
`NeutralKVPayload` or `DynamicCache` directly to the Receiver is not a valid KV
Data Plane shortcut. `total_wire_bytes` and the transport record must both
equal `len(packet)`.

## Size definitions and timing boundaries

- `raw_kv_tensor_bytes`: sum of `numel * element_size` over the Sender's actual
  K/V tensors.
- `wire_tensor_body_bytes`: sum of all K/V tensor data bodies in the packet.
- `wire_metadata_bytes`: total packet bytes minus tensor-body bytes.
- `total_wire_bytes`: exact `len(packet)`.

Because v1 is complete and uncompressed,
`wire_tensor_body_bytes == raw_kv_tensor_bytes`. Metadata is not included in the
body, and `total_wire_bytes = wire_metadata_bytes + wire_tensor_body_bytes`.

The adapter measures `device_to_host_ms` using the existing device synchronization
helper: synchronize, start, copy each K/V tensor to CPU, synchronize, stop. It
does not clone the complete cache on the accelerator. `protocol_encode_ms`
measures neutral host representation to final packet. `total_serialize_ms`
measures the adapter/codec operation as a whole. `deserialize_ms` covers packet
bytes to neutral immutable host data only; it excludes Receiver device upload,
cache reconstruction, and inference.

## Parser validation and safety limits

The parser treats packet bytes as untrusted. Before consuming declared bodies it
checks the fixed header, magic/version/flags/reserved fields, exact total packet
length, UTF-8 metadata lengths, supported dtype, rank, every positive dimension,
shape-product byte count, per-tensor limits, continuous unique layer indexes,
declared/config layer agreement, logical sequence length, truncation, and
trailing data.

Central v1 limits are:

- 4,096 layers;
- rank 2 through 8 (the sequence dimension is `shape[-2]`);
- each dimension at most `2^31 - 1`;
- model ID at most 4,096 bytes and device metadata at most 256 bytes;
- one tensor body at most 8 GiB;
- one packet at most 16 GiB.

Python arbitrary-precision arithmetic is used for checked shape products, while
wire tensor and total-length fields are uint64 so the protocol is not inherently
limited to 4 GiB.

Version 1 has no checksum, encryption, compression, quantization, or pruning. It
detects structural and length errors but cannot detect every same-length silent
bit corruption. A future transport may provide integrity protection, or a later
protocol version may add it.

## Version evolution and Receiver restore semantics

Magic and version are separate; an unknown version is rejected. Future versions
may add fingerprints, integrity checks, new dtypes, layout identifiers, or
position metadata only after real Receiver continuation experiments justify
them.

Phase 5.2 takes the already parsed `NeutralKVPayload`—full K/V values, ordered
layers, shapes, dtypes, actual cached length, and compatibility metadata—and
validates it against the Receiver before reconstructing CPU tensors or starting
device upload. Bfloat16 is restored by byte view without an FP32 intermediate.

The Transformers 5.17 adapter uses the public
`DynamicCache(ddp_cache_data=..., config=...)` constructor. That installed
version has no `from_legacy_cache`, and the adapter neither constructs a legacy
tuple-of-tuples nor assigns `cache.layers[*].keys/values`. After reconstruction,
the adapter validates cache type semantics, layer tensors and
`get_seq_length()`. Protocol v1 remains independent of this API choice.

For an inherited cache of length N and Receiver suffix of length M, Qwen2 5.17
derives positions from `Cache.get_seq_length()` while the model adapter supplies
an attention mask of length N+M. The restored path never needs prefix Token IDs
and never performs Receiver Prefill; clean prefix+suffix recomputation exists
only as a correctness oracle. A pending predicted Token can be materialized by
one explicit model forward before serialization. The pending ID is not added to
protocol v1, and that forward's cost is reported separately.

## 7.6A 外层 Incremental Message Envelope

`TKVI` envelope v1 嵌套本协议原封不动的 Delta KV bytes 和 Token Protocol v1 内容 IDs。logical_cached_token_count 在外层增量语义中是精确消息内容 Token 数，不是 Sender 全工作 Cache 长度。Canonical 模式仍采用 Clean Prefill(M)。增量模式从 Decode 已有 KV(M | H) 截取，排除 EOS/EOT/header/closing；SHA-256 损坏校验和实际权重/词表/RoPE配置合约由外层提供，不改变 KV v1 Tensor 编码。Receiver 本阶段仅恢复中立 payload；位置目标/修正与长期追加留 7.6B。详见 [增量消息格式](incremental-kv-transport.md)。


> 公开副本范围：本页保留阶段性说明；原始日志、全部冻结证据、历史提交和未选入截图不随本副本发布。缺省资料的链接已改为明确的文字标记，历史 SHA 不是本仓库提交。
