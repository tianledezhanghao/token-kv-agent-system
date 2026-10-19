"""Versioned incremental-message envelope around unchanged TOKEN/KV v1 bytes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
import re
from struct import Struct
from typing import Any

from .kv_protocol import (DTYPE_ELEMENT_BYTES, NeutralKVPayload,
                          deserialize_kv_packet, serialize_kv_payload)
from .token_protocol import deserialize_token_packet, serialize_token_ids

_HEADER = Struct('!4sBIII')  # magic, version, metadata/TOKEN/KV byte lengths
MAGIC = b'TKVI'
VERSION = 1
DIGEST_BYTES = 32
INCREMENTAL_WIRE_HEADER_BYTES = _HEADER.size
MAX_METADATA_BYTES = 64 * 1024


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('utf-8')


@dataclass(frozen=True)
class IncrementalLimits:
    max_packet_bytes: int = 64 * 1024**2
    max_message_tokens: int = 4096
    max_sender_positions: int = 1 << 20

    def __post_init__(self) -> None:
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError('incremental limits must be positive integers')


@dataclass(frozen=True)
class ModelContract:
    """Receiver-owned endpoint contract, not a shape-only weights assumption."""
    model_id: str
    identity_sha256: str
    weights_sha256: str
    config_sha256: str  # Entire loaded config, including RoPE/position settings.
    vocab_size: int
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype: str
    eos_token_ids: tuple[int, ...]
    special_token_ids: tuple[int, ...]
    layout: str = 'dense_batch1_BHSD'
    position_semantics: str = 'sender_absolute_dense_cache_indices_unmodified'


def inspect_model_contract(backend: Any) -> ModelContract:
    """Hash actual loaded parameters/buffers once at channel setup.

    Models must remain immutable for this channel's lifetime. Chunking bounds
    temporary host copies. Device placement is deliberately absent from identity.
    """
    from .model.backend_pair import inspect_backend_identity
    identity = inspect_backend_identity(backend)
    config = backend._model.config.to_dict()
    digest = hashlib.sha256()
    for name, tensor in sorted(backend._model.state_dict().items()):
        digest.update(canonical_json([name, str(tensor.dtype), list(tensor.shape)]))
        flat = tensor.detach().contiguous().reshape(-1)
        for offset in range(0, flat.numel(), 262144):
            digest.update(flat[offset:offset + 262144].cpu().view(backend._torch.uint8).numpy().tobytes())
    structure = backend.kv_model_config
    return ModelContract(backend.model_name, hashlib.sha256(canonical_json(asdict(identity))).hexdigest(),
                         digest.hexdigest(), hashlib.sha256(canonical_json(config)).hexdigest(),
                         backend.vocab_size, structure.num_hidden_layers,
                         structure.num_key_value_heads, structure.head_dim,
                         backend.metadata.dtype.removeprefix('torch.'),
                         tuple(backend.eos_token_ids), tuple(identity.all_special_ids))


@dataclass(frozen=True)
class IncrementalKVMessage:
    message_id: str
    sequence: int  # Per directed source/target stream.
    source_agent: str
    target_agent: str
    message_token_ids: tuple[int, ...]  # Content only, also carried in TOKEN v1.
    message_token_count: int
    termination_reason: str
    termination_token_id: int | None
    sender_span_start: int
    sender_span_length: int
    model_contract: ModelContract
    delta: NeutralKVPayload
    payload_mode: str = 'incremental_kv'
    payload_format_version: int = VERSION

    @property
    def raw_generated_token_ids(self) -> tuple[int, ...]:
        return self.message_token_ids + (() if self.termination_token_id is None
                                         else (self.termination_token_id,))

    def metadata(self) -> dict:
        metadata = {f.name: getattr(self, f.name) for f in fields(self)
                    if f.name not in ('message_token_ids', 'delta', 'model_contract')}
        return {**metadata, 'model_contract': asdict(self.model_contract)}


def validate_message(message: IncrementalKVMessage, *, expected_contract: ModelContract,
                     source: str, target: str, sequence: int,
                     limits: IncrementalLimits = IncrementalLimits()) -> None:
    if type(message.payload_format_version) is not int or message.payload_format_version != VERSION or message.payload_mode != 'incremental_kv':
        raise ValueError('unsupported incremental message version/mode')
    if (message.source_agent, message.target_agent, message.sequence) != (source, target, sequence):
        raise ValueError('incorrect source/target or duplicate/out-of-order sequence')
    if any(not isinstance(s, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', s)
           for s in (source, target)) or source == target:
        raise ValueError('invalid endpoint identity')
    if type(message.sequence) is not int or message.sequence <= 0 or message.message_id != f'{source}:{sequence}':
        raise ValueError('invalid message identity/sequence')
    for value in (message.message_token_count, message.sender_span_length):
        if type(value) is not int or not 0 < value <= limits.max_message_tokens:
            raise ValueError('invalid/oversized message count or span length')
    if type(message.sender_span_start) is not int or not 0 <= message.sender_span_start:
        raise ValueError('invalid sender span start')
    if message.sender_span_start + message.sender_span_length > limits.max_sender_positions:
        raise ValueError('sender span exceeds position limit')
    count = len(message.message_token_ids)
    if message.message_token_count != count or message.sender_span_length != count:
        raise ValueError('message IDs/count/span length mismatch')
    contract = message.model_contract
    if contract != expected_contract:
        raise ValueError('incompatible checkpoint/tokenizer/config/cache contract')
    if contract.dtype not in DTYPE_ELEMENT_BYTES or contract.layout != 'dense_batch1_BHSD':
        raise ValueError('unsupported model dtype/layout')
    if any(type(i) is not int or not 0 <= i < contract.vocab_size
           or i in (*contract.special_token_ids, *contract.eos_token_ids) for i in message.message_token_ids):
        raise ValueError('content Token ID outside vocabulary or is a control token')
    if message.termination_reason == 'eos':
        if type(message.termination_token_id) is not int or message.termination_token_id not in contract.eos_token_ids:
            raise ValueError('invalid EOS termination metadata')
    elif message.termination_reason not in ('max_new_tokens', 'context_limit') or message.termination_token_id is not None:
        raise ValueError('invalid termination metadata')
    payload = message.delta
    model = payload.model
    if ((model.model_id, model.num_hidden_layers, model.num_key_value_heads, model.head_dim) !=
            (contract.model_id, contract.num_hidden_layers, contract.num_key_value_heads, contract.head_dim)
            or len(payload.layers) != contract.num_hidden_layers
            or payload.logical_cached_token_count != count or payload.source_device != 'cpu'):
        raise ValueError('incompatible Delta model/layers/length/neutral device')
    for index, layer in enumerate(payload.layers):
        if layer.layer_index != index:
            raise ValueError('incorrect Delta layer order')
        for tensor in (layer.key, layer.value):
            shape = (1, contract.num_key_value_heads, count, contract.head_dim)
            if tensor.shape != shape or tensor.dtype != contract.dtype:
                raise ValueError('invalid Delta shape/dtype/head dimension')
            expected_bytes = contract.num_key_value_heads * count * contract.head_dim * DTYPE_ELEMENT_BYTES[contract.dtype]
            if not isinstance(tensor.data, bytes) or len(tensor.data) != expected_bytes:
                raise ValueError('invalid Delta tensor byte size')


@dataclass(frozen=True)
class IncrementalEncodingResult:
    packet: bytes
    kv_wire_bytes: int
    kv_serialize_ms: float


def encode_incremental_message(message: IncrementalKVMessage, *,
                               limits: IncrementalLimits = IncrementalLimits()) -> bytes:
    return encode_incremental_message_with_metrics(message, limits=limits).packet


def encode_incremental_message_with_metrics(message: IncrementalKVMessage, *,
                               limits: IncrementalLimits = IncrementalLimits()) -> IncrementalEncodingResult:
    validate_message(message, expected_contract=message.model_contract,
                     source=message.source_agent, target=message.target_agent,
                     sequence=message.sequence, limits=limits)
    metadata = canonical_json(message.metadata())
    if len(metadata) > MAX_METADATA_BYTES:
        raise ValueError('incremental metadata exceeds limit')
    # Bound before allocating the KV packet.
    if message.delta.raw_kv_tensor_bytes + len(metadata) + 4 * message.message_token_count > limits.max_packet_bytes:
        raise ValueError('incremental packet exceeds size limit')
    tokens = serialize_token_ids(message.message_token_ids)
    encoded = serialize_kv_payload(message.delta)
    kv = encoded.packet
    packet = _HEADER.pack(MAGIC, VERSION, len(metadata), len(tokens), len(kv)) + metadata + tokens + kv
    if len(packet) + DIGEST_BYTES > limits.max_packet_bytes:
        raise ValueError('incremental packet exceeds size limit')
    return IncrementalEncodingResult(packet + hashlib.sha256(packet).digest(),
                                     len(kv), encoded.protocol_encode_ms)


def _unique_object(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate JSON metadata key')
        result[key] = value
    return result


def decode_incremental_message(data: bytes, *, expected_contract: ModelContract,
                               source: str, target: str, sequence: int,
                               limits: IncrementalLimits = IncrementalLimits()) -> IncrementalKVMessage:
    if not isinstance(data, bytes):
        raise TypeError('incremental packet must be bytes')
    if not _HEADER.size + DIGEST_BYTES <= len(data) <= limits.max_packet_bytes:
        raise ValueError('truncated/oversized incremental packet')
    magic, version, nm, nt, nk = _HEADER.unpack_from(data)
    if magic != MAGIC or version != VERSION:
        raise ValueError('unsupported incremental magic/version')
    if nm > MAX_METADATA_BYTES or _HEADER.size + nm + nt + nk + DIGEST_BYTES != len(data):
        raise ValueError('invalid incremental lengths/truncated/trailing packet')
    if hashlib.sha256(data[:-DIGEST_BYTES]).digest() != data[-DIGEST_BYTES:]:
        raise ValueError('corrupt incremental checksum')
    offset = _HEADER.size
    try:
        metadata = json.loads(data[offset:offset + nm], object_pairs_hook=_unique_object)
        if not isinstance(metadata, dict):
            raise ValueError('incremental metadata must be an object')
        required = {f.name for f in fields(IncrementalKVMessage)} - {'message_token_ids', 'delta'}
        if set(metadata) != required:
            raise ValueError('missing or unknown incremental metadata fields')
        # Verify the local contract before parsing tensor bodies.
        if metadata.get('model_contract') != json.loads(canonical_json(asdict(expected_contract))):
            raise ValueError('incompatible checkpoint/tokenizer/config/cache contract')
        metadata['model_contract'] = expected_contract
        offset += nm
        if nt > 10 + 4 * limits.max_message_tokens:
            raise ValueError('oversized Token packet')
        ids = deserialize_token_packet(data[offset:offset + nt]).token_ids
        offset += nt
        delta = deserialize_kv_packet(data[offset:offset + nk]).payload
        message = IncrementalKVMessage(message_token_ids=ids, delta=delta, **metadata)
    except (TypeError, UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise ValueError('invalid incremental metadata') from exc
    validate_message(message, expected_contract=expected_contract, source=source, target=target,
                     sequence=sequence, limits=limits)
    return message
