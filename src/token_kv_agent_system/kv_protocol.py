"""Dependency-free KV Wire Protocol v1 codec and neutral payload types."""

from __future__ import annotations

from dataclasses import dataclass
from struct import Struct
from time import perf_counter


KV_PROTOCOL_MAGIC = b"TKVK"
KV_PROTOCOL_VERSION = 1
KV_PROTOCOL_FLAGS = 0

DTYPE_BFLOAT16 = 1
DTYPE_FLOAT16 = 2
DTYPE_FLOAT32 = 3

DTYPE_NAME_TO_CODE = {
    "bfloat16": DTYPE_BFLOAT16,
    "float16": DTYPE_FLOAT16,
    "float32": DTYPE_FLOAT32,
}
DTYPE_CODE_TO_NAME = {code: name for name, code in DTYPE_NAME_TO_CODE.items()}
DTYPE_ELEMENT_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}

MAX_LAYER_COUNT = 4_096
MAX_TENSOR_RANK = 8
MAX_DIMENSION = (1 << 31) - 1
MAX_LOGICAL_CACHED_TOKENS = (1 << 63) - 1
MAX_MODEL_ID_BYTES = 4_096
MAX_DEVICE_BYTES = 256
MAX_TENSOR_BYTES = 8 * 1024**3
MAX_PACKET_BYTES = 16 * 1024**3

# All metadata integers use network byte order. Possible large sizes are uint64.
# magic, version, flags, reserved, layers, logical tokens, total packet bytes,
# config layers, KV heads, head dim, model-id bytes, source-device bytes.
_HEADER = Struct("!4sBBHIQQIIIHH")
_LAYER = Struct("!I")
# dtype code, rank, reserved, raw tensor-body byte length.
_TENSOR = Struct("!BBHQ")
_DIMENSION = Struct("!Q")
KV_WIRE_FIXED_HEADER_BYTES = _HEADER.size


@dataclass(frozen=True)
class KVWireSizeEstimate:
    raw_kv_tensor_bytes: int
    wire_tensor_body_bytes: int
    wire_metadata_bytes: int
    total_wire_bytes: int


def estimate_kv_wire_size(
    token_count: int, model: KVModelCompatibility, *, dtype: str, source_device: str,
    batch_size: int = 1,
) -> KVWireSizeEstimate:
    """Exact dense v1 K/V packet size for uniform [batch, heads, tokens, dim]."""
    if not isinstance(model, KVModelCompatibility):
        raise TypeError("model must be KVModelCompatibility")
    _validate_packet_counts(model.num_hidden_layers, token_count, model.num_hidden_layers,
                            model.num_key_value_heads, model.head_dim)
    if dtype not in DTYPE_ELEMENT_BYTES:
        raise ValueError("unsupported KV dtype")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model_bytes = model.model_id.encode("utf-8")
    device_bytes = source_device.encode("utf-8")
    if not model_bytes or len(model_bytes) > MAX_MODEL_ID_BYTES or len(device_bytes) > MAX_DEVICE_BYTES:
        raise ValueError("invalid model/device metadata length")
    rank = 4
    tensor_count = model.num_hidden_layers * 2
    raw = tensor_count * batch_size * model.num_key_value_heads * token_count * model.head_dim * DTYPE_ELEMENT_BYTES[dtype]
    per_tensor = raw // tensor_count
    if per_tensor > MAX_TENSOR_BYTES:
        raise ValueError("tensor exceeds protocol maximum")
    metadata = (_HEADER.size + len(model_bytes) + len(device_bytes)
                + model.num_hidden_layers * _LAYER.size
                + tensor_count * (_TENSOR.size + rank * _DIMENSION.size))
    total = raw + metadata
    if total > MAX_PACKET_BYTES:
        raise ValueError("packet exceeds protocol maximum")
    return KVWireSizeEstimate(raw, raw, metadata, total)


class KVProtocolError(ValueError):
    """Base class for invalid neutral payloads or KV packets."""


class KVSerializationError(KVProtocolError):
    """Raised when a neutral payload cannot be represented by protocol v1."""


class KVDeserializationError(KVProtocolError):
    """Raised when bytes are not exactly one valid protocol v1 packet."""


class UnsupportedKVProtocolVersion(KVDeserializationError):
    """Raised when the packet version is not supported."""


@dataclass(frozen=True)
class KVModelCompatibility:
    """Structural compatibility facts; v1 also assumes identical checkpoint weights."""

    model_id: str
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int


@dataclass(frozen=True)
class NeutralKVTensor:
    """One canonical contiguous tensor represented by immutable little-endian bytes."""

    dtype: str
    shape: tuple[int, ...]
    data: bytes

    @property
    def byte_length(self) -> int:
        return len(self.data)


@dataclass(frozen=True)
class NeutralKVLayer:
    """One indexed Transformer layer, with K followed by V on the wire."""

    layer_index: int
    key: NeutralKVTensor
    value: NeutralKVTensor


@dataclass(frozen=True)
class NeutralKVPayload:
    """Framework-neutral, sender-independent layer-wise KV working memory."""

    model: KVModelCompatibility
    logical_cached_token_count: int
    source_device: str
    layers: tuple[NeutralKVLayer, ...]

    @property
    def raw_kv_tensor_bytes(self) -> int:
        return sum(
            layer.key.byte_length + layer.value.byte_length for layer in self.layers
        )


@dataclass(frozen=True)
class KVSerializationResult:
    """Actual packet and distinctly measured serialization stages."""

    packet: bytes
    raw_kv_tensor_bytes: int
    wire_tensor_body_bytes: int
    wire_metadata_bytes: int
    total_wire_bytes: int
    device_to_host_ms: float
    protocol_encode_ms: float
    total_serialize_ms: float

    @property
    def wire_overhead_ratio(self) -> float:
        return self.wire_metadata_bytes / self.wire_tensor_body_bytes


@dataclass(frozen=True)
class KVDeserializationResult:
    """Neutral parsed payload plus packet-size and decode timing evidence."""

    payload: NeutralKVPayload
    wire_tensor_body_bytes: int
    wire_metadata_bytes: int
    total_wire_bytes: int
    deserialize_ms: float


def serialize_kv_payload(payload: NeutralKVPayload) -> KVSerializationResult:
    """Encode one validated neutral payload without any device transfer."""

    started = perf_counter()
    packet, tensor_body_bytes = _encode_payload(payload)
    encode_ms = (perf_counter() - started) * 1_000
    total_bytes = len(packet)
    return KVSerializationResult(
        packet=packet,
        raw_kv_tensor_bytes=payload.raw_kv_tensor_bytes,
        wire_tensor_body_bytes=tensor_body_bytes,
        wire_metadata_bytes=total_bytes - tensor_body_bytes,
        total_wire_bytes=total_bytes,
        device_to_host_ms=0.0,
        protocol_encode_ms=encode_ms,
        total_serialize_ms=encode_ms,
    )


def deserialize_kv_packet(
    data: bytes | bytearray | memoryview,
) -> KVDeserializationResult:
    """Strictly parse exactly one untrusted KV Wire Protocol v1 packet."""

    started = perf_counter()
    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    packet = bytes(data)
    if len(packet) > MAX_PACKET_BYTES:
        raise KVDeserializationError("packet exceeds configured maximum size")
    if len(packet) < KV_WIRE_FIXED_HEADER_BYTES:
        raise KVDeserializationError("packet is too short for the fixed header")

    (
        magic,
        version,
        flags,
        reserved,
        layer_count,
        logical_tokens,
        declared_total_bytes,
        config_layers,
        kv_heads,
        head_dim,
        model_id_length,
        device_length,
    ) = _HEADER.unpack_from(packet)
    if magic != KV_PROTOCOL_MAGIC:
        raise KVDeserializationError("invalid KV protocol magic")
    if version != KV_PROTOCOL_VERSION:
        raise UnsupportedKVProtocolVersion(f"unsupported KV protocol version {version}")
    if flags != KV_PROTOCOL_FLAGS or reserved != 0:
        raise KVDeserializationError("unsupported flags or non-zero reserved field")
    _validate_packet_counts(
        layer_count, logical_tokens, config_layers, kv_heads, head_dim
    )
    if declared_total_bytes != len(packet):
        relation = "truncated" if len(packet) < declared_total_bytes else "trailing"
        raise KVDeserializationError(
            f"{relation} packet: declared {declared_total_bytes} bytes, "
            f"received {len(packet)}"
        )
    if model_id_length == 0 or model_id_length > MAX_MODEL_ID_BYTES:
        raise KVDeserializationError("invalid model identifier length")
    if device_length > MAX_DEVICE_BYTES:
        raise KVDeserializationError("invalid source device length")

    offset = KV_WIRE_FIXED_HEADER_BYTES
    model_bytes, offset = _read_exact(packet, offset, model_id_length, "model metadata")
    device_bytes, offset = _read_exact(packet, offset, device_length, "device metadata")
    try:
        model_id = model_bytes.decode("utf-8")
        source_device = device_bytes.decode("utf-8")
    except UnicodeDecodeError as error:
        raise KVDeserializationError("model/device metadata is not valid UTF-8") from error
    if not model_id:
        raise KVDeserializationError("model identifier cannot be empty")

    layers: list[NeutralKVLayer] = []
    tensor_body_bytes = 0
    for expected_index in range(layer_count):
        raw_layer, offset = _read_exact(packet, offset, _LAYER.size, "layer header")
        (layer_index,) = _LAYER.unpack(raw_layer)
        if layer_index != expected_index:
            raise KVDeserializationError(
                f"invalid layer ordering: expected {expected_index}, got {layer_index}"
            )
        key, offset = _decode_tensor(packet, offset, layer_index, "K")
        value, offset = _decode_tensor(packet, offset, layer_index, "V")
        tensor_body_bytes += key.byte_length + value.byte_length
        layers.append(NeutralKVLayer(layer_index, key, value))

    if offset != len(packet):
        raise KVDeserializationError(
            f"trailing bytes after layer records: {len(packet) - offset}"
        )
    payload = NeutralKVPayload(
        model=KVModelCompatibility(model_id, config_layers, kv_heads, head_dim),
        logical_cached_token_count=logical_tokens,
        source_device=source_device,
        layers=tuple(layers),
    )
    _validate_payload(payload, KVDeserializationError)
    elapsed_ms = (perf_counter() - started) * 1_000
    return KVDeserializationResult(
        payload=payload,
        wire_tensor_body_bytes=tensor_body_bytes,
        wire_metadata_bytes=len(packet) - tensor_body_bytes,
        total_wire_bytes=len(packet),
        deserialize_ms=elapsed_ms,
    )


def _encode_payload(payload: NeutralKVPayload) -> tuple[bytes, int]:
    _validate_payload(payload, KVSerializationError)
    model_id = payload.model.model_id.encode("utf-8")
    source_device = payload.source_device.encode("utf-8")
    if len(model_id) > MAX_MODEL_ID_BYTES:
        raise KVSerializationError("model identifier is too long")
    if len(source_device) > MAX_DEVICE_BYTES:
        raise KVSerializationError("source device is too long")

    records: list[bytes] = [model_id, source_device]
    tensor_body_bytes = 0
    for layer in payload.layers:
        records.append(_LAYER.pack(layer.layer_index))
        for tensor in (layer.key, layer.value):
            dtype_code = DTYPE_NAME_TO_CODE[tensor.dtype]
            records.append(
                _TENSOR.pack(dtype_code, len(tensor.shape), 0, tensor.byte_length)
            )
            records.extend(_DIMENSION.pack(dimension) for dimension in tensor.shape)
            records.append(tensor.data)
            tensor_body_bytes += tensor.byte_length

    body = b"".join(records)
    total_bytes = KV_WIRE_FIXED_HEADER_BYTES + len(body)
    if total_bytes > MAX_PACKET_BYTES:
        raise KVSerializationError("packet exceeds configured maximum size")
    header = _HEADER.pack(
        KV_PROTOCOL_MAGIC,
        KV_PROTOCOL_VERSION,
        KV_PROTOCOL_FLAGS,
        0,
        len(payload.layers),
        payload.logical_cached_token_count,
        total_bytes,
        payload.model.num_hidden_layers,
        payload.model.num_key_value_heads,
        payload.model.head_dim,
        len(model_id),
        len(source_device),
    )
    return header + body, tensor_body_bytes


def _validate_payload(payload: NeutralKVPayload, error_type: type[ValueError]) -> None:
    if not isinstance(payload, NeutralKVPayload):
        raise TypeError("payload must be NeutralKVPayload")
    model = payload.model
    if not isinstance(model, KVModelCompatibility):
        raise error_type("payload model must be KVModelCompatibility")
    if not isinstance(model.model_id, str) or not model.model_id:
        raise error_type("model identifier cannot be empty")
    if not isinstance(payload.source_device, str):
        raise error_type("source device must be a string")
    if not isinstance(payload.layers, tuple):
        raise error_type("layers must be an immutable tuple")
    layer_count = len(payload.layers)
    try:
        _validate_packet_counts(
            layer_count,
            payload.logical_cached_token_count,
            model.num_hidden_layers,
            model.num_key_value_heads,
            model.head_dim,
        )
    except KVDeserializationError as error:
        raise error_type(str(error)) from error
    if model.num_hidden_layers != layer_count:
        raise error_type("model layer count does not match payload layer count")

    for expected_index, layer in enumerate(payload.layers):
        if not isinstance(layer, NeutralKVLayer):
            raise error_type("layer records must be NeutralKVLayer values")
        if layer.layer_index != expected_index:
            raise error_type(
                f"layers must be complete and ordered: expected {expected_index}, "
                f"got {layer.layer_index}"
            )
        for label, tensor in (("K", layer.key), ("V", layer.value)):
            _validate_tensor(tensor, layer.layer_index, label, error_type)
        if layer.key.shape != layer.value.shape:
            raise error_type(f"layer {layer.layer_index} K/V shapes differ")
        if layer.key.shape[-2] != payload.logical_cached_token_count:
            raise error_type(
                f"layer {layer.layer_index} cache sequence length does not match "
                "logical_cached_token_count"
            )


def _validate_packet_counts(
    layer_count: int,
    logical_tokens: int,
    config_layers: int,
    kv_heads: int,
    head_dim: int,
) -> None:
    for name, value in (
        ("layer count", layer_count),
        ("logical cached token count", logical_tokens),
        ("config layer count", config_layers),
        ("KV head count", kv_heads),
        ("head dimension", head_dim),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise KVDeserializationError(f"invalid {name}")
    if layer_count == 0 or layer_count > MAX_LAYER_COUNT:
        raise KVDeserializationError("invalid or excessive layer count")
    if logical_tokens == 0 or logical_tokens > MAX_LOGICAL_CACHED_TOKENS:
        raise KVDeserializationError("invalid logical cached token count")
    for name, value in (
        ("config layer count", config_layers),
        ("KV head count", kv_heads),
        ("head dimension", head_dim),
    ):
        if value == 0 or value > MAX_DIMENSION:
            raise KVDeserializationError(f"invalid {name}")


def _validate_tensor(
    tensor: NeutralKVTensor,
    layer_index: int,
    label: str,
    error_type: type[ValueError],
) -> None:
    if not isinstance(tensor, NeutralKVTensor):
        raise error_type(f"layer {layer_index} {label} is not a neutral tensor")
    if tensor.dtype not in DTYPE_NAME_TO_CODE:
        raise error_type(f"layer {layer_index} {label} has unsupported dtype")
    if not isinstance(tensor.shape, tuple):
        raise error_type(f"layer {layer_index} {label} shape must be a tuple")
    if not isinstance(tensor.data, bytes):
        raise error_type(f"layer {layer_index} {label} data must be immutable bytes")
    rank = len(tensor.shape)
    if rank < 2 or rank > MAX_TENSOR_RANK:
        raise error_type(f"layer {layer_index} {label} has invalid rank")
    elements = 1
    for dimension in tensor.shape:
        if (
            not isinstance(dimension, int)
            or isinstance(dimension, bool)
            or dimension <= 0
            or dimension > MAX_DIMENSION
        ):
            raise error_type(f"layer {layer_index} {label} has invalid dimension")
        elements *= dimension
        if elements > MAX_TENSOR_BYTES:
            raise error_type(f"layer {layer_index} {label} shape is excessive")
    expected = elements * DTYPE_ELEMENT_BYTES[tensor.dtype]
    if expected != tensor.byte_length:
        raise error_type(
            f"layer {layer_index} {label} shape/dtype requires {expected} bytes, "
            f"got {tensor.byte_length}"
        )
    if tensor.byte_length > MAX_TENSOR_BYTES:
        raise error_type(f"layer {layer_index} {label} exceeds maximum tensor size")


def _decode_tensor(
    packet: bytes, offset: int, layer_index: int, label: str
) -> tuple[NeutralKVTensor, int]:
    raw_header, offset = _read_exact(packet, offset, _TENSOR.size, "tensor descriptor")
    dtype_code, rank, reserved, byte_length = _TENSOR.unpack(raw_header)
    if dtype_code not in DTYPE_CODE_TO_NAME:
        raise KVDeserializationError(
            f"layer {layer_index} {label} has unknown dtype code {dtype_code}"
        )
    if reserved != 0:
        raise KVDeserializationError("tensor descriptor reserved field must be zero")
    if rank < 2 or rank > MAX_TENSOR_RANK:
        raise KVDeserializationError(
            f"layer {layer_index} {label} has invalid rank {rank}"
        )
    if byte_length > MAX_TENSOR_BYTES:
        raise KVDeserializationError(
            f"layer {layer_index} {label} exceeds maximum tensor size"
        )
    shape: list[int] = []
    for _ in range(rank):
        raw_dimension, offset = _read_exact(
            packet, offset, _DIMENSION.size, "tensor dimensions"
        )
        (dimension,) = _DIMENSION.unpack(raw_dimension)
        if dimension == 0 or dimension > MAX_DIMENSION:
            raise KVDeserializationError(
                f"layer {layer_index} {label} has invalid dimension"
            )
        shape.append(dimension)
    body, offset = _read_exact(packet, offset, byte_length, "tensor body")
    tensor = NeutralKVTensor(DTYPE_CODE_TO_NAME[dtype_code], tuple(shape), body)
    _validate_tensor(tensor, layer_index, label, KVDeserializationError)
    return tensor, offset


def _read_exact(
    packet: bytes, offset: int, length: int, label: str
) -> tuple[bytes, int]:
    if length < 0 or offset > len(packet) or length > len(packet) - offset:
        raise KVDeserializationError(f"truncated {label}")
    end = offset + length
    return packet[offset:end], end
