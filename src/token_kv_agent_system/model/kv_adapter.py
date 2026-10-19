"""HuggingFace DynamicCache adapter for framework-neutral KV payloads."""

from __future__ import annotations

import sys
import inspect
from dataclasses import dataclass, field, replace
from importlib import import_module
from time import perf_counter
from types import ModuleType
from typing import Any

from ..kv_protocol import (
    DTYPE_ELEMENT_BYTES,
    KVModelCompatibility,
    KVSerializationResult,
    NeutralKVLayer,
    NeutralKVPayload,
    NeutralKVTensor,
    serialize_kv_payload,
)
from .cache_inspection import KVModelConfig
from .huggingface import synchronize_device


# Paired 8.3B acceptance: packed MPS BF16 within the measured byte range.
# CPU/CUDA, mixed dtype and larger payloads retain the established adapter path.
DEFAULT_TRANSFER_PATH = "auto"
MAX_AUTO_PACKED_TRANSFER_BYTES = 6 * 1024 * 1024
MAX_PACKED_TRANSFER_BYTES = 64 * 1024 * 1024


_TORCH_DTYPE_NAMES = {
    "torch.bfloat16": "bfloat16",
    "torch.float16": "float16",
    "torch.float32": "float32",
}


class HuggingFaceKVAdapterError(ValueError):
    """Raised when a runtime cache is incomplete or incompatible with v1."""


@dataclass(frozen=True)
class KVCacheCompatibilityResult:
    """Receiver compatibility facts validated before tensor reconstruction."""

    passed: bool
    model_id: str
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    dtype: str


@dataclass(frozen=True)
class KVCacheRestoreResult:
    """A short-lived Receiver cache plus separately observable restore stages."""

    cache: Any = field(repr=False, compare=False)
    logical_cached_token_count: int
    target_device: str
    compatibility: KVCacheCompatibilityResult
    cpu_tensor_reconstruction_ms: float
    host_to_device_ms: float
    cache_object_construction_ms: float
    total_restore_ms: float


def serialize_huggingface_kv_cache(
    cache: object,
    *,
    logical_cached_token_count: int,
    model_id: str,
    model_config: KVModelConfig,
    torch_module: ModuleType | Any | None = None,
    transfer_path: str | None = None,
) -> KVSerializationResult:
    """Copy an actual DynamicCache to host and encode one deterministic packet."""

    torch_module = torch_module or import_module("torch")
    total_started = perf_counter()
    payload, raw_bytes, device_to_host_ms = extract_huggingface_kv_payload(
        cache,
        logical_cached_token_count=logical_cached_token_count,
        model_id=model_id,
        model_config=model_config,
        torch_module=torch_module,
        transfer_path=transfer_path,
    )
    encoded = serialize_kv_payload(payload)
    total_ms = (perf_counter() - total_started) * 1_000
    if encoded.wire_tensor_body_bytes != raw_bytes:
        raise RuntimeError("wire tensor body bytes do not match source raw tensor bytes")
    return replace(
        encoded,
        raw_kv_tensor_bytes=raw_bytes,
        device_to_host_ms=device_to_host_ms,
        total_serialize_ms=total_ms,
    )


def extract_huggingface_kv_payload(
    cache: object,
    *,
    logical_cached_token_count: int,
    model_id: str,
    model_config: KVModelConfig,
    torch_module: ModuleType | Any | None = None,
    transfer_path: str | None = None,
) -> tuple[NeutralKVPayload, int, float]:
    """Validate layers and snapshot immutable bytes without any dtype conversion.

    ``transfer_path`` is a controlled legacy/packed comparison override. Default
    auto uses the validated MPS BF16 byte domain; it never catches packing errors
    or retries a failed transfer through a different path.
    """

    torch_module = torch_module or import_module("torch")
    if logical_cached_token_count <= 0:
        raise HuggingFaceKVAdapterError(
            "logical_cached_token_count must be greater than zero"
        )
    if not model_id.strip():
        raise HuggingFaceKVAdapterError("model_id cannot be empty")
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise HuggingFaceKVAdapterError(
            "unsupported cache: expected Transformers Cache.layers"
        )
    layers = tuple(layers)
    if len(layers) != model_config.num_hidden_layers:
        raise HuggingFaceKVAdapterError(
            f"cache has {len(layers)} layers; model config expects "
            f"{model_config.num_hidden_layers}"
        )
    get_seq_length = getattr(cache, "get_seq_length", None)
    if callable(get_seq_length):
        runtime_length = int(get_seq_length())
        if runtime_length != logical_cached_token_count:
            raise HuggingFaceKVAdapterError(
                f"cache runtime length {runtime_length} does not match logical "
                f"cached length {logical_cached_token_count}"
            )

    validated: list[tuple[int, Any, Any, str, str]] = []
    source_devices: set[str] = set()
    expected_shape: tuple[int, ...] | None = None
    for layer_index, layer in enumerate(layers):
        key = getattr(layer, "keys", None)
        value = getattr(layer, "values", None)
        if key is None or value is None:
            raise HuggingFaceKVAdapterError(
                f"cache layer {layer_index} has no initialized K/V tensors"
            )
        key_shape = _shape(key, layer_index, "K")
        value_shape = _shape(value, layer_index, "V")
        if len(key_shape) != 4 or len(value_shape) != 4:
            raise HuggingFaceKVAdapterError(
                "expected K/V layout [batch, KV heads, cached tokens, head dim]"
            )
        if key_shape != value_shape:
            raise HuggingFaceKVAdapterError(f"layer {layer_index} K/V shapes differ")
        if key_shape[2] != logical_cached_token_count:
            raise HuggingFaceKVAdapterError(
                f"layer {layer_index} sequence length {key_shape[2]} does not match "
                f"logical cached length {logical_cached_token_count}"
            )
        if key_shape[1] != model_config.num_key_value_heads:
            raise HuggingFaceKVAdapterError(
                f"layer {layer_index} KV head count does not match model config"
            )
        if key_shape[3] != model_config.head_dim:
            raise HuggingFaceKVAdapterError(
                f"layer {layer_index} head dimension does not match model config"
            )
        if expected_shape is None:
            expected_shape = key_shape
        elif key_shape != expected_shape:
            raise HuggingFaceKVAdapterError("cache shapes are inconsistent across layers")
        key_dtype = _dtype_name(key, layer_index, "K")
        value_dtype = _dtype_name(value, layer_index, "V")
        if key_dtype != value_dtype:
            raise HuggingFaceKVAdapterError(f"layer {layer_index} K/V dtypes differ")
        key_device = str(getattr(key, "device", ""))
        value_device = str(getattr(value, "device", ""))
        if not key_device or key_device != value_device:
            raise HuggingFaceKVAdapterError(f"layer {layer_index} K/V devices differ")
        source_devices.add(key_device)
        validated.append((layer_index, key, value, key_dtype, value_dtype))
    if len(source_devices) != 1:
        raise HuggingFaceKVAdapterError("cache tensors span multiple source devices")

    source_device = next(iter(source_devices))
    sync_device = source_device
    synchronize_device(torch_module, sync_device)
    copy_started = perf_counter()
    neutral_layers: list[NeutralKVLayer] = []
    raw_bytes = 0
    tensors = tuple(t for _, k, v, _, _ in validated for t in (k, v))
    transfer_bytes = sum(int(t.numel()) * int(t.element_size()) for t in tensors)
    uniform_dtype = len({dtype for _, _, _, dtype, _ in validated}) == 1
    path = _transfer_path(transfer_path, source_device, transfer_bytes, uniform_dtype, validated[0][3])
    if path == "packed":
        if not uniform_dtype:
            raise HuggingFaceKVAdapterError("packed export requires uniform dtype")
        raw_bytes = transfer_bytes
        _check_packed_size(raw_bytes)
        # stack copies logical BHSD order, including legal non-contiguous inputs.
        # No float arithmetic or dtype conversion: BF16/NaN/-0 bits survive.
        packed = torch_module.stack(tuple(t.detach() for t in tensors))
        cpu = packed.cpu()
        raw = cpu.view(dtype=torch_module.uint8).numpy().tobytes()
        width = DTYPE_ELEMENT_BYTES[validated[0][3]]
        if sys.byteorder == "big":
            raw = _reverse_elements(raw, width)
        if len(raw) != raw_bytes:
            raise HuggingFaceKVAdapterError("packed byte extraction size mismatch")
        offset = 0
        for layer_index, key, value, key_dtype, value_dtype in validated:
            size = int(key.numel()) * int(key.element_size())
            k = NeutralKVTensor(key_dtype, tuple(key.shape), raw[offset:offset + size])
            offset += size
            v = NeutralKVTensor(value_dtype, tuple(value.shape), raw[offset:offset + size])
            offset += size
            neutral_layers.append(NeutralKVLayer(layer_index, k, v))
    else:
        for layer_index, key, value, key_dtype, value_dtype in validated:
            key_payload = _tensor_to_neutral(key, key_dtype, layer_index, "K", torch_module)
            value_payload = _tensor_to_neutral(value, value_dtype, layer_index, "V", torch_module)
            raw_bytes += key_payload.byte_length + value_payload.byte_length
            neutral_layers.append(NeutralKVLayer(layer_index, key_payload, value_payload))
    synchronize_device(torch_module, sync_device)
    device_to_host_ms = (perf_counter() - copy_started) * 1_000

    payload = NeutralKVPayload(
        model=KVModelCompatibility(
            model_id=model_id,
            num_hidden_layers=model_config.num_hidden_layers,
            num_key_value_heads=model_config.num_key_value_heads,
            head_dim=model_config.head_dim,
        ),
        logical_cached_token_count=logical_cached_token_count,
        source_device=source_device,
        layers=tuple(neutral_layers),
    )
    return payload, raw_bytes, device_to_host_ms


def extract_huggingface_kv_span(cache: object, start: int, end: int, *,
                               model_id: str, model_config: KVModelConfig,
                               torch_module: Any = None) -> NeutralKVPayload:
    """Snapshot only a materialized Decode span using the unchanged v1 adapter.

    This view is serialization-only, never a model cache or Receiver history.
    Device metadata describes the resulting CPU representation, not placement.
    """
    from types import SimpleNamespace
    from .decode_kv import extract_span

    pairs = extract_span(cache, start, end)
    view = SimpleNamespace(
        layers=tuple(SimpleNamespace(keys=k, values=v) for k, v in pairs),
        get_seq_length=lambda: end - start)
    payload, _, _ = extract_huggingface_kv_payload(
        view, logical_cached_token_count=end - start, model_id=model_id,
        model_config=model_config, torch_module=torch_module)
    return replace(payload, source_device="cpu")


def restore_huggingface_kv_cache(
    payload: NeutralKVPayload,
    *,
    model_id: str,
    model_config: KVModelConfig,
    model_dtype: str,
    target_device: str,
    transformers_config: Any,
    torch_module: ModuleType | Any | None = None,
    transfer_path: str | None = None,
    dynamic_cache_class: type | None = None,
) -> KVCacheRestoreResult:
    """Restore neutral bytes into a Transformers 5.17-compatible DynamicCache.

    Compatibility and all tensor descriptors are validated before allocating CPU
    tensors or starting accelerator uploads. The public ``ddp_cache_data``
    constructor is deliberately used instead of assigning cache layer internals.
    Each packed call owns its upload buffer. Constructor inputs are nonoverlapping
    views; TF5.17 DynamicLayer.update cats them into separate owned cache tensors.
    """

    torch_module = torch_module or import_module("torch")
    total_started = perf_counter()
    compatibility = validate_huggingface_kv_compatibility(
        payload,
        model_id=model_id,
        model_config=model_config,
        model_dtype=model_dtype,
    )
    if dynamic_cache_class is None:
        dynamic_cache_class = import_module("transformers").DynamicCache
    parameters = inspect.signature(dynamic_cache_class).parameters
    if not {"ddp_cache_data", "config"}.issubset(parameters) and not any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    ):
        raise HuggingFaceKVAdapterError(
            "installed DynamicCache lacks the required public ddp_cache_data/config constructor; "
            "inspect the server Transformers API before adding adapter compatibility"
        )

    cpu_started = perf_counter()
    path = _transfer_path(transfer_path, target_device, payload.raw_kv_tensor_bytes, dtype=compatibility.dtype)
    if path == "packed":
        descriptors = tuple(t for layer in payload.layers for t in (layer.key, layer.value))
        _check_packed_size(sum(t.byte_length for t in descriptors))
        raw = b"".join(t.data for t in descriptors)
        if sys.byteorder == "big":
            raw = _reverse_elements(raw, DTYPE_ELEMENT_BYTES[compatibility.dtype])
        # Each call owns its bytearray; frombuffer retains its Python owner.
        packed_cpu = torch_module.frombuffer(bytearray(raw), dtype=torch_module.uint8).view(
            dtype=getattr(torch_module, compatibility.dtype))
        del raw
    else:
        cpu_layers = tuple(
            (_neutral_to_cpu_tensor(layer.key, torch_module),
             _neutral_to_cpu_tensor(layer.value, torch_module))
            for layer in payload.layers
        )
    cpu_ms = (perf_counter() - cpu_started) * 1_000

    sync_device = target_device
    synchronize_device(torch_module, sync_device)
    upload_started = perf_counter()
    if path == "packed":
        # A new allocation per restore, never a pool or a Sender tensor alias.
        packed_device = packed_cpu.to(target_device)
        views = packed_device.reshape(len(descriptors), *descriptors[0].shape).unbind(0)
        # Compatibility established equal BHSD shapes/dtype. Disjoint offsets
        # share a live refcounted storage; writes in one view cannot touch another.
        device_layers = tuple(zip(views[::2], views[1::2]))
    else:
        device_layers = tuple(
            (key.to(target_device), value.to(target_device)) for key, value in cpu_layers
        )
    synchronize_device(torch_module, sync_device)
    upload_ms = (perf_counter() - upload_started) * 1_000

    synchronize_device(torch_module, sync_device)
    construction_started = perf_counter()
    cache = dynamic_cache_class(
        ddp_cache_data=device_layers,
        config=transformers_config,
    )
    synchronize_device(torch_module, sync_device)
    construction_ms = (perf_counter() - construction_started) * 1_000
    _validate_restored_cache(
        cache,
        payload=payload,
        target_device=target_device,
    )
    return KVCacheRestoreResult(
        cache=cache,
        logical_cached_token_count=payload.logical_cached_token_count,
        target_device=target_device,
        compatibility=compatibility,
        cpu_tensor_reconstruction_ms=cpu_ms,
        host_to_device_ms=upload_ms,
        cache_object_construction_ms=construction_ms,
        total_restore_ms=(perf_counter() - total_started) * 1_000,
    )


def validate_huggingface_kv_compatibility(
    payload: NeutralKVPayload,
    *,
    model_id: str,
    model_config: KVModelConfig,
    model_dtype: str,
) -> KVCacheCompatibilityResult:
    """Reject an incompatible packet before any tensor allocation or upload."""

    if not isinstance(payload, NeutralKVPayload):
        raise TypeError("payload must be NeutralKVPayload")
    expected_dtype = model_dtype.removeprefix("torch.")
    if expected_dtype not in DTYPE_ELEMENT_BYTES:
        raise HuggingFaceKVAdapterError(
            f"Receiver model dtype {model_dtype!r} is unsupported by KV Protocol v1"
        )
    expected = (
        ("model identifier", payload.model.model_id, model_id),
        ("hidden layer count", payload.model.num_hidden_layers, model_config.num_hidden_layers),
        ("KV head count", payload.model.num_key_value_heads, model_config.num_key_value_heads),
        ("head dimension", payload.model.head_dim, model_config.head_dim),
    )
    for label, actual, wanted in expected:
        if actual != wanted:
            raise HuggingFaceKVAdapterError(
                f"Receiver {label} mismatch: packet has {actual!r}, expected {wanted!r}"
            )
    if len(payload.layers) != model_config.num_hidden_layers:
        raise HuggingFaceKVAdapterError("payload layer count does not match Receiver")

    for expected_index, layer in enumerate(payload.layers):
        if layer.layer_index != expected_index:
            raise HuggingFaceKVAdapterError("payload layers are not complete and ordered")
        for label, tensor in (("K", layer.key), ("V", layer.value)):
            shape = tensor.shape
            if len(shape) != 4:
                raise HuggingFaceKVAdapterError(
                    f"layer {expected_index} {label} must use [B,Hkv,S,D] layout"
                )
            if shape[0] != 1:
                raise HuggingFaceKVAdapterError("only batch size 1 cache restore is supported")
            if shape[1] != model_config.num_key_value_heads:
                raise HuggingFaceKVAdapterError(
                    f"layer {expected_index} {label} KV head count mismatch"
                )
            if shape[2] != payload.logical_cached_token_count:
                raise HuggingFaceKVAdapterError(
                    f"layer {expected_index} {label} cache length mismatch"
                )
            if shape[3] != model_config.head_dim:
                raise HuggingFaceKVAdapterError(
                    f"layer {expected_index} {label} head dimension mismatch"
                )
            if tensor.dtype != expected_dtype:
                raise HuggingFaceKVAdapterError(
                    f"layer {expected_index} {label} dtype {tensor.dtype!r} does not "
                    f"match Receiver dtype {expected_dtype!r}"
                )
            expected_bytes = _shape_product(shape) * DTYPE_ELEMENT_BYTES[tensor.dtype]
            if tensor.byte_length != expected_bytes:
                raise HuggingFaceKVAdapterError(
                    f"layer {expected_index} {label} byte length does not match shape"
                )
        if layer.key.shape != layer.value.shape or layer.key.dtype != layer.value.dtype:
            raise HuggingFaceKVAdapterError(
                f"layer {expected_index} K/V metadata differs"
            )
    return KVCacheCompatibilityResult(
        passed=True,
        model_id=model_id,
        num_hidden_layers=model_config.num_hidden_layers,
        num_key_value_heads=model_config.num_key_value_heads,
        head_dim=model_config.head_dim,
        dtype=expected_dtype,
    )


def _neutral_to_cpu_tensor(tensor: NeutralKVTensor, torch_module: Any) -> Any:
    raw = tensor.data
    element_bytes = DTYPE_ELEMENT_BYTES[tensor.dtype]
    if sys.byteorder == "big":
        raw = _reverse_elements(raw, element_bytes)
    # bytearray owns writable storage and avoids PyTorch's non-writable-buffer warning.
    byte_view = torch_module.frombuffer(bytearray(raw), dtype=torch_module.uint8)
    dtype = getattr(torch_module, tensor.dtype)
    restored = byte_view.view(dtype=dtype).reshape(tensor.shape)
    return restored.clone()


def _validate_restored_cache(
    cache: Any, *, payload: NeutralKVPayload, target_device: str
) -> None:
    layers = tuple(getattr(cache, "layers", ()))
    if len(layers) != len(payload.layers):
        raise HuggingFaceKVAdapterError("restored cache layer count is incorrect")
    if int(cache.get_seq_length()) != payload.logical_cached_token_count:
        raise HuggingFaceKVAdapterError("restored cache sequence length is incorrect")
    expected_device = target_device.split(":", 1)[0]
    for index, (runtime, neutral) in enumerate(zip(layers, payload.layers)):
        for label, runtime_tensor, descriptor in (
            ("K", getattr(runtime, "keys", None), neutral.key),
            ("V", getattr(runtime, "values", None), neutral.value),
        ):
            if runtime_tensor is None:
                raise HuggingFaceKVAdapterError(
                    f"restored layer {index} {label} tensor is missing"
                )
            if tuple(int(value) for value in runtime_tensor.shape) != descriptor.shape:
                raise HuggingFaceKVAdapterError(
                    f"restored layer {index} {label} shape is incorrect"
                )
            if str(runtime_tensor.dtype).removeprefix("torch.") != descriptor.dtype:
                raise HuggingFaceKVAdapterError(
                    f"restored layer {index} {label} dtype is incorrect"
                )
            actual_device = str(runtime_tensor.device)
            if (actual_device.split(":", 1)[0] != expected_device
                    or (":" in target_device and actual_device != target_device)):
                raise HuggingFaceKVAdapterError(
                    f"restored layer {index} {label} device is incorrect"
                )


def _shape_product(shape: tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _tensor_to_neutral(
    tensor: Any,
    dtype: str,
    layer_index: int,
    label: str,
    torch_module: ModuleType | Any,
) -> NeutralKVTensor:
    contiguous = tensor.detach()
    if not contiguous.is_contiguous():
        contiguous = contiguous.contiguous()
    cpu_tensor = contiguous.cpu()
    # Viewing bytes avoids any bfloat16 -> float conversion and preserves NaNs/-0.
    raw = cpu_tensor.view(dtype=torch_module.uint8).numpy().tobytes()
    element_bytes = DTYPE_ELEMENT_BYTES[dtype]
    if sys.byteorder == "big":
        raw = _reverse_elements(raw, element_bytes)
    expected = int(tensor.numel()) * int(tensor.element_size())
    if len(raw) != expected:
        raise HuggingFaceKVAdapterError(
            f"layer {layer_index} {label} byte extraction size mismatch"
        )
    return NeutralKVTensor(dtype=dtype, shape=_shape(tensor, layer_index, label), data=raw)


def _reverse_elements(data: bytes, width: int) -> bytes:
    return b"".join(
        data[offset : offset + width][::-1]
        for offset in range(0, len(data), width)
    )


def _shape(tensor: Any, layer_index: int, label: str) -> tuple[int, ...]:
    shape = getattr(tensor, "shape", None)
    if shape is None:
        raise HuggingFaceKVAdapterError(
            f"cache layer {layer_index} {label} has no tensor shape"
        )
    return tuple(int(dimension) for dimension in shape)


def _dtype_name(tensor: Any, layer_index: int, label: str) -> str:
    dtype = str(getattr(tensor, "dtype", ""))
    try:
        return _TORCH_DTYPE_NAMES[dtype]
    except KeyError as error:
        raise HuggingFaceKVAdapterError(
            f"cache layer {layer_index} {label} uses unsupported dtype {dtype!r}"
        ) from error


def _transfer_path(requested: str | None, device: str, size: int, uniform_dtype: bool = True,
                   dtype: str | None = None) -> str:
    path = DEFAULT_TRANSFER_PATH if requested is None else requested
    if path == "auto":
        return "packed" if (device.split(":", 1)[0] == "mps" and uniform_dtype
                            and dtype == "bfloat16"
                            and 0 < size <= MAX_AUTO_PACKED_TRANSFER_BYTES) else "legacy"
    if path not in ("legacy", "packed"):
        raise HuggingFaceKVAdapterError("transfer_path must be legacy, packed or auto")
    return path


def _check_packed_size(size: int) -> None:
    if not 0 < size <= MAX_PACKED_TRANSFER_BYTES:
        raise HuggingFaceKVAdapterError("packed transfer exceeds bounded buffer capacity")
