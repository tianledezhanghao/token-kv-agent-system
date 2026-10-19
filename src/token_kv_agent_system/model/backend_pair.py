"""Small same-checkpoint contract for independent model instances.

No weights fingerprint, model plugin registry, or selector profile is defined here.
Existing shared-backend callers retain their original identity guarantee.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Callable

from .cache_inspection import KVModelConfig
from .config import validate_device_spec


@dataclass(frozen=True)
class BackendIdentity:
    model_id: str
    model_type: str
    architecture: str
    dtype: str
    structure: KVModelConfig
    model_vocab_size: int
    tokenizer_class: str
    tokenizer_vocab_size: int
    tokenizer_length: int
    tokenizer_vocab_sha256: str
    special_token_ids: tuple[tuple[str, int | None], ...]
    all_special_ids: tuple[int, ...]
    eos_token_ids: tuple[int, ...]


def inspect_backend_identity(backend: Any) -> BackendIdentity:
    model, tokenizer = backend._model, backend._tokenizer
    config = model.config
    vocab = tokenizer.get_vocab()
    if not vocab or any(type(value) is not int or not 0 <= value < backend.vocab_size
                        for value in vocab.values()):
        raise ValueError("tokenizer vocabulary is outside the model vocabulary")
    if not getattr(config, "model_type", None) or not backend.metadata.dtype:
        raise ValueError("model type and actual loaded dtype are required")
    special = tuple((name, getattr(tokenizer, name, None)) for name in
                    ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"))
    all_special_ids = tuple(sorted(set(tokenizer.all_special_ids)))
    if any(not 0 <= token_id < backend.vocab_size for token_id in
           (*all_special_ids, *backend.eos_token_ids,
            *(value for _, value in special if value is not None))):
        raise ValueError("special Token ID is outside the model vocabulary")
    return BackendIdentity(
        model_id=backend.model_name, model_type=config.model_type,
        architecture=type(model).__name__, dtype=backend.metadata.dtype,
        structure=backend.kv_model_config, model_vocab_size=backend.vocab_size,
        tokenizer_class=type(tokenizer).__name__, tokenizer_vocab_size=tokenizer.vocab_size,
        tokenizer_length=len(tokenizer),
        tokenizer_vocab_sha256=hashlib.sha256(json.dumps(
            vocab, sort_keys=True, ensure_ascii=True, separators=(",", ":")
        ).encode()).hexdigest(),
        special_token_ids=special, all_special_ids=all_special_ids,
        eos_token_ids=tuple(backend.eos_token_ids),
    )


def validate_backend_compatibility(source: Any, target: Any) -> None:
    if source is target:
        return
    left = getattr(source, "compatibility_metadata", None)
    right = getattr(target, "compatibility_metadata", None)
    if not isinstance(left, BackendIdentity) or not isinstance(right, BackendIdentity):
        raise ValueError("Agents must share one model backend or provide complete backend compatibility metadata")
    for name in left.__dataclass_fields__:
        if getattr(left, name) != getattr(right, name):
            raise ValueError(f"independent backend compatibility mismatch: {name}")


@dataclass(frozen=True)
class IndependentBackendConfig:
    model_name: str
    agent_a_device: str = "cuda:0"
    agent_b_device: str = "cuda:1"

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        for device in (self.agent_a_device, self.agent_b_device):
            validate_device_spec(device)
            if device == "auto":
                raise ValueError("independent backends require explicit device placement")


@dataclass(frozen=True)
class IndependentBackends:
    agent_a_backend: Any
    agent_b_backend: Any

    def __post_init__(self) -> None:
        a, b = self.agent_a_backend, self.agent_b_backend
        if a is b or a._model is b._model or a._tokenizer is b._tokenizer:
            raise ValueError("independent backends require separate backend, model and tokenizer objects")
        validate_backend_compatibility(a, b)
        storages = []
        for backend in (a, b):
            tensors = tuple(backend._model.parameters()) + tuple(backend._model.buffers())
            if not tensors:
                raise ValueError("cannot verify independence without model parameter storage")
            device = str(tensors[0].device)
            expected = backend.device
            if device.split(":")[0] != expected.split(":")[0] or (
                ":" in expected and device != expected
            ) or any(str(t.device) != device for t in tensors):
                raise ValueError("all model parameters/buffers must reside on the assigned single device")
            storages.append({(str(t.device), t.untyped_storage().data_ptr())
                             for t in tensors if t.numel()})
        if not all(storages) or storages[0] & storages[1]:
            raise ValueError("independent model instances must not share parameter/buffer storage")

    @classmethod
    def load(cls, config: IndependentBackendConfig, *, torch_module=None,
             backend_factory: Callable | None = None) -> IndependentBackends:
        from importlib import import_module
        from .huggingface import HuggingFaceModelBackend, resolve_device

        torch_module = torch_module or import_module("torch")
        # Both devices must pass preflight before allocating either model.
        for device in (config.agent_a_device, config.agent_b_device):
            resolve_device(device, torch_module)
        factory = backend_factory or HuggingFaceModelBackend
        a = factory(config.model_name, device=config.agent_a_device, torch_module=torch_module)
        b = factory(config.model_name, device=config.agent_b_device, torch_module=torch_module)
        return cls(a, b)

    def to_dict(self) -> dict:
        a, b = self.agent_a_backend, self.agent_b_backend
        a_device = str(next(a._model.parameters()).device)
        b_device = str(next(b._model.parameters()).device)
        return {
            "mode": "independent", "identity": asdict(a.compatibility_metadata),
            "agent_a_device": a_device, "agent_b_device": b_device,
            "same_device": a_device == b_device,
            "separate_cuda_devices": a_device.startswith("cuda:") and
                b_device.startswith("cuda:") and a_device != b_device,
            "model_objects_independent": a._model is not b._model,
            "parameter_storage_independent": True,
            "weights_identity": "same-checkpoint assumption; no weights fingerprint",
        }
