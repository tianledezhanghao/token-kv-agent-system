from __future__ import annotations

from dataclasses import replace

import pytest


torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from token_kv_agent_system.kv_protocol import (  # noqa: E402
    KVModelCompatibility,
    NeutralKVLayer,
    NeutralKVPayload,
    NeutralKVTensor,
    deserialize_kv_packet,
)
from token_kv_agent_system.model import (  # noqa: E402
    HuggingFaceKVAdapterError,
    HuggingFaceModelBackend,
    KVModelConfig,
    restore_huggingface_kv_cache,
    serialize_huggingface_kv_cache,
    validate_huggingface_kv_compatibility,
)


class _Tokenizer:
    vocab_size = 64
    eos_token_id = 63
    bos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str):
        return [1, 2, 3]

    def decode(self, token_ids, **kwargs):
        return " ".join(str(value) for value in token_ids)


def _backend() -> HuggingFaceModelBackend:
    torch.manual_seed(7)
    config = transformers.Qwen2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        use_cache=True,
    )
    model = transformers.Qwen2ForCausalLM(config).eval()
    return HuggingFaceModelBackend(
        "tiny/qwen2",
        device="cpu",
        tokenizer=_Tokenizer(),
        model=model,
        torch_module=torch,
    )


def _wire_restore(backend: HuggingFaceModelBackend, prefix: tuple[int, ...]):
    sender = backend.prefill(input_token_ids=prefix)
    encoded = serialize_huggingface_kv_cache(
        sender.cache,
        logical_cached_token_count=len(prefix),
        model_id=backend.model_name,
        model_config=backend.kv_model_config,
        torch_module=torch,
    )
    decoded = deserialize_kv_packet(encoded.packet)
    return sender, backend.restore_kv_cache(decoded.payload)


def _tensor(dtype: str = "float32") -> NeutralKVTensor:
    element_bytes = {"bfloat16": 2, "float16": 2, "float32": 4}[dtype]
    return NeutralKVTensor(dtype, (1, 2, 3, 8), bytes(range(48 * element_bytes)))


def _payload(dtype: str = "float32") -> NeutralKVPayload:
    layers = tuple(
        NeutralKVLayer(index, _tensor(dtype), _tensor(dtype)) for index in range(2)
    )
    return NeutralKVPayload(
        model=KVModelCompatibility("tiny/qwen2", 2, 2, 8),
        logical_cached_token_count=3,
        source_device="cpu",
        layers=layers,
    )


def _model_config() -> KVModelConfig:
    return KVModelConfig(2, 4, 2, 32, 8, "config.head_dim")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("model_id", "other/model", "model identifier mismatch"),
        ("num_hidden_layers", 3, "hidden layer count mismatch"),
        ("num_key_value_heads", 4, "KV head count mismatch"),
        ("head_dim", 16, "head dimension mismatch"),
    ],
)
def test_compatibility_rejects_model_structure_before_tensor_work(
    field, value, message
):
    payload = _payload()
    kwargs = {
        "model_id": "tiny/qwen2",
        "model_config": _model_config(),
        "model_dtype": "torch.float32",
    }
    if field == "model_id":
        payload = replace(payload, model=replace(payload.model, model_id=value))
    else:
        payload = replace(payload, model=replace(payload.model, **{field: value}))
    with pytest.raises(HuggingFaceKVAdapterError, match=message):
        validate_huggingface_kv_compatibility(payload, **kwargs)


def test_compatibility_rejects_dtype_mismatch():
    with pytest.raises(HuggingFaceKVAdapterError, match="does not match Receiver dtype"):
        validate_huggingface_kv_compatibility(
            _payload("float16"),
            model_id="tiny/qwen2",
            model_config=_model_config(),
            model_dtype="torch.float32",
        )


def test_restore_rejects_incompatibility_before_cpu_reconstruction_or_upload():
    class NoTensorWork:
        def __getattr__(self, name):
            raise AssertionError(f"tensor/device work started through {name}")

    payload = replace(
        _payload(), model=replace(_payload().model, model_id="wrong/model")
    )
    with pytest.raises(HuggingFaceKVAdapterError, match="model identifier mismatch"):
        restore_huggingface_kv_cache(
            payload,
            model_id="tiny/qwen2",
            model_config=_model_config(),
            model_dtype="torch.float32",
            target_device="cpu",
            transformers_config=object(),
            torch_module=NoTensorWork(),
        )


def test_bfloat16_cpu_reconstruction_preserves_every_bit_and_shape():
    payload = _payload("bfloat16")
    config = transformers.Qwen2Config(
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )
    restored = restore_huggingface_kv_cache(
        payload,
        model_id="tiny/qwen2",
        model_config=_model_config(),
        model_dtype="torch.bfloat16",
        target_device="cpu",
        transformers_config=config,
        torch_module=torch,
    )
    assert restored.cache.get_seq_length() == 3
    assert len(restored.cache.layers) == 2
    assert restored.cache.layers[0].keys.dtype == torch.bfloat16
    assert restored.cache.layers[0].keys.device.type == "cpu"
    actual = restored.cache.layers[0].keys.view(torch.uint8).numpy().tobytes()
    assert actual == payload.layers[0].key.data
    assert restored.cpu_tensor_reconstruction_ms >= 0
    assert restored.host_to_device_ms >= 0
    assert restored.cache_object_construction_ms >= 0
    assert restored.total_restore_ms >= sum(
        (
            restored.cpu_tensor_reconstruction_ms,
            restored.host_to_device_ms,
            restored.cache_object_construction_ms,
        )
    )


@pytest.mark.parametrize("suffix", [(5,), (5, 6, 7)])
def test_wire_restored_cache_matches_original_and_full_recompute(suffix):
    backend = _backend()
    prefix = (1, 2, 3, 4)
    original = backend.prefill(input_token_ids=prefix)
    sender, restored = _wire_restore(backend, prefix)
    assert original.cache is not sender.cache
    assert sender.cache is not restored.cache

    original_result = backend.continue_from_cache(
        suffix,
        previous_cache=original.cache,
        logical_cached_token_count=len(prefix),
    )
    restored_result = backend.continue_from_cache(
        suffix,
        previous_cache=restored.cache,
        logical_cached_token_count=len(prefix),
    )
    full = backend.prefill(input_token_ids=prefix + suffix)

    assert original_result.logical_cached_token_count == len(prefix) + len(suffix)
    assert restored_result.logical_cached_token_count == len(prefix) + len(suffix)
    assert (
        original_result.predicted_next_token_id
        == restored_result.predicted_next_token_id
        == full.first_token_id
    )
    assert torch.equal(original_result.last_token_logits, restored_result.last_token_logits)
    assert torch.allclose(restored_result.last_token_logits, full.last_token_logits)
    assert (
        torch.max(
            torch.abs(restored_result.last_token_logits - full.last_token_logits)
        ).item()
        < 1e-5
    )


def test_receiver_suffix_uses_full_context_attention_mask_and_implicit_positions():
    backend = _backend()
    prefix = (1, 2, 3, 4)
    prefill = backend.prefill(input_token_ids=prefix)
    captured = {}

    def capture(module, args, kwargs):
        captured.update(kwargs)

    handle = backend._model.register_forward_pre_hook(capture, with_kwargs=True)
    try:
        result = backend.continue_from_cache(
            (5, 6, 7),
            previous_cache=prefill.cache,
            logical_cached_token_count=len(prefix),
        )
    finally:
        handle.remove()
    assert tuple(captured["attention_mask"].shape) == (1, 7)
    assert "position_ids" not in captured
    assert "cache_position" not in captured
    assert result.logical_cached_token_count == 7


def test_restored_cache_multi_step_greedy_decode_matches_original_exactly():
    backend = _backend()
    prefix = (1, 2, 3, 4)
    original = backend.prefill(input_token_ids=prefix)
    _, restored = _wire_restore(backend, prefix)
    original_cache, restored_cache = original.cache, restored.cache
    original_length = restored_length = len(prefix)
    token = 5
    original_ids = []
    restored_ids = []
    for _ in range(5):
        a = backend.continue_from_cache(
            (token,),
            previous_cache=original_cache,
            logical_cached_token_count=original_length,
        )
        b = backend.continue_from_cache(
            (token,),
            previous_cache=restored_cache,
            logical_cached_token_count=restored_length,
        )
        original_ids.append(a.predicted_next_token_id)
        restored_ids.append(b.predicted_next_token_id)
        assert torch.equal(a.last_token_logits, b.last_token_logits)
        original_cache, restored_cache = a.cache, b.cache
        original_length, restored_length = (
            a.logical_cached_token_count,
            b.logical_cached_token_count,
        )
        token = a.predicted_next_token_id
    assert original_ids == restored_ids
    assert restored_length == len(prefix) + 5


def test_pending_prediction_materialization_matches_full_recompute_next_token():
    backend = _backend()
    prefix = (1, 2, 3, 4)
    prefill = backend.prefill(input_token_ids=prefix)
    pending = prefill.first_token_id
    assert prefill.cache.get_seq_length() == len(prefix)
    materialized = backend.materialize_token_into_cache(
        pending,
        previous_cache=prefill.cache,
        logical_cached_token_count=len(prefix),
    )
    full = backend.prefill(input_token_ids=prefix + (pending,))
    assert materialized.logical_cached_token_count == len(prefix) + 1
    assert materialized.cache.get_seq_length() == len(prefix) + 1
    assert materialized.predicted_next_token_id == full.first_token_id
    assert torch.allclose(materialized.last_token_logits, full.last_token_logits)
    assert materialized.materialization_forward_ms >= 0


def test_restore_does_not_execute_receiver_prefill():
    backend = _backend()
    prefix = (1, 2, 3)
    sender = backend.prefill(input_token_ids=prefix)
    wire = serialize_huggingface_kv_cache(
        sender.cache,
        logical_cached_token_count=len(prefix),
        model_id=backend.model_name,
        model_config=backend.kv_model_config,
        torch_module=torch,
    )
    payload = deserialize_kv_packet(wire.packet).payload
    calls = 0

    def count_calls(module, args, output):
        nonlocal calls
        calls += 1

    handle = backend._model.register_forward_hook(count_calls)
    try:
        restored = backend.restore_kv_cache(payload)
    finally:
        handle.remove()
    assert calls == 0
    assert restored.cache.get_seq_length() == len(prefix)


def test_structured_demo_covers_all_continuation_oracles():
    from token_kv_agent_system.kv_restore_demo import run_kv_restore_demo

    result = run_kv_restore_demo(
        _backend(), prefix="prefix", receiver_suffix=" suffix", decode_tokens=4
    )
    assert result.one_token.all_next_token_ids_equal
    assert result.multi_token.all_next_token_ids_equal
    assert result.multi_step.original_restored_exact
    assert result.multi_step.all_paths_exact
    assert result.materialization.continuation_matches_full_recompute
