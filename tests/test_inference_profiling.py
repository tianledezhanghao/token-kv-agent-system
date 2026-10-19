from dataclasses import fields, is_dataclass
from math import prod
from types import SimpleNamespace

import pytest

from token_kv_agent_system.model import (
    DecodeStepResult,
    ModelMetadata,
    PrefillResult,
    ProfileConfig,
    measure_context_scaling,
    measure_inference_profile,
)


class FakeTensor:
    def __init__(self, shape, *, element_size=2):
        self.shape = tuple(shape)
        self.dtype = "torch.float16"
        self.device = "cpu"
        self._element_size = element_size

    def numel(self):
        return prod(self.shape)

    def element_size(self):
        return self._element_size


class FakeLayer:
    def __init__(self, tokens):
        self.keys = FakeTensor((1, 2, tokens, 4))
        self.values = FakeTensor((1, 2, tokens, 4))

    def grow(self, tokens):
        self.keys = FakeTensor((1, 2, tokens, 4))
        self.values = FakeTensor((1, 2, tokens, 4))


class FakeDynamicCache:
    def __init__(self, tokens, identity):
        self.identity = identity
        self.layers = [FakeLayer(tokens), FakeLayer(tokens)]

    def grow(self, tokens):
        for layer in self.layers:
            layer.grow(tokens)


class FakeProfilerBackend:
    def __init__(self, *, vocab_size=5, prefill_times=None):
        self.metadata = ModelMetadata(
            model_name="fake/profile-model",
            device="cpu",
            vocab_size=vocab_size,
            eos_token_id=None,
            bos_token_id=None,
            pad_token_id=None,
            load_wall_time_seconds=0.0,
            dtype="torch.float16",
        )
        self.eos_token_ids = ()
        self.prefill_times = iter(prefill_times or [10.0] * 100)
        self.created_caches = []
        self.decode_cache_ids = []

    @property
    def model_name(self):
        return self.metadata.model_name

    @property
    def device(self):
        return self.metadata.device

    @property
    def vocab_size(self):
        return self.metadata.vocab_size

    def encode(self, text):
        return tuple(index % self.vocab_size for index, _ in enumerate(text.split()))

    def decode(self, token_ids, *, skip_special_tokens=False):
        return " ".join(str(token_id) for token_id in token_ids)

    def prefill(self, prompt=None, *, input_token_ids=None):
        assert prompt is None
        ids = tuple(input_token_ids)
        cache = FakeDynamicCache(len(ids), len(self.created_caches))
        self.created_caches.append(cache)
        return PrefillResult(
            input_token_ids=ids,
            first_token_id=1,
            cache=cache,
            prefill_forward_ms=next(self.prefill_times),
            logical_cached_token_count=len(ids),
            logits_shape=(1, len(ids), self.vocab_size),
            last_token_logits_shape=(self.vocab_size,),
            last_token_logits=SimpleNamespace(),
        )

    def decode_step(
        self, input_token_id, *, previous_cache, logical_cached_token_count
    ):
        self.decode_cache_ids.append(previous_cache.identity)
        new_count = logical_cached_token_count + 1
        previous_cache.grow(new_count)
        return DecodeStepResult(
            input_token_id=input_token_id,
            predicted_next_token_id=(input_token_id + 1) % self.vocab_size,
            cache=previous_cache,
            decode_forward_ms=2.0 + input_token_id,
            logical_cached_token_count=new_count,
        )


def _contains_runtime_object(value):
    if isinstance(value, (FakeTensor, FakeDynamicCache)):
        return True
    if is_dataclass(value):
        return any(_contains_runtime_object(getattr(value, field.name)) for field in fields(value))
    if isinstance(value, (tuple, list)):
        return any(_contains_runtime_object(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_runtime_object(item) for item in value.values())
    return False


def test_profile_aggregates_token_prefill_decode_kv_and_ratio_metadata():
    backend = FakeProfilerBackend(prefill_times=[10.0, 30.0, 20.0])

    profile = measure_inference_profile(
        backend,
        "one two three",
        ProfileConfig(max_new_tokens=3, warmup_runs=0, measurement_runs=3),
    )

    assert profile.measured.model.model_name == "fake/profile-model"
    assert profile.measured.model.dtype == "torch.float16"
    assert profile.measured.input.input_token_ids == (0, 1, 2)
    assert profile.measured.input.input_token_count == 3
    assert profile.derived.theoretical_bits_per_token == 3
    assert profile.derived.theoretical_token_payload_bits == 9
    assert profile.derived.theoretical_token_payload_bytes == 2
    assert profile.derived.prefill_forward_samples_ms == (10.0, 30.0, 20.0)
    assert profile.derived.median_prefill_forward_ms == 20.0

    sample = profile.measured.samples[0]
    assert sample.prefill.cached_token_count_after_prefill == 3
    assert sample.kv.kv_layer_count == 2
    assert sample.kv.key_bytes == 96
    assert sample.kv.value_bytes == 96
    assert sample.kv.raw_kv_tensor_bytes == 192
    assert sample.kv.raw_kv_tensor_bytes_after_decode_steps == (256, 320)
    assert sample.decode.generated_token_count == 3
    assert sample.decode.decode_forward_count == 2
    assert sample.decode.per_step_decode_forward_ms == (3.0, 4.0)
    assert profile.derived.decode_forward_samples_ms == (3.0, 4.0) * 3
    assert profile.derived.mean_decode_forward_ms == 3.5
    assert profile.derived.median_decode_forward_ms == 3.5
    assert profile.derived.per_run_total_decode_forward_ms == (7.0, 7.0, 7.0)
    assert profile.derived.median_total_decode_forward_ms == 7.0
    assert profile.derived.raw_kv_bytes_per_cached_token == 64
    assert profile.derived.raw_kv_growth_bytes_per_decode_step == (64, 64) * 3
    assert profile.derived.raw_kv_to_theoretical_token_payload_ratio == 96.0


def test_profile_is_json_serializable_and_contains_no_tensor_or_cache():
    profile = measure_inference_profile(
        FakeProfilerBackend(),
        "one two three",
        ProfileConfig(max_new_tokens=2, warmup_runs=0, measurement_runs=1),
    )

    serialized = profile.to_json()

    assert '"measured"' in serialized
    assert '"derived"' in serialized
    assert '"raw_kv_tensor_bytes": 192' in serialized
    assert not _contains_runtime_object(profile)


def test_warmup_and_formal_measurements_each_create_an_independent_cache():
    backend = FakeProfilerBackend(prefill_times=[999.0, 10.0, 20.0, 30.0])

    profile = measure_inference_profile(
        backend,
        "one two three",
        ProfileConfig(max_new_tokens=2, warmup_runs=1, measurement_runs=3),
    )

    assert len(backend.created_caches) == 4
    assert len({cache.identity for cache in backend.created_caches}) == 4
    assert backend.decode_cache_ids == [0, 1, 2, 3]
    assert profile.derived.prefill_forward_samples_ms == (10.0, 20.0, 30.0)
    assert 999.0 not in profile.derived.prefill_forward_samples_ms


def test_context_scaling_profiles_requested_lengths_without_cache_reuse():
    backend = FakeProfilerBackend()

    scaling = measure_context_scaling(
        backend,
        target_token_counts=(2, 5, 9),
        config=ProfileConfig(max_new_tokens=1, warmup_runs=0, measurement_runs=2),
        seed_text="one two ",
    )

    actual_counts = tuple(
        entry.profile.measured.input.input_token_count for entry in scaling.entries
    )
    assert actual_counts == (2, 6, 10)
    assert len(backend.created_caches) == 6
    assert len({cache.identity for cache in backend.created_caches}) == 6
    assert all(
        entry.profile.derived.raw_kv_bytes_per_cached_token == 64
        for entry in scaling.entries
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"warmup_runs": -1},
        {"measurement_runs": 0},
    ],
)
def test_invalid_profile_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        ProfileConfig(**kwargs)


def test_empty_prompt_and_invalid_scaling_configuration_are_rejected():
    backend = FakeProfilerBackend()
    config = ProfileConfig(warmup_runs=0, measurement_runs=1)

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        measure_inference_profile(backend, "", config)
    with pytest.raises(ValueError, match="cannot be empty"):
        measure_context_scaling(backend, target_token_counts=(), config=config)
    with pytest.raises(ValueError, match="positive"):
        measure_context_scaling(backend, target_token_counts=(0,), config=config)
    with pytest.raises(ValueError, match="strictly increasing"):
        measure_context_scaling(backend, target_token_counts=(5, 2), config=config)
