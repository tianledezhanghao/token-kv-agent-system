from dataclasses import fields, is_dataclass

import pytest

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.model import (
    DecodeStepResult,
    GenerationResult,
    IncrementalCausalLMBackend,
    ModelMetadata,
    PrefillResult,
    TokenIdGenerationResult,
)
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.token_cost import (
    TokenCostMeasurementConfig,
    measure_token_handoff_cost,
)
from token_kv_agent_system.token_handoff import TokenHandoffError
from token_kv_agent_system.transport import InMemoryByteTransport


class FakeCostBackend(IncrementalCausalLMBackend):
    def __init__(self, prefill_times=(999.0, 10.0, 20.0, 30.0)):
        self.prefill_times = iter(prefill_times)
        self.prefill_inputs = []
        self.caches = []
        self._metadata = ModelMetadata(
            model_name="fake/cost",
            device="cpu",
            vocab_size=16,
            eos_token_id=None,
            bos_token_id=None,
            pad_token_id=None,
            load_wall_time_seconds=0.0,
        )

    @property
    def metadata(self):
        return self._metadata

    @property
    def eos_token_ids(self):
        return ()

    def encode(self, text):
        return (1, 2, 3)

    def decode(self, token_ids, *, skip_special_tokens=False):
        return " ".join(map(str, token_ids))

    def generate(self, prompt, config=None):
        return GenerationResult(prompt, (1,), "", (1,), 0.0, (1, 1))

    def prefill(self, prompt=None, *, input_token_ids=None):
        ids = tuple(input_token_ids or ())
        cache = object()
        self.prefill_inputs.append(ids)
        self.caches.append(cache)
        return PrefillResult(
            input_token_ids=ids,
            first_token_id=4,
            cache=cache,
            prefill_forward_ms=next(self.prefill_times),
            logical_cached_token_count=len(ids),
            logits_shape=(1, len(ids), 16),
            last_token_logits_shape=(16,),
            last_token_logits=object(),
        )

    def decode_step(self, input_token_id, *, previous_cache, logical_cached_token_count):
        return DecodeStepResult(
            input_token_id=input_token_id,
            predicted_next_token_id=input_token_id + 1,
            cache=previous_cache,
            decode_forward_ms=2.0,
            logical_cached_token_count=logical_cached_token_count + 1,
        )

    def manual_generate(self, prompt, config=None, *, warmup_runs=1):
        raise NotImplementedError

    def generate_from_token_ids(self, input_token_ids, config=None):
        prefill = self.prefill(input_token_ids=input_token_ids)
        step = self.decode_step(
            prefill.first_token_id,
            previous_cache=prefill.cache,
            logical_cached_token_count=prefill.logical_cached_token_count,
        )
        return TokenIdGenerationResult(
            input_token_ids=prefill.input_token_ids,
            output_token_ids=(prefill.first_token_id, step.predicted_next_token_id),
            generated_text="4 5",
            prefill=prefill,
            decode_steps=(step,),
            stop_reason="max_new_tokens",
        )


class MillisecondClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        current = self.value
        self.value += 0.001
        return current


def _measure(backend=None):
    backend = backend or FakeCostBackend()
    return measure_token_handoff_cost(
        source=LLMAgent("agent_a", backend),
        target=LLMAgent("agent_b", backend),
        token_ids=(1, 2, 3),
        transport=InMemoryByteTransport(),
        networks=(
            SimulatedNetwork(20, base_latency_ms=5),
            SimulatedNetwork(1_000, base_latency_ms=5),
        ),
        config=TokenCostMeasurementConfig(
            warmup_runs=1, measurement_runs=3, receiver_max_new_tokens=2
        ),
        _clock=MillisecondClock(),
    )


def test_measurement_separates_measured_stages_simulated_links_and_decode():
    backend = FakeCostBackend()
    result = _measure(backend)

    assert result.token_count == 3
    assert result.config.measurement_runs == 3
    assert result.theoretical_token_payload_bytes == 2
    assert result.wire_header_bytes == 10
    assert result.wire_body_bytes == 12
    assert result.wire_total_bytes == 22
    assert result.serialization_samples_ms == pytest.approx((1.0, 1.0, 1.0))
    assert result.serialization_median_ms == pytest.approx(1.0)
    assert result.local_byte_transfer_ms == pytest.approx(1.0)
    assert result.deserialization_median_ms == pytest.approx(1.0)
    assert result.first_warmup_prefill_ms == 999.0
    assert result.receiver_prefill_samples_ms == (10.0, 20.0, 30.0)
    assert result.receiver_prefill_median_ms == 20.0
    assert result.receiver_generated_token_count == 2
    assert result.receiver_decode_forward_count == 1
    assert result.receiver_decode_samples_ms == (2.0, 2.0, 2.0)
    assert result.receiver_decode_mean_ms == 2.0
    assert result.receiver_decode_median_ms == 2.0
    assert result.sender_ids_match_received is True
    assert result.received_ids_match_prefill_input is True
    assert len({id(cache) for cache in backend.caches}) == 4

    slow, fast = result.simulated_links
    assert slow.simulated_link_latency_ms == pytest.approx(5.0088)
    assert fast.simulated_link_latency_ms == pytest.approx(5.000176)
    # Formal cost excludes both the 1 ms local transfer and Decode.
    assert slow.token_handoff_cost_ms == pytest.approx(27.0088)
    assert fast.token_handoff_cost_ms == pytest.approx(27.000176)


def test_measurement_is_json_safe_and_holds_no_runtime_heavy_objects():
    result = _measure()
    serialized = result.to_json()

    assert '"wire_total_bytes": 22' in serialized
    assert '"local_transfer_is_diagnostic_only": true' in serialized
    names = {field.name.lower() for field in fields(result)}
    assert not any(word in name for name in names for word in ("tensor", "cache", "model"))
    assert is_dataclass(result)
    assert not any(isinstance(value, bytes) for value in result.__dict__.values())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"warmup_runs": -1},
        {"measurement_runs": 0},
        {"receiver_max_new_tokens": 0},
    ],
)
def test_invalid_measurement_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        TokenCostMeasurementConfig(**kwargs)


def test_empty_networks_and_incompatible_agents_are_rejected():
    backend = FakeCostBackend()
    common = dict(
        source=LLMAgent("agent_a", backend),
        target=LLMAgent("agent_b", backend),
        token_ids=(1,),
        transport=InMemoryByteTransport(),
        config=TokenCostMeasurementConfig(warmup_runs=0, measurement_runs=1),
    )
    with pytest.raises(ValueError, match="networks cannot be empty"):
        measure_token_handoff_cost(networks=(), **common)

    with pytest.raises(TokenHandoffError, match="share one model backend"):
        measure_token_handoff_cost(
            networks=(SimulatedNetwork(20),),
            **{**common, "target": LLMAgent("agent_b", FakeCostBackend())},
        )


def test_corrupted_ids_fail_before_performance_is_reported():
    backend = FakeCostBackend()

    def wrong_deserializer(data):
        from token_kv_agent_system.token_protocol import deserialize_token_packet

        packet = deserialize_token_packet(data)
        return type(packet)(
            packet.protocol_version,
            packet.token_width_bits,
            (9, 9, 9),
            packet.wire_bytes,
        )

    with pytest.raises(TokenHandoffError, match="do not match"):
        measure_token_handoff_cost(
            source=LLMAgent("agent_a", backend),
            target=LLMAgent("agent_b", backend),
            token_ids=(1, 2, 3),
            transport=InMemoryByteTransport(),
            networks=(SimulatedNetwork(20),),
            config=TokenCostMeasurementConfig(warmup_runs=0, measurement_runs=1),
            deserializer=wrong_deserializer,
        )
