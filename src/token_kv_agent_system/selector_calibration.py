"""Offline device calibration and prediction validation; never used by Selector."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median

from .agent import LLMAgent
from .kv_handoff import CanonicalHandoffSpan, perform_kv_handoff
from .kv_protocol import KV_PROTOCOL_VERSION, KVModelCompatibility, estimate_kv_wire_size
from .model import HuggingFaceModelBackend
from .model.cache_inspection import inspect_kv_cache
from .network import SimulatedNetwork
from .real_cost import CalibrationPoint, PROFILE_SCHEMA_VERSION, RealCommunicationCostProfile
from .token_cost import TokenCostMeasurementConfig, measure_token_handoff_cost
from .token_protocol import TOKEN_PROTOCOL_VERSION, estimate_token_wire_bytes
from .transport import InMemoryByteTransport


SEED = "Canonical handoff preserves the same logical token sequence. "


def canonical_ids(backend: HuggingFaceModelBackend, count: int) -> tuple[int, ...]:
    seed = backend.encode(SEED)
    if not seed:
        raise ValueError("calibration seed encoded to no tokens")
    return (seed * ((count + len(seed) - 1) // len(seed)))[:count]


def calibrate_backend(
    backend: HuggingFaceModelBackend, *, profile_name: str,
    counts: tuple[int, ...] = (8, 32, 128), warmup_runs: int = 1,
    measurement_runs: int = 3,
) -> RealCommunicationCostProfile:
    if warmup_runs < 1 or measurement_runs < 2:
        raise ValueError("calibration requires warmup and multiple measured runs")
    source, target = LLMAgent("calibration_sender", backend), LLMAgent("calibration_receiver", backend)
    network = SimulatedNetwork(100.0, 5.0)
    transport = InMemoryByteTransport()
    suffix = backend.encode(" Continue briefly.")
    if not suffix:
        raise ValueError("receiver suffix encoded to no tokens")
    points = []
    source_device = None
    kv_dtype = None
    for count in sorted(counts):
        ids = canonical_ids(backend, count)
        # Read runtime KV layout once; this is calibration, never decision time.
        snapshot = inspect_kv_cache(backend.prefill(input_token_ids=ids).cache, count)
        current_device = snapshot.layers[0].key_device
        current_dtype = snapshot.layers[0].key_dtype.removeprefix("torch.")
        if source_device is None:
            source_device, kv_dtype = current_device, current_dtype
        elif (source_device, kv_dtype) != (current_device, current_dtype):
            raise ValueError("KV layout changed across calibration points")
        token = measure_token_handoff_cost(
            source=source, target=target, token_ids=ids, transport=transport,
            networks=(network,), config=TokenCostMeasurementConfig(
                warmup_runs=warmup_runs, measurement_runs=measurement_runs,
                receiver_max_new_tokens=1))
        kv_span = CanonicalHandoffSpan(ids)
        for _ in range(warmup_runs):
            perform_kv_handoff(source=source, target=target, span=kv_span,
                               receiver_suffix_token_ids=suffix, transport=transport,
                               networks=(network,), receiver_max_new_tokens=1)
        kv_runs = tuple(perform_kv_handoff(
            source=source, target=target, span=kv_span,
            receiver_suffix_token_ids=suffix, transport=transport,
            networks=(network,), receiver_max_new_tokens=1)
            for _ in range(measurement_runs))
        expected_token = estimate_token_wire_bytes(count)
        expected_kv = estimate_kv_wire_size(
            count, backend_model(backend), dtype=kv_dtype, source_device=source_device)
        if token.wire_total_bytes != expected_token:
            raise ValueError("Token wire estimator disagrees with real packet")
        for run in kv_runs:
            if (run.raw_kv_tensor_bytes, run.wire_tensor_body_bytes,
                run.wire_metadata_bytes, run.wire_total_bytes) != (
                    expected_kv.raw_kv_tensor_bytes, expected_kv.wire_tensor_body_bytes,
                    expected_kv.wire_metadata_bytes, expected_kv.total_wire_bytes):
                raise ValueError("KV wire estimator disagrees with real packet")
        points.append(CalibrationPoint(
            count, token.serialization_median_ms, token.deserialization_median_ms,
            token.receiver_prefill_median_ms,
            median(run.sender_kv_prepare_ms for run in kv_runs),
            median(run.total_serialize_ms for run in kv_runs),
            median(run.deserialize_ms for run in kv_runs),
            median(run.total_restore_ms for run in kv_runs)))
    config = backend.kv_model_config
    return RealCommunicationCostProfile(
        PROFILE_SCHEMA_VERSION, backend.model_name, config.num_hidden_layers,
        config.num_key_value_heads, config.head_dim, kv_dtype, 1,
        backend.device.split(":")[0], profile_name, source_device,
        TOKEN_PROTOCOL_VERSION, KV_PROTOCOL_VERSION,
        "warmup then independent repeated real handoffs; median per compute stage; simulated link excluded; receiver continuation excluded",
        warmup_runs, measurement_runs, tuple(points))


def backend_model(backend: HuggingFaceModelBackend):
    config = backend.kv_model_config
    return KVModelCompatibility(backend.model_name, config.num_hidden_layers,
                                config.num_key_value_heads, config.head_dim)


@dataclass(frozen=True)
class PredictionValidation:
    token_count: int
    bandwidth_mbps: float
    predicted_token_ms: float
    measured_token_ms: float
    token_absolute_error_ms: float
    token_relative_error: float
    predicted_kv_ms: float
    measured_kv_ms: float
    kv_absolute_error_ms: float
    kv_relative_error: float
    predicted_winner: str
    measured_winner: str


def validate_prediction(backend: HuggingFaceModelBackend, model, *,
                        count: int, bandwidth_mbps: float, base_latency_ms: float = 5.0,
                        warmup_runs: int = 1, measurement_runs: int = 3) -> PredictionValidation:
    """Predict first, then run both real paths solely for offline validation."""
    from .real_cost import RealCostBasedSelector, RealSelectionContext
    context = RealSelectionContext.from_profile(model.profile, count, bandwidth_mbps, base_latency_ms)
    decision = RealCostBasedSelector(model).select(context)
    source, target = LLMAgent("validation_sender", backend), LLMAgent("validation_receiver", backend)
    ids = canonical_ids(backend, count)
    network = SimulatedNetwork(bandwidth_mbps, base_latency_ms)
    transport = InMemoryByteTransport()
    token = measure_token_handoff_cost(source=source, target=target, token_ids=ids,
        transport=transport, networks=(network,), config=TokenCostMeasurementConfig(
            warmup_runs=warmup_runs, measurement_runs=measurement_runs, receiver_max_new_tokens=1))
    suffix = backend.encode(" Continue briefly.")
    span = CanonicalHandoffSpan(ids)
    for _ in range(warmup_runs):
        perform_kv_handoff(source=source, target=target, span=span,
            receiver_suffix_token_ids=suffix, transport=transport, networks=(network,), receiver_max_new_tokens=1)
    runs = tuple(perform_kv_handoff(source=source, target=target, span=span,
        receiver_suffix_token_ids=suffix, transport=transport, networks=(network,), receiver_max_new_tokens=1)
        for _ in range(measurement_runs))
    measured_token = token.simulated_links[0].token_handoff_cost_ms
    measured_kv = (median(run.sender_kv_prepare_ms for run in runs)
                   + median(run.total_serialize_ms for run in runs)
                   + network.estimate_transmission_latency_ms(runs[0].wire_total_bytes)
                   + median(run.deserialize_ms for run in runs)
                   + median(run.total_restore_ms for run in runs))
    predicted_token = decision.token_estimate.total_ms
    predicted_kv = decision.kv_estimate.message_ready_total_ms
    return PredictionValidation(count, bandwidth_mbps, predicted_token, measured_token,
        abs(predicted_token - measured_token), abs(predicted_token - measured_token) / measured_token,
        predicted_kv, measured_kv, abs(predicted_kv - measured_kv),
        abs(predicted_kv - measured_kv) / measured_kv, decision.selected_mode.value,
        "TOKEN" if measured_token <= measured_kv else "KV")
