"""Canonical same-span real KV handoff and cost measurement."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict, dataclass
import json
from statistics import median
from time import perf_counter
from typing import Any, Callable, Mapping

from .agent import LLMAgent
from .kv_protocol import deserialize_kv_packet
from .model import serialize_huggingface_kv_cache
from .network import SimulatedNetwork
from .transport import BaseByteTransport


KV_HANDOFF_SCHEMA_VERSION = 1
KVHandoffStageListener = Callable[[str, Mapping[str, object]], None]


class KVHandoffError(ValueError):
    """Raised when a canonical KV handoff cannot preserve its contract."""


@dataclass(frozen=True)
class CanonicalHandoffSpan:
    """The one logical Token sequence represented by either TOKEN or KV."""

    handoff_token_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.handoff_token_ids:
            raise ValueError("handoff_token_ids cannot be empty")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in self.handoff_token_ids):
            raise ValueError("handoff_token_ids must contain non-negative integers")

    @property
    def token_count(self) -> int:
        return len(self.handoff_token_ids)


@dataclass(frozen=True)
class KVSimulatedLinkCost:
    """One derived link scenario using the actual KV packet length."""

    bandwidth_mbps: float
    base_latency_ms: float
    simulated_link_latency_ms: float
    kv_handoff_cost_ms: float
    kv_end_to_end_representation_cost_ms: float


@dataclass(frozen=True)
class KVHandoffResult:
    """JSON-safe evidence for one real bytes-only KV handoff."""

    schema_version: int
    source_agent_id: str
    target_agent_id: str
    handoff_token_ids: tuple[int, ...]
    receiver_suffix_token_ids: tuple[int, ...]
    token_count: int
    logical_cached_tokens: int
    raw_kv_tensor_bytes: int
    wire_tensor_body_bytes: int
    wire_metadata_bytes: int
    wire_total_bytes: int
    received_byte_count: int
    sender_kv_prepare_ms: float
    materialization_forward_ms: float | None
    device_to_host_ms: float
    protocol_encode_ms: float
    total_serialize_ms: float
    local_transfer_ms: float
    deserialize_ms: float
    cpu_tensor_reconstruction_ms: float
    host_to_device_ms: float
    cache_construction_ms: float
    total_restore_ms: float
    receiver_continuation_forward_ms: float
    receiver_output_token_ids: tuple[int, ...]
    receiver_generated_text: str
    receiver_next_token_id: int
    restored_cache_length_before_continuation: int
    restored_cache_length_after_suffix: int
    transport_name: str
    compatibility_verified: bool
    wire_identity_verified: bool
    canonical_cache_complete: bool
    receiver_prefix_prefill_executed: bool
    local_transfer_is_diagnostic_only: bool
    simulated_links: tuple[KVSimulatedLinkCost, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass(frozen=True)
class CanonicalEquivalenceResult:
    """TOKEN-prefill, wire-restored KV, and full-recompute oracle evidence."""

    handoff_token_ids: tuple[int, ...]
    receiver_suffix_token_ids: tuple[int, ...]
    token_path_next_token_id: int
    kv_path_next_token_id: int
    full_recompute_next_token_id: int
    token_vs_kv_max_abs_diff: float
    token_vs_kv_mean_abs_diff: float
    token_vs_kv_close: bool
    kv_vs_full_max_abs_diff: float
    kv_vs_full_mean_abs_diff: float
    kv_vs_full_close: bool
    all_next_token_ids_equal: bool
    kv_receiver_prefix_prefill_executed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class KVScalingMeasurement:
    """Median stage costs for one canonical span length."""

    token_count: int
    raw_kv_tensor_bytes: int
    wire_total_bytes: int
    sender_kv_prepare_median_ms: float
    total_serialize_median_ms: float
    deserialize_median_ms: float
    total_restore_median_ms: float
    links: tuple[KVSimulatedLinkCost, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def perform_kv_handoff(
    *,
    source: LLMAgent,
    target: LLMAgent,
    span: CanonicalHandoffSpan,
    receiver_suffix_token_ids: tuple[int, ...] | list[int],
    transport: BaseByteTransport,
    networks: Iterable[SimulatedNetwork],
    receiver_max_new_tokens: int = 8,
    stage_listener: KVHandoffStageListener | None = None,
) -> KVHandoffResult:
    """Run the real KV byte path without a Receiver prefix Prefill."""

    _validate_agents(source, target)
    suffix_ids = tuple(receiver_suffix_token_ids)
    if not suffix_ids:
        raise KVHandoffError("receiver suffix cannot be empty")
    if receiver_max_new_tokens <= 0:
        raise ValueError("receiver_max_new_tokens must be greater than zero")
    network_scenarios = tuple(networks)
    if not network_scenarios:
        raise ValueError("networks cannot be empty")
    backend = source.backend
    _require_kv_backend(backend)

    # Canonical v1 preparation is exactly one clean Prefill over the complete
    # span. Its predicted next Token is irrelevant and is never materialized.
    prepared = backend.prefill(input_token_ids=span.handoff_token_ids)
    runtime_length = _cache_length(prepared.cache)
    if (
        prepared.input_token_ids != span.handoff_token_ids
        or prepared.logical_cached_token_count != span.token_count
        or runtime_length != span.token_count
    ):
        raise KVHandoffError("Sender clean Prefill did not materialize the canonical span")
    _notify_stage(
        stage_listener,
        "prepared",
        {
            "token_count": span.token_count,
            "sender_kv_prepare_ms": prepared.prefill_forward_ms,
            "canonical_cache_complete": True,
        },
    )

    serialized = serialize_huggingface_kv_cache(
        prepared.cache,
        logical_cached_token_count=span.token_count,
        model_id=backend.model_name,
        model_config=backend.kv_model_config,
    )
    if serialized.total_wire_bytes != len(serialized.packet):
        raise KVHandoffError("KV wire byte count does not equal len(packet)")
    _notify_stage(
        stage_listener,
        "serialized",
        {
            "token_count": span.token_count,
            "raw_kv_tensor_bytes": serialized.raw_kv_tensor_bytes,
            "wire_tensor_body_bytes": serialized.wire_tensor_body_bytes,
            "wire_metadata_bytes": serialized.wire_metadata_bytes,
            "wire_total_bytes": serialized.total_wire_bytes,
            "device_to_host_ms": serialized.device_to_host_ms,
            "protocol_encode_ms": serialized.protocol_encode_ms,
            "total_serialize_ms": serialized.total_serialize_ms,
        },
    )

    transfer_started = perf_counter()
    delivered = transport.transfer(
        serialized.packet, source=source.agent_id, target=target.agent_id
    )
    local_transfer_ms = (perf_counter() - transfer_started) * 1_000.0
    if not isinstance(delivered, bytes):
        raise TypeError("byte transport must deliver bytes")
    if len(delivered) != len(serialized.packet):
        raise KVHandoffError("transport changed KV packet byte count")
    _notify_stage(
        stage_listener,
        "transferred",
        {
            "wire_total_bytes": len(delivered),
            "local_transfer_ms": local_transfer_ms,
            "local_transfer_is_diagnostic_only": True,
        },
    )

    decoded = deserialize_kv_packet(delivered)
    if decoded.total_wire_bytes != len(delivered):
        raise KVHandoffError("deserializer byte count does not match delivered packet")
    _notify_stage(
        stage_listener,
        "deserialized",
        {
            "wire_total_bytes": decoded.total_wire_bytes,
            "deserialize_ms": decoded.deserialize_ms,
        },
    )
    restored = target.backend.restore_kv_cache(decoded.payload)
    if restored.logical_cached_token_count != span.token_count:
        raise KVHandoffError("Receiver restored cache length differs from canonical span")
    _notify_stage(
        stage_listener,
        "restored",
        {
            "logical_cached_tokens": restored.logical_cached_token_count,
            "cpu_tensor_reconstruction_ms": restored.cpu_tensor_reconstruction_ms,
            "host_to_device_ms": restored.host_to_device_ms,
            "cache_construction_ms": restored.cache_object_construction_ms,
            "total_restore_ms": restored.total_restore_ms,
            "receiver_prefix_prefill_executed": False,
            "compatibility_verified": bool(restored.compatibility.passed),
        },
    )

    first, output_ids = _continue_greedy(
        target.backend,
        restored.cache,
        span.token_count,
        suffix_ids,
        receiver_max_new_tokens,
    )
    links = tuple(
        _derive_link_cost(
            network,
            serialized.total_wire_bytes,
            prepared.prefill_forward_ms,
            serialized.total_serialize_ms,
            decoded.deserialize_ms,
            restored.total_restore_ms,
        )
        for network in network_scenarios
    )
    return KVHandoffResult(
        schema_version=KV_HANDOFF_SCHEMA_VERSION,
        source_agent_id=source.agent_id,
        target_agent_id=target.agent_id,
        handoff_token_ids=span.handoff_token_ids,
        receiver_suffix_token_ids=suffix_ids,
        token_count=span.token_count,
        logical_cached_tokens=restored.logical_cached_token_count,
        raw_kv_tensor_bytes=serialized.raw_kv_tensor_bytes,
        wire_tensor_body_bytes=serialized.wire_tensor_body_bytes,
        wire_metadata_bytes=serialized.wire_metadata_bytes,
        wire_total_bytes=serialized.total_wire_bytes,
        received_byte_count=len(delivered),
        sender_kv_prepare_ms=prepared.prefill_forward_ms,
        materialization_forward_ms=None,
        device_to_host_ms=serialized.device_to_host_ms,
        protocol_encode_ms=serialized.protocol_encode_ms,
        total_serialize_ms=serialized.total_serialize_ms,
        local_transfer_ms=local_transfer_ms,
        deserialize_ms=decoded.deserialize_ms,
        cpu_tensor_reconstruction_ms=restored.cpu_tensor_reconstruction_ms,
        host_to_device_ms=restored.host_to_device_ms,
        cache_construction_ms=restored.cache_object_construction_ms,
        total_restore_ms=restored.total_restore_ms,
        receiver_continuation_forward_ms=first.forward_ms,
        receiver_output_token_ids=output_ids,
        receiver_generated_text=target.backend.decode(
            output_ids, skip_special_tokens=True
        ),
        receiver_next_token_id=first.predicted_next_token_id,
        restored_cache_length_before_continuation=span.token_count,
        restored_cache_length_after_suffix=first.logical_cached_token_count,
        transport_name=type(transport).__name__,
        compatibility_verified=bool(restored.compatibility.passed),
        wire_identity_verified=(serialized.packet == delivered),
        canonical_cache_complete=(runtime_length == span.token_count),
        receiver_prefix_prefill_executed=False,
        local_transfer_is_diagnostic_only=True,
        simulated_links=links,
    )


def compare_canonical_token_kv(
    *,
    source: LLMAgent,
    target: LLMAgent,
    span: CanonicalHandoffSpan,
    receiver_suffix_token_ids: tuple[int, ...] | list[int],
    transport: BaseByteTransport,
) -> CanonicalEquivalenceResult:
    """Compare equal logical spans without putting an oracle Prefill in KV path."""

    _validate_agents(source, target)
    suffix_ids = tuple(receiver_suffix_token_ids)
    if not suffix_ids:
        raise KVHandoffError("receiver suffix cannot be empty")
    backend = source.backend
    _require_kv_backend(backend)

    token_prefix = target.backend.prefill(input_token_ids=span.handoff_token_ids)
    token_result = target.backend.continue_from_cache(
        suffix_ids,
        previous_cache=token_prefix.cache,
        logical_cached_token_count=span.token_count,
    )

    sender_prefix = backend.prefill(input_token_ids=span.handoff_token_ids)
    encoded = serialize_huggingface_kv_cache(
        sender_prefix.cache,
        logical_cached_token_count=span.token_count,
        model_id=backend.model_name,
        model_config=backend.kv_model_config,
    )
    delivered = transport.transfer(
        encoded.packet, source=source.agent_id, target=target.agent_id
    )
    restored = target.backend.restore_kv_cache(deserialize_kv_packet(delivered).payload)
    kv_result = target.backend.continue_from_cache(
        suffix_ids,
        previous_cache=restored.cache,
        logical_cached_token_count=span.token_count,
    )
    full = target.backend.prefill(
        input_token_ids=span.handoff_token_ids + suffix_ids
    )

    token_kv = _logits_difference(backend, token_result.last_token_logits, kv_result.last_token_logits)
    kv_full = _logits_difference(backend, kv_result.last_token_logits, full.last_token_logits)
    next_ids = (
        token_result.predicted_next_token_id,
        kv_result.predicted_next_token_id,
        full.first_token_id,
    )
    result = CanonicalEquivalenceResult(
        handoff_token_ids=span.handoff_token_ids,
        receiver_suffix_token_ids=suffix_ids,
        token_path_next_token_id=next_ids[0],
        kv_path_next_token_id=next_ids[1],
        full_recompute_next_token_id=next_ids[2],
        token_vs_kv_max_abs_diff=token_kv[0],
        token_vs_kv_mean_abs_diff=token_kv[1],
        token_vs_kv_close=token_kv[2],
        kv_vs_full_max_abs_diff=kv_full[0],
        kv_vs_full_mean_abs_diff=kv_full[1],
        kv_vs_full_close=kv_full[2],
        all_next_token_ids_equal=len(set(next_ids)) == 1,
    )
    if not result.all_next_token_ids_equal:
        raise KVHandoffError("TOKEN/KV/Full Recompute next-token oracle diverged")
    if not result.token_vs_kv_close or not result.kv_vs_full_close:
        raise KVHandoffError("canonical continuation logits exceed tolerance")
    return result


def measure_kv_scaling(
    *,
    source: LLMAgent,
    target: LLMAgent,
    spans: Iterable[CanonicalHandoffSpan],
    receiver_suffix_token_ids: tuple[int, ...] | list[int],
    transport: BaseByteTransport,
    networks: Iterable[SimulatedNetwork],
    measurement_runs: int = 3,
) -> tuple[KVScalingMeasurement, ...]:
    """Repeat real handoffs and aggregate medians without retaining runtime state."""

    if measurement_runs <= 0:
        raise ValueError("measurement_runs must be greater than zero")
    network_scenarios = tuple(networks)
    measurements: list[KVScalingMeasurement] = []
    for span in spans:
        runs = tuple(
            perform_kv_handoff(
                source=source,
                target=target,
                span=span,
                receiver_suffix_token_ids=receiver_suffix_token_ids,
                transport=transport,
                networks=network_scenarios,
                receiver_max_new_tokens=1,
            )
            for _ in range(measurement_runs)
        )
        first = runs[0]
        if any(
            (run.token_count, run.raw_kv_tensor_bytes, run.wire_total_bytes)
            != (first.token_count, first.raw_kv_tensor_bytes, first.wire_total_bytes)
            for run in runs[1:]
        ):
            raise KVHandoffError("KV size changed between scaling measurements")
        prepare = median(run.sender_kv_prepare_ms for run in runs)
        serialize = median(run.total_serialize_ms for run in runs)
        deserialize = median(run.deserialize_ms for run in runs)
        restore = median(run.total_restore_ms for run in runs)
        links = tuple(
            _derive_link_cost(
                network,
                first.wire_total_bytes,
                prepare,
                serialize,
                deserialize,
                restore,
            )
            for network in network_scenarios
        )
        measurements.append(
            KVScalingMeasurement(
                token_count=first.token_count,
                raw_kv_tensor_bytes=first.raw_kv_tensor_bytes,
                wire_total_bytes=first.wire_total_bytes,
                sender_kv_prepare_median_ms=prepare,
                total_serialize_median_ms=serialize,
                deserialize_median_ms=deserialize,
                total_restore_median_ms=restore,
                links=links,
            )
        )
    return tuple(measurements)


class BidirectionalKVHandoffSession:
    """Alternate A->B and B->A only after successful canonical handoffs."""

    def __init__(
        self,
        agent_a: LLMAgent,
        agent_b: LLMAgent,
        transport: BaseByteTransport,
        networks: Iterable[SimulatedNetwork],
    ) -> None:
        _validate_agents(agent_a, agent_b)
        self._agents = (agent_a, agent_b)
        self._transport = transport
        self._networks = tuple(networks)
        if not self._networks:
            raise ValueError("networks cannot be empty")
        self._direction_index = 0

    @property
    def next_direction(self) -> tuple[str, str]:
        source = self._agents[self._direction_index]
        target = self._agents[1 - self._direction_index]
        return source.agent_id, target.agent_id

    def handoff_next(
        self,
        handoff_token_ids: tuple[int, ...] | list[int],
        *,
        receiver_suffix_token_ids: tuple[int, ...] | list[int],
        receiver_max_new_tokens: int = 8,
        stage_listener: KVHandoffStageListener | None = None,
        advance_direction: bool = True,
    ) -> KVHandoffResult:
        source = self._agents[self._direction_index]
        target = self._agents[1 - self._direction_index]
        result = perform_kv_handoff(
            source=source,
            target=target,
            span=CanonicalHandoffSpan(tuple(handoff_token_ids)),
            receiver_suffix_token_ids=receiver_suffix_token_ids,
            transport=self._transport,
            networks=self._networks,
            receiver_max_new_tokens=receiver_max_new_tokens,
            stage_listener=stage_listener,
        )
        if advance_direction:
            self.advance_direction()
        return result

    def advance_direction(self) -> None:
        """Commit a successful handoff direction after caller-side checks."""
        self._direction_index = 1 - self._direction_index


def _continue_greedy(
    backend: Any,
    cache: object,
    cached_tokens: int,
    suffix_ids: tuple[int, ...],
    max_new_tokens: int,
) -> tuple[Any, tuple[int, ...]]:
    first = backend.continue_from_cache(
        suffix_ids,
        previous_cache=cache,
        logical_cached_token_count=cached_tokens,
    )
    output = [first.predicted_next_token_id]
    current_cache = first.cache
    current_length = first.logical_cached_token_count
    eos_ids = set(getattr(backend, "eos_token_ids", ()))
    while len(output) < max_new_tokens and output[-1] not in eos_ids:
        step = backend.continue_from_cache(
            (output[-1],),
            previous_cache=current_cache,
            logical_cached_token_count=current_length,
        )
        output.append(step.predicted_next_token_id)
        current_cache = step.cache
        current_length = step.logical_cached_token_count
    return first, tuple(output)


def _derive_link_cost(
    network: SimulatedNetwork,
    wire_bytes: int,
    sender_prepare_ms: float,
    total_serialize_ms: float,
    deserialize_ms: float,
    total_restore_ms: float,
) -> KVSimulatedLinkCost:
    stages = (sender_prepare_ms, total_serialize_ms, deserialize_ms, total_restore_ms)
    if any(value < 0 for value in stages):
        raise ValueError("KV handoff stage timings cannot be negative")
    link_ms = network.estimate_transmission_latency_ms(wire_bytes)
    ready_cost = total_serialize_ms + link_ms + deserialize_ms + total_restore_ms
    return KVSimulatedLinkCost(
        bandwidth_mbps=network.state.bandwidth_mbps,
        base_latency_ms=network.state.base_latency_ms,
        simulated_link_latency_ms=link_ms,
        kv_handoff_cost_ms=ready_cost,
        kv_end_to_end_representation_cost_ms=sender_prepare_ms + ready_cost,
    )


def _logits_difference(backend: Any, left: Any, right: Any) -> tuple[float, float, bool]:
    torch_module = getattr(backend, "_torch", None)
    if torch_module is None:
        raise TypeError("backend does not expose the tensor runtime needed for logits comparison")
    absolute = torch_module.abs(left.float() - right.float())
    return (
        float(torch_module.max(absolute).item()),
        float(torch_module.mean(absolute).item()),
        bool(torch_module.allclose(left.float(), right.float(), rtol=2e-2, atol=5e-1)),
    )


def _validate_agents(source: LLMAgent, target: LLMAgent) -> None:
    if source.agent_id == target.agent_id:
        raise KVHandoffError("source and target must be different Agents")
    from .model.backend_pair import validate_backend_compatibility
    try:
        validate_backend_compatibility(source.backend, target.backend)
    except ValueError as error:
        raise KVHandoffError(str(error)) from error
    _require_kv_backend(source.backend)
    _require_kv_backend(target.backend)


def _require_kv_backend(backend: Any) -> None:
    required = (
        "prefill",
        "continue_from_cache",
        "restore_kv_cache",
        "kv_model_config",
    )
    missing = [name for name in required if not hasattr(backend, name)]
    if missing:
        raise TypeError(f"backend lacks real KV capabilities: {', '.join(missing)}")


def _cache_length(cache: object) -> int:
    get_seq_length = getattr(cache, "get_seq_length", None)
    if not callable(get_seq_length):
        raise KVHandoffError("prepared cache does not expose get_seq_length()")
    return int(get_seq_length())


def _notify_stage(
    listener: KVHandoffStageListener | None,
    stage: str,
    metadata: Mapping[str, object],
) -> None:
    if listener is not None:
        listener(stage, metadata)


__all__ = [
    "BidirectionalKVHandoffSession",
    "CanonicalEquivalenceResult",
    "CanonicalHandoffSpan",
    "KVHandoffError",
    "KVHandoffResult",
    "KVHandoffStageListener",
    "KVScalingMeasurement",
    "KVSimulatedLinkCost",
    "compare_canonical_token_kv",
    "measure_kv_scaling",
    "perform_kv_handoff",
]
