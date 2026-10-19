"""Structured measurement of the real Token handoff data path."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
import json
from statistics import mean, median
from time import perf_counter

from .agent import LLMAgent
from .model import GenerationConfig
from .models import TokenPayload
from .network import SimulatedNetwork
from .token_handoff import EmptyTokenSequenceError, TokenHandoffError
from .token_protocol import (
    TokenWirePacket,
    deserialize_token_packet,
    measure_token_wire_size,
    serialize_token_ids,
)
from .transport import BaseByteTransport


TOKEN_HANDOFF_MEASUREMENT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class TokenCostMeasurementConfig:
    """Small repeated-measurement configuration."""

    warmup_runs: int = 1
    measurement_runs: int = 5
    receiver_max_new_tokens: int = 8

    def __post_init__(self) -> None:
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")
        if self.measurement_runs <= 0:
            raise ValueError("measurement_runs must be greater than zero")
        if self.receiver_max_new_tokens <= 0:
            raise ValueError("receiver_max_new_tokens must be greater than zero")


@dataclass(frozen=True)
class SimulatedLinkMeasurement:
    """Derived virtual-link cost for one network state; never wall-clock measured."""

    bandwidth_mbps: float
    base_latency_ms: float
    simulated_link_latency_ms: float
    token_handoff_cost_ms: float


@dataclass(frozen=True)
class TokenHandoffMeasurement:
    """JSON-safe Token cost evidence with measured and simulated stages separated."""

    schema_version: int
    config: TokenCostMeasurementConfig
    source_agent_id: str
    target_agent_id: str
    token_count: int
    theoretical_token_payload_bytes: int
    wire_header_bytes: int
    wire_body_bytes: int
    wire_total_bytes: int
    serialization_samples_ms: tuple[float, ...]
    serialization_median_ms: float
    local_byte_transfer_ms: float
    deserialization_samples_ms: tuple[float, ...]
    deserialization_median_ms: float
    first_warmup_prefill_ms: float | None
    receiver_prefill_samples_ms: tuple[float, ...]
    receiver_prefill_median_ms: float
    receiver_generated_token_count: int
    receiver_decode_forward_count: int
    receiver_decode_samples_ms: tuple[float, ...]
    receiver_decode_mean_ms: float | None
    receiver_decode_median_ms: float | None
    simulated_links: tuple[SimulatedLinkMeasurement, ...]
    sender_ids_match_received: bool
    received_ids_match_prefill_input: bool
    transport_name: str
    local_transfer_is_diagnostic_only: bool = True

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


Serializer = Callable[[Iterable[int]], bytes]
Deserializer = Callable[[bytes], TokenWirePacket]
Clock = Callable[[], float]


def measure_token_handoff_cost(
    *,
    source: LLMAgent,
    target: LLMAgent,
    token_ids: tuple[int, ...] | list[int],
    transport: BaseByteTransport,
    networks: Iterable[SimulatedNetwork],
    config: TokenCostMeasurementConfig | None = None,
    serializer: Serializer = serialize_token_ids,
    deserializer: Deserializer = deserialize_token_packet,
    _clock: Clock = perf_counter,
) -> TokenHandoffMeasurement:
    """Measure an already-generated Token representation through Receiver Prefill.

    The formal cost uses the simulated link and deliberately excludes both Sender
    generation and the diagnostic in-memory transfer wall time.
    """

    config = config or TokenCostMeasurementConfig()
    if source.agent_id == target.agent_id:
        raise TokenHandoffError("source and target must be different Agents")
    from .model.backend_pair import validate_backend_compatibility
    try:
        validate_backend_compatibility(source.backend, target.backend)
    except ValueError as error:
        raise TokenHandoffError(str(error)) from error
    sender_ids = tuple(token_ids)
    if not sender_ids:
        raise EmptyTokenSequenceError("sender generated no Token IDs")
    network_scenarios = tuple(networks)
    if not network_scenarios:
        raise ValueError("networks cannot be empty")

    serialization_samples: list[float] = []
    serialized_packets: list[bytes] = []
    for _ in range(config.measurement_runs):
        started = _clock()
        serialized = serializer(sender_ids)
        serialization_samples.append((_clock() - started) * 1_000.0)
        if not isinstance(serialized, bytes):
            raise TypeError("Token serializer must return bytes")
        serialized_packets.append(serialized)
    if any(packet != serialized_packets[0] for packet in serialized_packets[1:]):
        raise TokenHandoffError("Token serializer produced inconsistent packets")
    wire_bytes = serialized_packets[0]

    transfer_started = _clock()
    delivered = transport.transfer(
        wire_bytes, source=source.agent_id, target=target.agent_id
    )
    local_transfer_ms = (_clock() - transfer_started) * 1_000.0
    if not isinstance(delivered, bytes):
        raise TypeError("byte transport must deliver bytes")

    deserialization_samples: list[float] = []
    restored_packets: list[TokenWirePacket] = []
    for _ in range(config.measurement_runs):
        started = _clock()
        packet = deserializer(delivered)
        deserialization_samples.append((_clock() - started) * 1_000.0)
        if packet.token_ids != sender_ids:
            raise TokenHandoffError("transported Token IDs do not match sender output")
        restored_packets.append(packet)
    restored = restored_packets[0]

    receiver_config = GenerationConfig(
        max_new_tokens=config.receiver_max_new_tokens,
        do_sample=False,
        repetition_penalty=1.0,
    )
    first_warmup_prefill_ms: float | None = None
    for run_index in range(config.warmup_runs):
        warmup = target.generate_from_token_ids(restored.token_ids, receiver_config)
        _require_prefill_identity(restored.token_ids, warmup.input_token_ids)
        if run_index == 0:
            first_warmup_prefill_ms = warmup.prefill_forward_ms

    receiver_runs = tuple(
        target.generate_from_token_ids(restored.token_ids, receiver_config)
        for _ in range(config.measurement_runs)
    )
    for result in receiver_runs:
        _require_prefill_identity(restored.token_ids, result.input_token_ids)
    generated_counts = {len(result.output_token_ids) for result in receiver_runs}
    decode_counts = {len(result.decode_steps) for result in receiver_runs}
    if len(generated_counts) != 1 or len(decode_counts) != 1:
        raise TokenHandoffError("Receiver generation shape changed between measurements")

    logical_payload = TokenPayload.from_token_ids(
        sender_ids, vocab_size=source.backend.vocab_size
    )
    wire_size = measure_token_wire_size(logical_payload, restored)
    if len(delivered) != wire_size.total_wire_bytes:
        raise TokenHandoffError("transport changed Token packet byte count")

    serialization_median = median(serialization_samples)
    deserialization_median = median(deserialization_samples)
    prefill_samples = tuple(result.prefill_forward_ms for result in receiver_runs)
    prefill_median = median(prefill_samples)
    decode_samples = tuple(
        latency
        for result in receiver_runs
        for latency in result.decode_forward_latencies_ms
    )
    fixed_stage_cost_ms = (
        serialization_median + deserialization_median + prefill_median
    )
    links = tuple(
        _measure_simulated_link(network, len(delivered), fixed_stage_cost_ms)
        for network in network_scenarios
    )

    return TokenHandoffMeasurement(
        schema_version=TOKEN_HANDOFF_MEASUREMENT_SCHEMA_VERSION,
        config=config,
        source_agent_id=source.agent_id,
        target_agent_id=target.agent_id,
        token_count=wire_size.token_count,
        theoretical_token_payload_bytes=wire_size.theoretical_minimum_bytes,
        wire_header_bytes=wire_size.wire_header_bytes,
        wire_body_bytes=wire_size.wire_token_data_bytes,
        wire_total_bytes=wire_size.total_wire_bytes,
        serialization_samples_ms=tuple(serialization_samples),
        serialization_median_ms=serialization_median,
        local_byte_transfer_ms=local_transfer_ms,
        deserialization_samples_ms=tuple(deserialization_samples),
        deserialization_median_ms=deserialization_median,
        first_warmup_prefill_ms=first_warmup_prefill_ms,
        receiver_prefill_samples_ms=prefill_samples,
        receiver_prefill_median_ms=prefill_median,
        receiver_generated_token_count=next(iter(generated_counts)),
        receiver_decode_forward_count=next(iter(decode_counts)),
        receiver_decode_samples_ms=decode_samples,
        receiver_decode_mean_ms=mean(decode_samples) if decode_samples else None,
        receiver_decode_median_ms=median(decode_samples) if decode_samples else None,
        simulated_links=links,
        sender_ids_match_received=sender_ids == restored.token_ids,
        received_ids_match_prefill_input=all(
            restored.token_ids == result.input_token_ids for result in receiver_runs
        ),
        transport_name=type(transport).__name__,
    )


def _require_prefill_identity(
    received_ids: tuple[int, ...], prefill_input_ids: tuple[int, ...]
) -> None:
    if received_ids != prefill_input_ids:
        raise TokenHandoffError("Receiver Prefill Token IDs do not match received IDs")


def _measure_simulated_link(
    network: SimulatedNetwork, wire_bytes: int, fixed_stage_cost_ms: float
) -> SimulatedLinkMeasurement:
    link_ms = network.estimate_transmission_latency_ms(wire_bytes)
    return SimulatedLinkMeasurement(
        bandwidth_mbps=network.state.bandwidth_mbps,
        base_latency_ms=network.state.base_latency_ms,
        simulated_link_latency_ms=link_ms,
        token_handoff_cost_ms=fixed_stage_cost_ms + link_ms,
    )


def derive_single_run_token_handoff_cost(
    *,
    network: SimulatedNetwork,
    wire_bytes: int,
    serialization_ms: float,
    deserialization_ms: float,
    receiver_prefill_ms: float,
) -> SimulatedLinkMeasurement:
    """Apply the 4.3 formal cost boundary to one interactive handoff run."""
    if min(serialization_ms, deserialization_ms, receiver_prefill_ms) < 0:
        raise ValueError("measured Token handoff stages cannot be negative")
    fixed_stage_cost_ms = serialization_ms + deserialization_ms + receiver_prefill_ms
    return _measure_simulated_link(network, wire_bytes, fixed_stage_cost_ms)
