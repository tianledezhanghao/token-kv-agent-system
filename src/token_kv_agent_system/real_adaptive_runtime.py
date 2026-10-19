"""Predict before communication and execute exactly one real handoff path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any
from collections.abc import Callable

from .agent import LLMAgent
from .kv_handoff import CanonicalHandoffSpan, perform_kv_handoff
from .model import GenerationConfig
from .models import CommunicationMode
from .network import SimulatedNetwork
from .real_cost import (MESSAGE_READY, RealCommunicationCostModel,
                        RealCommunicationCostProfile, RealCostBasedSelector,
                        RealSelectionContext)
from .selection_stability import SelectionStabilityConfig, StableRealCostSelector
from .token_handoff import TokenHandoffSession
from .transport import InMemoryByteTransport


class RealAdaptiveRuntimeBusyError(RuntimeError):
    pass


@dataclass(frozen=True)
class AdaptiveHandoffResult:
    source: str
    target: str
    canonical_token_count: int
    cost_boundary: str
    bandwidth_mbps: float
    base_latency_ms: float
    token_estimated_ms: float
    kv_estimated_ms: float
    selected_mode: str
    estimated_saving_ms: float
    estimated_saving_ratio: float
    crossover_bandwidth_mbps: float | None
    actually_executed_mode: str
    predicted_selected_cost_ms: float
    actual_selected_cost_ms: float
    absolute_error_ms: float
    relative_error: float
    sender_generation_ms: float
    sender_text: str
    receiver_text: str
    correctness: bool
    selected_measurement: dict[str, float | int | bool]
    stability: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class RealAdaptiveRuntime:
    """The sole owner of direction and network mutation during a communication."""

    def __init__(self, agent_a: LLMAgent, agent_b: LLMAgent,
                 profile: RealCommunicationCostProfile, network: SimulatedNetwork,
                 *, initial_prompt: str = "Explain KV cache in one short sentence.",
                 max_new_tokens: int = 8,
                 receiver_suffix_token_ids: tuple[int, ...] = (13,),
                 selector: Any | None = None,
                 stability_config: SelectionStabilityConfig = SelectionStabilityConfig(),
                 token_handoff: Any | None = None,
                 kv_handoff: Any | None = None,
                 transport: InMemoryByteTransport | None = None) -> None:
        if agent_a.backend is not agent_b.backend:
            raise ValueError("adaptive Agents must share one backend")
        if agent_a.agent_id == agent_b.agent_id:
            raise ValueError("adaptive Agents need distinct identities")
        if max_new_tokens <= 0 or not receiver_suffix_token_ids:
            raise ValueError("invalid continuation configuration")
        self.agents = (agent_a, agent_b)
        self.backend = agent_a.backend
        self.profile = profile
        self._validate_profile()
        self.network = network
        self.transport = transport or InMemoryByteTransport()
        self.cost_model = RealCommunicationCostModel(profile)
        self.selector = selector or (StableRealCostSelector(RealCostBasedSelector(self.cost_model), stability_config)
                                     if stability_config.enabled else RealCostBasedSelector(self.cost_model))
        self.token_handoff = token_handoff or TokenHandoffSession(self.agents, self.transport)
        self._default_kv_handoff = kv_handoff is None
        self.kv_handoff = kv_handoff or self._perform_kv
        self.initial_prompt = initial_prompt
        self.max_new_tokens = max_new_tokens
        self.receiver_suffix_token_ids = tuple(receiver_suffix_token_ids)
        self._operation_lock = Lock()
        self._next_source_index = 0
        self._continuation_ids: tuple[int, ...] | None = None
        self._continuation_text = ""
        self.running = False

    @classmethod
    def from_profile_path(cls, agent_a: LLMAgent, agent_b: LLMAgent,
                          profile_path: str | Path, network: SimulatedNetwork,
                          **kwargs: Any) -> RealAdaptiveRuntime:
        profile = RealCommunicationCostProfile.from_json(
            Path(profile_path).read_text(encoding="utf-8"))
        return cls(agent_a, agent_b, profile, network, **kwargs)

    def _validate_profile(self) -> None:
        config = self.backend.kv_model_config
        metadata = self.backend.metadata
        expected = {
            "model_id": metadata.model_name,
            "num_hidden_layers": config.num_hidden_layers,
            "num_key_value_heads": config.num_key_value_heads,
            "head_dim": config.head_dim,
            "kv_dtype": metadata.dtype.removeprefix("torch.") if metadata.dtype else None,
            "batch_size": 1,
            "device_type": metadata.device.split(":")[0],
            "source_device": (f"{metadata.device}:0" if metadata.device in {"mps", "cuda"}
                              else metadata.device),
        }
        for field, value in expected.items():
            if getattr(self.profile, field) != value:
                raise ValueError(f"calibration profile incompatible with runtime {field}: "
                                 f"profile={getattr(self.profile, field)!r}, runtime={value!r}")

    @property
    def next_direction(self) -> tuple[str, str]:
        index = self._next_source_index
        return self.agents[index].agent_id, self.agents[1 - index].agent_id

    def set_bandwidth_mbps(self, bandwidth_mbps: float) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise RealAdaptiveRuntimeBusyError("communication is running")
        try:
            self.network.set_bandwidth_mbps(bandwidth_mbps)
        finally:
            self._operation_lock.release()

    def _perform_kv(self, source: LLMAgent, target: LLMAgent,
                    ids: tuple[int, ...], network: SimulatedNetwork,
                    stage_listener: Callable[[str, dict[str, Any]], None] | None = None) -> Any:
        return perform_kv_handoff(
            source=source, target=target, span=CanonicalHandoffSpan(ids),
            receiver_suffix_token_ids=self.receiver_suffix_token_ids,
            transport=self.transport, networks=(network,),
            receiver_max_new_tokens=self.max_new_tokens, stage_listener=stage_listener)

    def communicate_next(self, *, stage_listener: Callable[[str, dict[str, Any]], None] | None = None) -> AdaptiveHandoffResult:
        if not self._operation_lock.acquire(blocking=False):
            raise RealAdaptiveRuntimeBusyError("communication is running")
        self.running = True
        try:
            index = self._next_source_index
            source, target = self.agents[index], self.agents[1 - index]
            if self._continuation_ids is None:
                generation = source.generate_reply(
                    self.initial_prompt,
                    GenerationConfig(max_new_tokens=self.max_new_tokens,
                                     do_sample=False, repetition_penalty=1.0))
                ids = tuple(generation.output_token_ids)
                sender_text = generation.generated_text
                generation_ms = generation.generation_wall_time_seconds * 1000
            else:
                ids = self._continuation_ids
                sender_text = self._continuation_text
                generation_ms = 0.0  # The previous receiver already produced these IDs.
            span = CanonicalHandoffSpan(ids)
            if stage_listener:
                stage_listener("sender_ready", {"source": source.agent_id, "target": target.agent_id,
                                                "text": sender_text, "token_count": span.token_count,
                                                "generation_ms": generation_ms,
                                                "reused_continuation": generation_ms == 0.0})
            state = self.network.state
            context = RealSelectionContext.from_profile(
                self.profile, span.token_count, state.bandwidth_mbps, state.base_latency_ms)
            decision = self.selector.select(context, boundary=MESSAGE_READY)
            if decision.cost_boundary != MESSAGE_READY or decision.selected_mode not in (CommunicationMode.TOKEN, CommunicationMode.KV):
                raise ValueError("selector returned an unsupported execution decision")
            if decision.token_estimate.token_count != span.token_count or decision.kv_estimate.token_count != span.token_count:
                raise ValueError("selector estimates do not match canonical message")
            if self.network.state != state:
                raise RuntimeError("network changed during selection")
            if stage_listener:
                stage_listener("selected", {
                    "source": source.agent_id, "target": target.agent_id,
                    "token_count": span.token_count, "bandwidth_mbps": state.bandwidth_mbps,
                    "base_latency_ms": state.base_latency_ms, "cost_boundary": MESSAGE_READY,
                    "selector_kind": "real_cost_based", "selected_mode": decision.selected_mode.value,
                    "estimated_saving_ms": decision.estimated_saving_ms,
                    "estimated_saving_ratio": decision.estimated_saving_ratio,
                    "token_estimated_ms": decision.token_estimate.total_ms,
                    "kv_estimated_ms": decision.kv_estimate.message_ready_total_ms,
                    "token_estimate": asdict(decision.token_estimate),
                    "kv_estimate": asdict(decision.kv_estimate),
                    "stability": decision.stability_metadata() if hasattr(decision, "stability_metadata") else {},
                })
                stage_listener("path_started", {"source": source.agent_id, "target": target.agent_id,
                                                "selected_mode": decision.selected_mode.value})
            def path_stage(stage: str, metadata: Any) -> None:
                if stage_listener:
                    stage_listener("path_stage", {"source": source.agent_id, "target": target.agent_id,
                                                  "selected_mode": decision.selected_mode.value,
                                                  "stage": stage, "measurement": dict(metadata)})
            if decision.selected_mode == CommunicationMode.TOKEN:
                handoff = self.token_handoff.handoff_token_ids(
                    source.agent_id, target.agent_id, span.handoff_token_ids,
                    sender_generated_text=sender_text,
                    receiver_config=GenerationConfig(max_new_tokens=self.max_new_tokens,
                                                     do_sample=False, repetition_penalty=1.0),
                    stage_listener=path_stage if stage_listener else None)
                correct = handoff.token_identity_preserved and bool(handoff.receiver_output_token_ids)
                wire = handoff.wire_byte_count
                link = self.network.estimate_transmission_latency_ms(wire)
                actual = (handoff.serialization_ms + link + handoff.deserialization_ms
                          + handoff.receiver_prefill_forward_ms)
                predicted = decision.token_estimate.total_ms
                measurement: dict[str, float | int | bool] = {
                    "wire_bytes": wire, "serialize_ms": handoff.serialization_ms,
                    "link_ms": link, "deserialize_ms": handoff.deserialization_ms,
                    "receiver_prefill_ms": handoff.receiver_prefill_forward_ms}
            else:
                if self._default_kv_handoff and stage_listener:
                    handoff = self.kv_handoff(source, target, span.handoff_token_ids,
                                              self.network, stage_listener=path_stage)
                else:
                    handoff = self.kv_handoff(source, target, span.handoff_token_ids, self.network)
                correct = (handoff.handoff_token_ids == span.handoff_token_ids
                           and handoff.compatibility_verified and handoff.wire_identity_verified
                           and handoff.canonical_cache_complete
                           and not handoff.receiver_prefix_prefill_executed
                           and bool(handoff.receiver_output_token_ids))
                link = self.network.estimate_transmission_latency_ms(handoff.wire_total_bytes)
                actual = (handoff.sender_kv_prepare_ms + handoff.total_serialize_ms
                          + link + handoff.deserialize_ms + handoff.total_restore_ms)
                predicted = decision.kv_estimate.message_ready_total_ms
                measurement = {
                    "raw_kv_bytes": handoff.raw_kv_tensor_bytes,
                    "wire_tensor_body_bytes": getattr(handoff, "wire_tensor_body_bytes", 0),
                    "wire_metadata_bytes": getattr(handoff, "wire_metadata_bytes", 0),
                    "wire_bytes": handoff.wire_total_bytes,
                    "sender_prepare_ms": handoff.sender_kv_prepare_ms,
                    "device_to_host_ms": getattr(handoff, "device_to_host_ms", 0.0),
                    "encode_ms": getattr(handoff, "protocol_encode_ms", 0.0),
                    "serialize_ms": handoff.total_serialize_ms, "link_ms": link,
                    "deserialize_ms": handoff.deserialize_ms,
                    "cpu_reconstruction_ms": getattr(handoff, "cpu_tensor_reconstruction_ms", 0.0),
                    "host_to_device_ms": getattr(handoff, "host_to_device_ms", 0.0),
                    "cache_construction_ms": getattr(handoff, "cache_construction_ms", 0.0),
                    "receiver_prefill_skipped": not handoff.receiver_prefix_prefill_executed,
                    "restore_ms": handoff.total_restore_ms}
            if self.network.state != state:
                raise RuntimeError("network changed during execution")
            if not correct:
                raise RuntimeError("selected handoff failed correctness checks")
            error = abs(actual - predicted)
            result = AdaptiveHandoffResult(
                source.agent_id, target.agent_id, span.token_count, MESSAGE_READY,
                state.bandwidth_mbps, state.base_latency_ms,
                decision.token_estimate.total_ms, decision.kv_estimate.message_ready_total_ms,
                decision.selected_mode.value, decision.estimated_saving_ms,
                decision.estimated_saving_ratio,
                self.cost_model.crossover_bandwidth_mbps(context),
                decision.selected_mode.value, predicted, actual, error,
                error / predicted if predicted else 0.0, generation_ms,
                sender_text, handoff.receiver_generated_text, True, measurement,
                decision.stability_metadata() if hasattr(decision, "stability_metadata") else {})
            if hasattr(self.selector, "record_success"):
                self.selector.record_success(decision.selected_mode)
            self._continuation_ids = tuple(handoff.receiver_output_token_ids)
            self._continuation_text = handoff.receiver_generated_text
            self._next_source_index = 1 - index
            if stage_listener:
                stage_listener("finished", result.to_dict())
            return result
        finally:
            self.running = False
            self._operation_lock.release()
