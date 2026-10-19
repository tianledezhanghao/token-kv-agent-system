"""In-memory runtimes for user-driven Mock, real Token, and real KV experiments."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from threading import Lock, RLock
from typing import Any

from token_kv_agent_system.agent import Agent, LLMAgent, MockAgent
from token_kv_agent_system.kv_handoff import (
    BidirectionalKVHandoffSession,
    KVHandoffResult,
    KVSimulatedLinkCost,
)
from token_kv_agent_system.metrics import CommunicationEvent, EventType, MetricCollector
from token_kv_agent_system.model import (
    DEFAULT_MODEL_NAME,
    BaseModelBackend,
    GenerationConfig,
    HuggingFaceModelBackend,
    IncrementalCausalLMBackend,
)
from token_kv_agent_system.models import CommunicationMode
from token_kv_agent_system.real_adaptive_runtime import (
    AdaptiveHandoffResult, RealAdaptiveRuntime, RealAdaptiveRuntimeBusyError,
)
from token_kv_agent_system.network import SimulatedNetwork
from token_kv_agent_system.orchestrator import CommunicationOrchestrator, CommunicationResult
from token_kv_agent_system.selector import LatencyBasedSelector
from token_kv_agent_system.token_cost import derive_single_run_token_handoff_cost
from token_kv_agent_system.token_handoff import TokenHandoffResult, TokenHandoffSession
from token_kv_agent_system.transport import (
    InMemoryByteTransport,
    KVTransport,
    TokenTransport,
)


class RealAdaptiveDashboardRuntime:
    """Thin Control Plane adapter around the sole adaptive Data Plane runtime."""

    def __init__(self, runtime: RealAdaptiveRuntime, collector: MetricCollector) -> None:
        self.runtime = runtime
        self.collector = collector
        self.network = runtime.network
        self.agents = runtime.agents
        self._last_error: str | None = None
        self._active_mode: str | None = None

    @classmethod
    def create_default(cls, *, selector_profile: str, model_name: str = DEFAULT_MODEL_NAME,
                       device: str = "auto", max_new_tokens: int = 8,
                       initial_prompt: str = "Explain KV cache briefly.") -> "RealAdaptiveDashboardRuntime":
        try:
            backend = HuggingFaceModelBackend(model_name, device=device)
        except RuntimeError as error:
            if "Local LLM support is not installed" in str(error):
                raise RuntimeError("Real Adaptive runtime requires the 'llm' optional dependencies. Install with pip install -e '.[llm]'") from error
            raise
        collector = MetricCollector()
        network = SimulatedNetwork(20, 5, sleep_enabled=False, collector=collector)
        agents = (LLMAgent("Agent A", backend), LLMAgent("Agent B", backend))
        runtime = RealAdaptiveRuntime.from_profile_path(
            agents[0], agents[1], selector_profile, network,
            max_new_tokens=max_new_tokens, initial_prompt=initial_prompt)
        return cls(runtime, collector)

    def snapshot(self) -> dict[str, Any]:
        profile = self.runtime.profile
        source, target = self.runtime.next_direction
        return {
            "mode": "real_adaptive", "status": "running" if self.runtime.running else ("error" if self._last_error else "ready"),
            "running": self.runtime.running, "next_source": source, "next_target": target,
            "active_communication_mode": self._active_mode,
            "selector_status": "active", "selector_kind": "real_cost_based",
            "cost_boundary": "MESSAGE_READY",
            "selector_profile": {"profile_name": profile.profile_name,
                                 "device_type": profile.device_type, "model_id": profile.model_id,
                                 "cost_boundary": "MESSAGE_READY",
                                 "calibration_token_counts": [point.token_count for point in profile.points],
                                 "development_calibration": "development" in profile.profile_name.lower()},
            "model": profile.model_id, "device": self.runtime.backend.metadata.device,
            "max_new_tokens": self.runtime.max_new_tokens, "last_error": self._last_error,
        }

    def set_bandwidth_mbps(self, bandwidth_mbps: float) -> None:
        try:
            self.runtime.set_bandwidth_mbps(bandwidth_mbps)
        except RealAdaptiveRuntimeBusyError as error:
            raise InteractiveRuntimeBusyError(str(error)) from error

    def _stage(self, stage: str, data: dict[str, Any]) -> None:
        source, target = data["source"], data["target"]
        event_types = {
            "sender_ready": EventType.AGENT_GENERATION_FINISHED,
            "selected": EventType.REAL_SELECTION_MADE,
            "path_started": EventType.SELECTED_PATH_STARTED,
            "finished": EventType.REAL_ADAPTIVE_HANDOFF_FINISHED,
        }
        if stage == "path_stage":
            path_events = {
                "TOKEN": {"serialized": EventType.TOKEN_SERIALIZED,
                          "transferred": EventType.TOKEN_TRANSFERRED,
                          "deserialized": EventType.TOKEN_DESERIALIZED,
                          "receiver_prefill_finished": EventType.RECEIVER_PREFILL_FINISHED},
                "KV": {"prepared": EventType.KV_PREPARED,
                       "serialized": EventType.KV_SERIALIZED,
                       "transferred": EventType.KV_TRANSFERRED,
                       "deserialized": EventType.KV_DESERIALIZED,
                       "restored": EventType.KV_RESTORED},
            }
            self.collector.record(CommunicationEvent(
                event_type=path_events[data["selected_mode"]][data["stage"]],
                source_agent_id=source, target_agent_id=target,
                mode=CommunicationMode(data["selected_mode"]), metadata=data["measurement"]))
            return
        if stage == "selected":
            self._active_mode = data["selected_mode"]
        metadata = dict(data)
        if stage == "sender_ready":
            metadata = {"text": data["text"], "token_count": data["token_count"],
                        "generation_wall_time_ms": data["generation_ms"],
                        "reused_continuation": data["reused_continuation"], "real_adaptive": True}
        self.collector.record(CommunicationEvent(
            event_type=event_types[stage], source_agent_id=source, target_agent_id=target,
            mode=CommunicationMode(data["selected_mode"]) if "selected_mode" in data else None,
            payload_bytes=data.get("selected_measurement", {}).get("wire_bytes") if stage == "finished" else None,
            metadata=metadata))

    def communicate_next(self) -> tuple[str, str, AdaptiveHandoffResult]:
        source, target = self.runtime.next_direction
        previous_mode = self._active_mode
        self._last_error = None
        self.collector.record(CommunicationEvent(EventType.REAL_ADAPTIVE_HANDOFF_STARTED,
                                                  source_agent_id=source, target_agent_id=target))
        try:
            result = self.runtime.communicate_next(stage_listener=self._stage)
            return result.source, result.target, result
        except RealAdaptiveRuntimeBusyError as error:
            raise InteractiveRuntimeBusyError(str(error)) from error
        except Exception as error:
            self._last_error = str(error)
            self._active_mode = previous_mode
            self.collector.record(CommunicationEvent(EventType.REAL_ADAPTIVE_HANDOFF_FAILED,
                                                      source_agent_id=source, target_agent_id=target,
                                                      metadata={"error": str(error),
                                                                "active_communication_mode": previous_mode}))
            raise


class InteractiveRuntimeBusyError(RuntimeError):
    """Raised when a control operation would overlap a communication."""


class InteractiveRuntime:
    """Own the real core objects used by one interactive Dashboard session."""

    def __init__(
        self,
        *,
        agents: tuple[Agent, Agent],
        network: SimulatedNetwork,
        orchestrator: CommunicationOrchestrator,
        collector: MetricCollector,
    ) -> None:
        if agents[0].agent_id == agents[1].agent_id:
            raise ValueError("interactive agents must have distinct ids")
        self.agents = agents
        self.network = network
        self.orchestrator = orchestrator
        self.collector = collector
        self._operation_lock = Lock()
        self._state_lock = RLock()
        self._next_source_index = 0
        self._running = False

    @classmethod
    def create_default(
        cls,
        *,
        bandwidth_mbps: float = 20,
        base_latency_ms: float = 5,
    ) -> InteractiveRuntime:
        collector = MetricCollector()
        network = SimulatedNetwork(
            bandwidth_mbps=bandwidth_mbps,
            base_latency_ms=base_latency_ms,
            sleep_enabled=False,
            collector=collector,
        )
        transports = {
            CommunicationMode.TOKEN: TokenTransport(network, collector),
            CommunicationMode.KV: KVTransport(network, collector),
        }
        orchestrator = CommunicationOrchestrator(
            network=network,
            selector=LatencyBasedSelector(),
            transports=transports,
            collector=collector,
        )
        agents = (
            MockAgent("Agent A", collector=collector),
            MockAgent("Agent B", collector=collector),
        )
        return cls(
            agents=agents,
            network=network,
            orchestrator=orchestrator,
            collector=collector,
        )

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            source_index = self._next_source_index
            return {
                "mode": "interactive",
                "running": self._running,
                "next_source": self.agents[source_index].agent_id,
                "next_target": self.agents[1 - source_index].agent_id,
            }

    def set_bandwidth_mbps(self, bandwidth_mbps: float) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise InteractiveRuntimeBusyError("communication is already running")
        try:
            self.network.set_bandwidth_mbps(bandwidth_mbps)
        finally:
            self._operation_lock.release()

    def communicate_next(self) -> tuple[str, str, CommunicationResult]:
        if not self._operation_lock.acquire(blocking=False):
            raise InteractiveRuntimeBusyError("communication is already running")
        with self._state_lock:
            self._running = True
            source_index = self._next_source_index
            source = self.agents[source_index]
            target = self.agents[1 - source_index]
        try:
            prompt = (
                "提出一个确定性方案"
                if source_index == 0
                else "审查已提出的方案"
            )
            result = self.orchestrator.communicate(source, target, prompt)
            with self._state_lock:
                self._next_source_index = 1 - source_index
            return source.agent_id, target.agent_id, result
        finally:
            with self._state_lock:
                self._running = False
            self._operation_lock.release()


@dataclass(frozen=True)
class RealTokenCommunicationResult:
    """One user-triggered, single-run real Token handoff observation."""

    handoff: TokenHandoffResult
    simulated_link_latency_ms: float
    token_handoff_cost_ms: float
    receiver_decode_mean_ms: float | None
    receiver_decode_median_ms: float | None


class RealTokenRuntime:
    """Orchestrate a fixed TOKEN direct-causal-continuation experiment."""

    def __init__(
        self,
        *,
        agents: tuple[LLMAgent, LLMAgent],
        backend: BaseModelBackend,
        session: TokenHandoffSession,
        transport: InMemoryByteTransport,
        network: SimulatedNetwork,
        collector: MetricCollector,
        max_new_tokens: int = 8,
        initial_prompt: str = "Explain KV cache briefly.",
        warmup_performed: bool = False,
    ) -> None:
        if agents[0].agent_id == agents[1].agent_id:
            raise ValueError("real Token agents must have distinct ids")
        if agents[0].backend is not backend or agents[1].backend is not backend:
            raise ValueError("real Token agents must share the runtime backend")
        if not isinstance(backend, IncrementalCausalLMBackend):
            raise TypeError("Real Token runtime requires an incremental model backend")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if not initial_prompt.strip():
            raise ValueError("initial_prompt cannot be empty")
        self.agents = agents
        self.backend = backend
        self.session = session
        self.transport = transport
        self.network = network
        self.collector = collector
        self.max_new_tokens = max_new_tokens
        self.initial_prompt = initial_prompt
        self.warmup_performed = warmup_performed
        self._operation_lock = Lock()
        self._state_lock = RLock()
        self._next_source_index = 0
        self._running = False
        self._last_error: str | None = None
        self._continuation_ids: tuple[int, ...] | None = None
        self._continuation_text: str | None = None

    @classmethod
    def create_default(
        cls,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        max_new_tokens: int = 8,
        initial_prompt: str = "Explain KV cache briefly.",
        bandwidth_mbps: float = 20,
        base_latency_ms: float = 5,
        warmup: bool = True,
    ) -> "RealTokenRuntime":
        try:
            backend = HuggingFaceModelBackend(model_name, device=device)
        except RuntimeError as error:
            if "Local LLM support is not installed" in str(error):
                raise RuntimeError(
                    "Real Token runtime requires the 'llm' optional dependencies. "
                    "Install with pip install -e '.[llm]'"
                ) from error
            raise
        return cls.create_with_backend(
            backend=backend,
            max_new_tokens=max_new_tokens,
            initial_prompt=initial_prompt,
            bandwidth_mbps=bandwidth_mbps,
            base_latency_ms=base_latency_ms,
            warmup=warmup,
        )

    @classmethod
    def create_with_backend(
        cls,
        *,
        backend: BaseModelBackend,
        max_new_tokens: int = 8,
        initial_prompt: str = "Explain KV cache briefly.",
        bandwidth_mbps: float = 20,
        base_latency_ms: float = 5,
        warmup: bool = False,
    ) -> "RealTokenRuntime":
        collector = MetricCollector()
        network = SimulatedNetwork(
            bandwidth_mbps=bandwidth_mbps,
            base_latency_ms=base_latency_ms,
            sleep_enabled=False,
            collector=collector,
        )
        agents = (
            LLMAgent("Agent A", backend),
            LLMAgent("Agent B", backend),
        )
        transport = InMemoryByteTransport()
        session = TokenHandoffSession(agents, transport)
        runtime = cls(
            agents=agents,
            backend=backend,
            session=session,
            transport=transport,
            network=network,
            collector=collector,
            max_new_tokens=max_new_tokens,
            initial_prompt=initial_prompt,
        )
        if warmup:
            runtime.warmup()
        return runtime

    def warmup(self) -> None:
        """Warm shared generation/Prefill paths without events or history."""
        config = self._generation_config()
        generation = self.agents[0].generate_reply(self.initial_prompt, config)
        if not generation.output_token_ids:
            raise RuntimeError("Real Token warmup generated no Token IDs")
        self.agents[1].generate_from_token_ids(generation.output_token_ids, config)
        self.warmup_performed = True

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            source_index = self._next_source_index
            return {
                "mode": "real_token",
                "status": (
                    "running"
                    if self._running
                    else ("error" if self._last_error else "ready")
                ),
                "running": self._running,
                "next_source": self.agents[source_index].agent_id,
                "next_target": self.agents[1 - source_index].agent_id,
                "active_communication_mode": CommunicationMode.TOKEN.value,
                "selector_status": "not_applicable",
                "model": self.backend.model_name,
                "device": self.backend.device,
                "max_new_tokens": self.max_new_tokens,
                "warmup_performed": self.warmup_performed,
                "last_error": self._last_error,
            }

    def set_bandwidth_mbps(self, bandwidth_mbps: float) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise InteractiveRuntimeBusyError("communication is already running")
        try:
            self.network.set_bandwidth_mbps(bandwidth_mbps)
        finally:
            self._operation_lock.release()

    def communicate_next(self) -> tuple[str, str, RealTokenCommunicationResult]:
        if not self._operation_lock.acquire(blocking=False):
            raise InteractiveRuntimeBusyError("communication is already running")
        with self._state_lock:
            self._running = True
            self._last_error = None
            source_index = self._next_source_index
            source = self.agents[source_index]
            target = self.agents[1 - source_index]
        self._record(
            EventType.REAL_TOKEN_HANDOFF_STARTED,
            source=source.agent_id,
            target=target.agent_id,
            metadata={
                "communication_mode": CommunicationMode.TOKEN.value,
                "measurement_strategy": "single_run",
                "direct_causal_continuation": True,
            },
        )
        try:
            sender_ids, sender_text = self._sender_representation(source)
            handoff = self.session.handoff_token_ids(
                source.agent_id,
                target.agent_id,
                sender_ids,
                sender_generated_text=sender_text,
                receiver_config=self._generation_config(),
                stage_listener=lambda stage, metadata: self._record_stage(
                    stage, source.agent_id, target.agent_id, metadata
                ),
            )
            if not handoff.token_identity_preserved:
                raise RuntimeError("Token identity verification failed")
            cost = derive_single_run_token_handoff_cost(
                network=self.network,
                wire_bytes=handoff.wire_byte_count,
                serialization_ms=handoff.serialization_ms,
                deserialization_ms=handoff.deserialization_ms,
                receiver_prefill_ms=handoff.receiver_prefill_forward_ms,
            )
            decode = handoff.receiver_decode_forward_latencies_ms
            result = RealTokenCommunicationResult(
                handoff=handoff,
                simulated_link_latency_ms=cost.simulated_link_latency_ms,
                token_handoff_cost_ms=cost.token_handoff_cost_ms,
                receiver_decode_mean_ms=mean(decode) if decode else None,
                receiver_decode_median_ms=median(decode) if decode else None,
            )
            self._record_finished(result)
            with self._state_lock:
                self._continuation_ids = handoff.receiver_output_token_ids
                self._continuation_text = handoff.receiver_generated_text
                self._next_source_index = 1 - source_index
            return source.agent_id, target.agent_id, result
        except Exception as error:
            with self._state_lock:
                self._last_error = str(error)
            self._record(
                EventType.REAL_TOKEN_HANDOFF_FAILED,
                source=source.agent_id,
                target=target.agent_id,
                metadata={"error": str(error)},
            )
            raise
        finally:
            with self._state_lock:
                self._running = False
            self._operation_lock.release()

    def _sender_representation(self, source: LLMAgent) -> tuple[tuple[int, ...], str]:
        if self._continuation_ids is not None:
            return self._continuation_ids, self._continuation_text or ""
        self._record(
            EventType.AGENT_GENERATION_STARTED,
            source=source.agent_id,
            metadata={"prompt": self.initial_prompt, "real_token": True},
        )
        generation = source.generate_reply(self.initial_prompt, self._generation_config())
        self._record(
            EventType.AGENT_GENERATION_FINISHED,
            source=source.agent_id,
            metadata={
                "text": generation.generated_text,
                "token_count": len(generation.output_token_ids),
                "generation_wall_time_ms": generation.generation_wall_time_seconds * 1_000,
                "real_token": True,
            },
        )
        return generation.output_token_ids, generation.generated_text

    def _generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )

    def _record_stage(
        self,
        stage: str,
        source: str,
        target: str,
        metadata: Any,
    ) -> None:
        event_types = {
            "serialized": EventType.TOKEN_SERIALIZED,
            "transferred": EventType.TOKEN_TRANSFERRED,
            "deserialized": EventType.TOKEN_DESERIALIZED,
            "receiver_prefill_finished": EventType.RECEIVER_PREFILL_FINISHED,
        }
        self._record(event_types[stage], source=source, target=target, metadata=metadata)

    def _record_finished(self, result: RealTokenCommunicationResult) -> None:
        handoff = result.handoff
        self._record(
            EventType.REAL_TOKEN_HANDOFF_FINISHED,
            source=handoff.source_agent_id,
            target=handoff.target_agent_id,
            payload_bytes=handoff.wire_byte_count,
            transmission_latency_ms=result.simulated_link_latency_ms,
            estimated_e2e_latency_ms=result.token_handoff_cost_ms,
            metadata={
                "communication_mode": CommunicationMode.TOKEN.value,
                "measurement_strategy": "single_run",
                "token_count": handoff.token_count,
                "theoretical_token_bytes": handoff.theoretical_token_logical_bytes,
                "wire_bytes": handoff.wire_byte_count,
                "serialization_ms": handoff.serialization_ms,
                "local_byte_transfer_ms": handoff.local_byte_transfer_ms,
                "deserialization_ms": handoff.deserialization_ms,
                "receiver_prefill_ms": handoff.receiver_prefill_forward_ms,
                "simulated_link_ms": result.simulated_link_latency_ms,
                "token_handoff_cost_ms": result.token_handoff_cost_ms,
                "receiver_decode_mean_ms": result.receiver_decode_mean_ms,
                "receiver_decode_median_ms": result.receiver_decode_median_ms,
                "sender_generated_text": handoff.sender_generated_text,
                "receiver_generated_text": handoff.receiver_generated_text,
                "token_identity_verified": handoff.token_identity_preserved,
                "protocol_version": handoff.protocol_version,
                "transport": handoff.transport_name,
            },
        )

    def _record(
        self,
        event_type: EventType,
        *,
        source: str | None = None,
        target: str | None = None,
        payload_bytes: int | None = None,
        transmission_latency_ms: float | None = None,
        estimated_e2e_latency_ms: float | None = None,
        metadata: Any = None,
    ) -> None:
        self.collector.record(
            CommunicationEvent(
                event_type=event_type,
                source_agent_id=source,
                target_agent_id=target,
                mode=CommunicationMode.TOKEN,
                payload_bytes=payload_bytes,
                transmission_latency_ms=transmission_latency_ms,
                estimated_e2e_latency_ms=estimated_e2e_latency_ms,
                metadata=metadata or {},
            )
        )


@dataclass(frozen=True)
class RealKVCommunicationResult:
    """One user-triggered, single-run canonical KV handoff observation."""

    handoff: KVHandoffResult
    link: KVSimulatedLinkCost
    sender_generated_text: str
    continuation_verified: bool


class RealKVRuntime:
    """Orchestrate a fixed canonical KV direct-causal-continuation experiment."""

    def __init__(
        self,
        *,
        agents: tuple[LLMAgent, LLMAgent],
        backend: BaseModelBackend,
        session: BidirectionalKVHandoffSession,
        transport: InMemoryByteTransport,
        network: SimulatedNetwork,
        collector: MetricCollector,
        receiver_suffix_token_ids: tuple[int, ...],
        max_new_tokens: int = 8,
        initial_prompt: str = "Explain KV cache briefly.",
        warmup_performed: bool = False,
    ) -> None:
        if agents[0].agent_id == agents[1].agent_id:
            raise ValueError("real KV agents must have distinct ids")
        if agents[0].backend is not backend or agents[1].backend is not backend:
            raise ValueError("real KV agents must share the runtime backend")
        if not isinstance(backend, IncrementalCausalLMBackend):
            raise TypeError("Real KV runtime requires an incremental model backend")
        if not all(
            hasattr(backend, name)
            for name in ("restore_kv_cache", "continue_from_cache", "kv_model_config")
        ):
            raise TypeError("Real KV runtime requires KV restore capabilities")
        if not receiver_suffix_token_ids:
            raise ValueError("receiver_suffix_token_ids cannot be empty")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if not initial_prompt.strip():
            raise ValueError("initial_prompt cannot be empty")
        self.agents = agents
        self.backend = backend
        self.session = session
        self.transport = transport
        self.network = network
        self.collector = collector
        self.receiver_suffix_token_ids = receiver_suffix_token_ids
        self.max_new_tokens = max_new_tokens
        self.initial_prompt = initial_prompt
        self.warmup_performed = warmup_performed
        self._operation_lock = Lock()
        self._state_lock = RLock()
        self._next_source_index = 0
        self._running = False
        self._last_error: str | None = None
        self._continuation_ids: tuple[int, ...] | None = None
        self._continuation_text: str | None = None

    @classmethod
    def create_default(
        cls,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str = "auto",
        max_new_tokens: int = 8,
        initial_prompt: str = "Explain KV cache briefly.",
        receiver_suffix: str = " Continue briefly.",
        bandwidth_mbps: float = 20,
        base_latency_ms: float = 5,
        warmup: bool = True,
    ) -> "RealKVRuntime":
        try:
            backend = HuggingFaceModelBackend(model_name, device=device)
        except RuntimeError as error:
            if "Local LLM support is not installed" in str(error):
                raise RuntimeError(
                    "Real KV runtime requires the 'llm' optional dependencies. "
                    "Install with pip install -e '.[llm]'"
                ) from error
            raise
        return cls.create_with_backend(
            backend=backend,
            max_new_tokens=max_new_tokens,
            initial_prompt=initial_prompt,
            receiver_suffix=receiver_suffix,
            bandwidth_mbps=bandwidth_mbps,
            base_latency_ms=base_latency_ms,
            warmup=warmup,
        )

    @classmethod
    def create_with_backend(
        cls,
        *,
        backend: BaseModelBackend,
        max_new_tokens: int = 8,
        initial_prompt: str = "Explain KV cache briefly.",
        receiver_suffix: str = " Continue briefly.",
        bandwidth_mbps: float = 20,
        base_latency_ms: float = 5,
        warmup: bool = False,
    ) -> "RealKVRuntime":
        suffix_ids = tuple(backend.encode(receiver_suffix))
        if not suffix_ids:
            raise RuntimeError("Real KV Receiver suffix encoded to no Token IDs")
        collector = MetricCollector()
        network = SimulatedNetwork(
            bandwidth_mbps=bandwidth_mbps,
            base_latency_ms=base_latency_ms,
            sleep_enabled=False,
            collector=collector,
        )
        agents = (LLMAgent("Agent A", backend), LLMAgent("Agent B", backend))
        transport = InMemoryByteTransport()
        session = BidirectionalKVHandoffSession(
            agents[0], agents[1], transport, (network,)
        )
        runtime = cls(
            agents=agents,
            backend=backend,
            session=session,
            transport=transport,
            network=network,
            collector=collector,
            receiver_suffix_token_ids=suffix_ids,
            max_new_tokens=max_new_tokens,
            initial_prompt=initial_prompt,
        )
        if warmup:
            runtime.warmup()
        return runtime

    def warmup(self) -> None:
        """Warm shared generation and clean Prefill without events or history."""
        generation = self.agents[0].generate_reply(
            self.initial_prompt, self._generation_config()
        )
        if not generation.output_token_ids:
            raise RuntimeError("Real KV warmup generated no Token IDs")
        self.backend.prefill(input_token_ids=generation.output_token_ids)
        self.warmup_performed = True

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            source_index = self._next_source_index
            return {
                "mode": "real_kv",
                "status": "running" if self._running else ("error" if self._last_error else "ready"),
                "running": self._running,
                "next_source": self.agents[source_index].agent_id,
                "next_target": self.agents[1 - source_index].agent_id,
                "active_communication_mode": CommunicationMode.KV.value,
                "selector_status": "not_applicable",
                "model": self.backend.model_name,
                "device": self.backend.device,
                "max_new_tokens": self.max_new_tokens,
                "warmup_performed": self.warmup_performed,
                "canonical_handoff_semantics": True,
                "last_error": self._last_error,
            }

    def set_bandwidth_mbps(self, bandwidth_mbps: float) -> None:
        if not self._operation_lock.acquire(blocking=False):
            raise InteractiveRuntimeBusyError("communication is already running")
        try:
            self.network.set_bandwidth_mbps(bandwidth_mbps)
        finally:
            self._operation_lock.release()

    def communicate_next(self) -> tuple[str, str, RealKVCommunicationResult]:
        if not self._operation_lock.acquire(blocking=False):
            raise InteractiveRuntimeBusyError("communication is already running")
        with self._state_lock:
            self._running = True
            self._last_error = None
            source_index = self._next_source_index
            source = self.agents[source_index]
            target = self.agents[1 - source_index]
        self._record(
            EventType.REAL_KV_HANDOFF_STARTED,
            source=source.agent_id,
            target=target.agent_id,
            metadata={
                "communication_mode": CommunicationMode.KV.value,
                "measurement_strategy": "single_run",
                "canonical_handoff_semantics": True,
            },
        )
        try:
            sender_ids, sender_text = self._sender_representation(source)
            handoff = self.session.handoff_next(
                sender_ids,
                receiver_suffix_token_ids=self.receiver_suffix_token_ids,
                receiver_max_new_tokens=self.max_new_tokens,
                stage_listener=lambda stage, metadata: self._record_stage(
                    stage, source.agent_id, target.agent_id, metadata
                ),
                advance_direction=False,
            )
            if not handoff.simulated_links:
                raise RuntimeError("Real KV handoff produced no simulated link cost")
            verified = (
                handoff.compatibility_verified
                and handoff.wire_identity_verified
                and handoff.canonical_cache_complete
                and not handoff.receiver_prefix_prefill_executed
                and bool(handoff.receiver_output_token_ids)
            )
            if not verified:
                raise RuntimeError("Real KV continuation verification failed")
            result = RealKVCommunicationResult(
                handoff=handoff,
                link=handoff.simulated_links[0],
                sender_generated_text=sender_text,
                continuation_verified=True,
            )
            self._record_finished(result)
            self.session.advance_direction()
            with self._state_lock:
                self._continuation_ids = handoff.receiver_output_token_ids
                self._continuation_text = handoff.receiver_generated_text
                self._next_source_index = 1 - source_index
            return source.agent_id, target.agent_id, result
        except Exception as error:
            with self._state_lock:
                self._last_error = str(error)
            self._record(
                EventType.REAL_KV_HANDOFF_FAILED,
                source=source.agent_id,
                target=target.agent_id,
                metadata={"error": str(error)},
            )
            raise
        finally:
            with self._state_lock:
                self._running = False
            self._operation_lock.release()

    def _sender_representation(self, source: LLMAgent) -> tuple[tuple[int, ...], str]:
        if self._continuation_ids is not None:
            return self._continuation_ids, self._continuation_text or ""
        self._record(
            EventType.AGENT_GENERATION_STARTED,
            source=source.agent_id,
            metadata={"prompt": self.initial_prompt, "real_kv": True},
        )
        generation = source.generate_reply(self.initial_prompt, self._generation_config())
        self._record(
            EventType.AGENT_GENERATION_FINISHED,
            source=source.agent_id,
            metadata={
                "text": generation.generated_text,
                "token_count": len(generation.output_token_ids),
                "generation_wall_time_ms": generation.generation_wall_time_seconds * 1_000,
                "real_kv": True,
            },
        )
        return generation.output_token_ids, generation.generated_text

    def _generation_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            repetition_penalty=1.0,
        )

    def _record_stage(
        self, stage: str, source: str, target: str, metadata: Any
    ) -> None:
        event_types = {
            "prepared": EventType.KV_PREPARED,
            "serialized": EventType.KV_SERIALIZED,
            "transferred": EventType.KV_TRANSFERRED,
            "deserialized": EventType.KV_DESERIALIZED,
            "restored": EventType.KV_RESTORED,
        }
        self._record(event_types[stage], source=source, target=target, metadata=metadata)

    def _record_finished(self, result: RealKVCommunicationResult) -> None:
        handoff = result.handoff
        link = result.link
        self._record(
            EventType.REAL_KV_HANDOFF_FINISHED,
            source=handoff.source_agent_id,
            target=handoff.target_agent_id,
            payload_bytes=handoff.wire_total_bytes,
            transmission_latency_ms=link.simulated_link_latency_ms,
            estimated_e2e_latency_ms=link.kv_end_to_end_representation_cost_ms,
            metadata={
                "communication_mode": CommunicationMode.KV.value,
                "measurement_strategy": "single_run",
                "canonical_handoff_semantics": True,
                "token_count": handoff.token_count,
                "raw_kv_bytes": handoff.raw_kv_tensor_bytes,
                "wire_tensor_body_bytes": handoff.wire_tensor_body_bytes,
                "wire_metadata_bytes": handoff.wire_metadata_bytes,
                "wire_bytes": handoff.wire_total_bytes,
                "sender_kv_prepare_ms": handoff.sender_kv_prepare_ms,
                "device_to_host_ms": handoff.device_to_host_ms,
                "protocol_encode_ms": handoff.protocol_encode_ms,
                "total_serialize_ms": handoff.total_serialize_ms,
                "local_transfer_ms": handoff.local_transfer_ms,
                "deserialize_ms": handoff.deserialize_ms,
                "cpu_tensor_reconstruction_ms": handoff.cpu_tensor_reconstruction_ms,
                "host_to_device_ms": handoff.host_to_device_ms,
                "cache_construction_ms": handoff.cache_construction_ms,
                "total_restore_ms": handoff.total_restore_ms,
                "receiver_continuation_forward_ms": handoff.receiver_continuation_forward_ms,
                "bandwidth_mbps": link.bandwidth_mbps,
                "base_latency_ms": link.base_latency_ms,
                "simulated_link_ms": link.simulated_link_latency_ms,
                "representation_ready_cost_ms": link.kv_handoff_cost_ms,
                "message_ready_e2e_cost_ms": link.kv_end_to_end_representation_cost_ms,
                "receiver_prefix_prefill_executed": False,
                "sender_generated_text": result.sender_generated_text,
                "receiver_generated_text": handoff.receiver_generated_text,
                "compatibility_verified": handoff.compatibility_verified,
                "wire_identity_verified": handoff.wire_identity_verified,
                "canonical_cache_complete": handoff.canonical_cache_complete,
                "continuation_verified": result.continuation_verified,
                "transport": handoff.transport_name,
                "protocol_version": handoff.schema_version,
            },
        )

    def _record(
        self,
        event_type: EventType,
        *,
        source: str | None = None,
        target: str | None = None,
        payload_bytes: int | None = None,
        transmission_latency_ms: float | None = None,
        estimated_e2e_latency_ms: float | None = None,
        metadata: Any = None,
    ) -> None:
        self.collector.record(
            CommunicationEvent(
                event_type=event_type,
                source_agent_id=source,
                target_agent_id=target,
                mode=CommunicationMode.KV,
                payload_bytes=payload_bytes,
                transmission_latency_ms=transmission_latency_ms,
                estimated_e2e_latency_ms=estimated_e2e_latency_ms,
                metadata=metadata or {},
            )
        )
