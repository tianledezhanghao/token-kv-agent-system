"""FastAPI application factory for the Dashboard Control Plane."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from token_kv_agent_system.demo import run_demo
from token_kv_agent_system.metrics import CommunicationEvent, MetricCollector
from token_kv_agent_system.model import DEFAULT_MODEL_NAME
from token_kv_agent_system.models import NetworkState

from .discussion_service import (DiscussionService, DiscussionStartRequest, PromptPreviewRequest, DiscussionNetworkRequest,
                                 DiscussionBusyError, DiscussionUnavailableError)
from .serialization import serialize_event
from .runtime import (
    InteractiveRuntime,
    InteractiveRuntimeBusyError,
    RealAdaptiveDashboardRuntime,
    RealKVCommunicationResult,
    RealKVRuntime,
    RealTokenCommunicationResult,
    RealTokenRuntime,
)
from .state import DashboardStateStore
from .websocket import EventBroadcaster

DemoRunner = Callable[[], Any]


class ModelSwitchRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    model: Literal['0.5B', '1.5B', '3B']


class NetworkControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bandwidth_mbps: float = Field(gt=0, allow_inf_nan=False)


class CommunicationControlRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


def create_app(
    *,
    collector: MetricCollector | None = None,
    initial_network_state: NetworkState | None = None,
    agent_ids: tuple[str, ...] = (),
    demo_runner: DemoRunner | None = None,
    demo_delay_seconds: float = 1.0,
    interactive_runtime: InteractiveRuntime | RealTokenRuntime | RealKVRuntime | RealAdaptiveDashboardRuntime | None = None,
    runtime_mode: str = "observer",
    discussion_service: DiscussionService | None = None,
    distributed_service=None,
) -> FastAPI:
    collector = collector or MetricCollector()
    discussion_service = discussion_service or DiscussionService(collector=collector)
    if discussion_service.collector is not collector:
        raise ValueError("Discussion and Dashboard must share their event collector")
    from .distributed_service import DistributedDashboardService
    distributed_service = distributed_service or DistributedDashboardService(collector=collector)
    if distributed_service.collector is not collector:
        raise ValueError('Distributed backend must share event collector')
    legacy_backend = getattr(interactive_runtime, "backend", None)
    if isinstance(interactive_runtime, RealAdaptiveDashboardRuntime):
        legacy_backend = interactive_runtime.runtime.backend
    if legacy_backend is not None and legacy_backend is discussion_service.backend:
        raise ValueError("Discussion backend must not be shared with an independently locked Canonical runtime")
    state = DashboardStateStore(
        initial_network_state=initial_network_state,
        agent_ids=agent_ids,
    )
    broadcaster = EventBroadcaster()

    for recorded_event in collector.events:
        state.consume(recorded_event)

    def consume(event: CommunicationEvent) -> None:
        state.consume(event)
        broadcaster.publish(serialize_event(event))

    collector.subscribe(consume)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task: asyncio.Task[Any] | None = None
        if demo_runner is not None:
            async def run_delayed_demo() -> None:
                await asyncio.sleep(demo_delay_seconds)
                await asyncio.to_thread(demo_runner)

            task = asyncio.create_task(run_delayed_demo())
        yield
        await asyncio.to_thread(distributed_service.close)
        await asyncio.to_thread(discussion_service.close)
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="Token/KV Agent Dashboard API", version="0.2.0", lifespan=lifespan)
    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, exc):
        # Invalid nonfinite JSON values cannot be echoed into JSONResponse.
        # Preserve the normal error contract without echoing arbitrary inputs.
        return JSONResponse(status_code=422, content={'detail': [
            {k: error[k] for k in ('type', 'loc', 'msg') if k in error}
            for error in exc.errors()]})

    app.state.collector = collector
    app.state.dashboard_state = state
    app.state.event_broadcaster = broadcaster
    app.state.interactive_runtime = interactive_runtime
    app.state.discussion_service = discussion_service

    def runtime_snapshot() -> dict[str, Any]:
        if interactive_runtime is not None:
            return interactive_runtime.snapshot()
        return {
            "mode": runtime_mode,
            "running": False,
            "next_source": None,
            "next_target": None,
        }

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/state")
    def dashboard_state() -> dict[str, Any]:
        return {**state.snapshot(), "runtime": runtime_snapshot()}

    @app.get("/api/discussions/latest")
    def latest_discussion() -> dict[str, Any]:
        return discussion_service.latest()

    @app.post('/api/models/switch', status_code=202)
    def switch_model(request: ModelSwitchRequest):
        if not hasattr(discussion_service, 'switch_model'):
            raise HTTPException(409, detail='Model switching requires --discussion')
        try:
            return discussion_service.switch_model(request.model)
        except DiscussionBusyError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, detail=f'{type(exc).__name__}: {exc}') from exc

    @app.post("/api/discussions/prompts")
    def preview_discussion_prompts(request: PromptPreviewRequest) -> dict[str, Any]:
        try:
            return discussion_service.preview_prompts(request)
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.get("/api/discussions/{session_id}")
    def get_discussion(session_id: str) -> dict[str, Any]:
        try:
            return discussion_service.get(session_id)
        except KeyError as exc:
            raise HTTPException(404, detail="Discussion Session not found (process-local history)") from exc

    @app.get('/api/discussions/{session_id}/network')
    def get_discussion_network(session_id: str) -> dict[str, Any]:
        try:
            return discussion_service.network(session_id)
        except KeyError as exc:
            raise HTTPException(404, detail='Discussion Session not found (process-local history)') from exc

    @app.patch('/api/discussions/{session_id}/network')
    def update_discussion_network(session_id: str, request: DiscussionNetworkRequest) -> dict[str, Any]:
        try:
            return discussion_service.update_network(session_id, request)
        except KeyError as exc:
            raise HTTPException(404, detail='Discussion Session not found (process-local history)') from exc
        except DiscussionBusyError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    @app.post("/api/discussions", status_code=202)
    def start_discussion(request: DiscussionStartRequest) -> dict[str, Any]:
        try:
            return discussion_service.start(request)
        except DiscussionBusyError as exc:
            raise HTTPException(409, detail=str(exc)) from exc
        except DiscussionUnavailableError as exc:
            raise HTTPException(503, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, detail=str(exc)) from exc

    # Explicit namespace freezes backend identity; never route by browser draft.
    @app.get('/api/distributed/discussions/latest')
    def distributed_latest():
        return distributed_service.latest()

    @app.get('/api/distributed/health')
    def distributed_health():
        return distributed_service.health(force=True)

    @app.post('/api/distributed/discussions/prompts')
    def distributed_prompts(request: PromptPreviewRequest):
        return distributed_service.preview_prompts(request)

    @app.get('/api/distributed/discussions/{session_id}')
    def distributed_get(session_id: str):
        try:
            return distributed_service.get(session_id)
        except KeyError as exc:
            raise HTTPException(404, 'Distributed Session not found') from exc

    @app.post('/api/distributed/discussions', status_code=202)
    def distributed_start(request: DiscussionStartRequest):
        try:
            return distributed_service.start(request)
        except DiscussionUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except DiscussionBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get('/api/distributed/discussions/{session_id}/network')
    def distributed_network(session_id: str):
        try:
            return distributed_service.network(session_id)
        except KeyError as exc:
            raise HTTPException(404, 'Distributed Session not found') from exc

    @app.patch('/api/distributed/discussions/{session_id}/network')
    def distributed_patch(session_id: str, request: DiscussionNetworkRequest):
        try:
            return distributed_service.update_network(session_id, request)
        except KeyError as exc:
            raise HTTPException(404, 'Distributed Session not found') from exc
        except DiscussionBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.post('/api/distributed/discussions/{session_id}/release')
    def distributed_release(session_id: str):
        try:
            return distributed_service.release(session_id)
        except KeyError as exc:
            raise HTTPException(404, 'Distributed Session not found') from exc
        except DiscussionBusyError as exc:
            raise HTTPException(409, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get('/api/distributed/events')
    def distributed_events(limit: int = Query(default=100, ge=1, le=200)):
        items = [e for e in state.recent_events(200) if e.get('metadata', {}).get('architecture') == 'distributed'][-limit:]
        return dict(events=items, count=len(items))

    @app.get("/api/events")
    def events(limit: int = Query(default=50, ge=1, le=200)) -> dict[str, Any]:
        items = state.recent_events(limit)
        return {"events": items, "count": len(items)}

    def require_interactive_runtime() -> InteractiveRuntime | RealTokenRuntime | RealKVRuntime | RealAdaptiveDashboardRuntime:
        if interactive_runtime is None:
            raise HTTPException(
                status_code=409,
                detail="interactive mode is required for control operations",
            )
        return interactive_runtime

    @app.patch("/api/control/network")
    def update_network(request: NetworkControlRequest) -> dict[str, Any]:
        runtime = require_interactive_runtime()
        try:
            runtime.set_bandwidth_mbps(request.bandwidth_mbps)
        except InteractiveRuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        return {
            "network": {
                "bandwidth_mbps": runtime.network.state.bandwidth_mbps,
                "base_latency_ms": runtime.network.state.base_latency_ms,
            }
        }

    @app.post("/api/control/communication")
    def trigger_communication(
        request: CommunicationControlRequest | None = None,
    ) -> dict[str, Any]:
        del request
        runtime = require_interactive_runtime()
        try:
            source, target, result = runtime.communicate_next()
        except InteractiveRuntimeBusyError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(status_code=500, detail=str(error)) from error
        if isinstance(result, RealTokenCommunicationResult):
            return {
                "source": source,
                "target": target,
                "communication_mode": "TOKEN",
                "selected_mode": None,
                "real_token_metrics": {
                    "token_count": result.handoff.token_count,
                    "wire_bytes": result.handoff.wire_byte_count,
                    "serialization_ms": result.handoff.serialization_ms,
                    "simulated_link_ms": result.simulated_link_latency_ms,
                    "deserialization_ms": result.handoff.deserialization_ms,
                    "receiver_prefill_ms": result.handoff.receiver_prefill_forward_ms,
                    "token_handoff_cost_ms": result.token_handoff_cost_ms,
                    "token_identity_verified": result.handoff.token_identity_preserved,
                },
                "runtime": runtime.snapshot(),
            }
        if isinstance(runtime, RealAdaptiveDashboardRuntime):
            return {"source": source, "target": target,
                    "communication_mode": result.actually_executed_mode,
                    "selected_mode": result.selected_mode,
                    "adaptive_execution": state.snapshot()["adaptive_execution"],
                    "runtime": runtime.snapshot()}
        if isinstance(result, RealKVCommunicationResult):
            handoff = result.handoff
            link = result.link
            return {
                "source": source,
                "target": target,
                "communication_mode": "KV",
                "selected_mode": None,
                "real_kv_metrics": {
                    "token_count": handoff.token_count,
                    "raw_kv_bytes": handoff.raw_kv_tensor_bytes,
                    "wire_bytes": handoff.wire_total_bytes,
                    "sender_kv_prepare_ms": handoff.sender_kv_prepare_ms,
                    "total_serialize_ms": handoff.total_serialize_ms,
                    "simulated_link_ms": link.simulated_link_latency_ms,
                    "total_restore_ms": handoff.total_restore_ms,
                    "representation_ready_cost_ms": link.kv_handoff_cost_ms,
                    "message_ready_e2e_cost_ms": link.kv_end_to_end_representation_cost_ms,
                    "receiver_prefix_prefill_executed": False,
                    "continuation_verified": result.continuation_verified,
                },
                "runtime": runtime.snapshot(),
            }
        return {
            "source": source,
            "target": target,
            "communication_mode": result.decision.mode.value,
            "selected_mode": result.decision.mode.value,
            "transmission": {
                "payload_bytes": result.transmission.payload_bytes,
                "transmission_latency_ms": result.transmission.transmission_latency_ms,
                "estimated_e2e_latency_ms": result.transmission.estimated_e2e_latency_ms,
            },
            "runtime": runtime.snapshot(),
        }

    @app.websocket("/ws/distributed/events")
    @app.websocket("/ws/events")
    async def event_stream(websocket: WebSocket) -> None:
        await websocket.accept()
        subscription = broadcaster.subscribe()
        receive_task = asyncio.create_task(websocket.receive())
        try:
            while not receive_task.done():
                event_task = asyncio.create_task(subscription.queue.get())
                done, _ = await asyncio.wait(
                    (receive_task, event_task),
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if receive_task in done:
                    event_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await event_task
                    break
                await websocket.send_json(event_task.result())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            if not receive_task.done():
                receive_task.cancel()
                with suppress(asyncio.CancelledError):
                    await receive_task
            broadcaster.unsubscribe(subscription)

    return app


def create_integrated_demo_app(*, demo_delay_seconds: float = 1.0) -> FastAPI:
    """Create a backend whose real Mock Demo shares the same collector."""
    collector = MetricCollector()
    return create_app(
        collector=collector,
        initial_network_state=NetworkState(20, 5),
        agent_ids=("Agent A", "Agent B"),
        demo_runner=lambda: run_demo(collector=collector),
        demo_delay_seconds=demo_delay_seconds,
        runtime_mode="auto",
    )


def create_interactive_app() -> FastAPI:
    """Create a backend that waits for Dashboard control requests."""
    runtime = InteractiveRuntime.create_default()
    return create_app(
        collector=runtime.collector,
        initial_network_state=runtime.network.state,
        agent_ids=tuple(agent.agent_id for agent in runtime.agents),
        interactive_runtime=runtime,
    )


def create_real_token_app(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    max_new_tokens: int = 8,
    initial_prompt: str = "Explain KV cache briefly.",
) -> FastAPI:
    """Synchronously load/warm one shared backend, then await browser control."""
    runtime = RealTokenRuntime.create_default(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        initial_prompt=initial_prompt,
        warmup=True,
    )
    return create_app(
        collector=runtime.collector,
        initial_network_state=runtime.network.state,
        agent_ids=tuple(agent.agent_id for agent in runtime.agents),
        interactive_runtime=runtime,
        runtime_mode="real_token",
    )


def create_real_kv_app(
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    device: str = "auto",
    max_new_tokens: int = 8,
    initial_prompt: str = "Explain KV cache briefly.",
) -> FastAPI:
    """Synchronously load/warm one shared backend, then await real KV controls."""
    runtime = RealKVRuntime.create_default(
        model_name=model_name,
        device=device,
        max_new_tokens=max_new_tokens,
        initial_prompt=initial_prompt,
        warmup=True,
    )
    return create_app(
        collector=runtime.collector,
        initial_network_state=runtime.network.state,
        agent_ids=tuple(agent.agent_id for agent in runtime.agents),
        interactive_runtime=runtime,
        runtime_mode="real_kv",
    )


def create_real_adaptive_app(*, selector_profile: str, model_name: str = DEFAULT_MODEL_NAME,
                             device: str = "auto", max_new_tokens: int = 8,
                             initial_prompt: str = "Explain KV cache briefly.") -> FastAPI:
    runtime = RealAdaptiveDashboardRuntime.create_default(
        selector_profile=selector_profile, model_name=model_name, device=device,
        max_new_tokens=max_new_tokens, initial_prompt=initial_prompt)
    return create_app(collector=runtime.collector,
                      initial_network_state=runtime.network.state,
                      agent_ids=tuple(agent.agent_id for agent in runtime.agents),
                      interactive_runtime=runtime, runtime_mode="real_adaptive")


def create_discussion_app(*, model_name: str = DEFAULT_MODEL_NAME, device: str = "mps") -> FastAPI:
    """Own one offline MPS model process; retain Mock Canonical controls."""
    import os
    from .model_worker import ModelWorkerService, MODELS, cached_model
    if device != 'mps':
        raise ValueError('Discussion model worker requires MPS; CPU fallback is disabled')
    os.environ["HF_HUB_OFFLINE"] = "1"
    for size, name in MODELS.items():
        if model_name == name:
            model_name = cached_model(size)
            break
    if not Path(model_name).is_dir():
        raise ValueError('Discussion requires an existing local model snapshot')
    runtime = InteractiveRuntime.create_default()
    service = ModelWorkerService(model_name, collector=runtime.collector)
    return create_app(collector=runtime.collector, initial_network_state=runtime.network.state,
                      agent_ids=tuple(agent.agent_id for agent in runtime.agents),
                      interactive_runtime=runtime, discussion_service=service)
