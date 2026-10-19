"""Decision-time cost prediction from device-specific real communication calibration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite

from .kv_protocol import KVModelCompatibility, KV_PROTOCOL_VERSION, estimate_kv_wire_size
from .models import CommunicationMode
from .network import SimulatedNetwork
from .token_protocol import TOKEN_PROTOCOL_VERSION, estimate_token_wire_bytes


PROFILE_SCHEMA_VERSION = 1
MESSAGE_READY = "MESSAGE_READY"
REPRESENTATION_READY = "REPRESENTATION_READY"


@dataclass(frozen=True)
class CalibrationPoint:
    token_count: int
    token_serialize_ms: float
    token_deserialize_ms: float
    receiver_prefill_ms: float
    sender_prepare_ms: float
    kv_serialize_ms: float
    kv_deserialize_ms: float
    kv_restore_ms: float

    def __post_init__(self) -> None:
        if isinstance(self.token_count, bool) or not isinstance(self.token_count, int) or self.token_count <= 0:
            raise ValueError("calibration token_count must be positive")
        for name, value in asdict(self).items():
            if name != "token_count" and (not isinstance(value, (int, float)) or not isfinite(value) or value < 0):
                raise ValueError(f"invalid calibration stage {name}")


@dataclass(frozen=True)
class RealCommunicationCostProfile:
    schema_version: int
    model_id: str
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    kv_dtype: str
    batch_size: int
    device_type: str
    profile_name: str
    source_device: str
    token_protocol_version: int
    kv_protocol_version: int
    measurement_policy: str
    warmup_runs: int
    measurement_runs: int
    points: tuple[CalibrationPoint, ...]

    def __post_init__(self) -> None:
        if self.schema_version != PROFILE_SCHEMA_VERSION:
            raise ValueError("unsupported cost profile schema")
        if self.token_protocol_version != TOKEN_PROTOCOL_VERSION or self.kv_protocol_version != KV_PROTOCOL_VERSION:
            raise ValueError("profile protocol version mismatch")
        if not self.model_id or not self.profile_name or not self.measurement_policy:
            raise ValueError("missing profile identity or measurement policy")
        if self.device_type not in {"cpu", "mps", "cuda"} or self.source_device.split(":")[0] != self.device_type:
            raise ValueError("profile device type/source device mismatch")
        if self.warmup_runs < 1 or self.measurement_runs < 2:
            raise ValueError("profile requires warmup and multiple measured runs")
        if len(self.points) < 2 or any(a.token_count >= b.token_count for a, b in zip(self.points, self.points[1:])):
            raise ValueError("calibration points must be unique and ascending")
        estimate_kv_wire_size(self.points[0].token_count, self.model, dtype=self.kv_dtype,
                              source_device=self.source_device, batch_size=self.batch_size)

    @property
    def model(self) -> KVModelCompatibility:
        return KVModelCompatibility(self.model_id, self.num_hidden_layers,
                                    self.num_key_value_heads, self.head_dim)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> RealCommunicationCostProfile:
        values = dict(data)
        values["points"] = tuple(CalibrationPoint(**p) for p in values["points"])
        return cls(**values)

    @classmethod
    def from_json(cls, data: str) -> RealCommunicationCostProfile:
        return cls.from_dict(json.loads(data))


@dataclass(frozen=True)
class RealSelectionContext:
    handoff_token_count: int
    bandwidth_mbps: float
    base_latency_ms: float
    model_id: str
    num_hidden_layers: int
    num_key_value_heads: int
    head_dim: int
    kv_dtype: str
    batch_size: int
    device_type: str
    profile_name: str
    source_device: str
    token_protocol_version: int = TOKEN_PROTOCOL_VERSION
    kv_protocol_version: int = KV_PROTOCOL_VERSION

    @classmethod
    def from_profile(cls, profile: RealCommunicationCostProfile, token_count: int,
                     bandwidth_mbps: float, base_latency_ms: float) -> RealSelectionContext:
        return cls(token_count, bandwidth_mbps, base_latency_ms, profile.model_id,
                   profile.num_hidden_layers, profile.num_key_value_heads, profile.head_dim,
                   profile.kv_dtype, profile.batch_size, profile.device_type,
                   profile.profile_name, profile.source_device,
                   profile.token_protocol_version, profile.kv_protocol_version)


@dataclass(frozen=True)
class TokenCostEstimate:
    token_count: int
    wire_bytes: int
    serialize_ms: float
    link_ms: float
    deserialize_ms: float
    receiver_prefill_ms: float
    total_ms: float
    calibration_position: str


@dataclass(frozen=True)
class KVCostEstimate:
    token_count: int
    raw_kv_bytes: int
    wire_tensor_body_bytes: int
    wire_metadata_bytes: int
    wire_bytes: int
    sender_prepare_ms: float
    serialize_ms: float
    link_ms: float
    deserialize_ms: float
    restore_ms: float
    message_ready_total_ms: float
    representation_ready_total_ms: float
    calibration_position: str


@dataclass(frozen=True)
class RealCostEstimates:
    token: TokenCostEstimate
    kv: KVCostEstimate


@dataclass(frozen=True)
class SelectionDecision:
    selected_mode: CommunicationMode
    cost_boundary: str
    token_estimate: TokenCostEstimate
    kv_estimate: KVCostEstimate
    estimated_saving_ms: float
    estimated_saving_ratio: float


class RealCommunicationCostModel:
    def __init__(self, profile: RealCommunicationCostProfile) -> None:
        self.profile = profile

    def _validate_context(self, context: RealSelectionContext) -> None:
        for name in ("model_id", "num_hidden_layers", "num_key_value_heads", "head_dim",
                     "kv_dtype", "batch_size", "device_type", "profile_name", "source_device",
                     "token_protocol_version", "kv_protocol_version"):
            if getattr(context, name) != getattr(self.profile, name):
                raise ValueError(f"calibration profile incompatible with runtime {name}")
        if context.handoff_token_count <= 0:
            raise ValueError("handoff token count must be positive")

    def _stages(self, count: int) -> tuple[dict[str, float], str]:
        points = self.profile.points
        fields = tuple(k for k in asdict(points[0]) if k != "token_count")
        for point in points:
            if point.token_count == count:
                return {field: getattr(point, field) for field in fields}, "exact"
        if count < points[0].token_count:
            left, right = points[:2]
            position = "extrapolated"
        elif count > points[-1].token_count:
            left, right = points[-2:]
            position = "extrapolated"
        else:
            left, right = next((a, b) for a, b in zip(points, points[1:]) if a.token_count < count < b.token_count)
            position = "interpolated"
        fraction = (count - left.token_count) / (right.token_count - left.token_count)
        return {field: max(0.0, getattr(left, field) + fraction * (getattr(right, field) - getattr(left, field))) for field in fields}, position

    def estimate(self, context: RealSelectionContext) -> RealCostEstimates:
        self._validate_context(context)
        count = context.handoff_token_count
        stages, position = self._stages(count)
        network = SimulatedNetwork(context.bandwidth_mbps, context.base_latency_ms)
        token_bytes = estimate_token_wire_bytes(count)
        kv_bytes = estimate_kv_wire_size(count, self.profile.model, dtype=self.profile.kv_dtype,
                                         source_device=self.profile.source_device,
                                         batch_size=self.profile.batch_size)
        token_link = network.estimate_transmission_latency_ms(token_bytes)
        kv_link = network.estimate_transmission_latency_ms(kv_bytes.total_wire_bytes)
        token_total = stages["token_serialize_ms"] + token_link + stages["token_deserialize_ms"] + stages["receiver_prefill_ms"]
        kv_rep = stages["kv_serialize_ms"] + kv_link + stages["kv_deserialize_ms"] + stages["kv_restore_ms"]
        return RealCostEstimates(
            TokenCostEstimate(count, token_bytes, stages["token_serialize_ms"], token_link,
                              stages["token_deserialize_ms"], stages["receiver_prefill_ms"], token_total, position),
            KVCostEstimate(count, kv_bytes.raw_kv_tensor_bytes, kv_bytes.wire_tensor_body_bytes,
                           kv_bytes.wire_metadata_bytes, kv_bytes.total_wire_bytes,
                           stages["sender_prepare_ms"], stages["kv_serialize_ms"], kv_link,
                           stages["kv_deserialize_ms"], stages["kv_restore_ms"],
                           stages["sender_prepare_ms"] + kv_rep, kv_rep, position))

    def crossover_bandwidth_mbps(self, context: RealSelectionContext, *, boundary: str = MESSAGE_READY) -> float | None:
        if boundary not in {MESSAGE_READY, REPRESENTATION_READY}:
            raise ValueError("unknown cost boundary")
        estimates = self.estimate(context)
        token_compute = estimates.token.total_ms - estimates.token.link_ms
        kv_compute = (estimates.kv.message_ready_total_ms if boundary == MESSAGE_READY else estimates.kv.representation_ready_total_ms) - estimates.kv.link_ms
        delta_compute = token_compute - kv_compute
        delta_bytes = estimates.kv.wire_bytes - estimates.token.wire_bytes
        if delta_bytes <= 0 or delta_compute <= 0:
            return None
        return delta_bytes * 8 * 1000 / (1_000_000 * delta_compute)


class RealCostBasedSelector:
    """Pure selection: no model, packet, cache or Data Plane access."""

    def __init__(self, cost_model: RealCommunicationCostModel) -> None:
        self.cost_model = cost_model

    def select(self, context: RealSelectionContext, *, boundary: str = MESSAGE_READY) -> SelectionDecision:
        if boundary not in {MESSAGE_READY, REPRESENTATION_READY}:
            raise ValueError("unknown cost boundary")
        estimates = self.cost_model.estimate(context)
        token_total = estimates.token.total_ms
        kv_total = estimates.kv.message_ready_total_ms if boundary == MESSAGE_READY else estimates.kv.representation_ready_total_ms
        selected = CommunicationMode.TOKEN if token_total <= kv_total else CommunicationMode.KV
        saving = abs(token_total - kv_total)
        return SelectionDecision(selected, boundary, estimates.token, estimates.kv,
                                 saving, saving / max(token_total, kv_total) if max(token_total, kv_total) else 0.0)
