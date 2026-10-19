"""Explicitly synthetic profile for mechanism tests and demos only."""

from .real_cost import CalibrationPoint, RealCommunicationCostProfile, PROFILE_SCHEMA_VERSION
from .token_protocol import TOKEN_PROTOCOL_VERSION
from .kv_protocol import KV_PROTOCOL_VERSION


def synthetic_crossover_profile() -> RealCommunicationCostProfile:
    points = tuple(CalibrationPoint(
        token_count=count, token_serialize_ms=0, token_deserialize_ms=0,
        receiver_prefill_ms=30, sender_prepare_ms=10, kv_serialize_ms=5,
        kv_deserialize_ms=0, kv_restore_ms=5) for count in (8, 32))
    return RealCommunicationCostProfile(
        PROFILE_SCHEMA_VERSION, "synthetic/crossover", 24, 2, 64, "bfloat16", 1,
        "cpu", "SYNTHETIC-DEMO-NOT-MEASURED", "cpu",
        TOKEN_PROTOCOL_VERSION, KV_PROTOCOL_VERSION, "synthetic fixture; no hardware measurement",
        1, 2, points)
