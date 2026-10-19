from dataclasses import fields

import pytest

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.model import (
    DecodeStepResult,
    GenerationConfig,
    GenerationResult,
    IncrementalCausalLMBackend,
    ModelMetadata,
    PrefillResult,
    TokenIdGenerationResult,
)
from token_kv_agent_system.token_handoff import (
    EmptyTokenSequenceError,
    TokenHandoffError,
    TokenHandoffSession,
    UnknownAgentError,
)
from token_kv_agent_system.token_protocol import (
    TokenDeserializationError,
    deserialize_token_packet,
    serialize_token_ids,
)
from token_kv_agent_system.transport import BaseByteTransport, InMemoryByteTransport


class FakeTokenBackend(IncrementalCausalLMBackend):
    def __init__(self, outputs=(4, 5)):
        self.outputs = tuple(outputs)
        self.prefill_inputs = []
        self.prefill_caches = []
        self.encode_calls = 0
        self.decode_calls = 0
        self._metadata = ModelMetadata(
            model_name="fake/shared",
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
        self.encode_calls += 1
        return (1,)

    def decode(self, token_ids, *, skip_special_tokens=False):
        self.decode_calls += 1
        return " ".join(map(str, token_ids))

    def generate(self, prompt, config=None):
        return GenerationResult(
            prompt=prompt,
            input_token_ids=(1,),
            generated_text=self.decode(self.outputs),
            output_token_ids=self.outputs,
            generation_wall_time_seconds=0.001,
            full_token_ids=(1,) + self.outputs,
        )

    def prefill(self, prompt=None, *, input_token_ids=None):
        assert prompt is None
        normalized = tuple(input_token_ids or ())
        cache = object()
        self.prefill_inputs.append(normalized)
        self.prefill_caches.append(cache)
        return PrefillResult(
            input_token_ids=normalized,
            first_token_id=6,
            cache=cache,
            prefill_forward_ms=1.25,
            logical_cached_token_count=len(normalized),
            logits_shape=(1, len(normalized), 16),
            last_token_logits_shape=(16,),
            last_token_logits=object(),
        )

    def decode_step(self, input_token_id, *, previous_cache, logical_cached_token_count):
        return DecodeStepResult(
            input_token_id=input_token_id,
            predicted_next_token_id=7,
            cache=previous_cache,
            decode_forward_ms=0.5,
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
        output_ids = (prefill.first_token_id, step.predicted_next_token_id)
        return TokenIdGenerationResult(
            input_token_ids=prefill.input_token_ids,
            output_token_ids=output_ids,
            generated_text=self.decode(output_ids),
            prefill=prefill,
            decode_steps=(step,),
            stop_reason="max_new_tokens",
        )


class TruncatingTransport(BaseByteTransport):
    def transfer(self, data, *, source, target):
        return data[:-1]


def make_session(*, transport=None, outputs=(4, 5), serializer=None, deserializer=None):
    backend = FakeTokenBackend(outputs)
    agents = [LLMAgent("agent_a", backend), LLMAgent("agent_b", backend)]
    kwargs = {}
    if serializer is not None:
        kwargs["serializer"] = serializer
    if deserializer is not None:
        kwargs["deserializer"] = deserializer
    session = TokenHandoffSession(
        agents, transport or InMemoryByteTransport(), **kwargs
    )
    return session, backend


def test_in_memory_byte_transport_accepts_bytes_and_preserves_identity_and_metadata():
    transport = InMemoryByteTransport()
    data = b"\x00token-packet\xff"

    delivered = transport.transfer(data, source="agent_a", target="agent_b")

    assert delivered is data
    assert transport.records[0].source == "agent_a"
    assert transport.records[0].target == "agent_b"
    assert transport.records[0].byte_count == len(data)
    with pytest.raises(TypeError, match="must be bytes"):
        transport.transfer(bytearray(data), source="agent_a", target="agent_b")


def test_handoff_uses_codec_and_exact_ids_reach_receiver_prefill():
    calls = []

    def serializer(token_ids):
        calls.append(("serialize", tuple(token_ids)))
        return serialize_token_ids(token_ids)

    def deserializer(data):
        calls.append(("deserialize", data))
        return deserialize_token_packet(data)

    session, backend = make_session(serializer=serializer, deserializer=deserializer)

    result = session.generate_and_handoff(
        "agent_a",
        "agent_b",
        "prompt",
        sender_config=GenerationConfig(max_new_tokens=2),
        receiver_config=GenerationConfig(max_new_tokens=2),
    )

    assert calls[0] == ("serialize", (4, 5))
    assert calls[1][0] == "deserialize"
    assert result.sender_output_token_ids == (4, 5)
    assert result.received_token_ids == (4, 5)
    assert result.receiver_prefill_input_token_ids == (4, 5)
    assert backend.prefill_inputs == [(4, 5)]
    assert result.token_identity_preserved is True
    assert result.receiver_output_token_ids == (6, 7)
    assert result.receiver_prefill_forward_ms == 1.25
    assert result.wire_byte_count == len(serialize_token_ids((4, 5))) == 18
    assert result.received_byte_count == result.wire_byte_count
    assert result.wire_header_bytes == 10
    assert result.wire_body_bytes == 8
    assert result.kv_bytes_transferred == 0


def test_bidirectional_handoff_preserves_direction_and_uses_shared_backend():
    transport = InMemoryByteTransport()
    session, backend = make_session(transport=transport)

    a_to_b = session.generate_and_handoff("agent_a", "agent_b", "prompt")
    b_to_a = session.handoff_token_ids(
        "agent_b",
        "agent_a",
        a_to_b.receiver_output_token_ids,
        sender_generated_text=a_to_b.receiver_generated_text,
    )

    assert (a_to_b.source_agent_id, a_to_b.target_agent_id) == ("agent_a", "agent_b")
    assert (b_to_a.source_agent_id, b_to_a.target_agent_id) == ("agent_b", "agent_a")
    assert [record.source for record in transport.records] == ["agent_a", "agent_b"]
    assert [record.target for record in transport.records] == ["agent_b", "agent_a"]
    assert a_to_b.token_identity_preserved and b_to_a.token_identity_preserved
    assert backend.prefill_caches[0] is not backend.prefill_caches[1]


def test_receiver_inference_does_not_encode_received_ids():
    session, backend = make_session()

    session.generate_and_handoff("agent_a", "agent_b", "prompt")

    assert backend.encode_calls == 0
    assert backend.prefill_inputs == [(4, 5)]


def test_transport_corruption_is_rejected_by_protocol():
    session, _ = make_session(transport=TruncatingTransport())

    with pytest.raises(TokenDeserializationError, match="truncated"):
        session.generate_and_handoff("agent_a", "agent_b", "prompt")


def test_empty_sender_output_is_rejected_before_transport():
    transport = InMemoryByteTransport()
    session, _ = make_session(transport=transport, outputs=())

    with pytest.raises(EmptyTokenSequenceError, match="no Token IDs"):
        session.generate_and_handoff("agent_a", "agent_b", "prompt")
    assert transport.records == ()


def test_unknown_same_agent_and_different_backends_are_rejected():
    session, _ = make_session()
    with pytest.raises(UnknownAgentError):
        session.generate_and_handoff("missing", "agent_b", "prompt")
    with pytest.raises(TokenHandoffError, match="different"):
        session.generate_and_handoff("agent_a", "agent_a", "prompt")

    agents = [LLMAgent("agent_a", FakeTokenBackend()), LLMAgent("agent_b", FakeTokenBackend())]
    incompatible = TokenHandoffSession(agents, InMemoryByteTransport())
    with pytest.raises(TokenHandoffError, match="share one model backend"):
        incompatible.generate_and_handoff("agent_a", "agent_b", "prompt")


def test_handoff_result_has_no_runtime_cache_tensor_model_or_packet_bytes_fields():
    session, _ = make_session()
    result = session.generate_and_handoff("agent_a", "agent_b", "prompt")

    names = {field.name.lower() for field in fields(result)}
    assert not any("cache" in name or "tensor" in name or "model" in name for name in names)
    assert "wire_packet_bytes" not in names
    assert all(not isinstance(value, bytes) for value in result.__dict__.values())
