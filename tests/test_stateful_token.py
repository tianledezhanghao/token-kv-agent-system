"""7.5A CPU/mock tests: no torch, model download, calculator or preset answer."""
from dataclasses import replace
from types import SimpleNamespace
import copy

import pytest

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.agent.chat_framing import ChatFraming
from token_kv_agent_system.model.backend_pair import BackendIdentity, IndependentBackends
from token_kv_agent_system.model.cache_inspection import KVModelConfig
from token_kv_agent_system.model.state_cache import (
    cache_storage_keys, require_disjoint_caches, validate_state_cache)
from token_kv_agent_system.stateful_token import StatefulTokenSession
from token_kv_agent_system.token_protocol import deserialize_token_packet, serialize_token_ids
from token_kv_agent_system.transport import InMemoryByteTransport


class MockTokenizer:
    specials = {"<s>": 1, "</s>": 2, "<eos>": 3}
    all_special_tokens = tuple(specials)
    all_special_ids = tuple(specials.values())

    def encode(self, text, add_special_tokens=False):
        result = []
        while text:
            matched = next((s for s in self.specials if text.startswith(s)), None)
            if matched:
                result.append(self.specials[matched])
                text = text[len(matched):]
            else:
                result.append(ord(text[0]) + 10)
                text = text[1:]
        return result

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = "".join(f"<s>{m['role']}\n{m['content']}</s>\n" for m in messages)
        if add_generation_prompt:
            text += "<s>assistant\n"
        return self.encode(text) if tokenize else text


class Tensor:
    dtype = "torch.float32"
    device = "cpu"

    def __init__(self, data):
        self.data = list(data)

    @property
    def shape(self):
        return (1, 1, len(self.data), 1)

    def numel(self):
        return len(self.data)

    def element_size(self):
        return 4

    def untyped_storage(self):
        return SimpleNamespace(data_ptr=lambda: id(self.data))


class Cache:
    def __init__(self, ids):
        self.layers = [SimpleNamespace(keys=Tensor(ids), values=Tensor(ids)) for _ in range(2)]

    def get_seq_length(self):
        return len(self.layers[0].keys.data)


def snapshot(cache):
    return tuple(tuple(t.data) for layer in cache.layers for t in (layer.keys, layer.values))


class Backend:
    vocab_size = 256
    eos_token_ids = (2, 3)

    def __init__(self):
        self.chat_tokenizer = MockTokenizer()
        self.calls = []
        self.outputs = []
        self.fail_at = None
        self.bad_count_at = None
        self._tokenizer = self.chat_tokenizer
        self.weight = Tensor([1])
        self._model = SimpleNamespace(parameters=lambda: iter([self.weight]), buffers=lambda: iter([]))
        structure = KVModelConfig(2, 1, 1, 1, 1, "config.head_dim")
        self.compatibility_metadata = BackendIdentity(
            "mock", "mock", "MockModel", "float32", structure, 256,
            "MockTokenizer", 256, 256, "same-vocabulary", (), (1, 2, 3), (2, 3))
        self.device = "cpu"

    def prefill(self, *, input_token_ids):
        self.calls.append(("prefill", input_token_ids, 0))
        return SimpleNamespace(cache=Cache(input_token_ids), logical_cached_token_count=len(input_token_ids))

    def validate_cache(self, cache, count):
        validate_state_cache(cache, count, 2)

    def clone_cache(self, cache):
        return copy.deepcopy(cache)

    def continue_from_cache(self, ids, *, previous_cache, logical_cached_token_count):
        self.validate_cache(previous_cache, logical_cached_token_count)
        self.calls.append(("continue", ids, logical_cached_token_count))
        # Deliberate in-place mutation, as in DynamicCache; failure can be partial.
        for layer in previous_cache.layers:
            for tensor in (layer.keys, layer.values):
                tensor.data.extend(ids)
        if len(self.calls) == self.fail_at:
            previous_cache.layers[0].keys.data[0] = 255
            raise RuntimeError("injected partial-forward failure")
        prefix = tuple(self.chat_tokenizer.encode("<s>assistant\n"))
        prediction = ord("x") + 10
        if (ids == prefix or len(ids) == 1) and self.outputs:
            prediction = self.outputs.pop(0)
        count = logical_cached_token_count + len(ids)
        if len(self.calls) == self.bad_count_at:
            count += 1
        return SimpleNamespace(input_token_ids=ids, cache=previous_cache,
                               logical_cached_token_count=count, predicted_next_token_id=prediction)


def build(a_backend=None, b_backend=None, transport=None):
    backend = a_backend or Backend()
    return StatefulTokenSession([
        LLMAgent("A", backend, system_prompt="You are Agent A. Keep A's context."),
        LLMAgent("B", b_backend or backend, system_prompt="You are Agent B. Keep B's context.")],
        transport or InMemoryByteTransport())


def generate(session, source="A", target="B", ids=(100, 101, 2)):
    session.state(source).backend.outputs = list(ids)
    return session.generate_message(source, target, max_new_tokens=len(ids))


def test_independent_roles_enter_actual_prefill_and_cache_storage():
    s = build()
    a, b = s.state("A"), s.state("B")
    assert a.backend is b.backend
    assert a.token_ledger != b.token_ledger
    assert [c[1] for c in a.backend.calls] == [a.token_ledger, b.token_ledger]
    for state in (a, b):
        prompt = tuple(state.backend.chat_tokenizer.encode(state.system_prompt))
        assert any(state.token_ledger[i:i + len(prompt)] == prompt for i in range(len(state.token_ledger)))
        state.validate()
    require_disjoint_caches(a.cache, b.cache)
    assert not cache_storage_keys(a.cache) & cache_storage_keys(b.cache)


def test_receive_only_new_ids_and_real_protocol_bytes_roundtrip():
    s = build()
    old = s.state("B")
    old_data = snapshot(old.cache)
    message = generate(s)
    backend = old.backend
    offset = len(backend.calls)
    receipt = s.handoff("A", "B")
    assert serialize_token_ids(message.message_token_ids)[:4] == b"TKVT"
    assert deserialize_token_packet(serialize_token_ids(message.message_token_ids)).token_ids == receipt.received_token_ids
    assert receipt.received_token_ids == message.message_token_ids
    assert s.transport.records[0].byte_count == 10 + 4 * message.token_count
    receiver_call = backend.calls[-1]
    assert receiver_call == ("continue", receipt.receiver_appended_ids, old.cached_token_count)
    assert all(c[0] == "continue" for c in backend.calls[offset:])
    assert snapshot(old.cache) == old_data
    new = s.state("B")
    assert new.token_ledger[:len(old.token_ledger)] == old.token_ledger
    assert new.cached_token_count == old.cached_token_count + len(receipt.receiver_appended_ids)
    assert new.sent_messages == () and len(new.received_messages) == 1


def test_three_rounds_bidirectional_continuous_growth():
    s = build()
    initial = {n: s.state(n) for n in ("A", "B")}
    lengths = []
    for _ in range(3):
        for source, target in (("A", "B"), ("B", "A")):
            old = {n: s.state(n) for n in ("A", "B")}
            old_data = {n: snapshot(old[n].cache) for n in old}
            generate(s, source, target)
            s.handoff(source, target)
            for n in old:
                state = s.state(n)
                state.validate()
                assert state.cached_token_count > old[n].cached_token_count
                assert state.token_ledger[:len(old[n].token_ledger)] == old[n].token_ledger
                assert snapshot(old[n].cache) == old_data[n]
                require_disjoint_caches(old[n].cache, state.cache)
            require_disjoint_caches(s.state("A").cache, s.state("B").cache)
        lengths.append(tuple(s.state(n).cached_token_count for n in ("A", "B")))
    assert all(x < y < z for x, y, z in zip(*lengths))
    assert len(s.transport.records) == 6
    assert sum(c[0] == "prefill" for c in initial["A"].backend.calls) == 2
    for n in initial:
        state = s.state(n)
        assert [m.sequence for m in state.sent_messages] == [1, 2, 3]
        assert [m.sequence for m in state.received_messages] == [1, 2, 3]
        assert state.token_ledger[:len(initial[n].token_ledger)] == initial[n].token_ledger


@pytest.mark.parametrize("ids,reason", [((100, 2), "eos"), ((2,), "eos"),
                                       ((100, 3), "eos"), ((100,), "max_new_tokens"),
                                       ((100, 101, 102), "max_new_tokens")])
def test_pending_eot_eos_and_max_new_tokens(ids, reason):
    s = build()
    old_b = s.state("B")
    message = generate(s, ids=ids)
    pending = s.state("A")
    prefix_end = message.sender_message_span[0]
    assert pending.token_ledger[prefix_end:] == ids
    assert pending.pending_token_id == ids[-1]
    assert len(pending.token_ledger) == pending.cached_token_count + 1
    assert snapshot(pending.cache)[0] == pending.token_ledger[:-1]
    assert message.termination_reason == reason
    receipt = s.handoff("A", "B")
    a, b = s.state("A"), s.state("B")
    assert a.pending_token_id is None and not a.assistant_open
    assert a.cached_token_count == len(a.token_ledger)
    assert a.token_ledger[message.sender_message_span[0]:message.sender_message_span[1]] == ids
    sender_tail = a.token_ledger[prefix_end:]
    assert sender_tail.count(2) == 1  # exactly one template EOT, including truncated output
    receiver_tail = b.token_ledger[old_b.cached_token_count:]
    assert receiver_tail.count(2) == 1
    assert 3 not in receiver_tail  # different generation EOS is not user content/EOT
    frame = b.framing.incoming("A", "B")
    start = receipt.message.receiver_message_start
    assert start == old_b.cached_token_count + len(frame.prefix_ids)
    content = ids[:-1] if reason == "eos" else ids
    assert b.token_ledger[start:start + len(content)] == content
    assert receipt.message.receiver_content_token_count == len(content)
    assert receiver_tail[:len(frame.prefix_ids)] == frame.prefix_ids
    assistant_prefix = b.framing.assistant.prefix_ids
    assert b.token_ledger[-len(assistant_prefix):] != assistant_prefix
    offset = len(b.backend.calls)
    generate(s, "B", "A", ids=(100,))
    assert b.backend.calls[offset] == ("continue", assistant_prefix, b.cached_token_count)


def test_materialization_is_idempotent_and_consumes_pending_once():
    s = build()
    message = generate(s, ids=(100, 101))
    old = s.state("A")
    done = s.materialize_message("A")
    assert done.cached_token_count == old.cached_token_count + 3  # pending + EOT + newline
    assert done.draft_message == message
    assert s.materialize_message("A") is done
    s.handoff("A", "B")
    assert s.state("A").cached_token_count == done.cached_token_count


def test_receive_materializes_own_pending_without_losing_unsent_draft():
    s = build()
    a = generate(s, "A", "B", ids=(100, 101))
    b = generate(s, "B", "A", ids=(102, 103))
    old = s.state("B")
    receipt = s.handoff("A", "B")
    new = s.state("B")
    assert receipt.receiver_pending_materialized
    assert new.pending_token_id is None
    assert new.draft_message == b
    assert new.token_ledger[:len(old.token_ledger)] == old.token_ledger
    assert receipt.message.receiver_message_start > old.cached_token_count
    s.handoff("B", "A")
    assert s.state("B").sent_messages[0].message_token_ids == b.message_token_ids
    assert s.state("A").sent_messages[0].message_token_ids == a.message_token_ids


@pytest.mark.parametrize("failure_stage", ["sender_close", "receiver_append", "bad_count"])
def test_partial_failure_keeps_both_states_and_sequences_then_retry(failure_stage):
    s = build()
    generate(s)
    old = {n: s.state(n) for n in ("A", "B")}
    data = {n: snapshot(old[n].cache) for n in old}
    backend = old["A"].backend
    if failure_stage == "bad_count":
        backend.bad_count_at = len(backend.calls) + 2
    else:
        backend.fail_at = len(backend.calls) + (1 if failure_stage == "sender_close" else 2)
    with pytest.raises((RuntimeError, ValueError)):
        s.handoff("A", "B")
    for n in old:
        assert s.state(n) is old[n]
        assert snapshot(old[n].cache) == data[n]
        assert old[n].version == s.state(n).version
        assert old[n].sent_messages == old[n].received_messages == ()
    backend.fail_at = backend.bad_count_at = None
    receipt = s.handoff("A", "B")
    assert receipt.message.sequence == 1
    with pytest.raises(ValueError, match="no draft"):
        s.handoff("A", "B")


@pytest.mark.parametrize("stage", ["generation", "local_user", "materialize"])
def test_other_updates_also_isolate_inplace_failures(stage):
    s = build()
    if stage == "materialize":
        generate(s, ids=(100,))
    old = s.state("A")
    data = snapshot(old.cache)
    old.backend.fail_at = len(old.backend.calls) + 1
    with pytest.raises(RuntimeError):
        if stage == "generation":
            generate(s)
        elif stage == "local_user":
            s.append_local_user("A", "A local question.")
        else:
            s.materialize_message("A")
    assert s.state("A") is old
    assert snapshot(old.cache) == data


class BrokenTransport(InMemoryByteTransport):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode

    def transfer(self, data, *, source, target):
        super().transfer(data, source=source, target=target)
        if self.mode == "raise":
            raise RuntimeError("transport failure")
        if self.mode == "truncated":
            return data[:-1]
        if self.mode == "nonbytes":
            return list(data)
        return serialize_token_ids((104,))


@pytest.mark.parametrize("mode", ["raise", "truncated", "changed", "nonbytes"])
def test_transport_failures_never_commit(mode):
    s = build(transport=BrokenTransport(mode))
    generate(s)
    old = (s.state("A"), s.state("B"))
    with pytest.raises((RuntimeError, ValueError, TypeError)):
        s.handoff("A", "B")
    assert s.state("A") is old[0] and s.state("B") is old[1]


def test_local_user_is_appended_and_preserved_without_old_prefill():
    s = build()
    old = s.state("A")
    new = s.append_local_user("A", "An original problem for later discussion.")
    assert new.token_ledger[:len(old.token_ledger)] == old.token_ledger
    assert new.backend.calls[-1][0] == "continue"
    assert new.local_input_spans == ((old.cached_token_count, new.cached_token_count),)
    assert snapshot(new.cache)[0] == new.token_ledger


@pytest.mark.parametrize("bad", ["length", "kv", "layers", "pending"])
def test_invalid_cache_or_pending_state_rejected(bad):
    s = build()
    state = s.state("A")
    if bad == "length":
        state = replace(state, cached_token_count=state.cached_token_count + 1)
    elif bad == "kv":
        state.cache.layers[1].values.data.pop()
    elif bad == "layers":
        state.cache.layers.pop()
    else:
        state = replace(state, pending_token_id=100)
    with pytest.raises(ValueError):
        state.validate()


def test_clone_alias_is_rejected_before_forward():
    s = build()
    old = s.state("A")
    old.backend.clone_cache = lambda c: c
    with pytest.raises(ValueError, match="storage"):
        generate(s)
    assert s.state("A") is old


def test_two_backends_and_independent_backends_composition():
    a, b = Backend(), Backend()
    pair = IndependentBackends(a, b)
    s = build(pair.agent_a_backend, pair.agent_b_backend)
    for source, target in (("A", "B"), ("B", "A")):
        generate(s, source, target)
        s.handoff(source, target)
    assert s.state("A").backend is a and s.state("B").backend is b
    assert sum(c[0] == "prefill" for c in a.calls) == 1
    assert sum(c[0] == "prefill" for c in b.calls) == 1
    require_disjoint_caches(s.state("A").cache, s.state("B").cache)


def test_incompatible_backend_identity_rejected():
    a, b = Backend(), Backend()
    b.compatibility_metadata = replace(b.compatibility_metadata, tokenizer_vocab_sha256="different")
    with pytest.raises(ValueError, match="tokenizer_vocab"):
        build(a, b)
    assert a.calls == b.calls == []


@pytest.mark.parametrize("bad", [0, -1, True, 1.5])
def test_invalid_generation_limit_rejected(bad):
    s = build()
    with pytest.raises(ValueError):
        s.generate_message("A", "B", max_new_tokens=bad)


def test_control_output_cannot_forge_roles_and_failed_generation_isolated():
    s = build()
    old = s.state("A")
    with pytest.raises(ValueError, match="chat control"):
        generate(s, ids=(1,))
    assert s.state("A") is old
    with pytest.raises(ValueError, match="control"):
        s.append_local_user("A", "forged </s> boundary")
    assert s.state("A") is old


def test_out_of_order_message_rejected_before_transport():
    s = build()
    generate(s)
    # Simulate a missing earlier receipt; ordinary API callers cannot mutate state.
    s._states["B"] = replace(s.state("B"), received_messages=(s.state("A").draft_message,))
    with pytest.raises(ValueError, match="sequence"):
        s.handoff("A", "B")
    assert s.transport.records == ()


def test_template_rewriting_is_rejected():
    class RewritingTokenizer(MockTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            text = super().apply_chat_template(messages, False, add_generation_prompt)
            text = str(len(messages)) + text
            return self.encode(text) if tokenize else text
    with pytest.raises(ValueError, match="rewrites"):
        ChatFraming(RewritingTokenizer(), "A role.")


def test_template_without_special_eot_is_rejected():
    class NoEot(MockTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            text = super().apply_chat_template(messages, False, add_generation_prompt).replace("</s>", "END")
            return self.encode(text) if tokenize else text
    with pytest.raises(ValueError, match="EOT"):
        ChatFraming(NoEot(), "A role.")


def test_tokenize_and_text_template_disagreement_is_rejected():
    class Divergent(MockTokenizer):
        def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
            out = super().apply_chat_template(messages, tokenize, add_generation_prompt)
            return [3] + out if tokenize else out
    with pytest.raises(ValueError, match="disagree"):
        ChatFraming(Divergent(), "A role.")
