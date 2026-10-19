"""Real tiny CPU Decode plus controlled predictions: no pretrained weights or oracle Prefill(M)."""
from dataclasses import asdict, replace

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers')

from test_stateful_token_cache import backend, TinyTokenizer, copies, unchanged
from token_kv_agent_system.discussion import DiscussionConfig, DiscussionRuntime
from token_kv_agent_system.incremental_kv import IncrementalKVChannel, prepare_incremental_message
from token_kv_agent_system.incremental_kv_protocol import inspect_model_contract
from token_kv_agent_system.incremental_kv_transport_probe import run_probe
from token_kv_agent_system.transport import InMemoryByteTransport


class ContractTokenizer(TinyTokenizer):
    def get_vocab(self): return {str(i): i for i in range(256)}
    def __len__(self): return 256


def make_backend():
    b = backend()
    b._tokenizer = ContractTokenizer()
    b._eos_token_ids = (2, 3)
    original = b.continue_from_cache
    b.outputs = []
    b.calls = []
    def tracked(ids, **kwargs):
        b.calls.append((tuple(ids), kwargs['logical_cached_token_count']))
        out = original(ids, **kwargs)
        if len(ids) == 1 or tuple(ids) == tuple(b._tokenizer.encode('<s>assistant\n')):
            return replace(out, predicted_next_token_id=b.outputs.pop(0) if b.outputs else 100)
        return out
    b.continue_from_cache = tracked
    return b


def make_runtime():
    b = make_backend()
    runtime = DiscussionRuntime(b, config=DiscussionConfig(rounds=2, max_new_tokens=2))
    channel = IncrementalKVChannel(b, b, InMemoryByteTransport())
    # Any later clean Prefill, including Prefill(M), is a hard failure.
    b.prefill = lambda **kw: pytest.fail('unexpected clean Prefill after role initialization')
    return b, runtime, channel


@pytest.mark.parametrize('raw,missing', [((100, 101, 2), 0), ((100, 101, 3), 0),
                                       ((100, 101), 1), ((100,), 1)])
def test_precise_decode_span_control_tokens_and_minimal_pending(raw, missing):
    b, runtime, channel = make_runtime()
    b.outputs = list(raw)
    message = runtime.session.generate_message('A', 'B', max_new_tokens=len(raw))
    states = {n: runtime.session.state(n) for n in ('A', 'B')}
    saved = {n: copies(states[n].cache) for n in states}
    before_calls = len(b.calls)
    decoded, diag = channel.send(states['A'], message)
    e = diag.export
    content = raw[:-1] if raw[-1] in (2, 3) else raw
    assert decoded.message_token_ids == content
    assert decoded.raw_generated_token_ids == raw
    assert e.supplemented_token_count == missing
    assert e.cached_message_token_count + missing == len(content)
    assert b.calls[before_calls:] == ([(content[-1:], states['A'].cached_token_count)] if missing else [])
    assert e.sender_span_start == message.sender_message_span[0]
    assert e.sender_span_length == len(content) == decoded.delta.logical_cached_token_count
    assert e.sender_span_length < states['A'].cached_token_count
    assert diag.bitwise_exact
    assert e.delta_raw_bytes == 2 * 2 * 1 * len(content) * 8 * 4
    for n in states:
        assert runtime.session.state(n) is states[n]
        unchanged(states[n].cache, saved[n])
    # Existing materialized positions are byte-for-byte those computed in Decode.
    start = e.sender_span_start
    cached = e.cached_message_token_count
    for layer, source in zip(decoded.delta.layers, states['A'].cache.layers):
        for neutral, tensor in ((layer.key, source.keys), (layer.value, source.values)):
            expected = tensor[..., start:start+cached, :].contiguous().view(torch.uint8).numpy().tobytes()
            assert neutral.data[:len(expected)] == expected
    assert channel.transport.records[0].byte_count == diag.envelope_total_bytes
    assert isinstance(channel.transport.records[0].byte_count, int)


def test_already_materialized_message_never_supplements_twice():
    b, runtime, channel = make_runtime()
    b.outputs = [100, 101]
    runtime.session.generate_message('A', 'B', max_new_tokens=2)
    receipt = runtime.session.handoff('A', 'B')
    state = runtime.session.state('A')
    before = len(b.calls)
    for _ in range(2):
        message, info = prepare_incremental_message(state, receipt.message, contract=channel.contract)
        assert info.supplemented_token_count == 0
    assert len(b.calls) == before


def test_four_bidirectional_spans_and_unchanged_discussion_output_path():
    b, runtime, channel = make_runtime()
    # Preserve baseline role states; TOKEN history/IDs/output must match exactly.
    _, baseline, _ = make_runtime()
    baseline_result = baseline.run()
    result, records = run_probe(runtime, channel)
    actual, expected = asdict(result), asdict(baseline_result)
    # Wall-clock instrumentation differs; protocol, state and outputs must not.
    for record in (actual, expected):
        for handoff in record['handoffs']:
            handoff.pop('observed_local_ms')
    assert actual == expected
    assert result.status == 'completed' and len(records) == 4
    assert [(r['source_agent'], r['sequence']) for r in records] == [('A', 1), ('B', 1), ('A', 2), ('B', 2)]
    assert len(channel.transport.records) == 4
    assert len(runtime.session.transport.records) == 4  # Still real TOKEN bytes.
    for source in ('A', 'B'):
        rows = [r for r in records if r['source_agent'] == source]
        spans = [r['diagnostics']['export']['sender_span_start'] for r in rows]
        assert spans[1] > spans[0]
        for row in rows:
            d = row['diagnostics']; e = d['export']
            assert row['receiver_content_ids_exact'] and row['kv_receiver_cache_appended_count'] == 0
            assert e['sender_span_length'] == 2 and e['delta_raw_bytes'] == 256
            assert e['supplemented_token_count'] == 0  # Reuse TOKEN close's cached pending.
            assert row['token_path_materialized_content_count'] == 1
            assert d['bitwise_exact'] and d['envelope_total_bytes'] > d['kv_protocol_wire_bytes']
    for n in ('A', 'B'):
        actual, reference = runtime.session.state(n), baseline.session.state(n)
        assert actual.token_ledger == reference.token_ledger
        for a, r in zip(actual.cache.layers, reference.cache.layers):
            assert torch.equal(a.keys, r.keys) and torch.equal(a.values, r.values)


def test_failed_pending_forward_does_not_mutate_either_agent():
    b, runtime, channel = make_runtime()
    runtime.session.generate_message('A', 'B', max_new_tokens=2)
    states = {n: runtime.session.state(n) for n in ('A', 'B')}
    saved = {n: copies(states[n].cache) for n in states}
    original = b.materialize_token_into_cache
    def fail(*args, **kwargs):
        original(*args, **kwargs)
        with torch.inference_mode():
            kwargs['previous_cache'].layers[0].keys.zero_()
        raise RuntimeError('partial pending failure')
    b.materialize_token_into_cache = fail
    with pytest.raises(RuntimeError, match='pending'):
        channel.send(states['A'], states['A'].draft_message)
    assert channel.receivers[('A', 'B')].next_sequence == 1
    for n in states:
        assert runtime.session.state(n) is states[n]
        unchanged(states[n].cache, saved[n])


def test_transport_corruption_duplicate_and_failure_isolation():
    b, runtime, channel = make_runtime()
    runtime.session.generate_message('A', 'B', max_new_tokens=2)
    state = runtime.session.state('A'); saved = copies(state.cache)
    class Corrupt(InMemoryByteTransport):
        def transfer(self, data, **kwargs):
            assert isinstance(data, bytes)
            return super().transfer(data, **kwargs)[:-1]
    channel.transport = Corrupt()
    with pytest.raises(ValueError, match='altered'):
        channel.send(state, state.draft_message)
    unchanged(state.cache, saved)
    assert channel.receivers[('A', 'B')].next_sequence == 1
    channel.transport = InMemoryByteTransport()
    channel.send(state, state.draft_message)
    before = len(b.calls)
    with pytest.raises(ValueError, match='sequence'):
        channel.send(state, state.draft_message)
    assert len(b.calls) == before  # Reject duplicates before even a private supplement.
    unchanged(state.cache, saved)


def test_empty_content_and_forged_ledger_fail_explicitly():
    b, runtime, channel = make_runtime()
    b.outputs = [2]
    message = runtime.session.generate_message('A', 'B', max_new_tokens=1)
    with pytest.raises(ValueError, match='empty'):
        channel.send(runtime.session.state('A'), message)
    with pytest.raises(ValueError, match='accepted'):
        channel.send(runtime.session.state('A'), replace(message, sender_message_span=(1, 2)))


def test_actual_weights_and_rope_config_are_checked_for_independent_backends():
    a, b = make_backend(), make_backend()
    assert inspect_model_contract(a) == inspect_model_contract(b)
    IncrementalKVChannel(a, b, InMemoryByteTransport())
    with torch.no_grad():
        next(b._model.parameters()).flatten()[0] += 1
    assert inspect_model_contract(a).weights_sha256 != inspect_model_contract(b).weights_sha256
    with pytest.raises(ValueError, match='incompatible'):
        IncrementalKVChannel(a, b, InMemoryByteTransport())
    b = make_backend(); b._model.config.rope_theta = 9999
    with pytest.raises(ValueError, match='incompatible'):
        IncrementalKVChannel(a, b, InMemoryByteTransport())


def test_pending_delta_matches_actual_supplemental_forward_at_every_position():
    b, runtime, channel = make_runtime()
    message = runtime.session.generate_message('A', 'B', max_new_tokens=2)
    snapshots = []
    original = b.materialize_token_into_cache
    def tracked(*args, **kwargs):
        out = original(*args, **kwargs)
        snapshots.append(copies(out.cache))
        return out
    b.materialize_token_into_cache = tracked
    decoded, info = channel.send(runtime.session.state('A'), message)
    assert len(snapshots) == 1
    start = decoded.sender_span_start
    end = start + decoded.sender_span_length
    for layer, (k, v) in zip(decoded.delta.layers, snapshots[0]):
        assert layer.key.data == k[..., start:end, :].contiguous().view(torch.uint8).numpy().tobytes()
        assert layer.value.data == v[..., start:end, :].contiguous().view(torch.uint8).numpy().tobytes()


def test_sender_size_bound_and_bad_cache_fail_before_supplement():
    from token_kv_agent_system.incremental_kv_protocol import IncrementalLimits
    b, runtime, channel = make_runtime()
    message = runtime.session.generate_message('A', 'B', max_new_tokens=2)
    before = len(b.calls)
    state = runtime.session.state('A')
    with pytest.raises(ValueError, match='size'):
        prepare_incremental_message(state, message, contract=channel.contract,
                                    limits=IncrementalLimits(max_message_tokens=1))
    assert len(b.calls) == before
    corrupt_cache = b.clone_cache(state.cache)
    corrupt_cache.layers[1].values = corrupt_cache.layers[1].values[..., :-1, :]
    with pytest.raises(ValueError):
        channel.send(replace(state, cache=corrupt_cache), message)
    assert len(b.calls) == before
