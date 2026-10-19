"""No pretrained model, network, or accelerator required for 7.2A tests."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from token_kv_agent_system.experiments.incremental_kv import (
    MODES, ProbeTask, append_delta, cache_layers, clone_cache, complete_message,
    exact_answer, extract_span, greedy, rephase_key, render_receiver, rotate_half,
    run_receiver,
)


class FakeCache:
    def __init__(self, ddp_cache_data, config=None):
        self.layers = [SimpleNamespace(keys=k, values=v) for k, v in ddp_cache_data]

    def get_seq_length(self):
        return self.layers[0].keys.shape[2]


def make_cache(length=5, layers=3):
    return FakeCache(tuple((torch.full((1, 2, length, 8), float(i)),
                            torch.full((1, 2, length, 8), float(i + 10)))
                           for i in range(layers)))


@pytest.mark.parametrize("start,end", [(0, 1), (2, 5), (0, 5)])
def test_span_boundaries_and_layer_order(start, end):
    cache = make_cache()
    delta = extract_span(cache, start, end)
    assert len(delta) == 3
    for i, (k, v) in enumerate(delta):
        assert k.shape == (1, 2, end - start, 8)
        assert torch.all(k == i) and torch.all(v == i + 10)
        k.fill_(99)
    assert torch.all(cache.layers[0].keys == 0)


@pytest.mark.parametrize("start,end", [(-1, 3), (0, 6), (2, 2), (4, 3)])
def test_reject_missing_or_invalid_span(start, end):
    with pytest.raises(ValueError):
        extract_span(make_cache(), start, end)


def test_reject_incomplete_layers():
    cache = make_cache()
    cache.layers[1].keys = cache.layers[1].keys[..., :4, :]
    with pytest.raises(ValueError):
        cache_layers(cache)
    cache.layers[1].keys = None
    with pytest.raises(ValueError):
        cache_layers(cache)


class Materializer:
    def __init__(self):
        self.calls = []

    def materialize_token_into_cache(self, token, *, previous_cache, logical_cached_token_count):
        self.calls.append((token, logical_cached_token_count))
        result = make_cache(logical_cached_token_count + 1)
        return SimpleNamespace(cache=result, materialization_forward_ms=3.5)


@pytest.mark.parametrize("length,pending", [(6, 1), (7, 0)])
def test_final_token_completeness(length, pending):
    backend = Materializer()
    cache, info = complete_message(backend, make_cache(length), 5, (71, 72))
    assert cache.get_seq_length() == 7
    assert info["pending_token_count"] == pending
    assert backend.calls == ([(72, 6)] if pending else [])
    # Already materialized: do not duplicate final token.
    complete_message(backend, cache, 5, (71, 72))
    assert len(backend.calls) == pending


@pytest.mark.parametrize("length", [4, 5, 8])
def test_reject_more_than_one_missing_or_extra_tokens(length):
    with pytest.raises(ValueError):
        complete_message(Materializer(), make_cache(length), 5, (71, 72))


def test_materialization_wrong_growth_fails():
    backend = SimpleNamespace(materialize_token_into_cache=lambda *a, **kw:
                              SimpleNamespace(cache=make_cache(8), materialization_forward_ms=1))
    with pytest.raises(ValueError, match="exactly one"):
        complete_message(backend, make_cache(6), 5, (71, 72))


def test_append_preserves_order_length_and_storage_isolation():
    prefix, source = make_cache(5), make_cache(8)
    delta = extract_span(source, 5, 8)
    clones = [clone_cache(prefix, None, FakeCache) for _ in MODES]
    caches = [append_delta(c, delta, None, FakeCache) for c in clones]
    for c in caches:
        assert c.get_seq_length() == 8
        for i, (k, v) in enumerate(cache_layers(c)):
            assert torch.all(k == i) and torch.all(v == i + 10)
    caches[0].layers[1].keys.fill_(88)
    caches[0].layers[1].values.fill_(88)
    assert all(torch.all(c.layers[1].keys == 1) for c in caches[1:])
    assert torch.all(prefix.layers[1].keys == 1)
    assert torch.all(source.layers[1].keys == 1)
    assert torch.all(delta[1][0] == 1)
    clones[0].layers[0].keys.fill_(99)
    assert torch.all(clones[1].layers[0].keys == 0)


def test_append_rejects_wrong_layers_and_lengths():
    prefix = make_cache()
    delta = extract_span(make_cache(8), 5, 8)
    with pytest.raises(ValueError):
        append_delta(prefix, delta[:-1], None, FakeCache)
    uneven = delta[:1] + ((delta[1][0][..., :2, :], delta[1][1][..., :2, :]),) + delta[2:]
    with pytest.raises(ValueError):
        append_delta(prefix, uneven, None, FakeCache)


def tables(positions, dim=8, scale=1.0):
    frequencies = torch.tensor([1.0, 0.1, 0.01, 0.001])[:dim // 2]
    angles = torch.tensor(positions).float()[:, None] * frequencies
    angles = torch.cat((angles, angles), -1)[None]
    return angles.cos() * scale, angles.sin() * scale


def independent_rotation(x, positions, scale=1.0):
    # Separate scalar 2x2 rotation oracle; no call to rotate_half/rephase_key.
    import math
    out = x.clone()
    half = x.shape[-1] // 2
    for j, pos in enumerate(positions):
        for d, freq in enumerate([1.0, 0.1, 0.01, 0.001][:half]):
            c, s = math.cos(pos * freq) * scale, math.sin(pos * freq) * scale
            out[..., j, d] = x[..., j, d] * c - x[..., j, d + half] * s
            out[..., j, d + half] = x[..., j, d] * s + x[..., j, d + half] * c
    return out


@pytest.mark.parametrize("origin,target", [([3, 4, 5], [19, 20, 21]),
                                           ([19, 20, 21], [3, 4, 5]),
                                           ([3, 4, 5], [3, 4, 5])])
@pytest.mark.parametrize("scale", [1.0, 1.7])
def test_rope_against_independent_rotation_and_inverse(origin, target, scale):
    x = torch.randn((1, 2, 3, 8), generator=torch.Generator().manual_seed(7))
    original = independent_rotation(x, origin, scale)
    old, new = tables(origin, scale=scale), tables(target, scale=scale)
    actual = rephase_key(original, *old, *new)
    torch.testing.assert_close(actual, independent_rotation(x, target, scale), atol=1e-6, rtol=1e-6)
    back = rephase_key(actual, *new, *old)
    torch.testing.assert_close(back, original, atol=1e-6, rtol=1e-6)
    if scale == 1:
        torch.testing.assert_close(actual.norm(dim=-1), original.norm(dim=-1), atol=1e-6, rtol=1e-6)


def test_rope_partial_dim_and_no_mutation():
    key = torch.randn(1, 2, 3, 8)
    before = key.clone()
    old, new = tables([1, 2, 3], 4), tables([7, 8, 9], 4)
    changed = rephase_key(key, *old, *new)
    assert torch.equal(changed[..., 4:], key[..., 4:])
    assert torch.equal(key, before)


def test_rope_rejects_nonfinite_and_wrong_layout():
    key = torch.ones(1, 2, 3, 8)
    old, new = tables([1, 2, 3]), tables([4, 5, 6])
    with pytest.raises(ValueError):
        rephase_key(key, old[0][..., :7], old[1], *new)
    key.fill_(float("nan"))
    with pytest.raises(ValueError, match="nonfinite"):
        rephase_key(key, *old, *new)


class TraceBackend:
    device = "cpu"
    _torch = torch
    eos_token_ids = (9,)
    _model = SimpleNamespace(config=None)

    def __init__(self):
        self.calls = []

    def continue_from_cache(self, ids, *, previous_cache, logical_cached_token_count):
        self.calls.append(tuple(ids))
        cache = make_cache(logical_cached_token_count + len(ids))
        logits = torch.zeros(10)
        logits[9] = 2
        return SimpleNamespace(cache=cache, last_token_logits=logits,
                               forward_ms=0.2, predicted_next_token_id=9)

    def decode(self, ids, **kwargs):
        return "blue"


@pytest.mark.parametrize("mode", MODES)
def test_receiver_forwards_message_only_for_token_baseline(monkeypatch, mode):
    # Patch constructor only; use the real receiver orchestration/rotation/append.
    import token_kv_agent_system.experiments.incremental_kv as probe
    original_reconstruct = probe.reconstruct
    monkeypatch.setattr(probe, "reconstruct", lambda pairs, config, cache_class=None:
                        original_reconstruct(pairs, config, FakeCache))
    backend = TraceBackend()
    source, history = make_cache(8), make_cache(4)
    delta = extract_span(source, 5, 8)
    result, logits = run_receiver(backend, history, delta, (1, 2, 9), (4, 5), mode,
                                  tables([5, 6, 7]), tables([4, 5, 6]), 4, "blue")
    assert result["completed"] and result["correct"]
    assert result["cache_start"] == 4 and result["cache_after_message"] == 7
    assert result["cache_after_prompt"] == 9
    assert backend.calls == ([(1, 2, 9), (4, 5)] if mode == MODES[0] else [(4, 5)])
    assert history.get_seq_length() == 4 and source.get_seq_length() == 8


def test_greedy_eos_and_pending_prediction():
    backend = TraceBackend()
    logits = torch.zeros(10)
    logits[9] = 1
    ids, cache, trace = greedy(backend, SimpleNamespace(cache=make_cache(4), last_token_logits=logits), 4)
    assert ids == (9,) and cache.get_seq_length() == 4
    assert not backend.calls and trace["stop_reason"] == "eos"
    assert trace["final_prediction_cached"] is False


def test_scoring_is_not_fluency_or_substring():
    assert exact_answer(' 7392. ', '7392')
    assert not exact_answer('7392 or 1111', '7392')
    assert not exact_answer('The cup is red and blue.', 'blue')


def test_real_dynamic_cache_public_constructor_with_config():
    transformers = pytest.importorskip("transformers")
    config = transformers.Qwen2Config(num_hidden_layers=3, hidden_size=16,
                                      num_attention_heads=2, num_key_value_heads=2)
    prefix = clone_cache(make_cache(5), config)
    joined = append_delta(prefix, extract_span(make_cache(8), 5, 8), config)
    assert joined.get_seq_length() == 8
    assert len(joined.layers) == 3 and prefix.get_seq_length() == 5


def test_greedy_limit_has_exactly_one_pending_token():
    backend = TraceBackend()
    backend.eos_token_ids = ()
    logits = torch.zeros(10)
    logits[9] = 1
    ids, cache, trace = greedy(backend, SimpleNamespace(cache=make_cache(4), last_token_logits=logits), 3)
    assert ids == (9, 9, 9) and backend.calls == [(9,), (9,)]
    assert trace["cache_lengths"] == [4, 5, 6]
    assert cache.get_seq_length() == 4 + len(ids) - 1
    assert trace["stop_reason"] == "max_new_tokens"


class BoundaryTokenizer:
    all_special_ids = (9,)

    def apply_chat_template(self, messages, **kwargs):
        return ''.join('<' + m['role'] + '>' + m['content'] + '\x03\n' for m in messages) + '<assistant>'

    def encode(self, text, **kwargs):
        return [9 if c == '\x03' else ord(c) + 10 for c in text]

    def decode(self, ids):
        return ''.join('\x03' if i == 9 else chr(i - 10) for i in ids)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("message", [(7, 9), (7, 8)])
def test_receiver_boundary_without_duplicate_eos_or_message_leak(shared, message):
    tokenizer = BoundaryTokenizer()
    task = ProbeTask('test', 'secret 7392', 'Which code?', '7392')
    text, history, suffix_text, suffix = render_receiver(tokenizer, task, message, 'A history', shared)
    assert '7392' not in text and '7392' not in suffix_text
    assert 'Which code?' in suffix_text
    assert (suffix[0] == 9) == (message[-1] != 9)
    if shared:
        assert text == 'A history'


def test_receiver_boundary_mismatch_rejected():
    tokenizer = BoundaryTokenizer()
    tokenizer.all_special_ids = (8, 9)
    with pytest.raises(ValueError, match="boundary"):
        render_receiver(tokenizer, ProbeTask('t', 'report', 'Question?', 'answer'), (7, 8), 'history', False)


def test_receiver_reports_nan_and_exception(monkeypatch):
    import token_kv_agent_system.experiments.incremental_kv as probe
    original = probe.reconstruct
    monkeypatch.setattr(probe, 'reconstruct', lambda pairs, config, cache_class=None:
                        original(pairs, config, FakeCache))
    backend = TraceBackend()
    normal_forward = backend.continue_from_cache

    def bad_forward(*args, **kwargs):
        result = normal_forward(*args, **kwargs)
        result.last_token_logits.fill_(float('nan'))
        return result

    backend.continue_from_cache = bad_forward
    path, logits = run_receiver(backend, make_cache(4), extract_span(make_cache(8), 5, 8),
                                (1, 2, 9), (4, 5), MODES[1],
                                tables([5, 6, 7]), tables([4, 5, 6]), 4, 'blue')
    assert not path['completed'] and not path['correct'] and logits is None
    assert 'nonfinite' in path['exception']['message']


def test_qwen_rotary_dim64_positions_against_scalar_oracle():
    import math
    x = torch.randn(1, 2, 3, 64, generator=torch.Generator().manual_seed(23))
    old_positions, new_positions = [86, 87, 88], [66, 67, 68]

    def scalar_rotate(positions):
        result = x.clone()
        for j, pos in enumerate(positions):
            for d in range(32):
                freq = 1 / (1_000_000 ** (2 * d / 64))
                c, s = math.cos(pos * freq), math.sin(pos * freq)
                result[..., j, d] = x[..., j, d] * c - x[..., j, d + 32] * s
                result[..., j, d + 32] = x[..., j, d] * s + x[..., j, d + 32] * c
        return result

    def actual_tables(positions):
        freq = 1 / (1_000_000 ** (torch.arange(0, 64, 2).float() / 64))
        angles = torch.tensor(positions).float()[:, None] * freq
        angles = torch.cat((angles, angles), -1)[None]
        return angles.cos(), angles.sin()

    shifted = rephase_key(scalar_rotate(old_positions), *actual_tables(old_positions),
                          *actual_tables(new_positions))
    torch.testing.assert_close(shifted, scalar_rotate(new_positions), atol=2e-5, rtol=2e-5)
