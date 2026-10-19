"""Real dense DynamicCache on random tiny Qwen2/CPU, without checkpoint weights."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.model import HuggingFaceModelBackend
from token_kv_agent_system.model.state_cache import (
    clone_state_cache, require_disjoint_caches, validate_state_cache)
from token_kv_agent_system.stateful_token import StatefulTokenSession
from token_kv_agent_system.transport import InMemoryByteTransport
from test_stateful_token import MockTokenizer


class TinyTokenizer(MockTokenizer):
    eos_token_id = 2
    bos_token_id = None
    pad_token_id = 3
    vocab_size = 256

    def decode(self, token_ids, **kwargs):
        return " ".join(map(str, token_ids))


def backend():
    torch.manual_seed(75)
    config = transformers.Qwen2Config(
        vocab_size=256, hidden_size=16, intermediate_size=32,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=8, max_position_embeddings=2048, eos_token_id=2)
    return HuggingFaceModelBackend("random/tiny-qwen2", device="cpu",
                                  tokenizer=TinyTokenizer(),
                                  model=transformers.Qwen2ForCausalLM(config).eval(),
                                  torch_module=torch)


def copies(cache):
    return [(layer.keys.clone(), layer.values.clone()) for layer in cache.layers]


def unchanged(cache, saved):
    for layer, (key, value) in zip(cache.layers, saved):
        assert torch.equal(layer.keys, key)
        assert torch.equal(layer.values, value)


def test_cpu_dynamiccache_clone_and_incremental_forward_match_clean_oracle():
    b = backend()
    initial = b.prefill(input_token_ids=(30, 31, 32))
    saved = copies(initial.cache)
    work = b.clone_cache(initial.cache)
    require_disjoint_caches(initial.cache, work)
    out = b.continue_from_cache((33, 34), previous_cache=work, logical_cached_token_count=3)
    full = b.prefill(input_token_ids=(30, 31, 32, 33, 34))
    assert torch.allclose(out.last_token_logits, full.last_token_logits, atol=1e-6, rtol=1e-5)
    validate_state_cache(out.cache, 5, 2)
    unchanged(initial.cache, saved)


def test_cpu_actual_session_three_rounds_and_full_ledger_oracle():
    b = backend()
    s = StatefulTokenSession([
        LLMAgent("A", b, system_prompt="You are Agent A."),
        LLMAgent("B", b, system_prompt="You are Agent B.")], InMemoryByteTransport())
    prefill_calls = []
    original = b.prefill
    def tracked(*args, **kwargs):
        prefill_calls.append(kwargs.get("input_token_ids"))
        return original(*args, **kwargs)
    b.prefill = tracked
    for _ in range(3):
        for source, target in (("A", "B"), ("B", "A")):
            old = {n: s.state(n) for n in ("A", "B")}
            saved = {n: copies(old[n].cache) for n in old}
            s.generate_message(source, target, max_new_tokens=2)
            s.handoff(source, target)
            for n in old:
                s.state(n).validate()
                assert s.state(n).cached_token_count > old[n].cached_token_count
                unchanged(old[n].cache, saved[n])
            require_disjoint_caches(s.state("A").cache, s.state("B").cache)
    assert prefill_calls == []  # all six receives and generations are incremental
    for n in ("A", "B"):
        state = s.state(n)
        # Full Prefill is an independent TEST oracle only.
        full = original(input_token_ids=state.token_ledger)
        for layer, reference in zip(state.cache.layers, full.cache.layers):
            assert torch.allclose(layer.keys, reference.keys, atol=1e-6, rtol=1e-5)
            assert torch.allclose(layer.values, reference.values, atol=1e-6, rtol=1e-5)


def test_cpu_partial_dynamiccache_failure_preserves_effective_state_and_retry():
    b = backend()
    s = StatefulTokenSession([
        LLMAgent("A", b, system_prompt="You are Agent A."),
        LLMAgent("B", b, system_prompt="You are Agent B.")], InMemoryByteTransport())
    s.generate_message("A", "B", max_new_tokens=2)
    old = {n: s.state(n) for n in ("A", "B")}
    saved = {n: copies(old[n].cache) for n in old}
    original = b._model.forward
    calls = 0
    def failing(**kwargs):
        nonlocal calls
        calls += 1
        out = original(**kwargs)
        if calls == 2:  # receiver grew before the simulated exception
            out.past_key_values.layers[0].keys.zero_()
            raise RuntimeError("after in-place DynamicCache update")
        return out
    b._model.forward = failing
    with pytest.raises(RuntimeError, match="model forward failed") as error:
        s.handoff("A", "B")
    assert "in-place" in str(error.value.__cause__)
    for n in old:
        assert s.state(n) is old[n]
        unchanged(old[n].cache, saved[n])
    b._model.forward = original
    s.handoff("A", "B")
    assert s.state("B").received_messages[0].sequence == 1


def test_cpu_rejects_non_dense_cache_and_incomplete_layer():
    b = backend()
    initial = b.prefill(input_token_ids=(30, 31, 32))
    class UnknownCache(type(initial.cache)):
        pass
    unknown = UnknownCache(ddp_cache_data=tuple(copies(initial.cache)), config=b._model.config)
    with pytest.raises(ValueError, match="dense"):
        clone_state_cache(unknown, b._model.config)
    initial.cache.layers[1].values = initial.cache.layers[1].values[..., :-1, :]
    with pytest.raises(ValueError, match="inconsistent"):
        b.validate_cache(initial.cache, 3)
