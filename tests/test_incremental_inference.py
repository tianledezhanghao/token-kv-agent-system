from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from token_kv_agent_system.model import GenerationConfig, HuggingFaceModelBackend
from token_kv_agent_system.model.huggingface import synchronize_device


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeVector:
    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def argmax(self, *, dim):
        assert dim == -1
        return FakeScalar(max(range(len(self.values)), key=self.values.__getitem__))


class FakeLogits:
    def __init__(self, sequence_length, predicted_token_id, vocab_size=8):
        self.shape = (1, sequence_length, vocab_size)
        self.vector = FakeVector(
            [10.0 if token_id == predicted_token_id else float(token_id) for token_id in range(vocab_size)]
        )

    def __getitem__(self, key):
        assert key == (0, -1, slice(None, None, None))
        return self.vector


class FakeTensor:
    def __init__(self, values):
        self.values = [list(row) for row in values]
        self.moves = []

    @property
    def shape(self):
        return (len(self.values), len(self.values[0]))

    def to(self, device):
        self.moves.append(device)
        return self

    def tolist(self):
        return self.values


class FakeCache:
    def __init__(self, count):
        self.count = count


class FakeTokenizer:
    vocab_size = 8
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = None

    def __init__(self, torch_module):
        self.torch = torch_module

    def encode(self, text):
        return [6, 7, 1]

    def decode(self, token_ids, *, skip_special_tokens=False):
        return ",".join(str(value) for value in token_ids if not skip_special_tokens or value != 2)

    def __call__(self, text, *, return_tensors):
        assert text and return_tensors == "pt"
        return {
            "input_ids": self.torch.tensor([[6, 7, 1]]),
            "attention_mask": self.torch.ones((1, 3)),
        }


class FakeIncrementalModel:
    config = SimpleNamespace(vocab_size=8, eos_token_id=2)
    generation_config = SimpleNamespace(eos_token_id=[2, 3])

    def __init__(self, events=None):
        self.calls = []
        self.created_caches = []
        self.events = events

    def to(self, device):
        return self

    def eval(self):
        pass

    def __call__(self, **kwargs):
        if self.events is not None:
            self.events.append("forward")
        input_ids = kwargs["input_ids"]
        previous_cache = kwargs.get("past_key_values")
        self.calls.append(kwargs)
        sequence_length = input_ids.shape[1]
        if previous_cache is None:
            cache = FakeCache(sequence_length)
            self.created_caches.append(cache)
            predicted = 4
        else:
            cache = previous_cache
            cache.count += sequence_length
            predicted = 2 if input_ids.values[0][0] == 5 else 5
        return SimpleNamespace(
            logits=FakeLogits(sequence_length, predicted),
            past_key_values=cache,
        )

    def generate(self, **kwargs):
        return FakeTensor([[6, 7, 1, 4, 5, 2]])


class FakeTorch:
    long = "long"

    def __init__(self, *, mps_available=False, events=None):
        self.cpu_sync_calls = 0
        self.mps_sync_calls = 0
        self.cuda_sync_calls = 0
        self.events = events
        self.cuda = SimpleNamespace(
            is_available=lambda: False,
            synchronize=self._cuda_synchronize,
        )
        self.mps = SimpleNamespace(synchronize=self._mps_synchronize)
        self.backends = SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps_available)
        )
        self.inference_mode = nullcontext

    def tensor(self, values, **kwargs):
        return FakeTensor(values)

    def ones(self, shape, **kwargs):
        return FakeTensor([[1] * shape[1] for _ in range(shape[0])])

    def ones_like(self, tensor):
        return self.ones(tensor.shape)

    def _mps_synchronize(self):
        self.mps_sync_calls += 1
        if self.events is not None:
            self.events.append("sync")

    def _cuda_synchronize(self, device=None):
        self.cuda_sync_calls += 1


def make_backend():
    torch_module = FakeTorch()
    model = FakeIncrementalModel()
    backend = HuggingFaceModelBackend(
        "fake/incremental",
        device="cpu",
        tokenizer=FakeTokenizer(torch_module),
        model=model,
        torch_module=torch_module,
    )
    return backend, model, torch_module


def test_prefill_forwards_full_prompt_once_and_uses_last_logits():
    backend, model, _ = make_backend()

    result = backend.prefill("prompt")

    assert len(model.calls) == 1
    assert model.calls[0]["input_ids"].shape == (1, 3)
    assert model.calls[0]["use_cache"] is True
    assert "past_key_values" not in model.calls[0]
    assert result.input_token_ids == (6, 7, 1)
    assert result.first_token_id == 4
    assert result.logical_cached_token_count == 3
    assert result.logits_shape == (1, 3, 8)
    assert result.last_token_logits_shape == (8,)
    assert result.cache.count == 3


def test_prefill_accepts_ids_without_text_round_trip():
    backend, model, _ = make_backend()

    result = backend.prefill(input_token_ids=[6, 7])

    assert model.calls[0]["input_ids"].values == [[6, 7]]
    assert result.input_token_ids == (6, 7)


def test_generate_from_token_ids_directly_prefills_ids_and_starts_clean_each_time():
    backend, model, _ = make_backend()
    backend._tokenizer.encode = lambda text: (_ for _ in ()).throw(
        AssertionError("encode must not be called")
    )
    backend._tokenizer.__call__ = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("tokenizer text path must not be called")
    )

    first = backend.generate_from_token_ids(
        [6, 7], GenerationConfig(max_new_tokens=3)
    )
    second = backend.generate_from_token_ids(
        [6, 7], GenerationConfig(max_new_tokens=1)
    )

    assert first.input_token_ids == (6, 7)
    assert first.prefill.input_token_ids == (6, 7)
    assert first.output_token_ids == (4, 5, 2)
    assert model.calls[0]["input_ids"].values == [[6, 7]]
    assert "past_key_values" not in model.calls[0]
    assert "past_key_values" not in model.calls[3]
    assert len(model.created_caches) == 2
    assert model.created_caches[0] is not model.created_caches[1]
    assert second.output_token_ids == (4,)


@pytest.mark.parametrize("token_ids", [[], [-1], [8], [True], [1.5]])
def test_generate_from_token_ids_rejects_empty_or_invalid_model_ids(token_ids):
    backend, _, _ = make_backend()

    with pytest.raises(ValueError):
        backend.generate_from_token_ids(token_ids, GenerationConfig(max_new_tokens=1))


def test_decode_inputs_one_token_reuses_cache_and_grows_logical_length():
    backend, model, _ = make_backend()
    prefill = backend.prefill("prompt")

    step = backend.decode_step(
        prefill.first_token_id,
        previous_cache=prefill.cache,
        logical_cached_token_count=prefill.logical_cached_token_count,
    )

    call = model.calls[-1]
    assert call["input_ids"].values == [[4]]
    assert call["input_ids"].shape == (1, 1)
    assert call["past_key_values"] is prefill.cache
    assert call["attention_mask"].shape == (1, 4)
    assert step.input_token_id == 4
    assert step.predicted_next_token_id == 5
    assert step.cache is prefill.cache
    assert step.logical_cached_token_count == 4


def test_manual_generation_one_token_performs_no_decode_forward():
    backend, model, _ = make_backend()

    result = backend.manual_generate(
        "prompt", GenerationConfig(max_new_tokens=1), warmup_runs=0
    )

    assert result.output_token_ids == (4,)
    assert result.decode_steps == ()
    assert len(model.calls) == 1
    assert result.stop_reason == "max_new_tokens"


def test_manual_generation_runs_at_most_n_minus_one_decode_forwards():
    backend, model, _ = make_backend()

    result = backend.manual_generate(
        "prompt", GenerationConfig(max_new_tokens=3), warmup_runs=0
    )

    assert result.output_token_ids == (4, 5, 2)
    assert len(result.decode_steps) == 2
    assert len(model.calls) == 3
    assert result.stop_reason == "eos"
    assert result.generated_text == "4,5"
    assert result.input_token_ids == (6, 7, 1)


def test_eos_predicted_by_prefill_stops_without_decode():
    backend, model, _ = make_backend()
    model.generation_config.eos_token_id = [4, 3]
    backend._eos_token_ids = backend._resolve_eos_token_ids()

    result = backend.manual_generate(
        "prompt", GenerationConfig(max_new_tokens=4), warmup_runs=0
    )

    assert result.output_token_ids == (4,)
    assert result.stop_reason == "eos"
    assert len(model.calls) == 1


def test_warmup_uses_independent_prefill_cache():
    backend, model, _ = make_backend()

    result = backend.manual_generate(
        "prompt", GenerationConfig(max_new_tokens=1), warmup_runs=1
    )

    assert len(model.calls) == 2
    assert len(model.created_caches) == 2
    assert result.prefill.cache is model.created_caches[1]
    assert result.prefill.cache is not model.created_caches[0]
    assert result.warmup_runs == 1


def test_cpu_synchronization_is_a_no_op_and_accelerators_are_supported():
    torch_module = FakeTorch()

    synchronize_device(torch_module, "cpu")
    assert torch_module.mps_sync_calls == 0
    assert torch_module.cuda_sync_calls == 0

    synchronize_device(torch_module, "mps")
    synchronize_device(torch_module, "cuda")
    assert torch_module.mps_sync_calls == 1
    assert torch_module.cuda_sync_calls == 1


def test_mps_forward_is_synchronized_immediately_before_and_after_model_call():
    events = []
    torch_module = FakeTorch(mps_available=True, events=events)
    model = FakeIncrementalModel(events=events)
    backend = HuggingFaceModelBackend(
        "fake/incremental",
        device="mps",
        tokenizer=FakeTokenizer(torch_module),
        model=model,
        torch_module=torch_module,
    )

    backend.prefill("prompt")

    assert events == ["sync", "forward", "sync"]
