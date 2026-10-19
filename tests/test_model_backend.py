from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.model import (
    BaseModelBackend,
    GenerationConfig,
    GenerationResult,
    HuggingFaceModelBackend,
    ModelMetadata,
)
from token_kv_agent_system.model.huggingface import resolve_device


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self

    def tolist(self):
        return self.values


class FakeTokenizer:
    vocab_size = 1001
    eos_token_id = 2
    bos_token_id = 1
    pad_token_id = None

    def __init__(self):
        self.decode_calls = []

    def encode(self, text):
        assert text
        return ["10", 20, 30]

    def decode(self, token_ids, *, skip_special_tokens=False):
        self.decode_calls.append((token_ids, skip_special_tokens))
        return "decoded:" + ",".join(str(token_id) for token_id in token_ids)

    def __call__(self, text, *, return_tensors):
        assert text and return_tensors == "pt"
        return {
            "input_ids": FakeTensor([[10, 20, 30]]),
            "attention_mask": FakeTensor([[1, 1, 1]]),
        }


class FakeModel:
    config = SimpleNamespace(vocab_size=1001)

    def __init__(self):
        self.device_moves = []
        self.generate_kwargs = None
        self.eval_called = False

    def to(self, device):
        self.device_moves.append(device)
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return FakeTensor([[10, 20, 30, 40, 41]])


class MPSFailOnceModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.generate_calls = 0

    def generate(self, **kwargs):
        self.generate_calls += 1
        if self.generate_calls == 1:
            raise RuntimeError("unsupported MPS operation")
        return super().generate(**kwargs)


def fake_torch(*, cuda=False, mps=False):
    return SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda),
        backends=SimpleNamespace(
            mps=SimpleNamespace(is_available=lambda: mps)
        ),
        inference_mode=nullcontext,
    )


class MinimalBackend(BaseModelBackend):
    @property
    def metadata(self):
        return ModelMetadata("fake", "cpu", 8, None, None, None, 0.0)

    def encode(self, text):
        return (1,)

    def decode(self, token_ids, *, skip_special_tokens=False):
        return "ok"

    def generate(self, prompt, config=None):
        return GenerationResult(prompt, (1,), "ok", (2,), 0.0, (1, 2))


def make_backend():
    tokenizer = FakeTokenizer()
    model = FakeModel()
    backend = HuggingFaceModelBackend(
        "fake/model",
        device="cpu",
        tokenizer=tokenizer,
        model=model,
        torch_module=fake_torch(),
    )
    return backend, tokenizer, model


def test_model_backend_abstraction_can_be_implemented():
    assert MinimalBackend().generate("prompt").output_token_ids == (2,)


def test_huggingface_encode_decode_and_metadata_use_normalized_ids():
    backend, tokenizer, model = make_backend()

    assert backend.encode("hello") == (10, 20, 30)
    assert backend.decode((10, 20)) == "decoded:10,20"
    assert tokenizer.decode_calls[-1] == ([10, 20], False)
    assert backend.metadata.vocab_size == 1001
    assert backend.vocab_size == 1001
    assert backend.metadata.eos_token_id == 2
    assert backend.metadata.bos_token_id == 1
    assert backend.metadata.pad_token_id is None
    assert backend.metadata.device == "cpu"
    assert model.eval_called


def test_generation_splits_new_tokens_and_omits_sampling_only_options():
    backend, tokenizer, model = make_backend()

    result = backend.generate(
        "hello",
        GenerationConfig(
            max_new_tokens=2,
            do_sample=False,
            temperature=0.7,
            top_p=0.9,
        ),
    )

    assert result.input_token_ids == (10, 20, 30)
    assert result.output_token_ids == (40, 41)
    assert result.full_token_ids == (10, 20, 30, 40, 41)
    assert result.generated_text == "decoded:40,41"
    assert result.input_token_count == 3
    assert result.output_token_count == 2
    assert model.generate_kwargs["max_new_tokens"] == 2
    assert model.generate_kwargs["do_sample"] is False
    assert "temperature" not in model.generate_kwargs
    assert "top_p" not in model.generate_kwargs
    assert tokenizer.decode_calls[-1] == ([40, 41], True)


def test_two_llm_agents_share_one_backend_instance():
    backend = MinimalBackend()
    agent_a = LLMAgent("Agent A", backend)
    agent_b = LLMAgent("Agent B", backend)

    assert agent_a.backend is agent_b.backend
    assert agent_a.generate_reply("hello").generated_text == "ok"


@pytest.mark.parametrize(
    ("cuda", "mps", "expected"),
    [(True, True, "cuda"), (False, True, "mps"), (False, False, "cpu")],
)
def test_auto_device_policy_is_centralized_in_model_layer(cuda, mps, expected):
    assert resolve_device("auto", fake_torch(cuda=cuda, mps=mps)) == expected


def test_unavailable_explicit_device_has_clear_error():
    with pytest.raises(RuntimeError, match="MPS was requested"):
        resolve_device("mps", fake_torch())


def test_auto_device_falls_back_to_cpu_if_mps_generation_fails():
    model = MPSFailOnceModel()
    backend = HuggingFaceModelBackend(
        "fake/model",
        device="auto",
        tokenizer=FakeTokenizer(),
        model=model,
        torch_module=fake_torch(mps=True),
    )

    result = backend.generate("hello", GenerationConfig(max_new_tokens=2))

    assert result.output_token_ids == (40, 41)
    assert backend.device == "cpu"
    assert backend.metadata.device_fallback_reason == (
        "MPS generation failed: unsupported MPS operation"
    )
    assert model.device_moves == ["mps", "cpu"]
