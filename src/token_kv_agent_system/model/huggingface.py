"""HuggingFace Transformers implementation of the model backend contract."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace
from importlib import import_module
from time import perf_counter
from types import ModuleType
from typing import Any

from .base import IncrementalCausalLMBackend
from .cache_inspection import KVModelConfig, inspect_kv_model_config
from .config import GenerationConfig, validate_device_spec
from .types import (
    CachedContinuationResult,
    DecodeStepResult,
    GenerationResult,
    ManualGenerationResult,
    ModelMetadata,
    PrefillResult,
    TokenMaterializationResult,
    TokenIdGenerationResult,
)


def resolve_device(requested_device: str, torch_module: ModuleType | Any) -> str:
    """Resolve all hardware policy inside the model layer."""

    validate_device_spec(requested_device)
    if requested_device == "auto":
        if torch_module.cuda.is_available():
            return "cuda"
        mps = getattr(torch_module.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested_device.split(":")[0] == "cuda":
        if not torch_module.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if ":" in requested_device:
            index = int(requested_device.split(":")[1])
            if index >= torch_module.cuda.device_count():
                raise RuntimeError(f"CUDA device {requested_device} is not visible/available")
            # Query the requested device before model loading, never silently fall back.
            torch_module.cuda.get_device_properties(index)
    if requested_device == "mps":
        mps = getattr(torch_module.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
    return requested_device


def synchronize_device(torch_module: ModuleType | Any, device: str) -> None:
    """Wait for queued accelerator work; CPU execution is already synchronous."""

    device_type = str(device).split(":", 1)[0]
    if device_type == "mps":
        torch_module.mps.synchronize()
    elif device_type == "cuda":
        torch_module.cuda.synchronize(device)


def _normalize_single_sequence(value: Any) -> tuple[int, ...]:
    raw = value.tolist() if hasattr(value, "tolist") else value
    if isinstance(raw, tuple):
        raw = list(raw)
    if raw and isinstance(raw[0], (list, tuple)):
        if len(raw) != 1:
            raise ValueError("only batch size 1 is supported")
        raw = raw[0]
    return tuple(int(token_id) for token_id in raw)


class HuggingFaceModelBackend(IncrementalCausalLMBackend):
    """Load and run a standard decoder-only HuggingFace causal LM."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        tokenizer: Any | None = None,
        model: Any | None = None,
        torch_module: ModuleType | Any | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")
        if (tokenizer is None) != (model is None):
            raise ValueError("tokenizer and model must be injected together")

        torch_module = torch_module or self._import_optional_dependency("torch")
        self._torch = torch_module
        self._requested_device = device
        self._device_fallback_reason: str | None = None
        resolved_device = resolve_device(device, torch_module)

        load_started = perf_counter()
        if tokenizer is None:
            transformers = self._import_optional_dependency("transformers")
            tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
            model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
        self._tokenizer = tokenizer
        self._model = model
        self._device = self._move_model(resolved_device)
        if hasattr(self._model, "eval"):
            self._model.eval()
        load_wall_time_seconds = perf_counter() - load_started

        model_vocab_size = getattr(getattr(self._model, "config", None), "vocab_size", None)
        tokenizer_vocab_size = getattr(self._tokenizer, "vocab_size", None)
        vocab_size = model_vocab_size or tokenizer_vocab_size
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise ValueError("loaded model/tokenizer does not expose a valid vocab_size")
        self._metadata = ModelMetadata(
            model_name=model_name,
            device=self._device,
            vocab_size=vocab_size,
            eos_token_id=self._optional_int("eos_token_id"),
            bos_token_id=self._optional_int("bos_token_id"),
            pad_token_id=self._optional_int("pad_token_id"),
            load_wall_time_seconds=load_wall_time_seconds,
            device_fallback_reason=self._device_fallback_reason,
            dtype=(
                str(model_dtype)
                if (model_dtype := getattr(self._model, "dtype", None)) is not None
                else None
            ),
        )
        self._eos_token_ids = self._resolve_eos_token_ids()

    @staticmethod
    def _import_optional_dependency(name: str) -> ModuleType:
        try:
            return import_module(name)
        except ImportError as error:
            raise RuntimeError(
                "Local LLM support is not installed; run "
                "pip install -e '.[llm]'"
            ) from error

    def _move_model(self, device: str) -> str:
        try:
            self._model.to(device)
            return device
        except RuntimeError as error:
            if self._requested_device == "auto" and device == "mps":
                self._model.to("cpu")
                self._device_fallback_reason = f"MPS model loading failed: {error}"
                return "cpu"
            raise RuntimeError(f"failed to move model to requested device {device}") from error

    def _optional_int(self, attribute: str) -> int | None:
        value = getattr(self._tokenizer, attribute, None)
        return int(value) if value is not None else None

    def _resolve_eos_token_ids(self) -> tuple[int, ...]:
        candidates = (
            getattr(
                getattr(self._model, "generation_config", None),
                "eos_token_id",
                None,
            ),
            getattr(getattr(self._model, "config", None), "eos_token_id", None),
            getattr(self._tokenizer, "eos_token_id", None),
        )
        for value in candidates:
            if value is None:
                continue
            values = value if isinstance(value, (list, tuple, set)) else (value,)
            normalized = tuple(dict.fromkeys(int(token_id) for token_id in values))
            if normalized:
                return normalized
        return ()

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def eos_token_ids(self) -> tuple[int, ...]:
        return self._eos_token_ids

    @property
    def kv_model_config(self) -> KVModelConfig:
        """Cache-relevant model config with derived fields explicitly labeled."""

        return inspect_kv_model_config(self._model.config)

    @property
    def compatibility_metadata(self):
        """Compute identity from actual config/tokenizer; never infer a checkpoint."""
        from .backend_pair import inspect_backend_identity

        return inspect_backend_identity(self)

    def encode(self, text: str) -> tuple[int, ...]:
        return tuple(int(token_id) for token_id in self._tokenizer.encode(text))

    def decode(
        self, token_ids: tuple[int, ...] | list[int], *, skip_special_tokens: bool = False
    ) -> str:
        return self._tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=skip_special_tokens,
        )

    def generate(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> GenerationResult:
        config = config or GenerationConfig()
        encoded = self._tokenizer(prompt, return_tensors="pt")
        inputs = {name: value.to(self._device) for name, value in encoded.items()}
        input_token_ids = _normalize_single_sequence(inputs["input_ids"])

        generation_started = perf_counter()
        generated = self._generate_with_optional_mps_fallback(inputs, config)
        generation_wall_time_seconds = perf_counter() - generation_started
        full_token_ids = _normalize_single_sequence(generated)
        output_token_ids = full_token_ids[len(input_token_ids) :]
        generated_text = self.decode(output_token_ids, skip_special_tokens=True)
        return GenerationResult(
            prompt=prompt,
            input_token_ids=input_token_ids,
            generated_text=generated_text,
            output_token_ids=output_token_ids,
            generation_wall_time_seconds=generation_wall_time_seconds,
            full_token_ids=full_token_ids,
        )

    def prefill(
        self,
        prompt: str | None = None,
        *,
        input_token_ids: tuple[int, ...] | list[int] | None = None,
    ) -> PrefillResult:
        if (prompt is None) == (input_token_ids is None):
            raise ValueError("provide exactly one of prompt or input_token_ids")

        if prompt is not None:
            encoded = self._tokenizer(prompt, return_tensors="pt")
            inputs = {name: value.to(self._device) for name, value in encoded.items()}
            normalized_ids = _normalize_single_sequence(inputs["input_ids"])
        else:
            normalized_ids = self._validate_input_token_ids(input_token_ids or ())
            if not normalized_ids:
                raise ValueError("input_token_ids cannot be empty")
            input_tensor = self._torch.tensor(
                [normalized_ids], dtype=self._torch.long, device=self._device
            )
            inputs = {
                "input_ids": input_tensor,
                "attention_mask": self._torch.ones_like(input_tensor),
            }
        if not normalized_ids:
            raise ValueError("prompt must encode to at least one token")

        outputs, forward_ms = self._timed_forward(
            **inputs,
            use_cache=True,
        )
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("model forward did not return a cache with use_cache=True")
        logits = outputs.logits
        last_token_logits = logits[0, -1, :]
        first_token_id = int(last_token_logits.argmax(dim=-1).item())
        return PrefillResult(
            input_token_ids=normalized_ids,
            first_token_id=first_token_id,
            cache=cache,
            prefill_forward_ms=forward_ms,
            logical_cached_token_count=len(normalized_ids),
            logits_shape=tuple(int(size) for size in logits.shape),
            last_token_logits_shape=tuple(int(size) for size in last_token_logits.shape),
            last_token_logits=last_token_logits,
        )

    def decode_step(
        self,
        input_token_id: int,
        *,
        previous_cache: object,
        logical_cached_token_count: int,
    ) -> DecodeStepResult:
        if previous_cache is None:
            raise ValueError("previous_cache cannot be None")
        if logical_cached_token_count <= 0:
            raise ValueError("logical_cached_token_count must be greater than zero")

        continuation = self.continue_from_cache(
            (int(input_token_id),),
            previous_cache=previous_cache,
            logical_cached_token_count=logical_cached_token_count,
        )
        return DecodeStepResult(
            input_token_id=int(input_token_id),
            predicted_next_token_id=continuation.predicted_next_token_id,
            cache=continuation.cache,
            decode_forward_ms=continuation.forward_ms,
            logical_cached_token_count=continuation.logical_cached_token_count,
        )

    def restore_kv_cache(self, payload: Any) -> Any:
        """Restore a neutral v1 payload using this Receiver backend's metadata."""

        from .kv_adapter import restore_huggingface_kv_cache

        model_dtype = self.metadata.dtype
        if model_dtype is None:
            raise RuntimeError("Receiver model does not expose a dtype")
        return restore_huggingface_kv_cache(
            payload,
            model_id=self.model_name,
            model_config=self.kv_model_config,
            model_dtype=model_dtype,
            target_device=self.device,
            transformers_config=self._model.config,
            torch_module=self._torch,
        )

    @property
    def chat_tokenizer(self) -> Any:
        """Actual tokenizer used by the separate persistent-state path."""
        return self._tokenizer

    @property
    def context_window_tokens(self) -> int | None:
        """Configured position budget; no cache truncation is implied."""
        return getattr(self._model.config, "max_position_embeddings", None)

    def clone_cache(self, cache: Any) -> Any:
        from .state_cache import clone_state_cache

        return clone_state_cache(cache, self._model.config)

    def validate_cache(self, cache: Any, count: int) -> None:
        from .state_cache import validate_state_cache

        validate_state_cache(cache, count, self.kv_model_config.num_hidden_layers)

    def continue_from_cache(
        self,
        input_token_ids: tuple[int, ...] | list[int],
        *,
        previous_cache: object,
        logical_cached_token_count: int,
    ) -> CachedContinuationResult:
        """Append one or more exact Token IDs to inherited KV working memory."""

        if previous_cache is None:
            raise ValueError("previous_cache cannot be None")
        if logical_cached_token_count <= 0:
            raise ValueError("logical_cached_token_count must be greater than zero")
        normalized = self._validate_input_token_ids(input_token_ids)
        if not normalized:
            raise ValueError("input_token_ids cannot be empty")
        get_seq_length = getattr(previous_cache, "get_seq_length", None)
        runtime_length = int(get_seq_length()) if callable(get_seq_length) else None
        if runtime_length is not None and runtime_length != logical_cached_token_count:
            raise ValueError(
                f"cache length {runtime_length} does not match logical length "
                f"{logical_cached_token_count}"
            )
        input_ids = self._torch.tensor(
            [normalized], dtype=self._torch.long, device=self._device
        )
        # Standard causal cache continuation uses inherited length and a full mask.
        # Actual model/version position semantics must pass the server smoke gate.
        attention_mask = self._torch.ones(
            (1, logical_cached_token_count + len(normalized)),
            dtype=self._torch.long,
            device=self._device,
        )
        outputs, forward_ms = self._timed_forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=previous_cache,
            use_cache=True,
        )
        cache = outputs.past_key_values
        if cache is None:
            raise RuntimeError("cached model forward did not return an updated cache")
        expected_length = logical_cached_token_count + len(normalized)
        updated_get_seq_length = getattr(cache, "get_seq_length", None)
        if (
            callable(updated_get_seq_length)
            and int(updated_get_seq_length()) != expected_length
        ):
            raise RuntimeError("cached model forward produced an unexpected cache length")
        last_logits = outputs.logits[0, -1, :]
        return CachedContinuationResult(
            input_token_ids=normalized,
            predicted_next_token_id=int(last_logits.argmax(dim=-1).item()),
            cache=cache,
            forward_ms=forward_ms,
            logical_cached_token_count=expected_length,
            last_token_logits=last_logits,
        )

    def materialize_token_into_cache(
        self,
        token_id: int,
        *,
        previous_cache: object,
        logical_cached_token_count: int,
    ) -> TokenMaterializationResult:
        """Consume one pending prediction so its K/V becomes part of the cache."""

        result = self.continue_from_cache(
            (int(token_id),),
            previous_cache=previous_cache,
            logical_cached_token_count=logical_cached_token_count,
        )
        return TokenMaterializationResult(
            materialized_token_id=int(token_id),
            predicted_next_token_id=result.predicted_next_token_id,
            cache=result.cache,
            logical_cached_token_count=result.logical_cached_token_count,
            materialization_forward_ms=result.forward_ms,
            last_token_logits=result.last_token_logits,
        )

    def manual_generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        warmup_runs: int = 1,
    ) -> ManualGenerationResult:
        config = config or GenerationConfig()
        if config.do_sample:
            raise ValueError(
                "manual generation supports deterministic greedy decoding only"
            )
        if config.repetition_penalty not in (None, 1.0):
            raise ValueError(
                "manual generation uses raw-logit argmax; repetition_penalty must be 1.0"
            )
        if warmup_runs < 0:
            raise ValueError("warmup_runs cannot be negative")

        for _ in range(warmup_runs):
            self._manual_generate_once(prompt, config, warmup_runs=0)
        return self._manual_generate_once(prompt, config, warmup_runs=warmup_runs)

    def generate_from_token_ids(
        self,
        input_token_ids: tuple[int, ...] | list[int],
        config: GenerationConfig | None = None,
    ) -> TokenIdGenerationResult:
        """Clean-Prefill exact IDs and continue causally without re-tokenization."""

        config = self._validate_manual_config(config or GenerationConfig())
        prefill = self.prefill(input_token_ids=input_token_ids)
        output_ids, decode_steps, stop_reason = self._decode_from_prefill(prefill, config)
        return TokenIdGenerationResult(
            input_token_ids=prefill.input_token_ids,
            output_token_ids=output_ids,
            generated_text=self.decode(output_ids, skip_special_tokens=True),
            prefill=prefill,
            decode_steps=decode_steps,
            stop_reason=stop_reason,
        )

    def _manual_generate_once(
        self,
        prompt: str,
        config: GenerationConfig,
        *,
        warmup_runs: int,
    ) -> ManualGenerationResult:
        self._validate_manual_config(config)
        prefill = self.prefill(prompt)
        output_token_ids, decode_steps, stop_reason = self._decode_from_prefill(
            prefill, config
        )
        return ManualGenerationResult(
            prompt=prompt,
            input_token_ids=prefill.input_token_ids,
            output_token_ids=output_token_ids,
            generated_text=self.decode(output_token_ids, skip_special_tokens=True),
            prefill=prefill,
            decode_steps=decode_steps,
            stop_reason=stop_reason,
            warmup_runs=warmup_runs,
        )

    def _decode_from_prefill(
        self, prefill: PrefillResult, config: GenerationConfig
    ) -> tuple[tuple[int, ...], tuple[DecodeStepResult, ...], str]:
        output_token_ids = [prefill.first_token_id]
        decode_steps: list[DecodeStepResult] = []
        stop_reason = (
            "eos"
            if prefill.first_token_id in self.eos_token_ids
            else "max_new_tokens"
        )
        cache = prefill.cache
        cached_count = prefill.logical_cached_token_count

        while stop_reason != "eos" and len(output_token_ids) < config.max_new_tokens:
            step = self.decode_step(
                output_token_ids[-1],
                previous_cache=cache,
                logical_cached_token_count=cached_count,
            )
            decode_steps.append(step)
            cache = step.cache
            cached_count = step.logical_cached_token_count
            output_token_ids.append(step.predicted_next_token_id)
            if step.predicted_next_token_id in self.eos_token_ids:
                stop_reason = "eos"

        return (
            tuple(output_token_ids),
            tuple(decode_steps),
            stop_reason,
        )

    def _validate_manual_config(self, config: GenerationConfig) -> GenerationConfig:
        if config.do_sample:
            raise ValueError(
                "manual generation supports deterministic greedy decoding only"
            )
        if config.repetition_penalty not in (None, 1.0):
            raise ValueError(
                "manual generation uses raw-logit argmax; repetition_penalty must be 1.0"
            )
        return config

    def _validate_input_token_ids(
        self, input_token_ids: tuple[int, ...] | list[int]
    ) -> tuple[int, ...]:
        normalized: list[int] = []
        for index, token_id in enumerate(input_token_ids):
            if not isinstance(token_id, int) or isinstance(token_id, bool):
                raise ValueError(f"input Token ID at index {index} must be an integer")
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"input Token ID at index {index} is outside model vocabulary "
                    f"[0, {self.vocab_size})"
                )
            normalized.append(token_id)
        return tuple(normalized)

    def _timed_forward(self, **model_inputs: Any) -> tuple[Any, float]:
        inference_mode = getattr(self._torch, "inference_mode", nullcontext)
        try:
            with inference_mode():
                synchronize_device(self._torch, self._device)
                started = perf_counter()
                outputs = self._model(**model_inputs)
                synchronize_device(self._torch, self._device)
                elapsed_ms = (perf_counter() - started) * 1000.0
        except RuntimeError as error:
            raise RuntimeError(f"model forward failed on device {self._device}") from error
        return outputs, elapsed_ms

    def _generate_with_optional_mps_fallback(
        self, inputs: dict[str, Any], config: GenerationConfig
    ) -> Any:
        inference_mode = getattr(self._torch, "inference_mode", nullcontext)
        try:
            with inference_mode():
                return self._model.generate(**inputs, **config.as_generate_kwargs())
        except RuntimeError as error:
            if self._requested_device != "auto" or self._device != "mps":
                raise RuntimeError(f"generation failed on device {self._device}") from error
            self._model.to("cpu")
            self._device = "cpu"
            self._device_fallback_reason = f"MPS generation failed: {error}"
            self._metadata = replace(
                self._metadata,
                device="cpu",
                device_fallback_reason=self._device_fallback_reason,
            )
            cpu_inputs = {name: value.to("cpu") for name, value in inputs.items()}
            with inference_mode():
                return self._model.generate(
                    **cpu_inputs, **config.as_generate_kwargs()
                )
