"""Stable language-model backend contract."""

from __future__ import annotations

from abc import ABC, abstractmethod

from .config import GenerationConfig
from .types import (
    DecodeStepResult,
    GenerationResult,
    ManualGenerationResult,
    ModelMetadata,
    PrefillResult,
    TokenIdGenerationResult,
)


class BaseModelBackend(ABC):
    @property
    @abstractmethod
    def metadata(self) -> ModelMetadata:
        """Loaded model, tokenizer, device, vocabulary, and timing metadata."""

    @property
    def model_name(self) -> str:
        return self.metadata.model_name

    @property
    def device(self) -> str:
        return self.metadata.device

    @property
    def vocab_size(self) -> int:
        return self.metadata.vocab_size

    @property
    def eos_token_id(self) -> int | None:
        return self.metadata.eos_token_id

    @property
    def bos_token_id(self) -> int | None:
        return self.metadata.bos_token_id

    @property
    def pad_token_id(self) -> int | None:
        return self.metadata.pad_token_id

    @abstractmethod
    def encode(self, text: str) -> tuple[int, ...]:
        """Encode plain text into normalized Python integer token IDs."""

    @abstractmethod
    def decode(
        self, token_ids: tuple[int, ...] | list[int], *, skip_special_tokens: bool = False
    ) -> str:
        """Decode token IDs into text."""

    @abstractmethod
    def generate(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> GenerationResult:
        """Run causal generation and return only newly generated output IDs."""


class IncrementalCausalLMBackend(BaseModelBackend):
    """Capability contract for backends with real cached incremental decoding."""

    @property
    @abstractmethod
    def eos_token_ids(self) -> tuple[int, ...]:
        """All token IDs that terminate generation for the loaded model."""

    @abstractmethod
    def prefill(
        self,
        prompt: str | None = None,
        *,
        input_token_ids: tuple[int, ...] | list[int] | None = None,
    ) -> PrefillResult:
        """Forward one complete prompt and predict the first new token."""

    @abstractmethod
    def decode_step(
        self,
        input_token_id: int,
        *,
        previous_cache: object,
        logical_cached_token_count: int,
    ) -> DecodeStepResult:
        """Forward one new token with a previous opaque cache."""

    @abstractmethod
    def manual_generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        *,
        warmup_runs: int = 1,
    ) -> ManualGenerationResult:
        """Generate using explicit prefill and cached one-token decode calls."""

    @abstractmethod
    def generate_from_token_ids(
        self,
        input_token_ids: tuple[int, ...] | list[int],
        config: GenerationConfig | None = None,
    ) -> TokenIdGenerationResult:
        """Start from a clean Prefill of exact Token IDs, then cached Decode."""
