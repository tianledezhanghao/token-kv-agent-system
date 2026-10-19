"""Small, dependency-free configuration objects for local LLM inference."""

from __future__ import annotations

from dataclasses import dataclass
import re


DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


def validate_device_spec(device: str) -> str:
    """Validate syntax without importing torch or probing hardware."""
    if device not in {"auto", "cpu", "mps", "cuda"} and not (
        isinstance(device, str) and re.fullmatch(r"cuda:(0|[1-9][0-9]*)", device)
    ):
        raise ValueError("device must be auto, cpu, mps, cuda, or cuda:<nonnegative index>")
    return device


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 32
    do_sample: bool = False
    temperature: float | None = None
    top_p: float | None = None
    repetition_penalty: float | None = None

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
        if self.temperature is not None and self.temperature <= 0:
            raise ValueError("temperature must be greater than zero")
        if self.top_p is not None and not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in the interval (0, 1]")
        if self.repetition_penalty is not None and self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be greater than zero")

    def as_generate_kwargs(self) -> dict[str, int | float | bool]:
        kwargs: dict[str, int | float | bool] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.do_sample,
        }
        if self.do_sample:
            if self.temperature is not None:
                kwargs["temperature"] = self.temperature
            if self.top_p is not None:
                kwargs["top_p"] = self.top_p
        if self.repetition_penalty is not None:
            kwargs["repetition_penalty"] = self.repetition_penalty
        return kwargs


@dataclass(frozen=True)
class LLMConfig:
    model_name: str = DEFAULT_MODEL_NAME
    device: str = "auto"
    max_new_tokens: int = 32

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ValueError("model_name cannot be empty")
        validate_device_spec(self.device)
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be greater than zero")
