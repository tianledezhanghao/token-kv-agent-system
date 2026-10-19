"""Logical Agent wrapper around a reusable model backend."""

from __future__ import annotations

from token_kv_agent_system.model import (
    BaseModelBackend,
    GenerationConfig,
    GenerationResult,
    IncrementalCausalLMBackend,
    TokenIdGenerationResult,
)


class LLMAgent:
    """An Agent identity and prompt policy, independent of model ownership.

    This intentionally does not implement the Phase 1 ``Agent.generate``
    communication contract because that contract requires a KV payload. Phase
    3.1 has no real KV representation and does not connect this class to the
    communication orchestrator.
    """

    def __init__(
        self,
        agent_id: str,
        backend: BaseModelBackend,
        *,
        system_prompt: str | None = None,
    ) -> None:
        if not agent_id.strip():
            raise ValueError("agent_id cannot be empty")
        self._agent_id = agent_id
        self.backend = backend
        self.system_prompt = system_prompt

    @property
    def agent_id(self) -> str:
        return self._agent_id

    def generate_reply(
        self, prompt: str, config: GenerationConfig | None = None
    ) -> GenerationResult:
        effective_prompt = prompt
        if self.system_prompt:
            effective_prompt = f"{self.system_prompt}\n\n{prompt}"
        return self.backend.generate(effective_prompt, config)

    def generate_from_token_ids(
        self,
        input_token_ids: tuple[int, ...] | list[int],
        config: GenerationConfig | None = None,
    ) -> TokenIdGenerationResult:
        """Continue exact received IDs through a clean receiver-side Prefill."""

        if not isinstance(self.backend, IncrementalCausalLMBackend):
            raise TypeError("backend does not support incremental Token-ID inference")
        return self.backend.generate_from_token_ids(input_token_ids, config)
