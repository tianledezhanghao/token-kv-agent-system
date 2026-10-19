"""Manual local Tiny LLM integration demo."""

from __future__ import annotations

import argparse

from token_kv_agent_system.agent import LLMAgent
from token_kv_agent_system.model import (
    DEFAULT_MODEL_NAME,
    GenerationConfig,
    HuggingFaceModelBackend,
    LLMConfig,
)
from token_kv_agent_system.models import TokenPayload


DEFAULT_PROMPT = "Explain KV cache in one short sentence."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a real local Tiny LLM")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def run_llm_demo(config: LLMConfig, *, prompt: str = DEFAULT_PROMPT) -> None:
    backend = HuggingFaceModelBackend(config.model_name, device=config.device)
    agent = LLMAgent("Local LLM Agent", backend)
    result = agent.generate_reply(
        prompt, GenerationConfig(max_new_tokens=config.max_new_tokens)
    )
    metadata = backend.metadata
    input_payload = TokenPayload.from_token_ids(
        result.input_token_ids, vocab_size=metadata.vocab_size
    )
    output_payload = TokenPayload.from_token_ids(
        result.output_token_ids, vocab_size=metadata.vocab_size
    )

    print(f"Model: {metadata.model_name}")
    print(f"Device: {metadata.device}")
    if metadata.device_fallback_reason:
        print(f"Device fallback: {metadata.device_fallback_reason}")
    print(f"Prompt: {result.prompt}")
    print()
    print(f"Input token count: {result.input_token_count}")
    print(f"Input Token IDs: {list(result.input_token_ids)}")
    print(f"Decoded input: {backend.decode(result.input_token_ids)}")
    print()
    print(f"Generated text: {result.generated_text}")
    print(f"Output token count: {result.output_token_count}")
    print(f"Output Token IDs: {list(result.output_token_ids)}")
    print()
    print(f"Vocabulary size: {metadata.vocab_size}")
    print(
        "Theoretical bits per token: "
        f"{input_payload.theoretical_bits_per_token}"
    )
    print(
        "Theoretical input Token payload size: "
        f"{input_payload.theoretical_payload_bits} bits / "
        f"{input_payload.theoretical_payload_bytes} bytes"
    )
    print(
        "Theoretical output Token payload size: "
        f"{output_payload.theoretical_payload_bits} bits / "
        f"{output_payload.theoretical_payload_bytes} bytes"
    )
    print()
    print(f"Model load wall time: {metadata.load_wall_time_seconds:.3f} s")
    print(f"Generation wall time: {result.generation_wall_time_seconds:.3f} s")


def main() -> None:
    args = build_parser().parse_args()
    config = LLMConfig(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    run_llm_demo(config, prompt=args.prompt)


if __name__ == "__main__":
    main()
