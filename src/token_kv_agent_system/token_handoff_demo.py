"""Real single-process A -> B -> A Token-ID handoff demo."""

from __future__ import annotations

import argparse

from .agent import LLMAgent
from .model import DEFAULT_MODEL_NAME, GenerationConfig, HuggingFaceModelBackend
from .token_handoff import TokenHandoffResult, TokenHandoffSession
from .transport import InMemoryByteTransport


DEFAULT_PROMPT = "Explain Token communication in one short sentence."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real single-process bidirectional Token-ID handoff"
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def _print_handoff(label: str, result: TokenHandoffResult) -> None:
    print(f"=== {label} ===")
    print(f"Direction: {result.source_agent_id} -> {result.target_agent_id}")
    print(f"Sender output text: {result.sender_generated_text}")
    print(f"Sender Token IDs: {list(result.sender_output_token_ids)}")
    print(f"Token count: {result.token_count}")
    print(
        "Wire size: "
        f"{result.wire_byte_count} B "
        f"(header {result.wire_header_bytes} B + body {result.wire_body_bytes} B)"
    )
    print(
        "Theoretical Token logical bytes: "
        f"{result.theoretical_token_logical_bytes} B"
    )
    print(f"Transport: {result.transport_name}")
    print(f"Received bytes: {result.received_byte_count} B")
    print(f"Restored Token IDs: {list(result.received_token_ids)}")
    print(
        "Receiver Prefill input Token IDs: "
        f"{list(result.receiver_prefill_input_token_ids)}"
    )
    print(
        "Sender/restored/Prefill IDs exact match: "
        f"{result.token_identity_preserved}"
    )
    print(f"Receiver Prefill forward: {result.receiver_prefill_forward_ms:.3f} ms")
    print(f"Receiver output Token IDs: {list(result.receiver_output_token_ids)}")
    print(f"Receiver generated text: {result.receiver_generated_text}")
    print(f"KV bytes transferred: {result.kv_bytes_transferred}")


def run_token_handoff_demo(
    *, model_name: str, device: str, max_new_tokens: int, prompt: str
) -> tuple[TokenHandoffResult, TokenHandoffResult]:
    config = GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    backend = HuggingFaceModelBackend(model_name, device=device)
    agent_a = LLMAgent("agent_a", backend)
    agent_b = LLMAgent("agent_b", backend)
    transport = InMemoryByteTransport()
    session = TokenHandoffSession((agent_a, agent_b), transport)

    a_to_b = session.generate_and_handoff(
        "agent_a",
        "agent_b",
        prompt,
        sender_config=config,
        receiver_config=config,
    )
    b_to_a = session.handoff_token_ids(
        "agent_b",
        "agent_a",
        a_to_b.receiver_output_token_ids,
        sender_generated_text=a_to_b.receiver_generated_text,
        receiver_config=config,
    )

    print("=== INITIAL ===")
    print(f"Model: {backend.model_name}")
    print(f"Device: {backend.device}")
    print("Backend: shared")
    print(f"Agent A: {agent_a.agent_id}")
    print(f"Agent B: {agent_b.agent_id}")
    print(f"Initial prompt: {prompt}")
    print()
    _print_handoff("A -> B", a_to_b)
    print()
    _print_handoff("B -> A", b_to_a)
    print()
    print("=== RESULT ===")
    print(f"A -> B Token identity preserved: {a_to_b.token_identity_preserved}")
    print(f"B -> A Token identity preserved: {b_to_a.token_identity_preserved}")
    print("No text round-trip: True")
    print("No KV transferred: True")
    print("Network claim: none (single-process in-memory byte boundary)")
    return a_to_b, b_to_a


def main() -> None:
    args = build_parser().parse_args()
    run_token_handoff_demo(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
