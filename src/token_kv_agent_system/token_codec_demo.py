"""Real Qwen Token Protocol v1 serialization round-trip demo."""

from __future__ import annotations

import argparse

from .model import DEFAULT_MODEL_NAME, GenerationConfig, HuggingFaceModelBackend
from .models import TokenPayload
from .token_protocol import (
    TOKEN_BYTES,
    TOKEN_PROTOCOL_VERSION,
    TOKEN_WIDTH_BITS,
    TokenWirePacket,
    TokenWireSize,
    deserialize_token_packet,
    measure_token_wire_size,
    serialize_token_payload,
)


DEFAULT_PROMPT = "Explain KV cache in one short sentence."


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the real Token Protocol v1 demo")
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "mps", "cuda"), default="auto"
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    return parser


def _round_trip(
    token_ids: tuple[int, ...], *, vocab_size: int
) -> tuple[TokenPayload, TokenWirePacket, TokenWireSize]:
    logical_payload = TokenPayload.from_token_ids(token_ids, vocab_size=vocab_size)
    serialized = serialize_token_payload(logical_payload)
    packet = deserialize_token_packet(serialized)
    size = measure_token_wire_size(logical_payload, packet)
    return logical_payload, packet, size


def _print_size(
    label: str,
    payload: TokenPayload,
    packet: TokenWirePacket,
    size: TokenWireSize,
) -> None:
    print(f"{label} Token IDs: {list(payload.token_ids)}")
    print(f"{label} Token count: {payload.token_count}")
    if payload.token_ids:
        print(f"{label} Token ID range: {min(payload.token_ids)}..{max(payload.token_ids)}")
    print(
        f"{label} theoretical minimum: {payload.theoretical_payload_bits} bits / "
        f"{size.theoretical_minimum_bytes} B"
    )
    print(f"{label} wire header: {size.wire_header_bytes} B")
    print(f"{label} wire token body: {size.wire_token_data_bytes} B")
    print(f"{label} total wire packet: {size.total_wire_bytes} B")
    print(f"{label} wire overhead vs theoretical: {size.wire_overhead_bytes} B")
    ratio = size.wire_to_theoretical_ratio
    print(
        f"{label} wire/theoretical ratio: {ratio:.3f}x"
        if ratio is not None
        else f"{label} wire/theoretical ratio: n/a"
    )
    print(f"{label} round-trip: {packet.token_ids == payload.token_ids}")


def run_token_codec_demo(
    *, model_name: str, device: str, max_new_tokens: int, prompt: str
) -> None:
    backend = HuggingFaceModelBackend(model_name, device=device)
    input_ids = backend.encode(prompt)
    generated = backend.generate(
        prompt, GenerationConfig(max_new_tokens=max_new_tokens, do_sample=False)
    )
    input_payload, input_packet, input_size = _round_trip(
        input_ids, vocab_size=backend.vocab_size
    )
    output_payload, output_packet, output_size = _round_trip(
        generated.output_token_ids, vocab_size=backend.vocab_size
    )

    print(f"Model: {backend.metadata.model_name}")
    print(f"Device: {backend.metadata.device}")
    print(f"Prompt: {prompt}")
    print(f"Vocab size: {backend.vocab_size}")
    print(f"Theoretical bits/token: {input_payload.theoretical_bits_per_token}")
    print(f"Wire protocol: Token Protocol v{TOKEN_PROTOCOL_VERSION}")
    print(f"Token representation: uint{TOKEN_WIDTH_BITS} big-endian ({TOKEN_BYTES} B/token)")
    print()
    _print_size("Input", input_payload, input_packet, input_size)
    print()
    print(f"Generated text: {generated.generated_text}")
    _print_size("Generated", output_payload, output_packet, output_size)


def main() -> None:
    args = build_parser().parse_args()
    run_token_codec_demo(
        model_name=args.model,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        prompt=args.prompt,
    )


if __name__ == "__main__":
    main()
