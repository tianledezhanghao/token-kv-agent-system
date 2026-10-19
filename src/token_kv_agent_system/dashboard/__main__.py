"""Uvicorn command-line entry point for the Dashboard Backend."""

from __future__ import annotations

import argparse

import uvicorn

from token_kv_agent_system.model import DEFAULT_MODEL_NAME

from .app import (
    create_app,
    create_discussion_app,
    create_integrated_demo_app,
    create_interactive_app,
    create_real_kv_app,
    create_real_adaptive_app,
    create_real_token_app,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Token/KV Dashboard Backend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="wait for Dashboard network and communication controls",
    )
    parser.add_argument(
        "--real-token",
        action="store_true",
        help="load one shared real LLM backend and await TOKEN handoff controls",
    )
    parser.add_argument(
        "--real-kv",
        action="store_true",
        help="load one shared real LLM backend and await canonical KV handoff controls",
    )
    parser.add_argument("--real-adaptive", action="store_true",
                        help="load a real selector profile and execute only its selected path")
    parser.add_argument("--discussion", action="store_true",
                        help="load one offline real model for manual TOKEN/V0/V1 discussions")
    parser.add_argument("--selector-profile", help="required calibration JSON for --real-adaptive")
    parser.add_argument(
        "--no-demo",
        action="store_true",
        help="start an empty backend instead of the same-process Mock Demo",
    )
    parser.add_argument(
        "--demo-delay",
        default=1.0,
        type=float,
        help="seconds before the integrated demo starts (default: 1.0)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--initial-prompt", default="Explain KV cache briefly.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    selected_modes = sum((args.interactive, args.real_token, args.real_kv, args.real_adaptive, args.discussion, args.no_demo))
    if selected_modes > 1:
        raise SystemExit(
            "--interactive, --real-token, --real-kv, --real-adaptive, --discussion and --no-demo are mutually exclusive"
        )
    if args.real_adaptive and not args.selector_profile:
        raise SystemExit("--real-adaptive requires --selector-profile")
    if args.selector_profile and not args.real_adaptive:
        raise SystemExit("--selector-profile requires --real-adaptive")
    if args.discussion:
        app = create_discussion_app(model_name=args.model, device=args.device)
    elif args.real_adaptive:
        app = create_real_adaptive_app(
            selector_profile=args.selector_profile, model_name=args.model,
            device=args.device, max_new_tokens=args.max_new_tokens,
            initial_prompt=args.initial_prompt)
    elif args.real_kv:
        app = create_real_kv_app(
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            initial_prompt=args.initial_prompt,
        )
    elif args.real_token:
        app = create_real_token_app(
            model_name=args.model,
            device=args.device,
            max_new_tokens=args.max_new_tokens,
            initial_prompt=args.initial_prompt,
        )
    elif args.interactive:
        app = create_interactive_app()
    elif args.no_demo:
        app = create_app()
    else:
        app = create_integrated_demo_app(demo_delay_seconds=args.demo_delay)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
