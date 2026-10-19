"""Opt-in server migration preflight; defaults to environment inspection only."""

from token_kv_agent_system.server_preflight import main


if __name__ == "__main__":
    raise SystemExit(main())
