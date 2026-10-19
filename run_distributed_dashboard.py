"""Start bounded cached Mac Workers, dual Relay and Dashboard adapter; Ctrl-C stops owned processes."""
from token_kv_agent_system.dashboard.distributed_launcher import main

if __name__ == '__main__':
    raise SystemExit(main())
