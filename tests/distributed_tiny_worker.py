"""Explicit engineering fixture: tiny real CPU forward, controlled numeric argmax.

Never used by production Worker CLI or pretrained smoke evidence.
"""
import argparse
from dataclasses import replace
from pathlib import Path

from test_incremental_kv_transport import make_backend
from token_kv_agent_system.distributed_worker import serve_worker
from token_kv_agent_system.relay import address
from token_kv_agent_system.transport import SocketConfig

parser = argparse.ArgumentParser()
parser.add_argument('--agent',required=True)
parser.add_argument('--control-listen',type=address,required=True)
parser.add_argument('--data-listen',type=address,required=True)
parser.add_argument('--output-dir',type=Path,required=True)
parser.add_argument('--max-context-tokens',type=int,default=2048)
parser.add_argument('--timeout',type=float,default=30)
parser.add_argument('--chunk-bytes',type=int,default=4093)
args = parser.parse_args()
backend = make_backend()
original = backend.continue_from_cache
backend.continue_from_cache = lambda *a,**kw: replace(original(*a,**kw),predicted_next_token_id=59)
serve_worker(backend,agent=args.agent,control_listen=args.control_listen,data_listen=args.data_listen,
    output_dir=args.output_dir,config=SocketConfig(timeout_seconds=args.timeout,chunk_bytes=args.chunk_bytes),
    max_context_tokens=args.max_context_tokens)
