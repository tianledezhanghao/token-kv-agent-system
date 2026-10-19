"""Standalone relay / byte endpoint CLI. Default bind is loopback only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import threading

from .transport.socket_service import ByteService
from .transport.socket_wire import SocketConfig


def address(value: str) -> tuple[str, int]:
    try:
        host, port = value.rsplit(':', 1)
        host = host.strip('[]')
        port = int(port)
        if not host or not 0 <= port < 65536:
            raise ValueError()
        return host, port
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected host:port') from exc


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--listen', type=address, default=('127.0.0.1', 9101))
    role = parser.add_mutually_exclusive_group(required=True)
    role.add_argument('--forward', type=address, help='relay upstream host:port (no protocol parsing)')
    role.add_argument('--endpoint', help='byte verifier target identity (not a remote model Agent)')
    role.add_argument('--route', action='append', help='repeat LISTEN=UPSTREAM for two directions in one relay process')
    parser.add_argument('--max-payload-bytes', type=int, default=64 * 1024**2)
    parser.add_argument('--chunk-bytes', type=int, default=64 * 1024)
    parser.add_argument('--timeout', type=float, default=30)
    parser.add_argument('--max-connections', type=int, default=8)
    parser.add_argument('--log', type=Path, help='append JSONL socket measurements')
    args = parser.parse_args(argv)
    lock = threading.Lock()
    def emit(row):
        text = json.dumps(row, ensure_ascii=False)
        with lock:
            if args.log:
                with args.log.open('a', encoding='utf-8') as output:
                    output.write(text + '\n')
            print(text, flush=True)
    config = SocketConfig(args.max_payload_bytes, args.chunk_bytes, args.timeout)
    services, threads = [], []
    try:
        routes = [(args.listen, args.forward, args.endpoint)]
        if args.route:
            routes = []
            for route in args.route:
                listen, upstream = route.split('=', 1)
                routes.append((address(listen), address(upstream), None))
        for listen, upstream, target in routes:
            service = ByteService(listen, config=config, upstream=upstream,
                                  target=target, log=emit, max_connections=args.max_connections)
            services.append(service)
            thread = threading.Thread(target=service.serve_forever, kwargs={'poll_interval': 0.1}, daemon=True)
            thread.start()
            threads.append(thread)
        emit(dict(event='ready', pid=os.getpid(), listeners=[dict(listen=s.server_address,
            upstream=s.upstream, endpoint=s.target) for s in services]))
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for service in services:
            service.shutdown()
            service.server_close()
        for thread in threads:
            thread.join(timeout=2)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
