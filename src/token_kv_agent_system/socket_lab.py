"""Mac loopback laboratory: two relay processes and two byte verifier processes.

No models are loaded here. Ports are assigned by the OS; startup races fail
explicitly rather than substituting a route or transport.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading

from .transport import SocketByteTransport, SocketConfig


class SocketLab(AbstractContextManager):
    def __init__(self, output_dir: Path, *, config: SocketConfig = SocketConfig()):
        self.output_dir, self.config = Path(output_dir), config
        self.processes: list[subprocess.Popen] = []
        self.logs = []
        self.readers = []
        self.routes = {}

    def start(self, name: str, *arguments: str) -> dict:
        path = self.output_dir / f'{name}.jsonl'
        log = path.open('w', encoding='utf-8')
        self.logs.append(log)
        process = subprocess.Popen([sys.executable, '-m', 'token_kv_agent_system.relay',
            '--timeout', str(self.config.timeout_seconds), '--chunk-bytes', str(self.config.chunk_bytes),
            '--max-payload-bytes', str(self.config.max_payload_bytes), *arguments],
            stdout=subprocess.PIPE, stderr=log, text=True, env=os.environ.copy())
        self.processes.append(process)
        readiness = queue.Queue()
        def drain():
            for line in process.stdout:
                log.write(line)
                log.flush()
                row = json.loads(line)
                if row.get('event') == 'ready':
                    readiness.put(row)
            readiness.put(None)
        reader = threading.Thread(target=drain, daemon=True)
        reader.start()
        self.readers.append(reader)
        row = readiness.get(timeout=10)
        if row is None:
            raise RuntimeError(f'{name} failed to start; see {path}')
        return row

    def __enter__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        try:
            a = self.start('endpoint-a', '--listen', '127.0.0.1:0', '--endpoint', 'A')
            b = self.start('endpoint-b', '--listen', '127.0.0.1:0', '--endpoint', 'B')
            # Reserve ingress ports until processes are about to bind. The lab
            # checks startup and never silently chooses a different route on error.
            import socket
            with socket.socket() as reserve_a, socket.socket() as reserve_b:
                reserve_a.bind(('127.0.0.1', 0))
                reserve_b.bind(('127.0.0.1', 0))
                pa, pb = reserve_a.getsockname()[1], reserve_b.getsockname()[1]
            ea = a['listeners'][0]['listen'][1]
            eb = b['listeners'][0]['listen'][1]
            rb = self.start('relay-b', '--route', f'127.0.0.1:{pb}=127.0.0.1:{eb}',
                            '--route', f'127.0.0.1:0=127.0.0.1:{pa}')
            reverse_b = rb['listeners'][1]['listen'][1]
            ra = self.start('relay-a', '--route', f'127.0.0.1:{pa}=127.0.0.1:{ea}',
                            '--route', f'127.0.0.1:0=127.0.0.1:{pb}')
            forward_a = ra['listeners'][1]['listen'][1]
            self.routes = {'B': ('127.0.0.1', forward_a), 'A': ('127.0.0.1', reverse_b)}
            self.transport = SocketByteTransport(self.routes, config=self.config)
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        for process in reversed(self.processes):
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
        for process in reversed(self.processes):
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        for reader in self.readers:
            reader.join(timeout=2)
        for process in self.processes:
            if process.stdout:
                process.stdout.close()
        for log in self.logs:
            log.close()
        return False
