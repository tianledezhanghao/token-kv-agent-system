"""Owned two-worker/two-relay process lifecycle for loopback acceptance runs."""
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading

from .socket_lab import SocketLab
from .transport import SocketConfig
from .distributed_worker import resource_budget


class DistributedLab(SocketLab):
    def __init__(self, output_dir, *, model_a=None, model_b=None, device_a='mps', device_b='mps',
                 worker_command=None, config=SocketConfig(timeout_seconds=120), max_context_tokens=2048):
        super().__init__(output_dir, config=config)
        self.models = {'A':model_a,'B':model_b or model_a}
        self.devices = {'A':device_a,'B':device_b}
        # Test harness can explicitly inject a tiny real random-model executable;
        # production CLI exposes no mock model or fallback option.
        self.worker_command = worker_command
        self.max_context_tokens = max_context_tokens
        self.workers = {}

    def worker(self,name):
        log = (self.output_dir/f'worker-{name.lower()}.log').open('w')
        self.logs.append(log)
        output = self.output_dir/f'worker-{name.lower()}'
        command = self.worker_command or [sys.executable,'-m','token_kv_agent_system.distributed_worker']
        arguments = ['--agent',name,'--control-listen','127.0.0.1:0','--data-listen','127.0.0.1:0',
            '--output-dir',str(output),'--max-context-tokens',str(self.max_context_tokens),
            '--timeout',str(self.config.timeout_seconds),'--chunk-bytes',str(self.config.chunk_bytes)]
        if not self.worker_command:
            arguments += ['--model',str(self.models[name]),'--device',self.devices[name]]
        process = subprocess.Popen(command+arguments, stdout=subprocess.PIPE,stderr=log,text=True,env=os.environ.copy())
        self.processes.append(process)
        readiness = queue.Queue()
        def drain():
            for line in process.stdout:
                log.write(line);log.flush()
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if row.get('event') == 'ready':
                    readiness.put(row)
            readiness.put(None)
        reader = threading.Thread(target=drain,daemon=True)
        reader.start();self.readers.append(reader)
        row = readiness.get(timeout=120)
        if row is None:
            raise RuntimeError(f'Worker {name} failed; see {log.name}')
        return row

    def __enter__(self):
        self.output_dir.mkdir(parents=True,exist_ok=False)
        try:
            if not self.worker_command:
                # Check aggregate reservation BEFORE either model load. Workers
                # repeat per-load checks and keep live ResourceMonitor enabled.
                budgets = [resource_budget(self.models[n], max_context_tokens=self.max_context_tokens) for n in ('A','B')]
                if sum(b['estimated_reservation_bytes'] for b in budgets) > min(b['available_bytes'] for b in budgets):
                    raise RuntimeError('insufficient combined two-model resource budget')
                (self.output_dir/'budget.json').write_text(json.dumps(budgets,indent=2)+'\n')
            a,b = self.worker('A'),self.worker('B')
            self.workers = {'A':a['control'],'B':b['control']}
            # Two listeners per relay: inbound local delivery and outbound peer.
            import socket
            with socket.socket() as sa, socket.socket() as sb:
                sa.bind(('127.0.0.1',0));sb.bind(('127.0.0.1',0))
                pa,pb = sa.getsockname()[1],sb.getsockname()[1]
            rb = self.start('relay-b','--route',f'127.0.0.1:{pb}=127.0.0.1:{b["data"][1]}',
                            '--route',f'127.0.0.1:0=127.0.0.1:{pa}')
            ra = self.start('relay-a','--route',f'127.0.0.1:{pa}=127.0.0.1:{a["data"][1]}',
                            '--route',f'127.0.0.1:0=127.0.0.1:{pb}')
            self.routes = {'B':('127.0.0.1',ra['listeners'][1]['listen'][1]),
                           'A':('127.0.0.1',rb['listeners'][1]['listen'][1])}
            self.readiness = {'A':a,'B':b,'relay_a':ra,'relay_b':rb}
            return self
        except BaseException:
            self.__exit__(None,None,None)
            raise
