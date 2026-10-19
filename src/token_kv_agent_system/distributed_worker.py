"""Standalone offline model Worker: restricted HTTP control, Relay-only data."""
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import argparse
import json
import os
from pathlib import Path
import socketserver
import threading
import time
from urllib.request import Request, urlopen

from .distributed_agent import AgentExecutor
from .distributed_wire import RemoteAgentTransport
from .incremental_kv_protocol import canonical_json
from .relay import address
from .transport.socket_service import ByteService
from .transport.socket_wire import (SocketConfig, SocketCounters, REQUEST, VERIFIED,
                                   encode_frame, recv_frame, send_all)


def control_call(url, operation, *, timeout=120, **arguments):
    request = Request(url.rstrip('/')+'/control', data=canonical_json(dict(operation=operation, **arguments)),
                      headers={'Content-Type':'application/json'}, method='POST')
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(2*1024**2+1)
    if len(raw) > 2*1024**2:
        raise ValueError('control response exceeds limit')
    result = json.loads(raw)
    if not result.get('ok'):
        raise RuntimeError(result.get('error', 'Worker control failure'))
    return result['result']


class AgentDataService(ByteService):
    def __init__(self, listen, executor, *, config=SocketConfig(), log=None):
        self.executor = executor
        super().__init__(listen, target=executor.agent_id, config=config, log=log, max_connections=2)
        self.RequestHandlerClass = AgentDataHandler

    def emit(self, row):
        if self.log:
            self.log(dict(role='agent_data', listen=self.server_address, **row))


class AgentDataHandler(socketserver.BaseRequestHandler):
    def handle(self):
        server, counters = self.server, SocketCounters()
        deadline = time.monotonic()+server.config.timeout_seconds
        try:
            kind, identity, source, target, packet = recv_frame(self.request, server.config, deadline, counters)
            if kind != REQUEST or target != server.target:
                raise ValueError('invalid remote Agent data destination')
            result = server.executor.receive(packet, source, target)
            reply = encode_frame(canonical_json(result), source, target, identity, VERIFIED, server.config)
            send_all(self.request, reply, server.config, deadline, counters)
            server.emit(dict(event='application_ack', receipt=result, **asdict(counters)))
        except Exception as exc:
            # Could include an already committed import whose ACK timed out.
            server.emit(dict(event='data_error', error=f'{type(exc).__name__}: {exc}', **asdict(counters)))


# No arbitrary text/IDs/KV/prompt can be sent to an active Receiver via control.
CONTROL_FIELDS = {
    'status': set(), 'audit': {'session_id'}, 'selection': {'session_id'}, 'receipt': {'session_id','message_id'},
    'configure': {'route','peer_control'}, 'start': {'session_id','config'},
    'generate': {'session_id','stage','final'}, 'send': {'session_id','handoff','choice','auto','network'},
    'decide': {'session_id','handoff','choice','auto','network'},
    'reconcile': {'session_id'}, 'release': {'session_id'},
}


class WorkerControl:
    def __init__(self, executor, socket_config):
        self.executor, self.socket_config = executor, socket_config

    def dispatch(self, request):
        if not isinstance(request, dict) or 'operation' not in request:
            raise ValueError('control object/operation required')
        values = dict(request)
        operation = values.pop('operation')
        if operation not in CONTROL_FIELDS or set(values) != CONTROL_FIELDS[operation]:
            raise ValueError('unsupported control operation/fields; business bytes require data port')
        executor = self.executor
        with executor.lock:
            if operation == 'status':
                return executor.status()
            if operation == 'configure':
                route = values['route']
                if (not isinstance(route, list) or len(route) != 2 or not isinstance(route[0], str)
                        or type(route[1]) is not int or not 0 < route[1] < 65536):
                    raise ValueError('invalid relay route')
                peer_url = values['peer_control']
                if not isinstance(peer_url, str) or not peer_url.startswith('http://'):
                    raise ValueError('peer control requires explicit http URL')
                peer_query = lambda op, **args: control_call(peer_url, op, timeout=self.socket_config.timeout_seconds, **args)
                peer_status = peer_query('status')
                return executor.configure(RemoteAgentTransport({executor.peer:tuple(route)},
                    config=self.socket_config), peer_query, peer_status)
            method = {'selection':'selection_summary'}.get(operation, operation)
            return getattr(executor, method)(**values)


class ControlServer(ThreadingHTTPServer):
    daemon_threads = True
    block_on_close = False

    def __init__(self, listen, control):
        self.control = control
        self.slots = threading.BoundedSemaphore(4)
        super().__init__(listen, ControlHandler)

    def process_request(self, request, client_address):
        if not self.slots.acquire(False):
            request.close()
            return
        try:
            super().process_request(request, client_address)
        except Exception:
            self.slots.release()
            raise

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.slots.release()


class ControlHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        self.connection.settimeout(120)
        try:
            if self.path != '/control':
                raise ValueError('unknown control endpoint')
            length = int(self.headers.get('Content-Length', '0'))
            if not 0 < length <= 1024**2:
                raise ValueError('control request exceeds limit')
            result = self.server.control.dispatch(json.loads(self.rfile.read(length)))
            body = canonical_json(dict(ok=True, result=result))
        except Exception as exc:
            body = canonical_json(dict(ok=False, error=f'{type(exc).__name__}: {exc}'))
        try:
            self.send_response(200)
            self.send_header('Content-Type','application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except OSError:
            pass  # Command may have committed; clients query, never auto replay.


def sample_resources(torch=None, device='mps'):
    import sys
    if sys.platform == 'darwin':
        from .local_qwen_comparison import memory
        return memory(torch if device == 'mps' else None)
    # Linux deployment protection; no CUDA execution has been claimed on Mac.
    import resource
    info = {line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
    row = dict(monotonic=time.monotonic(), swap_used_mib=(info['SwapTotal']-info['SwapFree'])/1024,
        pressure_level=4 if info['MemAvailable'] < 256*1024 else 2 if info['MemAvailable'] < 1024*1024 else 1,
        rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if torch is not None and device.startswith('cuda'):
        free, total = torch.cuda.mem_get_info(device)
        row.update(cuda_free_bytes=free, cuda_total_bytes=total)
        if free < 256*1024**2:
            row['pressure_level'] = 4
    return row


def resource_budget(model_path, *, copies=1, max_context_tokens=4096):
    """Conservative pre-load estimate, not a substitute for live monitoring."""
    import sys
    import re
    import subprocess
    if type(max_context_tokens) is not int or max_context_tokens <= 0 or type(copies) is not int or copies <= 0:
        raise ValueError('invalid resource budget dimensions')
    path = Path(model_path)
    if not path.is_dir() or not (path/'config.json').is_file():
        raise ValueError('an existing complete local model directory is required; no downloads')
    weights = list(path.glob('*.safetensors'))
    if not weights:
        raise ValueError('local safetensors checkpoint missing')
    size = sum(p.stat().st_size for p in weights)
    config = json.loads((path/'config.json').read_text())
    head_dim = config.get('head_dim', config['hidden_size']//config['num_attention_heads'])
    kv = 2*config['num_hidden_layers']*config['num_key_value_heads']*head_dim*2*max_context_tokens
    estimate = copies*(size + 1024**3 + 5*kv)
    available = None
    if sys.platform == 'darwin':
        if size > 1200*1024**2:
            raise ValueError('Mac distributed validation only permits bounded 0.5B checkpoints')
        row = sample_resources(None)
        if row['pressure_level'] >= 4:
            raise RuntimeError('critical memory pressure before model load')
        vm = row['vm_stat']
        page = int(re.search(r'page size of (\d+) bytes', vm)[1])
        available = page*sum(int(re.search(label+r':\s+(\d+)', vm)[1])
            for label in ('Pages free','Pages inactive','Pages purgeable'))
    elif Path('/proc/meminfo').exists():
        info = Path('/proc/meminfo').read_text()
        available = int(re.search(r'MemAvailable:\s+(\d+)', info)[1])*1024
    if available is None or estimate > available:
        raise RuntimeError(f'insufficient resource budget: need {estimate}, available {available}')
    return dict(copies=copies, weight_file_bytes=size, kv_per_agent_bytes=kv,
                estimated_reservation_bytes=estimate, available_bytes=available)


def load_offline_backend(model_path, device, output_dir, max_context_tokens):
    from .model.config import validate_device_spec
    validate_device_spec(device)
    if device == 'auto':
        raise ValueError('Worker requires explicit placement; no CPU fallback')
    resource_budget(model_path, max_context_tokens=max_context_tokens)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
        PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    import transformers
    from .model.huggingface import HuggingFaceModelBackend, resolve_device
    from .local_qwen_resources import ResourceMonitor
    resolve_device(device, torch)
    monitor = ResourceMonitor(lambda:sample_resources(torch, device), sample_resources(torch, device), output_dir)
    monitor.start()
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                                 dtype=torch.bfloat16)
        backend = HuggingFaceModelBackend(str(model_path), device=device, tokenizer=tokenizer, model=model)
        if backend.metadata.device_fallback_reason or backend.metadata.dtype != 'torch.bfloat16':
            raise RuntimeError('explicit device/BF16 load contract failed')
        original = backend._timed_forward
        backend._timed_forward = lambda **kwargs: monitor.forward(original, **kwargs)
        monitor.check()
        return backend, monitor
    except BaseException:
        monitor.close()
        raise


def serve_worker(backend, *, agent, control_listen, data_listen, config, output_dir,
                 max_context_tokens=4096, max_sessions=32):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    journal_lock = threading.Lock()
    def emit(event, row):
        with journal_lock:
            with (output_dir/'events.jsonl').open('a') as stream:
                stream.write(json.dumps({**row, 'event':event, 'pid':os.getpid()}, ensure_ascii=False)+'\n')
    evidence_bytes = 0
    def packet_sink(direction, metadata, packet):
        nonlocal evidence_bytes
        if evidence_bytes + len(packet) > 512*1024**2:
            raise ValueError('Worker packet evidence resource limit (512 MiB) reached')
        from .distributed_wire import sha
        folder = output_dir/'packets'
        folder.mkdir(exist_ok=True)
        path = folder/f'{direction}-{metadata["session_id"]}-{metadata["handoff"]}-{sha(packet)}.bin'
        with path.open('xb') as stream:
            stream.write(packet)
        evidence_bytes += len(packet)
        emit('packet_saved', dict(direction=direction,path=path.name,bytes=len(packet),sha256=sha(packet)))
    executor = AgentExecutor(agent, backend, max_context_tokens=max_context_tokens,
                             max_sessions=max_sessions, evidence=emit, packet_sink=packet_sink)
    data = AgentDataService(data_listen, executor, config=config, log=lambda row:emit(row['event'],row))
    control = ControlServer(control_listen, WorkerControl(executor, config))
    threads = []
    try:
        for server in (data, control):
            thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval':.1}, daemon=True)
            thread.start()
            threads.append(thread)
        print(json.dumps(dict(event='ready', pid=os.getpid(), agent=agent, boot_id=executor.boot_id,
            control=f'http://{control.server_address[0]}:{control.server_address[1]}',
            data=data.server_address)), flush=True)
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        for server in (control, data):
            server.shutdown()
            server.server_close()
        for thread in threads:
            thread.join(timeout=2)
        # Process exit releases model, Cache, MPS/CUDA allocations. There is no
        # mutable cross-process model object or shared memory cache.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--agent', choices=('A','B'), required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--control-listen', type=address, default=('127.0.0.1',9201))
    parser.add_argument('--data-listen', type=address, default=('127.0.0.1',9301))
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--max-context-tokens', type=int, default=4096)
    parser.add_argument('--max-sessions', type=int, default=32)
    parser.add_argument('--timeout', type=float, default=120)
    parser.add_argument('--max-payload-bytes', type=int, default=64*1024**2)
    parser.add_argument('--chunk-bytes', type=int, default=64*1024)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    monitor = None
    try:
        config = SocketConfig(args.max_payload_bytes, args.chunk_bytes, args.timeout)
        backend, monitor = load_offline_backend(args.model, args.device, args.output_dir, args.max_context_tokens)
        serve_worker(backend, agent=args.agent, control_listen=args.control_listen, data_listen=args.data_listen,
            config=config, output_dir=args.output_dir, max_context_tokens=args.max_context_tokens,
            max_sessions=args.max_sessions)
        return 0
    except BaseException as exc:
        if not isinstance(exc, KeyboardInterrupt):
            (args.output_dir/'failure.json').write_text(json.dumps(dict(error=f'{type(exc).__name__}: {exc}'))+'\n')
        raise
    finally:
        if monitor:
            monitor.close()


if __name__ == '__main__':
    raise SystemExit(main())
