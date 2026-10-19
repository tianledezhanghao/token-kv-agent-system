"""Spawn-only model ownership. Join old owner before constructing its replacement."""
from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from threading import RLock, Thread

from .discussion_service import DiscussionService, DiscussionBusyError, DiscussionUnavailableError

MODELS = {size: f'Qwen/Qwen2.5-{size}-Instruct' for size in ('0.5B', '1.5B', '3B')}


def cached_model(size):
    """Resolve an existing complete snapshot, never contact the Hub."""
    import json
    from huggingface_hub.constants import HF_HUB_CACHE
    root = Path(HF_HUB_CACHE) / ('models--' + MODELS[size].replace('/', '--'))
    snapshots = root / 'snapshots'
    ref = root / 'refs' / 'main'
    paths = [snapshots / ref.read_text().strip()] if ref.is_file() else sorted(snapshots.glob('*'))
    complete = []
    for path in paths:
        if not all((path / name).is_file() for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json')):
            continue
        index = path / 'model.safetensors.index.json'
        weights = set(json.loads(index.read_text())['weight_map'].values()) if index.is_file() else {'model.safetensors'}
        if weights and all((path / name).is_file() for name in weights):
            complete.append(path)
    if len(complete) != 1:
        raise FileNotFoundError(f'Expected one complete local snapshot for {MODELS[size]}, found {len(complete)} in {snapshots}')
    return str(complete[0])


def worker_main(connection, events, path, history):
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', PYTORCH_ENABLE_MPS_FALLBACK='0')
    try:
        from ..model import HuggingFaceModelBackend
        service = DiscussionService(HuggingFaceModelBackend(path, device='mps'))
        if history:
            service._sessions, service._started_ids, service._latest = history
        service.collector.subscribe(events.send)
        connection.send((True, service.latest()))
        while True:
            method, args = connection.recv()
            if method == 'close':
                service.close()
                return
            try:
                result = ((service._sessions, service._started_ids, service._latest)
                          if method == 'history' else getattr(service, method)(*args))
                connection.send((True, result))
            except Exception as exc:
                connection.send((False, exc))
    except BaseException as exc:
        try:
            connection.send((False, RuntimeError(f'{type(exc).__name__}: {exc}')))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        connection.close()
        events.close()


class ModelWorkerService:
    """Control-plane proxy; no model, tensors or KV ever enter the API process."""
    backend = None

    def __init__(self, model_name, *, collector):
        self.collector = collector
        self._lock = RLock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='model-switch')
        self._process = self._connection = None
        self._history = None
        self._loading = False
        self._error = None
        self._closed = False
        self._snapshot = dict(available=False, model=None, status='idle', session=None, active_session_id=None)
        self._path = model_name
        # Startup uses the same single-owner lifecycle as switching.
        try:
            self._launch(model_name)
        except BaseException:
            self._stop()
            self._executor.shutdown(wait=True)
            raise

    def _launch(self, path):
        context = mp.get_context('spawn')
        parent, child = context.Pipe()
        event_reader, event_writer = context.Pipe(duplex=False)
        process = context.Process(target=worker_main, args=(child, event_writer, path, self._history), daemon=True)
        self._process, self._connection = process, parent
        process.start()
        child.close()
        event_writer.close()
        def forward():
            try:
                while True:
                    self.collector.record(event_reader.recv())
            except (EOFError, OSError):
                pass
            finally:
                event_reader.close()
        Thread(target=forward, daemon=True).start()
        if not parent.poll(180):
            raise RuntimeError('Model worker loading timed out after 180 seconds')
        ok, result = parent.recv()
        if not ok:
            raise result
        self._snapshot = result
        self._path = path

    def _stop(self):
        process = self._process
        if process is None:
            return
        if process.is_alive():
            try:
                self._connection.send(('close', ()))
            except (OSError, EOFError):
                pass
            process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        if process.is_alive():
            process.kill()
            process.join(10)
        if process.is_alive():
            raise RuntimeError('Old model worker did not exit; replacement loading refused')
        self._connection.close()
        process.close()
        self._process = self._connection = None

    def _rpc(self, method, *args):
        if self._loading:
            raise DiscussionBusyError('正在加载模型')
        if self._closed or not self._process or not self._process.is_alive():
            raise DiscussionUnavailableError('Model worker is unavailable')
        try:
            self._connection.send((method, args))
            if not self._connection.poll(30):
                raise DiscussionUnavailableError('Model worker response timed out')
            ok, result = self._connection.recv()
        except (EOFError, OSError) as exc:
            raise DiscussionUnavailableError(f'Model worker disconnected: {exc}') from exc
        if not ok:
            raise result
        return result

    def latest(self):
        with self._lock:
            if not self._loading and not self._closed and self._process and self._process.is_alive():
                self._snapshot = self._rpc('latest')
            elif not self._loading:
                self._snapshot.update(available=False, model=None)
            return {**deepcopy(self._snapshot), 'available': self._snapshot['available'] and not self._loading,
                    'model_switch': dict(loading=self._loading, error=self._error,
                                         options=list(MODELS), worker_pid=self._process.pid if self._process and not self._loading else None)}

    def switch_model(self, size):
        with self._lock:
            if self._closed or self._loading:
                raise DiscussionBusyError('Model service is closed or loading')
            snapshot = self.latest()
            if snapshot['active_session_id']:
                raise DiscussionBusyError('讨论进行中禁止切换模型')
            path = cached_model(size)  # Preflight before releasing the current model.
            if path == self._path and snapshot['available']:
                return snapshot
            if self._process and self._process.is_alive():
                self._history = self._rpc('history')
            self._loading, self._error = True, None
            previous = self._path
            self._executor.submit(self._switch, path, previous)
            return self.latest()

    def _switch(self, path, previous):
        try:
            self._stop()
            self._launch(path)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            try:
                self._stop()  # Failed target must exit before recovery.
                self._launch(previous)
                error += '; 已恢复原模型'
            except Exception as recovery:
                error += f'; 恢复失败: {type(recovery).__name__}: {recovery}'
                try:
                    self._stop()
                except Exception as cleanup:
                    error += f'; 清理失败: {cleanup}'
                self._snapshot.update(available=False, model=None)
            self._error = error
        finally:
            with self._lock:
                self._loading = False

    def _call(self, method, *args):
        with self._lock:
            return self._rpc(method, *args)

    def start(self, request):
        return self._call('start', request)

    def preview_prompts(self, request):
        # Prompt validation remains available while no model is loaded.
        with self._lock:
            if self._loading or not self._snapshot['available']:
                return DiscussionService.preview_prompts(self, request)
            return self._rpc('preview_prompts', request)

    def get(self, session_id):
        with self._lock:
            if self._loading or not self._snapshot['available']:
                if self._history is None:
                    raise KeyError(session_id)
                return deepcopy(self._history[0][session_id])
            return self._rpc('get', session_id)

    def network(self, session_id):
        return self.get(session_id)['network']

    def update_network(self, session_id, request):
        return self._call('update_network', session_id, request)

    def close(self):
        with self._lock:
            self._closed = True
        self._executor.shutdown(wait=True)
        with self._lock:
            self._stop()
