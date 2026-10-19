"""Lifecycle tests do not allocate models; real MPS smoke is recorded separately."""
from copy import deepcopy
from threading import Event
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from token_kv_agent_system.dashboard import model_worker as mw
from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard.discussion_service import DiscussionBusyError
from token_kv_agent_system.metrics import MetricCollector

REAL_RPC = mw.ModelWorkerService._rpc


@pytest.fixture
def harness(monkeypatch):
    operations = []
    alive = set()
    fail = set()
    gate = Event()
    gate.set()

    def launch(self, path):
        gate.wait(5)
        assert not alive, 'two models would be resident'
        operations.append(('load', path))
        alive.add(path)
        self._process = SimpleNamespace(is_alive=lambda: True, pid=123)
        if path in fail:
            raise RuntimeError(f'actual loader error: {path}')
        self._path = path
        self._snapshot = dict(available=True, model={'model_name': path, 'device': 'mps'},
                              status='idle', session=None, active_session_id=None)

    def stop(self):
        operations.append(('exit', next(iter(alive), None)))
        alive.clear()
        self._process = None

    def rpc(self, method, *args):
        if method == 'history':
            return ({'old': {'model': {'model_name': 'original'}, 'status': 'completed'}}, {'old'}, 'old')
        if method == 'latest':
            return deepcopy(self._snapshot)
        if method == 'start':
            if args[0] == 'old':
                raise DiscussionBusyError('Session cannot be restarted')
            self._snapshot['active_session_id'] = 'active'
            return {}
        if method == 'get':
            return deepcopy(self._history[0][args[0]])

    monkeypatch.setattr(mw.ModelWorkerService, '_launch', launch)
    monkeypatch.setattr(mw.ModelWorkerService, '_stop', stop)
    monkeypatch.setattr(mw.ModelWorkerService, '_rpc', rpc)
    monkeypatch.setattr(mw, 'cached_model', lambda size: size)
    service = mw.ModelWorkerService('original', collector=MetricCollector())
    yield service, operations, fail, gate
    gate.set()
    service.close()


def finish(service):
    # FIFO barrier after the switch job.
    service._executor.submit(lambda: None).result(5)


def test_single_owner_order_and_metadata_history(harness):
    service, operations, _, _ = harness
    service.switch_model('3B')
    finish(service)
    assert operations == [('load', 'original'), ('exit', 'original'), ('load', '3B')]
    assert service.latest()['model']['model_name'] == '3B'
    assert service.get('old')['model']['model_name'] == 'original'
    with pytest.raises(DiscussionBusyError, match='restarted'):
        service.start('old')


def test_failure_exits_target_before_recovering(harness):
    service, operations, fail, _ = harness
    fail.add('3B')
    service.switch_model('3B')
    finish(service)
    assert operations[-3:] == [('load', '3B'), ('exit', '3B'), ('load', 'original')]
    latest = service.latest()
    assert latest['available']
    assert latest['model']['model_name'] == 'original'
    assert 'actual loader error: 3B' in latest['model_switch']['error']


def test_failed_recovery_is_unavailable_and_retryable(harness):
    service, _, fail, _ = harness
    fail.update(('3B', 'original'))
    service.switch_model('3B')
    finish(service)
    assert service.latest()['model'] is None
    assert not service.latest()['available']
    assert '恢复失败' in service.latest()['model_switch']['error']
    fail.clear()
    service.switch_model('0.5B')
    finish(service)
    assert service.latest()['available']


def test_active_discussion_and_loading_reject_switch(harness):
    service, _, _, gate = harness
    service.start('new')
    with pytest.raises(DiscussionBusyError, match='讨论进行中'):
        service.switch_model('3B')
    service._snapshot['active_session_id'] = None
    gate.clear()
    service.switch_model('3B')
    assert service.latest()['model_switch']['loading']
    with pytest.raises(DiscussionBusyError):
        service.switch_model('1.5B')
    # Real _rpc gate applies to start and live updates.
    with pytest.raises(DiscussionBusyError):
        REAL_RPC(service, 'start')
    gate.set()
    finish(service)


def test_missing_cache_keeps_existing_worker(harness, monkeypatch):
    service, operations, _, _ = harness
    def missing(size):
        raise FileNotFoundError('cache shard missing')
    monkeypatch.setattr(mw, 'cached_model', missing)
    with pytest.raises(FileNotFoundError, match='shard missing'):
        service.switch_model('3B')
    assert operations == [('load', 'original')]
    assert service.latest()['available']


def test_switch_endpoint_validation_and_busy(harness):
    service, _, _, _ = harness
    app = create_app(collector=service.collector, discussion_service=service)
    with TestClient(app) as client:
        assert client.post('/api/models/switch', json={'model': '7B'}).status_code == 422
        service.start('new')
        response = client.post('/api/models/switch', json={'model': '3B'})
        assert response.status_code == 409
        assert '讨论进行中' in response.json()['detail']


def test_local_cache_without_main_ref_and_missing_shard(tmp_path, monkeypatch):
    import json
    import huggingface_hub.constants as constants
    monkeypatch.setattr(constants, 'HF_HUB_CACHE', str(tmp_path))
    snapshot = tmp_path / 'models--Qwen--Qwen2.5-3B-Instruct' / 'snapshots' / 'fixed-revision'
    snapshot.mkdir(parents=True)
    for filename in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        (snapshot / filename).write_text('{}')
    (snapshot / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'part1.safetensors', 'b': 'part2.safetensors'}}))
    (snapshot / 'part1.safetensors').write_bytes(b'local')
    with pytest.raises(FileNotFoundError, match='found 0'):
        mw.cached_model('3B')
    (snapshot / 'part2.safetensors').write_bytes(b'local')
    assert mw.cached_model('3B') == str(snapshot)


def test_stop_joins_before_clearing_owner():
    operations = []
    class Process:
        alive = True
        def is_alive(self): return self.alive
        def join(self, timeout): operations.append('join')
        def terminate(self): operations.append('terminate'); self.alive = False
        def close(self): operations.append('process closed')
    service = object.__new__(mw.ModelWorkerService)
    service._process = Process()
    service._connection = SimpleNamespace(send=lambda request: operations.append('close requested'), close=lambda: operations.append('pipe closed'))
    service._stop()
    assert operations == ['close requested', 'join', 'terminate', 'join', 'pipe closed', 'process closed']
    assert service._process is None


def test_refuses_replacement_if_owner_cannot_exit():
    service = object.__new__(mw.ModelWorkerService)
    service._process = SimpleNamespace(is_alive=lambda: True, join=lambda timeout: None, terminate=lambda: None, kill=lambda: None)
    service._connection = SimpleNamespace(send=lambda request: None)
    with pytest.raises(RuntimeError, match='replacement loading refused'):
        service._stop()
    assert service._process is not None
