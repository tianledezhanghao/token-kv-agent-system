"""HTTP updates during blocked real worker work; no sleeps or model weights."""
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from fastapi.testclient import TestClient

from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard.discussion_service import DiscussionService, DiscussionNetworkRequest
from token_kv_agent_system.metrics import MetricCollector
from token_kv_agent_system.stateful_token import StatefulTokenSession
from test_discussion import ScriptBackend


@pytest.fixture
def live():
    b = ScriptBackend([(100, 2)] * 7)
    c = MetricCollector()
    service = DiscussionService(b, collector=c)
    done = Event()
    c.subscribe(lambda e: done.set() if e.event_type.value in ('DISCUSSION_COMPLETED', 'DISCUSSION_FAILED') else None)
    with TestClient(create_app(collector=c, discussion_service=service)) as client:
        yield client, service, b, done


def test_update_during_inference_before_first_freeze_and_ws_recovery(live):
    client, service, b, done = live
    entered, release = Event(), Event()
    original = b.continue_from_cache
    def blocked(*a, **kw):
        entered.set()
        assert release.wait(10)
        return original(*a, **kw)
    b.continue_from_cache = blocked
    try:
        with client.websocket_connect('/ws/events') as ws:
            r = client.post('/api/discussions', json=dict(question='Q', max_new_tokens=3, mode='auto', rounds=2))
            sid = r.json()['session_id']
            assert entered.wait(2)
            response = client.patch(f'/api/discussions/{sid}/network', json=dict(bandwidth_mbps=50, base_latency_ms=7))
            assert response.status_code == 200  # returned while model forward is still blocked
            update = response.json()
            assert update['config_revision'] == 1 and update['effective_from_handoff_index'] == 1
            assert not release.is_set()
            assert client.get('/api/health').status_code == 200
            release.set()
            assert done.wait(10)
            events = []
            while not events or events[-1]['type'] not in ('DISCUSSION_COMPLETED', 'DISCUSSION_FAILED'):
                events.append(ws.receive_json())
        state = client.get(f'/api/discussions/{sid}').json()
        assert state['status'] == 'completed', state['error']
        assert [e['metadata']['revision'] for e in events] == list(range(1, state['revision'] + 1))
        assert [e['metadata']['network_update'] for e in events if e['type'] == 'DISCUSSION_NETWORK_UPDATED'] == state['network_updates']
        assert client.get(f'/api/discussions/{sid}/network').json() == state['network']
        for h, d in zip(state['handoffs'], state['auto_decisions']):
            assert h['network']['bandwidth_mbps'] == d['network']['bandwidth_mbps'] == 50
            assert h['network']['base_latency_ms'] == 7 and h['network']['config_revision'] == 1
            assert h['append_succeeded'] and h['executed_mode'] == 'token'
        assert client.patch(f'/api/discussions/{sid}/network', json=dict(bandwidth_mbps=20)).status_code == 409
    finally:
        release.set()


def test_frozen_handoff_preserved_two_updates_and_trace_override(live, monkeypatch):
    client, service, b, done = live
    entered, release = Event(), Event()
    original = StatefulTokenSession.handoff
    calls = 0
    def blocked(self, *a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set(); assert release.wait(10)
        return original(self, *a, **kw)
    monkeypatch.setattr(StatefulTokenSession, 'handoff', blocked)
    try:
        r = client.post('/api/discussions', json=dict(question='Q', rounds=2, max_new_tokens=3, mode='auto', network_mode='trace', bandwidth_trace=[20, 1000, 500]))
        sid = r.json()['session_id']; path = f'/api/discussions/{sid}/network'
        assert entered.wait(2)
        frozen = client.get(f'/api/discussions/{sid}').json()['active_handoff']
        assert client.patch(path, json=dict(bandwidth_mbps=50)).status_code == 422
        first = client.patch(path, json=dict(bandwidth_mbps=50, base_latency_ms=3, manual_override=True)).json()
        second = client.patch(path, json=dict(bandwidth_mbps=75, base_latency_ms=4)).json()
        assert (first['config_revision'], second['config_revision']) == (1, 2)
        assert first['mode_before'] == 'trace' and first['mode_after'] == 'manual'
        assert first['before']['config']['trace_mbps'] == [20, 1000, 500]
        assert first['effective_from_handoff_index'] == second['effective_from_handoff_index'] == 3
        assert client.get(f'/api/discussions/{sid}').json()['active_handoff'] == frozen
        release.set(); assert done.wait(10)
        state = client.get(f'/api/discussions/{sid}').json()
        assert state['status'] == 'completed', state['error']
        assert [h['network']['bandwidth_mbps'] for h in state['handoffs']] == [20, 1000, 75, 75]
        assert [h['network']['config_revision'] for h in state['handoffs']] == [0, 0, 2, 2]
        assert [h['network']['base_latency_ms'] for h in state['handoffs']] == [0, 0, 4, 4]
        assert state['auto_decisions'][1]['network'] == frozen['network']
        assert len(b.generation_histories) == 5  # no restarted/regenerated sender
    finally:
        release.set()


@pytest.mark.parametrize('bad', [{}, {'bandwidth_mbps': 0}, {'bandwidth_mbps': -1}, {'base_latency_ms': -1}, {'bandwidth_mbps': True}, {'bandwidth_mbps': '50'}, {'bandwidth_mbps': None}, {'bandwidth_mbps': 50, 'base_latency_ms': None}, {'bandwidth_mbps': 50, 'mode': 'manual'}, {'manual_override': True}, {'bandwidth_mbps': 50, 'manual_override': 1}])
def test_invalid_patch(live, bad):
    assert live[0].patch('/api/discussions/nonexistent/network', json=bad).status_code == 422


def test_nonfinite_and_missing_session(live):
    client = live[0]
    for value in ['NaN', 'Infinity', '-Infinity']:
        assert client.patch('/api/discussions/missing/network', content='{"bandwidth_mbps":'+value+'}', headers={'content-type': 'application/json'}).status_code == 422
    assert client.patch('/api/discussions/missing/network', json={'bandwidth_mbps': 50}).status_code == 404
    assert client.get('/api/discussions/missing/network').status_code == 404


def test_concurrent_accepts_have_whole_configs_monotonic_revision(live):
    client, service, b, done = live
    entered, release = Event(), Event()
    original = b.prefill
    def blocked(**kw):
        entered.set(); assert release.wait(10); return original(**kw)
    b.prefill = blocked
    try:
        sid = client.post('/api/discussions', json=dict(question='Q', max_new_tokens=3, rounds=1)).json()['session_id']
        assert entered.wait(2)
        def update(n): return service.update_network(sid, DiscussionNetworkRequest(bandwidth_mbps=float(n), base_latency_ms=float(n)))
        with ThreadPoolExecutor(max_workers=4) as pool: results = list(pool.map(update, range(1, 13)))
        updates = service.get(sid)['network_updates']
        assert [u['config_revision'] for u in updates] == list(range(1, 13))
        assert sorted(u['config_revision'] for u in results) == list(range(1, 13))
        for u in updates:
            assert u['after']['config']['bandwidth_mbps'] == u['after']['config']['base_latency_ms']
            assert u['after']['config_revision'] == u['before']['config_revision'] + 1
            assert u['effective_from_handoff_index'] == 1
        release.set(); assert done.wait(10)
        assert service.get(sid)['handoffs'][0]['network']['config_revision'] == 12
    finally: release.set()


def test_last_frozen_handoff_rejects_ineffective_update(live, monkeypatch):
    client, service, b, done = live
    entered, release = Event(), Event()
    original = StatefulTokenSession.handoff
    calls = 0
    def blocked(self, *a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            entered.set(); assert release.wait(10)
        return original(self, *a, **kw)
    monkeypatch.setattr(StatefulTokenSession, 'handoff', blocked)
    try:
        sid = client.post('/api/discussions', json=dict(question='Q', rounds=1, max_new_tokens=3)).json()['session_id']
        assert entered.wait(2)
        response = client.patch(f'/api/discussions/{sid}/network', json=dict(bandwidth_mbps=50))
        assert response.status_code == 409 and 'already frozen' in response.json()['detail']
        assert service.get(sid)['network_updates'] == []
    finally: release.set()
    assert done.wait(10)
