"""Control Plane tests: scripted opaque model, plus random tiny CPU KV (no weights)."""
from threading import Event
import weakref

import pytest
from fastapi.testclient import TestClient

from test_discussion import ScriptBackend
from token_kv_agent_system.dashboard.app import create_app, create_interactive_app
from token_kv_agent_system.dashboard.discussion_service import DiscussionService
from token_kv_agent_system.metrics import MetricCollector


@pytest.fixture
def harness():
    collector = MetricCollector()
    backend = ScriptBackend([(100, 2), (101, 2), (102, 2), (103, 2), (104, 2)] * 4)
    service = DiscussionService(backend, collector=collector)
    done = Event()
    collector.subscribe(lambda e: done.set() if e.event_type.value in ('DISCUSSION_COMPLETED', 'DISCUSSION_FAILED') else None)
    with TestClient(create_app(collector=collector, discussion_service=service)) as client:
        yield client, service, backend, done, collector


def start_wait(harness, **overrides):
    client, service, backend, done, collector = harness
    done.clear()
    response = client.post('/api/discussions', json={'question': '23 + 48 = ?', 'max_new_tokens': 3, **overrides})
    assert response.status_code == 202, response.text
    assert done.wait(10)
    return client.get('/api/discussions/' + response.json()['session_id']).json()


def test_start_transcript_summary_bytes_and_refresh_recovery(harness):
    client, service, backend, _, collector = harness
    result = start_wait(harness, summarizer='B')
    assert result['status'] == 'completed'
    assert [(t['agent'], t['round']) for t in result['turns']] == [('A',1),('B',1),('A',2),('B',2)]
    assert [t['text'] for t in result['turns']] == ['Z','[','\\',']']
    assert result['final_answer']['text'] == '^' and result['final_answer']['agent'] == 'B'
    assert result['summary']['total_wire_bytes'] == 4 * (10 + 2 * 4)
    assert result['summary']['token_handoffs'] == result['summary']['handoff_count'] == 4
    assert result['summary']['final_cache_lengths'] == result['cache_lengths']
    assert result['cache_released'] and service.latest()['active_session_id'] is None
    assert client.get('/api/discussions/latest').json()['session'] == result
    assert len(backend.generation_histories) == 5
    assert sum(c[0] == 'prefill' for c in backend.calls) == 2
    body = str(result)
    assert not any(k in body for k in ['message_token_ids', 'tensor', 'packet', 'token_ledger'])
    events = client.get('/api/events?limit=100').json()['events']
    expected = ['DISCUSSION_STARTED'] + [k for _ in range(4) for k in ['AGENT_TURN_STARTED', 'AGENT_TURN_COMPLETED', 'HANDOFF_STARTED', 'HANDOFF_COMPLETED']] + ['DISCUSSION_FINALIZING', 'DISCUSSION_COMPLETED']
    assert [e['type'] for e in events] == expected
    assert [e['metadata']['revision'] for e in events] == list(range(1, len(events)+1))
    assert all(e['metadata']['session_id'] == result['session_id'] and e['timestamp'] for e in events)
    assert events[-1]['metadata']['final_answer'] == result['final_answer']
    assert events[-1]['metadata']['summary'] == result['summary']
    for h in result['handoffs']:
        assert h['requested_mode'] == h['executed_mode'] == 'token'
        assert h['wire_bytes'] == 18 and h['receiver_message_prefill_count'] == 1
        assert h['fallback_reason'] is h['error'] is None


@pytest.mark.parametrize('bad', [
    {'question':''}, {'question':'  '}, {'question':'x'*513}, {'rounds':0}, {'rounds':4}, {'rounds':True},
    {'max_new_tokens':0}, {'max_new_tokens':True}, {'max_new_tokens':2.5}, {'summarizer':'C'},
    {'mode':'kv:v2','kv_recompute_tokens':-1}, {'mode':'kv:v4'}, {'mode':'auto','handoff_plan':[{'round':1,'source':'A','target':'B','mode':'token'}]}, {'mode':'mixed'},
    {'mode':'mixed','handoff_plan':[{'round':1,'source':'B','target':'A','mode':'token'}]*4},
    {'mode':'token','handoff_plan':[{'round':1,'source':'A','target':'B','mode':'token'}]},
    {'handoff_plan':[{'round':1,'source':'A','target':'B','mode':'kv:v2'}]}, {'unexpected':1}])
def test_invalid_requests_do_not_start_or_prefill(harness, bad):
    client, service, backend, _, _ = harness
    assert client.post('/api/discussions', json={'question':'Q', **bad}).status_code == 422
    assert not backend.calls and service.latest()['status'] == 'idle'


def test_same_session_cannot_restart_even_after_completion(harness):
    result = start_wait(harness, session_id='same-session')
    assert result['status'] == 'completed'
    assert harness[0].post('/api/discussions', json={'question':'Q','session_id':'same-session'}).status_code == 409


def test_only_one_active_run_health_ws_remain_live_and_disconnect_does_not_stop(harness):
    client, service, backend, done, _ = harness
    entered, release = Event(), Event()
    original = backend.continue_from_cache
    def blocked(*a, **kw):
        entered.set()
        assert release.wait(5)
        return original(*a, **kw)
    backend.continue_from_cache = blocked
    try:
        with client.websocket_connect('/ws/events') as ws:
            response = client.post('/api/discussions', json={'question':'Q', 'max_new_tokens':3})
            assert response.status_code == 202 and entered.wait(2)
            event = ws.receive_json()
            assert event['type'] == 'DISCUSSION_STARTED'
            assert event['metadata']['status'] == 'running'
            assert client.get('/api/health').json() == {'status':'ok'}
            assert client.post('/api/discussions', json={'question':'Q'}).status_code == 409
            assert client.get('/api/discussions/latest').json()['session']['status'] == 'running'
        assert service.latest()['session']['status'] == 'running'
    finally:
        release.set()
    assert done.wait(5)
    assert service.latest()['session']['status'] == 'completed'


def test_ws_order_and_terminal_rest_agree(harness):
    client, _, _, _, _ = harness
    with client.websocket_connect('/ws/events') as ws:
        response = client.post('/api/discussions', json={'question':'Q', 'rounds':1,'max_new_tokens':3})
        events = []
        while not events or events[-1]['type'] != 'DISCUSSION_COMPLETED':
            events.append(ws.receive_json())
        state = client.get('/api/discussions/' + response.json()['session_id']).json()
        assert state['revision'] == len(events)
        assert [e['metadata']['turn'] for e in events if e['type']=='AGENT_TURN_COMPLETED'] == state['turns']
        assert [e['metadata']['handoff'] for e in events if e['type']=='HANDOFF_COMPLETED'] == state['handoffs']
        assert events[-1]['metadata']['summary'] == state['summary']


@pytest.mark.parametrize('stage', ['initial', 'generation', 'summary', 'handoff'])
def test_errors_project_failed_and_keep_real_undelivered_turn(harness, monkeypatch, stage):
    from token_kv_agent_system.stateful_token import StatefulTokenSession
    client, _, backend, _, _ = harness
    def fail(*a, **kw): raise ValueError('injected execution failure')
    if stage == 'initial': backend.prefill = fail
    if stage == 'generation': backend.continue_from_cache = fail
    if stage == 'summary': monkeypatch.setattr(StatefulTokenSession, 'generate_local_message', fail)
    if stage == 'handoff': monkeypatch.setattr(StatefulTokenSession, 'handoff', fail)
    state = start_wait(harness)
    assert state['status'] == 'failed' and 'injected' in state['error']
    assert state['final_answer'] is None and state['cache_released']
    assert harness[-1].events[-1].event_type.value == 'DISCUSSION_FAILED'
    if stage == 'handoff':
        assert len(state['turns']) == 1 and len(state['handoffs']) == 1
        assert state['handoffs'][0]['executed_mode'] is None
        assert state['handoffs'][0]['error'] and not state['handoffs'][0]['append_succeeded']


def test_missing_model_and_session_not_found():
    with TestClient(create_app()) as client:
        assert client.get('/api/discussions/latest').json() == dict(available=False, model=None, status='idle', session=None, active_session_id=None)
        assert client.post('/api/discussions',json={'question':'Q'}).status_code == 503
        assert client.get('/api/discussions/missing').status_code == 404


def test_recent_history_bounded_and_new_sessions_use_fresh_roles(harness):
    for i in range(4): start_wait(harness, rounds=1, session_id=f'session-{i}')
    assert harness[0].get('/api/discussions/session-0').status_code == 404
    assert len(harness[1]._sessions) == 3
    assert harness[0].post('/api/discussions',json={'question':'Q','session_id':'session-0'}).status_code == 409
    assert sum(c[0] == 'prefill' for c in harness[2].calls) == 8
    assert 'cache' not in vars(harness[1])


def test_old_canonical_controls_and_snapshot_contract():
    with TestClient(create_interactive_app()) as client:
        old = client.get('/api/state').json()
        assert old['runtime']['mode'] == 'interactive'
        assert client.patch('/api/control/network',json={'bandwidth_mbps':1000}).status_code == 200
        assert client.post('/api/control/communication').json()['communication_mode'] == 'KV'
        assert client.get('/api/discussions/latest').json()['status'] == 'idle'


def test_worker_releases_runtime_and_agent_caches_before_terminal_event(harness, monkeypatch):
    from token_kv_agent_system.dashboard import discussion_service as module
    original = module.DiscussionRuntime
    references = []
    def tracked(*args, **kw):
        runtime = original(*args, **kw)
        references.extend([weakref.ref(runtime)] + [weakref.ref(runtime.session.state(n).cache) for n in ('A','B')])
        return runtime
    monkeypatch.setattr(module, 'DiscussionRuntime', tracked)
    start_wait(harness)
    assert references and all(ref() is None for ref in references)
