"""8.2B projection/API checks; scripted model is not real-model evidence."""
from copy import deepcopy
import pytest
from test_discussion_dashboard import harness, start_wait


def test_actual_message_ids_pending_decision_receipt_and_restore(harness):
    client, service, _, _, collector = harness
    snapshots = []
    collector.subscribe(lambda e: snapshots.append((e.event_type.value, service.get(e.metadata['session_id']))))
    result = start_wait(harness, mode='auto', rounds=3, network_mode='trace',
                        bandwidth_trace=[20.0, 1000.0, 50.0, 500.0], base_latency_ms=2.0)
    assert result['status'] == 'completed'
    assert len(result['auto_decisions']) == 6
    assert result['active_handoff'] is None
    assert [h['network']['bandwidth_mbps'] for h in result['handoffs']] == [20,1000,50,500,500,500]
    for turn, handoff, decision in zip(result['turns'],result['handoffs'],result['auto_decisions']):
        assert turn['message_id'] == handoff['message_id'] == decision['message_id']
        assert decision['handoff_index'] == handoff['sequence']
        assert handoff['requested_mode'] == 'auto'
        assert handoff['selected_mode'] == handoff['executed_mode'] == 'token'
        assert handoff['append_succeeded']
    assert result['turns'][0]['received_message_id'] is None
    assert [t['received_message_id'] for t in result['turns'][1:]] == [h['message_id'] for h in result['handoffs'][:-1]]
    assert result['final_answer']['message_id'] not in [h['message_id'] for h in result['handoffs']]
    for kind, snapshot in snapshots:
        if kind == 'AUTO_DECISION':
            pending = snapshot['active_handoff']
            assert pending['decision']['selected_candidate'] == 'token'
            assert pending['message_id'] == snapshot['auto_decisions'][-1]['message_id']
            assert len(snapshot['handoffs']) == len(snapshot['auto_decisions']) - 1
        if kind in ('HANDOFF_COMPLETED', 'DISCUSSION_COMPLETED'):
            assert snapshot['active_handoff'] is None
    assert client.get('/api/discussions/latest').json()['session'] == result
    assert client.get('/api/discussions/' + result['session_id']).json() == result


def test_failed_execution_has_id_but_no_delivery_or_fallback(harness, monkeypatch):
    from token_kv_agent_system.stateful_token import StatefulTokenSession
    def fail(*args, **kwargs): raise RuntimeError('fixture handoff failure')
    monkeypatch.setattr(StatefulTokenSession, 'handoff', fail)
    result = start_wait(harness, mode='auto')
    h = result['handoffs'][0]
    assert result['status'] == 'failed'
    assert h['message_id'] == result['turns'][0]['message_id']
    assert h['selected_mode'] == 'token' and h['executed_mode'] is None
    assert not h['append_succeeded'] and h['actual_execution_status'] == 'failed'
    assert h['auto_decision']['decision_status'] == 'execution_failed'
    assert h['fallback_reason'] is None and result['active_handoff'] is None
    events = harness[-1].events
    assert any(e.event_type.value == 'HANDOFF_FAILED' and e.metadata['handoff']['message_id'] == h['message_id'] for e in events)


@pytest.mark.parametrize('network', [dict(network_mode='manual',bandwidth_mbps=123.5,base_latency_ms=1.2), dict(network_mode='trace',bandwidth_trace=[20.0,1000.0])])
def test_manual_and_trace_accept_real_frontend_schema(harness, network):
    result = start_wait(harness, rounds=1, mode='token', **network)
    assert result['status'] == 'completed'
    assert all(h['network']['network_mode'] == network['network_mode'] for h in result['handoffs'])


def test_cost_identity_projection_and_no_server_profile_path(harness, monkeypatch):
    import token_kv_agent_system.dashboard.discussion_service as module
    identity = dict(model_id='fixture-only', device='cpu')
    monkeypatch.setattr(module,'backend_cost_identity', lambda backend: identity)
    assert harness[0].get('/api/discussions/latest').json()['cost_model_identity'] == dict(sender=identity,receiver=identity)
    response = harness[0].post('/api/discussions',json=dict(question='Q',mode='auto',cost_profile_path='/server/private.json'))
    assert response.status_code == 422
    assert harness[1].latest()['session'] is None


def test_recent_session_metadata_is_bounded_without_restarting_or_exposing_profiles(harness):
    ids=[]
    for i in range(4):
        result=start_wait(harness, rounds=1, mode='auto', session_id=f'history-{i}')
        ids.append(result['session_id'])
    latest=harness[0].get('/api/discussions/latest').json()
    assert [s['session_id'] for s in latest['recent_sessions']] == ids[:0:-1]
    assert all(set(s) == {'session_id','status','question','mode','created_at'} for s in latest['recent_sessions'])
    assert harness[0].get('/api/discussions/history-0').status_code == 404
    restored=harness[0].get('/api/discussions/history-1').json()
    assert restored['auto_decisions'] and restored['handoffs'][0]['message_id'] == restored['turns'][0]['message_id']
