"""Protocol integration with random tiny CPU weights; never load a pretrained model."""
from threading import Event
import pytest

from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard.discussion_service import DiscussionService
from token_kv_agent_system.metrics import MetricCollector
from token_kv_agent_system.stateful_token import StatefulTokenSession
from fastapi.testclient import TestClient


@pytest.mark.parametrize('mode', ['token','kv:v0','kv:v1','mixed'])
def test_api_actual_uniform_and_mixed_protocol_trace(mode, monkeypatch):
    from test_incremental_kv_transport import make_backend
    b = make_backend()
    collector = MetricCollector()
    service = DiscussionService(b, collector=collector)
    done = Event()
    collector.subscribe(lambda e: done.set() if e.event_type.value in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED') else None)
    original = StatefulTokenSession.handoff
    token_calls = []
    def token(*args):
        token_calls.append(1)
        return original(*args)
    monkeypatch.setattr(StatefulTokenSession,'handoff',token)
    plan = ['token','kv:v1','token','kv:v0'] if mode=='mixed' else [mode]*4
    config = {'question':'Q','rounds':2,'max_new_tokens':2,'summarizer':'B','mode':mode}
    if mode == 'mixed':
        config['handoff_plan'] = [dict(round=i//2+1,source='A' if i%2==0 else 'B',target='B' if i%2==0 else 'A',mode=m) for i,m in enumerate(plan)]
    with TestClient(create_app(collector=collector,discussion_service=service)) as client:
        assert client.post('/api/discussions',json=config).status_code == 202
        assert done.wait(20)
        result = client.get('/api/discussions/latest').json()['session']
        assert result['status']=='completed', result['error']
        assert result['final_answer']['agent']=='B'
        for h, expected in zip(result['handoffs'],plan):
            assert h['requested_mode']==h['executed_mode']==('token' if expected=='token' else 'kv')
            assert h['requested_kv_strategy']==h['executed_kv_strategy']
            assert h['wire_bytes']==h['token_protocol_bytes']+h['incremental_kv_envelope_bytes']
            assert h['receiver_message_prefill_count']==(2 if expected=='token' else 0)
            assert h['error'] is h['fallback_reason'] is None
            assert h['append_succeeded'] and h['cache_ledger_consistent']
        assert len(token_calls)==plan.count('token')
        assert result['summary']['total_wire_bytes']==sum(h['wire_bytes'] for h in result['handoffs'])


def test_kv_restore_failure_never_falls_back_to_token(monkeypatch):
    from test_incremental_kv_transport import make_backend
    from token_kv_agent_system import incremental_kv_receive
    b=make_backend()
    def fail(*a,**kw): raise ValueError('RoPE incompatible / append failed')
    monkeypatch.setattr(incremental_kv_receive,'receive_incremental_kv',fail)
    monkeypatch.setattr(StatefulTokenSession,'handoff',lambda *a,**kw: pytest.fail('hidden TOKEN fallback'))
    collector=MetricCollector(); service=DiscussionService(b,collector=collector); done=Event()
    collector.subscribe(lambda e: done.set() if e.event_type.value=='DISCUSSION_FAILED' else None)
    with TestClient(create_app(collector=collector,discussion_service=service)) as client:
        assert client.post('/api/discussions',json={'question':'Q','mode':'kv:v1','max_new_tokens':2}).status_code==202
        assert done.wait(20)
        result=client.get('/api/discussions/latest').json()['session']
        assert result['status']=='failed' and 'RoPE incompatible' in result['error']
        h=result['handoffs'][0]
        assert h['requested_mode']=='kv' and h['executed_mode'] is None
        assert h['wire_bytes']>0 and h['receiver_message_prefill_count'] is None
        assert h['fallback_reason'] is None and not h['append_succeeded']
