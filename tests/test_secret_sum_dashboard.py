from threading import Event
import pytest
from fastapi.testclient import TestClient
from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard.discussion_service import DiscussionService
from token_kv_agent_system.metrics import MetricCollector
from test_secret_sum import fixture_backend
from token_kv_agent_system.secret_sum import PUBLIC_TASK


@pytest.mark.parametrize('mode',['token','kv:v0','kv:v1','mixed'])
def test_secret_sum_api_persisted_scene_prompts_actual_delivery_and_score(mode):
    if mode=='token': b,_=fixture_backend(); limit=96
    else:
        from test_incremental_kv_transport import make_backend
        b=make_backend(); limit=2
    collector=MetricCollector(); service=DiscussionService(b,collector=collector); done=Event()
    collector.subscribe(lambda e: done.set() if e.event_type.value in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED') else None)
    config=dict(scenario='secret_sum',secret_a=17,secret_b=26,question='controller text must never reach the models',rounds=1,mode=mode,max_new_tokens=limit)
    if mode=='mixed': config['handoff_plan']=[dict(round=1,source='A',target='B',mode='token'),dict(round=1,source='B',target='A',mode='kv:v1')]
    with TestClient(create_app(collector=collector,discussion_service=service)) as client:
        started=client.post('/api/discussions',json=config)
        assert started.status_code==202,started.text
        assert done.wait(20)
        result=client.get('/api/discussions/latest').json()['session']
        assert result['status']=='completed',result['error']
        assert result['config']['scenario']=='secret_sum' and result['config']['question']==PUBLIC_TASK
        assert result['final_answer']['agent']=='A'
        assert result['prompt_config']['version']=='secret-sum-v1'
        assert '26' not in result['prompt_config']['agent_a']['text']
        assert '17' not in result['prompt_config']['agent_b']['text']
        assert [(h['source'],h['target']) for h in result['handoffs']]==[('A','B'),('B','A')]
        assert result['score']['engineering_success'] is True
        if mode=='token': assert result['score']['information_exchange_success'] is result['score']['task_success'] is True
        for h in result['handoffs']:
            assert h['wire_bytes']>0 and h['fallback_reason'] is None
            if h['executed_mode']=='kv': assert h['receiver_message_prefill_count']==0
        assert result['cache_released']


@pytest.mark.parametrize('extra',[
    dict(secret_a=0),dict(secret_b=100),dict(secret_a=True),dict(secret_b='26'),
    dict(rounds=2),dict(summarizer='B'),dict(agent_a_system_prompt='B secret is 26'),
    dict(agent_b_system_prompt='A secret is 17'),dict(scenario='other'),
    dict(mode='mixed',handoff_plan=[]),
])
def test_invalid_scene_request_rejected_before_inference(extra):
    b,_=fixture_backend(); service=DiscussionService(b)
    with TestClient(create_app(collector=service.collector,discussion_service=service)) as client:
        response=client.post('/api/discussions',json={**dict(scenario='secret_sum',question='Q',rounds=1),**extra})
        assert response.status_code==422,response.text
        assert not b.generation_histories and not b.calls
