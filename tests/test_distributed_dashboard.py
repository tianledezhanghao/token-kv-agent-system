"""Real tiny independent Worker/dual Relay Dashboard contract; no pretrained weights."""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json
import sys
import threading
import time

import pytest
from fastapi.testclient import TestClient

from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard.distributed_service import DistributedDashboardService, handoff_view
from token_kv_agent_system.dashboard.discussion_service import DiscussionStartRequest, DiscussionService
from token_kv_agent_system.distributed_lab import DistributedLab
from token_kv_agent_system.distributed_worker import control_call
from token_kv_agent_system.metrics import MetricCollector
from token_kv_agent_system.transport import SocketConfig
from test_discussion import ScriptBackend


@pytest.fixture(scope='module')
def lab(tmp_path_factory):
    with DistributedLab(tmp_path_factory.mktemp('distributed-ui')/'processes',
        worker_command=[sys.executable,str(Path(__file__).with_name('distributed_tiny_worker.py'))],
        config=SocketConfig(timeout_seconds=3)) as lab:
        yield lab


@contextmanager
def harness(lab, tmp_path):
    collector=MetricCollector()
    settings=dict(workers=lab.workers,routes=lab.routes,timeout=5,measurement_source='Mac TCP / loopback',
        relays=deepcopy({n:lab.readiness[n] for n in ('relay_a','relay_b')}))
    distributed=DistributedDashboardService(settings,collector=collector,history_path=tmp_path/'history.json')
    local=DiscussionService(ScriptBackend([(100,2)]*30),collector=collector)
    done=threading.Event()
    collector.subscribe(lambda e:done.set() if e.metadata.get('architecture')=='distributed' and e.event_type.value in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED') else None)
    with TestClient(create_app(collector=collector,discussion_service=local,distributed_service=distributed)) as client:
        yield client,distributed,done,collector


def run(client,done,**kw):
    done.clear()
    r=client.post('/api/distributed/discussions',json=dict(question='What is 1?',architecture='distributed',
        rounds=1,max_new_tokens=2,max_context_tokens=2048,**kw))
    assert r.status_code==202,r.text
    assert done.wait(20)
    return client.get('/api/distributed/discussions/'+r.json()['session_id']).json()


@pytest.mark.parametrize('mode',['token','kv:v0','kv:v1','kv:v2','kv:v3'])
def test_real_dashboard_worker_relay_receipt_projection_all_modes(lab,tmp_path,mode):
    with harness(lab,tmp_path) as (c,s,done,collector):
        result=run(c,done,mode=mode)
        assert result['status']=='completed',result.get('error')
        assert result['architecture']=='distributed' and 'Socket' in result['transport']
        assert len(result['turns'])==2 and result['final_answer']['text']
        assert result['cache_released']
        assert c.get('/api/discussions/'+result['session_id']).status_code==404
        assert c.get('/api/distributed/discussions/'+result['session_id']).json()==result
        for h in result['handoffs']:
            r=h['receiver_receipt']
            assert h['append_succeeded'] is True and r['state_committed'] and r['cache_imported']
            assert (r['session_id'],r['message_id'],r['handoff'],r['source'],r['target'])==(result['session_id'],h['message_id'],h['sequence'],h['source'],h['target'])
            assert h['protocol_bytes']<h['application_bytes']<h['socket_measurement']['sent_bytes']
            assert h['observed_handoff_wall_ms']>=h['socket_application_delivery_ms']>=h['receiver_processing_ms']>=0
            assert h['observed_local_ms'] is None
            assert h['measurement_source']=='Mac TCP / loopback'
            assert h['executed_mode']==('token' if mode=='token' else 'kv')
            if mode=='kv:v3':assert r['repaired_token_indices']==[0]
        events=[e for e in collector.events if e.metadata.get('architecture')=='distributed']
        kinds=[e.event_type.value for e in events]
        expected=['DISCUSSION_STARTED','DISCUSSION_WORKER_READY','DISCUSSION_WORKER_READY']+[
            k for _ in range(2) for k in ['AGENT_TURN_STARTED','AGENT_TURN_COMPLETED','HANDOFF_STARTED','HANDOFF_COMPLETED']]+['DISCUSSION_FINALIZING','DISCUSSION_COMPLETED']
        assert kinds==expected
        assert [e.metadata['revision'] for e in events]==list(range(1,len(events)+1))
        assert all(e.metadata['session_id']==result['session_id'] for e in events)
        assert c.get('/api/distributed/events').json()['count']==len(events)
        recovered=DistributedDashboardService(s.settings,history_path=tmp_path/'history.json')
        assert recovered.get(result['session_id'])==result
        with pytest.raises(Exception,match='already used'):
            recovered.start(DiscussionStartRequest(question='Q',architecture='distributed',session_id=result['session_id']))
        recovered.close()


def test_architecture_isolation_readiness_and_model_switch(lab,tmp_path):
    with harness(lab,tmp_path) as (c,s,done,_):
        health=c.get('/api/distributed/health').json()
        assert health['available'] and health['model_contract_consistent']
        assert health['workers']['A']['pid']!=health['workers']['B']['pid']
        assert all(v['status']=='ready' for v in health['relays'].values())
        assert c.post('/api/discussions',json=dict(question='Q',architecture='distributed')).status_code==422
        assert c.post('/api/distributed/discussions',json=dict(question='Q')).status_code==422
        assert c.post('/api/distributed/models/switch',json=dict(model='3B')).status_code==404
        assert c.post('/api/distributed/discussions',json=dict(question='Q',architecture='distributed',scenario='secret_sum',rounds=1)).status_code==422
        s.settings=None;s._health_at=0
        health=c.get('/api/distributed/health').json()
        assert health['coordinator']=='ready' and not health['available']
        assert c.post('/api/distributed/discussions',json=dict(question='Q',architecture='distributed')).status_code==503


def test_auto_real_patch_and_snapshot_freeze(lab,tmp_path):
    with harness(lab,tmp_path) as (c,s,done,collector):
        patch=[]
        def sink(event):
            if event.event_type.value=='HANDOFF_STARTED' and event.metadata.get('architecture')=='distributed' and event.metadata['sequence']==1:
                # Synchronously PATCH AFTER first freeze and BEFORE actual send.
                sid=event.metadata['session_id']
                patch.append(s.update_network(sid,__import__('token_kv_agent_system.dashboard.discussion_service',fromlist=['DiscussionNetworkRequest']).DiscussionNetworkRequest(bandwidth_mbps=.001)))
        collector.subscribe(sink)
        result=run(c,done,mode='auto',auto_policy='performance_experiment',idealized_simulation=True,bandwidth_mbps=100000.)
        assert result['status']=='completed',result.get('error')
        assert [h['executed_mode'] for h in result['handoffs']]==['kv','token']
        assert [h['network']['config_revision'] for h in result['handoffs']]==[0,1]
        assert result['handoffs'][0]['network']['bandwidth_mbps']==100000
        assert patch[0]['effective_from_handoff_index']==2
        assert result['network_updates'][0]['config_revision']==1
        assert all(d['candidate_set']==['token','kv:v0'] and d['idealized'] for d in result['auto_decisions'])
        assert all(h['receiver_receipt']['state_committed'] for h in result['handoffs'])
        assert c.patch('/api/distributed/discussions/'+result['session_id']+'/network',json=dict(bandwidth_mbps=20.)).status_code==409
        with pytest.raises(Exception,match='Local Profile'):
            s.start(DiscussionStartRequest(question='Q',architecture='distributed',mode='auto',cost_profile=json.loads(Path('docs/generalized-auto-85/evidence/measured-profile.json').read_text())))


def test_failures_do_not_create_success_receipts_and_explicit_release(lab,tmp_path):
    with harness(lab,tmp_path) as (c,s,done,_):
        original=__import__('token_kv_agent_system.distributed_discussion',fromlist=['control_call']).control_call
        from unittest.mock import patch
        def failing(url,op,**kw):
            if op=='send':raise TimeoutError('control response lost')
            return original(url,op,**kw)
        with patch('token_kv_agent_system.distributed_discussion.control_call',side_effect=failing):
            result=run(c,done)
        assert result['status']=='delivery_unknown'
        assert result['handoffs'][0]['append_succeeded'] is None
        assert result['handoffs'][0]['executed_mode'] is None
        assert not result['cache_released'] and not result['final_answer']
        assert c.post('/api/distributed/discussions',json=dict(question='Q',architecture='distributed')).status_code==409
        r=c.post('/api/distributed/discussions/'+result['session_id']+'/release')
        assert r.status_code==200 and r.json()['rollback'] is False


def test_projection_dedup_and_restart_boundary(lab,tmp_path):
    with harness(lab,tmp_path) as (c,s,done,_):
        result=run(c,done)
        sid=result['session_id'];revision=result['revision']
        s._event(sid,'AGENT_TURN_COMPLETED',dict(turn=result['turns'][0]))
        assert len(s.get(sid)['turns'])==2 and s.get(sid)['revision']==revision
        stored=s.get(sid);stored.update(status='running',cache_released=False)
        (tmp_path/'restart.json').write_text(json.dumps([stored]))
        restored=DistributedDashboardService(history_path=tmp_path/'restart.json')
        old=restored.get(sid)
        assert old['status']=='delivery_unknown' and not old['resumable']
        assert 'Coordinator restarted' in old['error']
        restored.close()


@pytest.mark.parametrize('outcome',['receiver_rejected','not_sent','delivery_unknown'])
def test_receipt_absence_cannot_be_success(lab,tmp_path,outcome):
    with harness(lab,tmp_path) as (c,s,done,_):
        result=run(c,done)
        row=result['raw_result']['turns'][0]
        row['delivery'].update(outcome=outcome,receipt=None)
        h=handoff_view(row,result['session_id'])
        assert h['append_succeeded'] is not True and h['executed_mode'] is None
        assert h['actual_execution_status']==outcome


def test_wrong_receipt_identity_never_success(lab,tmp_path):
    with harness(lab,tmp_path) as (c,s,done,_):
        result=run(c,done)
        row=result['raw_result']['turns'][0]
        row['delivery']['receipt']['session_id']='wrong'
        assert handoff_view(row,result['session_id'])['append_succeeded'] is None


def test_broken_relay_and_contract_are_not_ready(lab,tmp_path):
    from copy import deepcopy
    from unittest.mock import patch
    with harness(lab,tmp_path) as (c,s,done,_):
        original=deepcopy(s.settings)
        s.settings['relays']['relay_a']['listeners'][0]['upstream']=['127.0.0.1',1]
        health=s.health(force=True)
        assert not health['available'] and health['relays']['relay_a']['status']=='unavailable'
        s.settings=original
        def incompatible(url,operation,**kw):
            row=control_call(url,operation,**kw)
            if url==lab.workers['B']:row['contract']['weights_sha256']='wrong'
            return row
        with patch('token_kv_agent_system.dashboard.distributed_service.control_call',side_effect=incompatible):
            health=s.health(force=True)
        assert health['model_contract_consistent'] is False and not health['available']
        assert c.get('/api/discussions/latest').json()['available'] is True


def test_stop_waits_for_handoff_and_does_not_claim_rollback(lab,tmp_path):
    with harness(lab,tmp_path) as (c,s,done,collector):
        def stop(event):
            if event.event_type.value=='HANDOFF_COMPLETED' and event.metadata.get('architecture')=='distributed':
                s._runtime.request_stop()
        collector.subscribe(stop)
        result=run(c,done)
        assert result['status']=='failed' and result['failure_stage']=='stop_boundary'
        assert result['handoffs'][0]['append_succeeded'] is True
        assert len(result['turns'])==1 and not result['final_answer']
        assert not result['cache_released'] and 'not rolled back' in result['error']
        assert c.post('/api/distributed/discussions/'+result['session_id']+'/release').status_code==200


def test_port_preflight_never_kills_existing_listener():
    import socket
    from token_kv_agent_system.dashboard.distributed_launcher import check_port
    with socket.socket() as owned:
        owned.bind(('127.0.0.1',0));owned.listen()
        with pytest.raises(OSError):check_port('127.0.0.1',owned.getsockname()[1])
        with socket.create_connection(owned.getsockname(),timeout=.2):pass
