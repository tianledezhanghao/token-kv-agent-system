"""Independent worker integration plus model import/ACK uncertainty and isolation."""
from contextlib import contextmanager
from dataclasses import asdict, replace
import json
from pathlib import Path
import socket
import sys
import threading
import time
from uuid import uuid4

import pytest

from test_incremental_kv_transport import make_backend
from test_stateful_token_cache import copies, unchanged
from token_kv_agent_system.communication import HandoffChoice
from token_kv_agent_system.discussion import DiscussionConfig
from token_kv_agent_system.distributed_agent import AgentExecutor
from token_kv_agent_system.distributed_discussion import DistributedDiscussion, create_app
from token_kv_agent_system.distributed_lab import DistributedLab
from token_kv_agent_system.distributed_wire import RemoteAgentTransport, pack_delivery, unpack_delivery, sha
from token_kv_agent_system.distributed_worker import (
    AgentDataService, WorkerControl, control_call, resource_budget, CONTROL_FIELDS)
from token_kv_agent_system.incremental_kv_protocol import canonical_json
from token_kv_agent_system.network.handoff_network import HandoffNetwork, HandoffNetworkConfig
from token_kv_agent_system.transport import SocketConfig
from token_kv_agent_system.transport.socket_wire import recv_frame, SocketCounters


def make_executor(name):
    b = make_backend()
    b.outputs = [59]*100  # real tiny forward, explicit controlled digit predictions
    return AgentExecutor(name,b,max_context_tokens=2048)


@contextmanager
def pair():
    a,b = make_executor('A'),make_executor('B')
    servers,threads = [],[]
    try:
        for e in (a,b):
            server = AgentDataService(('127.0.0.1',0),e,config=SocketConfig(timeout_seconds=2))
            thread = threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
            thread.start();servers.append(server);threads.append(thread)
        for e,peer,server in ((a,b,servers[1]),(b,a,servers[0])):
            def query(op, peer=peer, **args):
                return getattr(peer,{'selection':'selection_summary'}.get(op,op))(**args)
            e.configure(RemoteAgentTransport({peer.agent_id:server.server_address},config=SocketConfig(timeout_seconds=2)),
                        query,peer.status())
            e.start('s',dict(question='What is 1?',rounds=2,max_new_tokens=2,max_context_tokens=2048))
        yield a,b,servers
    finally:
        for server in servers:server.shutdown();server.server_close()
        for thread in threads:thread.join(timeout=2)


def prepared(a,mode='token'):
    a.generate('s',1)
    return a.prepare('s',1,HandoffChoice.parse(mode))


@pytest.mark.parametrize('mode',['token','kv:v0','kv:v1','kv:v2','kv:v3'])
def test_remote_import_uses_exact_bytes_and_continues_both_caches(mode):
    with pair() as (a,b,_):
        assert set(a.session._states)=={'A'} and set(b.session._states)=={'B'}
        assert a.backend is not b.backend and a.backend._model is not b.backend._model
        for index,(sender,receiver) in enumerate(((a,b),(b,a),(a,b),(b,a)),1):
            sender.generate('s',index)
            result=sender.send('s',index,mode)
            assert result['outcome']=='receiver_committed'
            r=result['receipt']
            assert r['stages']==['request_received','protocol_verified','cache_imported_and_corrected','state_committed']
            assert r['protocol_bytes']==result['protocol_bytes']
            assert r['receiver_cache_after']>r['receiver_cache_before']
            assert receiver.status()['received'][-1]==r['message_id']
            assert sender.outgoing is None
            if mode=='kv:v2':assert r['message_prefill_count']==1
            if mode=='kv:v3':assert r['repaired_token_indices']==[0]
        assert a.generate('s',5,True)['message']['message_token_ids']
        for e in (a,b):
            audit=e.audit('s')
            assert audit['cached_tokens']==audit['ledger_tokens'] and audit['pending'] is None
        assert a.audit('s')['cache_sha256']!=b.audit('s')['cache_sha256']


@pytest.mark.parametrize('mode',['token','kv:v0','kv:v1','kv:v2','kv:v3'])
def test_receiver_import_failure_is_rejected_and_rolls_back(mode):
    with pair() as (a,b,_):
        a.generate('s',1)
        old=b.session.state('B');saved=copies(old.cache)
        def fail(*args,**kwargs):raise RuntimeError('injected receiver forward failure')
        b.backend.continue_from_cache=fail
        result=a.send('s',1,mode)
        assert result['outcome']=='receiver_rejected' and not result['measurement']['success']
        assert not result['receipt']['state_committed']
        assert b.session.state('B') is old and a.session.state('A').draft_message
        unchanged(old.cache,saved)
        with pytest.raises(ValueError,match='unresolved'):a.generate('s',1)


def test_duplicate_returns_original_receipt_without_forward_and_conflict_rejected():
    with pair() as (a,b,_):
        out=prepared(a,'kv:v2')
        r=b.receive(out['packet'],'A','B')
        before=b.audit('s');calls=len(b.backend.calls)
        assert b.receive(out['packet'],'A','B')==r
        assert b.audit('s')==before and len(b.backend.calls)==calls
        meta,protocol=unpack_delivery(out['packet'])
        meta['handoff']=3
        with pytest.raises(ValueError,match='conflicting'):
            b.receive(pack_delivery(meta,protocol),'A','B')


def test_lost_ack_is_unknown_blocks_resend_then_query_reconciles():
    with pair() as (a,b,_):
        def dropping_connector(address,timeout):
            client,peer=socket.socketpair()
            def consume():
                with peer:
                    _,_,s,t,packet=recv_frame(peer,SocketConfig(),time.monotonic()+2,SocketCounters())
                    b.receive(packet,s,t)  # Receiver commits, response is deliberately lost.
            threading.Thread(target=consume,daemon=True).start()
            return client
        a.transport.connector=dropping_connector
        a.generate('s',1)
        result=a.send('s',1,'kv:v0')
        assert result['outcome']=='delivery_unknown' and result['receipt'] is None
        assert b.status()['received']==['A:1'] and a.status()['sent']==[]
        with pytest.raises(ValueError,match='outgoing'):a.send('s',1,'kv:v0')
        assert len(a.transport.measurements)==1
        assert a.reconcile('s')['outcome']=='receiver_committed'
        assert a.status()['sent']==['A:1'] and a.outgoing is None
        b.generate('s',2)


def test_no_receipt_does_not_prove_failure_and_peer_restart_cannot_commit():
    with pair() as (a,b,_):
        prepared(a)
        assert a.reconcile('s')['outcome']=='delivery_unknown'
        out=a.outgoing
        a.peer_query=lambda *args,**kwargs:dict(status='committed',schema='remote_agent_commit_v1',
            **{**out['expected'],'receiver_boot_id':'different'},cache_imported=True,state_committed=True)
        with pytest.raises(ValueError,match='identity'):a.reconcile('s')
        assert a.status()['sent']==[]


@pytest.mark.parametrize('fault',['contract','session','direction','sequence','handoff','incarnation','bytes','v3_no_digit'])
def test_remote_validation_and_rejection(fault):
    with pair() as (a,b,_):
        if fault=='v3_no_digit':b.backend.decode=lambda *args,**kwargs:'no numeric content'
        out=prepared(a,'kv:v3' if fault=='v3_no_digit' else 'kv:v0')
        meta,protocol=unpack_delivery(out['packet']);old=b.session.state('B')
        source,target='A','B'
        if fault=='contract':meta['contract']['weights_sha256']='0'*64
        elif fault=='session':meta['session_id']='other'
        elif fault=='direction':source,target='B','A'
        elif fault=='sequence':meta['message']['sequence']=2;meta['message']['message_id']='A:2'
        elif fault=='handoff':meta['handoff']=3
        elif fault=='incarnation':meta['receiver_boot_id']='other'
        elif fault=='bytes':protocol=protocol[:-1]+bytes([protocol[-1]^1])
        try:
            r=b.receive(pack_delivery(meta,protocol),source,target)
            assert r['status']=='rejected' and r['state_committed'] is False
        except ValueError:
            pass
        assert b.session.state('B') is old and not b.status()['received']


def test_contract_mismatch_prevents_session_initialization():
    a,b=make_executor('A'),make_executor('B')
    status=b.status();status['contract']['config_sha256']='0'*64
    with pytest.raises(ValueError,match='ModelContract'):a.configure(None,None,status)
    assert a.session is None


@pytest.mark.parametrize('operation,extra',[
    ('generate',dict(local_prompt='secret bypass')),('send',dict(packet='tensor')),
    ('receive',dict(message='bypass')),('append_local_user',dict(text='bypass'))])
def test_control_plane_rejects_arbitrary_peer_content(operation,extra):
    control=WorkerControl(make_executor('B'),SocketConfig())
    request=dict(operation=operation,**{k:None for k in CONTROL_FIELDS.get(operation,())},**extra)
    with pytest.raises(ValueError,match='data port'):control.dispatch(request)


def test_session_release_tombstone_and_stage_guards():
    with pair() as (a,b,_):
        with pytest.raises(ValueError,match='stage'):b.generate('s',1)
        with pytest.raises(ValueError,match='stage'):a.generate('s',5,True)
        before=b.audit('s')
        b.release('s')
        assert b.session is None
        with pytest.raises(ValueError,match='reused'):b.start('s',{})
        with pytest.raises(ValueError,match='inactive'):b.audit('s')
        assert before['cached_tokens']>0


@pytest.mark.parametrize('fault',['no_route','disconnect','timeout','byte_verifier'])
def test_transport_failure_cannot_report_model_success(fault):
    with pair() as (a,b,servers):
        a.generate('s',1)
        if fault=='no_route':
            a.transport.routes={}
            with pytest.raises(KeyError):a.send('s',1,'token')
            assert not b.status()['received']
            return
        if fault=='byte_verifier':
            from test_socket_transport import service
            with service(target='B') as verifier:
                a.transport.routes={'B':verifier.server_address}
                result=a.send('s',1,'token')
        else:
            def connector(*args):
                client,peer=socket.socketpair()
                if fault=='disconnect':peer.close()
                else:
                    threading.Thread(target=lambda:(time.sleep(.15),peer.close()),daemon=True).start()
                return client
            a.transport.connector=connector
            a.transport.config=SocketConfig(timeout_seconds=.05)
            result=a.send('s',1,'token')
        assert result['outcome'] in ('delivery_unknown','not_sent') and result['receipt'] is None
        assert not a.status()['sent'] and not b.status()['received']


@pytest.fixture(scope='module')
def processes(tmp_path_factory):
    path=tmp_path_factory.mktemp('distributed')/'lab'
    with DistributedLab(path,worker_command=[sys.executable,str(Path(__file__).with_name('distributed_tiny_worker.py'))],
                        config=SocketConfig(timeout_seconds=20,chunk_bytes=997)) as lab:
        yield lab


@pytest.mark.parametrize('mode',['token','kv:v0','kv:v1','kv:v2','kv:v3'])
def test_independent_process_multi_round_discussion(processes,mode):
    lab=processes
    commands=[]
    cfg=DiscussionConfig(question='What is 1?',rounds=2,max_new_tokens=2,max_context_tokens=2048,
                         handoff_choice=HandoffChoice.parse(mode))
    runtime=DistributedDiscussion(lab.workers,routes=lab.routes,config=cfg,
                                 event_sink=lambda e,r:commands.append((e,r)))
    result=runtime.run()
    assert result['status']=='completed',result
    assert len(result['turns'])==4 and result['final']['message']['message_token_ids']
    pids=[r['pid'] for r in result['workers'].values()]
    import os
    assert len(set(pids))==2 and os.getpid() not in pids
    for name in ('A','B'):
        state=result['final_states'][name]
        assert len(state['sent'])==len(state['received'])==2
        assert state['pending'] is None and state['cached_tokens']==state['ledger_tokens']
        assert control_call(lab.workers[name],'status')['session_id'] is None
    for row in result['turns']:
        assert row['delivery']['receipt']['mode']==('token' if mode=='token' else 'kv')
        assert row['delivery']['measurement']['acknowledgement']=='remote_agent_commit_v1'
    for e,r in commands:
        if e=='control_command' and r['operation']=='generate':
            assert r['argument_keys']==['final','session_id','stage']
    for name in ('relay-a','relay-b'):
        rows=[json.loads(line) for line in (lab.output_dir/f'{name}.jsonl').read_text().splitlines()]
        assert sum(r.get('sent_bytes',0) for r in rows)>0


@pytest.mark.parametrize('bandwidth,expected',[(.001,'token'),(100000,'kv')])
def test_distributed_auto_idealized_selection_equals_actual_relay_execution(processes,bandwidth,expected):
    from token_kv_agent_system.incremental_auto import IncrementalAutoConfig
    runtime=DistributedDiscussion(processes.workers,routes=processes.routes,config=DiscussionConfig(
        question='What is 1?',rounds=1,max_new_tokens=2,max_context_tokens=2048,
        auto=IncrementalAutoConfig(policy='performance_experiment',idealized=True,candidates=('token','kv:v2')),
        network=HandoffNetworkConfig(bandwidth_mbps=bandwidth)))
    result=runtime.run()
    assert result['status']=='completed',result
    for turn in result['turns']:
        delivery=turn['delivery']
        assert delivery['receipt']['mode']==expected
        assert delivery['decision']['predicted_wire_bytes']==delivery['application_bytes']
        assert delivery['measurement']['payload_bytes']==delivery['application_bytes']
        assert delivery['decision']['cost_source']=='synthetic'


def test_quality_auto_and_network_patch(processes):
    from token_kv_agent_system.incremental_auto import IncrementalAutoConfig
    runtime=DistributedDiscussion(processes.workers,routes=processes.routes,config=DiscussionConfig(
        question='What is 1?',rounds=1,max_new_tokens=2,max_context_tokens=2048,
        auto=IncrementalAutoConfig()))
    def update(event,row):
        if event=='turn_completed' and row['source']=='A':
            runtime.network.update(HandoffNetworkConfig(bandwidth_mbps=100000,source='live_patch'))
    runtime.event_sink=update
    result=runtime.run()
    assert result['status']=='completed',result
    assert [t['network']['config_revision'] for t in result['turns']]==[0,1]
    assert all(t['delivery']['receipt']['mode']=='token' for t in result['turns'])


def test_rest_admission_and_config_validation(tmp_path):
    from fastapi.testclient import TestClient
    app=create_app(dict(workers={'A':'http://127.0.0.1:1','B':'http://127.0.0.1:2'},routes={}),tmp_path)
    with TestClient(app) as client:
        assert client.get('/state').json()['status']=='idle'
        assert client.patch('/network',json={}).status_code==409
        assert client.post('/discussion',json={'rounds':0}).status_code==422


def test_budget_missing_model_never_downloads(tmp_path):
    with pytest.raises(ValueError,match='local model'):resource_budget(tmp_path/'missing')
    with pytest.raises(ValueError,match='dimensions'):resource_budget(tmp_path,max_context_tokens=-1)


def test_profile_identity_explicitly_separates_single_and_distributed_execution():
    from token_kv_agent_system.incremental_auto import backend_cost_identity, IncrementalAutoConfig, IncrementalCostProfile, StageCost
    baseline=make_backend()
    single=backend_cost_identity(baseline)
    with pair() as (a,b,_):
        remote=backend_cost_identity(a.backend)
        assert remote['execution_scope']=='distributed_agent_commit_v1' and remote!=single
        a.generate('s',1)
        p=IncrementalCostProfile('old','synthetic/test_only',{'sender':single,'receiver':single},
            {'token_handoff':StageCost(20),'kv_base':StageCost(5)},0,(0,4096),(0,8192),'old local identity')
        with pytest.raises(ValueError,match='model_profile_mismatch'):
            a.send('s',1,None,auto=asdict(IncrementalAutoConfig(policy='performance_experiment',profile=p)),
                   network=asdict(HandoffNetwork().snapshot(1)))
        assert a.outgoing is None and not a.transport.measurements


def test_concurrent_identical_delivery_imports_once():
    with pair() as (a,b,_):
        out=prepared(a,'kv:v1')
        results=[]
        threads=[threading.Thread(target=lambda:results.append(b.receive(out['packet'],'A','B'))) for _ in range(4)]
        for t in threads:t.start()
        for t in threads:t.join(timeout=3)
        assert len(results)==4 and all(r==results[0] for r in results)
        assert b.status()['received']==['A:1'] and b.status()['last_handoff']==1


def test_actual_large_kv_import_over_new_application_socket():
    import torch
    import transformers
    from test_incremental_kv_transport import ContractTokenizer
    from token_kv_agent_system.model import HuggingFaceModelBackend
    from token_kv_agent_system.agent.state import AgentMessage, append_working_ids, working_state
    def large(name):
        torch.manual_seed(75)
        config=transformers.Qwen2Config(vocab_size=256,hidden_size=128,intermediate_size=160,
            num_hidden_layers=2,num_attention_heads=2,num_key_value_heads=2,head_dim=64,
            max_position_embeddings=4096,eos_token_id=2)
        backend=HuggingFaceModelBackend('fixture/large',device='cpu',tokenizer=ContractTokenizer(),
            model=transformers.Qwen2ForCausalLM(config).eval(),torch_module=torch)
        return AgentExecutor(name,backend,max_context_tokens=4096)
    a,b=large('A'),large('B')
    server=AgentDataService(('127.0.0.1',0),b,config=SocketConfig(timeout_seconds=20,chunk_bytes=4093))
    thread=threading.Thread(target=server.serve_forever,kwargs={'poll_interval':.01},daemon=True);thread.start()
    try:
        for e,peer in ((a,b),(b,a)):
            e.configure(RemoteAgentTransport({peer.agent_id:server.server_address},
                config=SocketConfig(timeout_seconds=20,chunk_bytes=4093)),None,peer.status())
            e.start('large',dict(question='1?',rounds=1,max_new_tokens=2,max_context_tokens=4096))
        old=a.session.state('A');start=old.cached_token_count
        ids=(59,)*1025
        # Engineering fixture: real batched source forward, not pretrained
        # generation evidence. Every exported KV position is ledger-proven.
        work=append_working_ids(working_state(old),ids)
        message=AgentMessage('A:1',1,'A','B',ids,'max_new_tokens',(start,start+len(ids)))
        a.session._states['A']=replace(work,draft_message=message)
        result=a.send('large',1,'kv:v0')
        assert result['outcome']=='receiver_committed' and result['protocol_bytes']>2*1024**2
        assert result['receipt']['receiver_cache_after']-result['receipt']['receiver_cache_before']>=1025
        assert b.generate('large',2)['message']['message_token_ids']
        assert result['measurement']['send_calls']>100
    finally:
        server.shutdown();server.server_close();thread.join(timeout=2)


def test_broken_real_relay_has_no_inmemory_fallback(tmp_path):
    with DistributedLab(tmp_path/'broken',worker_command=[sys.executable,str(Path(__file__).with_name('distributed_tiny_worker.py'))],
                        config=SocketConfig(timeout_seconds=2)) as lab:
        lab.processes[-1].terminate();lab.processes[-1].wait(timeout=3)
        runtime=DistributedDiscussion(lab.workers,routes=lab.routes,config=DiscussionConfig(
            question='1?',rounds=1,max_new_tokens=2,max_context_tokens=2048))
        result=runtime.run()
        assert result['status']=='failed'
        assert result['turns'][0]['delivery']['outcome']=='not_sent'
        a=control_call(lab.workers['A'],'status');b=control_call(lab.workers['B'],'status')
        assert a['draft_id']=='A:1' and a['sent']==[] and b['received']==[]
        assert len(result['turns'])==1 and result['final'] is None


def test_rest_runs_real_workers_and_patches_next_unfrozen_handoff(processes,tmp_path,monkeypatch):
    from fastapi.testclient import TestClient
    reached,proceed=threading.Event(),threading.Event()
    original=HandoffNetwork.snapshot
    def pause_second(network,index):
        if index==2:
            reached.set()
            assert proceed.wait(5)
        return original(network,index)
    monkeypatch.setattr(HandoffNetwork,'snapshot',pause_second)
    app=create_app(dict(workers=processes.workers,routes=processes.routes),tmp_path)
    with TestClient(app) as client:
        response=client.post('/discussion',json=dict(question='1?',rounds=1,max_new_tokens=2,max_context_tokens=2048,
            auto=dict(policy='performance_experiment',idealized=True)))
        assert response.status_code==200 and reached.wait(5)
        assert client.post('/discussion',json={}).status_code==409
        patched=client.patch('/network',json=dict(bandwidth_mbps=100000,source='rest_patch'))
        assert patched.status_code==200 and patched.json()['next_unfrozen_handoff_index']==2
        proceed.set()
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            state=client.get('/state').json()
            if state['status']=='completed':break
            time.sleep(.01)
        assert state['status']=='completed',state
        turns=state['result']['turns']
        assert [t['network']['config_revision'] for t in turns]==[0,1]
        assert turns[1]['delivery']['receipt']['mode']=='kv'
        assert turns[1]['delivery']['decision']['predicted_wire_bytes']==turns[1]['delivery']['application_bytes']


def test_manual_recompute_policy_and_b_summarizer_survive_control_serialization(processes):
    from token_kv_agent_system.communication import KVStrategy,CommunicationMode
    choice=HandoffChoice(CommunicationMode.KV,KVStrategy.SUFFIX_RECOMPUTE_DELTA,recompute_tokens=2)
    runtime=DistributedDiscussion(processes.workers,routes=processes.routes,config=DiscussionConfig(
        question='1?',rounds=1,max_new_tokens=3,max_context_tokens=2048,summarizer='B',handoff_choice=choice))
    result=runtime.run()
    assert result['status']=='completed',result
    assert result['final']['message']['source_agent']=='B'
    assert all(t['delivery']['receipt']['message_prefill_count']==2 for t in result['turns'])


@pytest.mark.parametrize('lost_control_response',[False,True])
def test_coordinator_unknown_is_preserved_when_diagnostic_queries_also_disconnect(lost_control_response):
    runtime=DistributedDiscussion({'A':'http://a:1','B':'http://b:2'},config=DiscussionConfig(rounds=1))
    def call(name,operation,**arguments):
        if operation=='status':
            return dict(pid=1 if name=='A' else 2,host_id='one-host',contract={},agent_id=name)
        if operation=='start':return {}
        if operation=='generate':return dict(message={'message_id':'A:1'},text='display only')
        if operation=='send':
            if lost_control_response:raise TimeoutError('send command response lost')
            return dict(outcome='delivery_unknown',application_bytes=100,receipt=None)
        if operation=='audit':raise OSError('diagnostic control channel also disconnected')
        pytest.fail(f'unexpected operation {operation}')
    runtime.call=call
    result=runtime.run()
    assert result['status']=='delivery_unknown' and len(result['cleanup_errors'])==2
    assert not result['released'] and result['final'] is None
