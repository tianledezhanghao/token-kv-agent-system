"""Opt-in single-loaded-model REST/WS dynamic smoke; no fabricated receipts."""
from dataclasses import asdict
import json
from pathlib import Path
import threading
import time

from .generalized_calibration import write_json


def discussion_smoke(backend,profile,args,output,check):
    import httpx
    import uvicorn
    from websockets.sync.client import connect
    from fastapi.staticfiles import StaticFiles
    from .dashboard.app import create_app
    from .dashboard.discussion_service import DiscussionService
    from .metrics import MetricCollector
    collector=MetricCollector();service=DiscussionService(backend,collector=collector)
    app=create_app(collector=collector,discussion_service=service)
    dist=Path(__file__).resolve().parents[2]/'frontend/dist'
    if dist.exists():
        app.mount('/',StaticFiles(directory=str(dist),html=True),name='validation-dashboard')
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=args.port,log_level='warning'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    started=time.monotonic()
    while not server.started:
        if not thread.is_alive() or time.monotonic()-started>15:
            raise RuntimeError('validation HTTP service did not start')
        time.sleep(.05)
    base=f'http://127.0.0.1:{args.port}'
    try:
        for label,request in [('measured',dict(question='23 + 48 = ?',rounds=1,max_new_tokens=8,
                mode='auto',auto_kv_version='v0',auto_policy='performance_experiment',cost_profile=asdict(profile),bandwidth_mbps=10000.0)),
            ('idealized',dict(question='23 + 48 = ?',rounds=2,max_new_tokens=8,mode='auto',auto_kv_version='v0',
                auto_policy='performance_experiment',idealized_simulation=True,bandwidth_mbps=10000.0))]:
            check();events=[];patch=None
            with httpx.Client(base_url=base,timeout=15) as client, connect(f'ws://127.0.0.1:{args.port}/ws/events',open_timeout=10) as ws:
                response=client.post('/api/discussions',json=request)
                assert response.status_code==202,response.text
                sid=response.json()['session_id']
                while True:
                    event=json.loads(ws.recv(timeout=30))
                    if event.get('metadata',{}).get('session_id')!=sid:
                        continue
                    events.append(event)
                    if label=='idealized' and event['type']=='HANDOFF_COMPLETED' and event['metadata']['sequence']==1:
                        accepted=client.patch(f'/api/discussions/{sid}/network',json={'bandwidth_mbps':.001})
                        assert accepted.status_code==200,accepted.text
                        patch=accepted.json()
                    if event['type'] in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED'):
                        break
                result=client.get('/api/discussions/'+sid).json()
                assert client.get('/api/discussions/latest').json()['session']==result
                receipt_events=[e['metadata']['handoff'] for e in events if e['type']=='HANDOFF_COMPLETED']
                assert receipt_events==[h for h in result['handoffs'] if h['append_succeeded']]
                for h in result['handoffs']:
                    assert h['actual_wire_bytes']==h['predicted_wire_bytes']==h['wire_bytes']
                    assert h['actual_execution_status']=='succeeded' and h['fallback_reason'] is None
                    assert h['observed_local_ms']>0
                if label=='idealized':
                    assert result['status']=='completed',result.get('error')
                    assert [h['executed_mode'] for h in result['handoffs']]==['kv','token','token','token']
                    assert [h['network']['config_revision'] for h in result['handoffs']]==[0,1,1,1]
                    assert result['handoffs'][0]['executed_kv_strategy']=='raw_delta'
                    assert patch['effective_from_handoff_index']==2
                    assert all(h['auto_decision']['profile_provenance']=='synthetic' for h in result['handoffs'])
                    assert len(result['turns'])==4 and result['final_answer'] is not None and result['cache_released']
                else:
                    # First handoff is required to be in the independently calibrated
                    # domain. Later out-of-domain rejection remains a real result.
                    assert result['handoffs'],'measured smoke first workload is outside the configured calibration domain: '+str(result.get('error'))
                    assert all(h['auto_decision']['profile_provenance']=='measured' for h in result['handoffs'])
                write_json(output/(label+'-discussion.json'),result)
                write_json(output/(label+'-ws.json'),events)
                if patch:write_json(output/'dynamic-patch.json',patch)
                print(label,'Discussion',result['status'],[h['executed_mode'] for h in result['handoffs']],flush=True)
        write_json(output/'browser-ready.json',dict(url=base,session_id=sid,serve_seconds=args.serve_seconds))
        print('Browser QA ready:',base,'for',args.serve_seconds,'seconds',flush=True)
        deadline=time.monotonic()+args.serve_seconds
        while time.monotonic()<deadline:
            check();time.sleep(.5)
    finally:
        server.should_exit=True;thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError('validation server failed to close')


def serve_dashboard(backend,args,check):
    """Fixed-model server entry for CUDA. Existing Mac switching is untouched."""
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from .dashboard.app import create_app
    from .dashboard.discussion_service import DiscussionService
    from .metrics import MetricCollector
    collector=MetricCollector()
    service=DiscussionService(backend,collector=collector)
    app=create_app(collector=collector,discussion_service=service)
    dist=Path(__file__).resolve().parents[2]/'frontend/dist'
    if dist.exists():
        app.mount('/',StaticFiles(directory=str(dist),html=True),name='generalized-dashboard')
    server=uvicorn.Server(uvicorn.Config(app,host='127.0.0.1',port=args.port,log_level='info'))
    thread=threading.Thread(target=server.run,daemon=True);thread.start()
    deadline=time.monotonic()+args.serve_seconds
    print(f'Fixed-model dashboard: http://127.0.0.1:{args.port}; import the calibrated JSON in AUTO settings.',flush=True)
    try:
        while thread.is_alive() and time.monotonic()<deadline:
            check();time.sleep(.5)
        if not server.started:
            raise RuntimeError('dashboard failed to bind/start')
    finally:
        server.should_exit=True;thread.join(timeout=30)
        if thread.is_alive():
            raise RuntimeError('dashboard did not close within bounded shutdown')
