"""Read-only acceptance audit/export. Never loads models or replaces failed runs."""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

from .distributed_smoke import verify_packets
from .distributed_wire import sha


def audit(root):
    root=Path(root)
    result=json.loads((root/'result.json').read_text())
    assert result['status']=='completed' and result['owned_processes_stopped']
    readiness=result['readiness']
    assert len({readiness[k]['pid'] for k in ('A','B','relay_a','relay_b')})==4
    assert result['pretrained'] and result['shared_weights'] is False
    assert [r['case'] for r in result['results']]==['token','kv:v0','kv:v1','kv:v2','kv:v3','auto_quality','auto_idealized']
    records=verify_packets(root/'processes',result['results'])
    assert records==result['packets'] and len(records)==18
    modes=Counter()
    terminations=Counter()
    bytes_total=application_total=0
    worker_events={}
    worker_forwards={}
    for name in ('A','B'):
        folder=root/'processes'/f'worker-{name.lower()}'
        worker_events[name]=[json.loads(line) for line in (folder/'events.jsonl').read_text().splitlines()]
        worker_forwards[name]=[json.loads(line) for line in (folder/'forwards.jsonl').read_text().splitlines()]
        assert len(worker_forwards[name])==result['worker_resources'][name.lower()]['forward_count']
        assert all('error' not in row and row['input_ids'] for row in worker_forwards[name])
    for run in result['results']:
        assert run['status']=='completed' and len(run['turns'])==(4 if run['case'] in ('token','kv:v0') else 2)
        assert run['final']['message']['message_token_ids']
        terminations[run['final']['message']['termination_reason']]+=1
        for who in ('A','B'):
            s=run['final_states'][who]
            assert s['pending'] is None and s['cached_tokens']==s['ledger_tokens']
            assert len(s['cache_sha256'])==64 and len(s['cache_layer_shapes'])==24
            assert s['contract']==run['workers'][who]['contract']
            assert s['model']['dtype']=='torch.bfloat16' and s['model']['device']=='mps'
            assert s['model']['device_fallback_reason'] is None
        for turn in run['turns']:
            delivery=turn['delivery'];r=delivery['receipt'];source,target=turn['source'],turn['target']
            mode=r['kv_strategy'] or r['mode'];modes[mode]+=1
            terminations[turn['generated']['message']['termination_reason']]+=1
            assert delivery['outcome']=='receiver_committed' and r['state_committed'] and r['cache_imported']
            assert r['receiver_cache_after']>r['receiver_cache_before']
            assert any(row['previous_cache_tokens']==r['receiver_cache_after'] for row in worker_forwards[target])
            assert any(e['event']=='receiver_result' and e['packet_sha256']==r['packet_sha256']
                       and e['status']=='committed' for e in worker_events[target])
            assert any(e['event']=='sender_committed' and e['receipt']==r for e in worker_events[source])
            assert delivery['measurement']['retries']==0 and delivery['measurement']['success']
            assert delivery['measurement']['acknowledgement']=='remote_agent_commit_v1'
            assert delivery['measurement']['payload_bytes']==delivery['application_bytes']
            if run['case'] in ('token','auto_quality'):assert mode=='token'
            if run['case']=='kv:v0':assert mode=='raw_delta'
            if run['case']=='kv:v1':assert mode=='rope_aligned_delta'
            if run['case']=='kv:v2':assert r['message_prefill_count']==1 and len(r['repaired_token_indices'])==1
            if run['case']=='kv:v3':assert len(r['repaired_token_indices'])==1 and len(r['repaired_layer_indices'])==24
            decision=delivery['decision']
            if decision:
                assert decision['predicted_wire_bytes']==delivery['application_bytes']
                assert (decision['selected_candidate']=='token')==(r['mode']=='token')
                assert decision['network']==turn['network']
            bytes_total+=delivery['protocol_bytes'];application_total+=delivery['application_bytes']
        if run['case']=='auto_idealized':
            assert [t['delivery']['receipt']['mode'] for t in run['turns']]==['kv','token']
            assert [t['network']['config_revision'] for t in run['turns']]==[0,1]
        for cmd in run['control_commands']:
            if cmd['operation']=='generate':assert cmd['argument_keys']==['final','session_id','stage']
            if cmd['operation']=='send':assert cmd['argument_keys']==['auto','choice','handoff','network','session_id']
    manifests={str(p.relative_to(root)):dict(sha256=sha(p.read_bytes()),bytes=p.stat().st_size)
               for p in sorted(root.rglob('*')) if p.is_file()}
    for name,expected in result['source_sha256'].items():
        assert sha(Path(__file__).with_name(name).read_bytes())==expected, f'smoke source drift: {name}'
    return dict(status='verified',cases=7,handoffs=18,generations=sum(terminations.values()),
        generation_terminations=dict(terminations),protocol_bytes=bytes_total,application_request_bytes=application_total,
        executed_modes=dict(modes),forwards=sum(len(rows) for rows in worker_forwards.values()),
        worker_pids={n:readiness[n]['pid'] for n in ('A','B')},
        scope='Mac loopback TCP, independently committed model states; no distributed atomicity claim',
        raw_manifest=manifests),result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args(argv)
    checked,result=audit(args.results_dir)
    if args.output_dir:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        for name,data in (('audit.json',checked),('smoke.json',result)):
            (args.output_dir/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
        for name in ('relay-a.jsonl','relay-b.jsonl','budget.json'):
            (args.output_dir/name).write_bytes((args.results_dir/'processes'/name).read_bytes())
    print(json.dumps({k:v for k,v in checked.items() if k!='raw_manifest'},ensure_ascii=False))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
