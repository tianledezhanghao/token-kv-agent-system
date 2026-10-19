"""Read-only SHA verification, offline analysis and compact 8.3A evidence export."""
import argparse
import ast
import csv
import json
import shutil
from pathlib import Path

from .local_qwen_comparison import file_hash
from .performance_characterization import offline_analysis
from .performance_validation import dump, summarize_group


def verify_index(root):
    rows=json.loads((root/'sha256-index.json').read_text())
    for row in rows:
        path=(root/row['path']).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file():
            raise ValueError('invalid/missing indexed path')
        if path.stat().st_size!=row['bytes'] or file_hash(path)!=row['sha256']:
            raise ValueError('raw SHA mismatch: '+row['path'])
    return rows


def export(results_dir, output_dir):
    index=verify_index(results_dir)
    groups=json.loads((results_dir/'groups.json').read_text())
    spec=json.loads((results_dir/'frozen-spec.json').read_text())
    settings=json.loads((results_dir/'settings.json').read_text())
    generations=json.loads((results_dir/'generations.json').read_text())
    resources=json.loads((results_dir/'resource-samples.json').read_text())
    samples=[];paired=[]
    for group in groups:
        c=group['config'];label=c['label']
        rows=[json.loads(line) for line in (results_dir/(label+'-samples.jsonl')).read_text().splitlines()]
        assert group['snapshot_unchanged'] and len(rows)==group['successful_handoffs']
        assert summarize_group(rows)==group['summary'],'statistics do not reproduce'
        assert all(f['device'].startswith('mps') for s in rows for f in s['forward_calls'])
        for mode in ('steady','instrumented'):
            for option in c['options']:
                branch=[s for s in rows if s['mode']==mode and s['candidate_label']==option['label']]
                assert len(branch)==spec['warmup']+spec['samples']
                assert sum(not s['warmup'] for s in branch)==spec['samples']
            if 'token' in group['summary'][mode]:
                deltas=[]
                for i in range(spec['warmup'],spec['warmup']+spec['samples']):
                    pair={s['candidate_label']:s['local_ms'] for s in rows if s['mode']==mode and s['index']==i}
                    deltas.append(pair['kv:v0']-pair['token'])
                paired.append(dict(label=label,mode=mode,kv_minus_token_ms=deltas,
                    median=__import__('statistics').median(deltas),min=min(deltas),max=max(deltas),
                    note='paired in time/order, same base inputs; not independent task success samples'))
        for sample in rows:
            receipt=sample['receipt']
            assert receipt['received_token_ids']==c['message_ids']
            assert receipt['wire_byte_count']==sample['wire_bytes']
            if sample['candidate']!='token':
                assert receipt['message']['payload_mode']=='KV'
                assert receipt['receiver_message_prefill_count']==len(receipt['repaired_token_indices'])
                assert receipt['receiver_target_span'][1]-receipt['receiver_target_span'][0]==c['message_tokens']
                if sample['candidate'] in ('kv:v0','kv:v1'):
                    assert receipt['receiver_message_prefill_count']==0
            if sample['mode']=='instrumented':
                total=sum(s['exclusive_ms'] for s in sample['stages'].values())
                assert abs(total-sample['stages']['handoff_residual_validation_commit']['inclusive_ms'])<1e-5
                assert 0 <= sample['local_ms']-total < 1,'external boundary or nested sum mismatch'
            samples.append(dict(group=label,mode=sample['mode'],candidate=sample['candidate_label'],index=sample['index'],
                warmup=sample['warmup'],local_ms=sample['local_ms'],wire_bytes=sample['wire_bytes'],
                recomputed_tokens=receipt.get('receiver_message_prefill_count',c['message_tokens']),
                sender_pending_supplement=receipt.get('sender_supplemented_token_count',0),
                repaired_token_indices=receipt.get('repaired_token_indices',[]),
                forward_calls=sample['forward_calls'],stages=sample['stages'],adapter_submetrics=sample['adapter_submetrics'],
                mps_allocated_after_bytes=sample['mps_allocated_after_bytes'],mps_driver_after_bytes=sample['mps_driver_after_bytes']))
    expected=2*(spec['warmup']+spec['samples'])*sum(len(r['config']['options']) for r in groups)
    assert len(groups)==11 and len(samples)==expected and len(generations)==2
    assert {r['config']['message_tokens'] for r in groups if r['config']['label'].startswith('length-')}=={2,8,32,128,256,512}
    assert {r['config']['receiver_cache_tokens'] for r in groups if r['config']['label'].startswith('history-')}=={512,1024,2048}
    assert all(g['message']['source_agent']=='A' for g in generations)
    assert settings['metadata']['device']=='mps' and settings['metadata']['dtype']=='torch.bfloat16'
    assert settings['metadata']['device_fallback_reason'] is None
    assert settings['revision']=='7ae557604adf67be50417f59c2c2f167def9a775'
    assert not any(r['stop_reason'] for r in resources)
    networks,profiles,decisions=offline_analysis(groups,settings)
    assert len(networks)==10 and len(profiles)==10 and len(decisions)==50
    # Frozen capture preserved after a post-measurement analysis failure.
    recovery=results_dir/'analysis-recovery'/'sources'
    source_audit=[]
    for path,sha in spec['source_sha256'].items():
        current=Path(path)
        same=file_hash(current)==sha
        if not same:
            captured=recovery/current.name
            assert captured.exists() and file_hash(captured)==sha,'missing measured source snapshot'
            protected={'performance_characterization.py':('StageRecorder','instrument','cache_sha256','real_prefix','measured_handoff'),
                       'performance_validation.py':('dump','pad_history','freeze_state','summarize_group','run_group')}
            def nodes(p):
                return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body
                        if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            old_nodes,new_nodes=nodes(captured),nodes(current)
            assert all(old_nodes[n]==new_nodes[n] for n in protected[current.name]),'measured operation changed'
        source_audit.append(dict(path=path,measured_sha256=sha,current_sha256=file_hash(current),unchanged=same,
            drift_scope='offline analysis refactor only; measured handoff/wrappers unchanged' if not same else None))
    failure=json.loads((results_dir/'failure.json').read_text()) if (results_dir/'failure.json').exists() else None
    audit=dict(status='verified_measurements_and_offline_analysis',core_measurements_complete=True,
        groups=len(groups),handoffs=len(samples),valid_samples=sum(not s['warmup'] for s in samples),
        warmups=sum(s['warmup'] for s in samples),generations=len(generations),raw_sha_verified=len(index),
        snapshot_unchanged=True,identity_verified=True,stage_sum_verified=True,
        protected_measurement_symbols_unchanged=True,
        model_failure_count=0,postmeasurement_failure=failure,offline_recovery_no_model_reexecution=bool(failure),
        source_audit=source_audit,resource=dict(samples=len(resources),stop_reason=None,
            pressure_levels=sorted({r['pressure_level'] for r in resources}),
            max_swap_growth_mib=max(r['swap_growth_mib'] for r in resources),
            max_mps_current_bytes=max(r['mps_current_bytes'] for r in resources),
            max_mps_driver_bytes=max(r['mps_driver_bytes'] for r in resources),
            max_rss_kib=max(r['rss_kib'] for r in resources)),
        nature='actual model local timings + actual bytes + offline mathematical network; no physical network')
    output_dir.mkdir(parents=True,exist_ok=False)
    for name,row in [('frozen-spec',spec),('settings',settings),('groups',groups),('generations',generations),
        ('network',networks),('profiles',profiles),('selector-decisions',decisions),('audit',audit),('paired-deltas',paired),('raw-index',index)]:
        dump(output_dir/(name+'.json'),row)
    drifted=[r for r in source_audit if not r['unchanged']]
    if drifted:
        destination=output_dir/'measured-sources'
        destination.mkdir()
        for row in drifted:
            name=Path(row['path']).name
            shutil.copyfile(recovery/name,destination/name)
    for source in results_dir.glob('*.json'):
        if (source.name.startswith(('short-A','short-B','long-source','history-')) and source.with_suffix('.kv').exists()):
            dump(output_dir/source.name,json.loads(source.read_text()))
    with (output_dir/'samples.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(samples[0]),lineterminator='\n')
        writer.writeheader()
        for row in samples:
            writer.writerow({k:json.dumps(v,separators=(',',':')) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    dump(output_dir/'evidence-sha256.json',[dict(path=str(p.relative_to(output_dir)),bytes=p.stat().st_size,sha256=file_hash(p))
        for p in sorted(output_dir.rglob('*')) if p.is_file()])
    return audit


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path('benchmark-results/performance-83a'))
    parser.add_argument('--output-dir',type=Path,default=Path('docs/performance-83a/evidence'))
    args=parser.parse_args(argv)
    print(json.dumps(export(args.results_dir,args.output_dir),ensure_ascii=False,indent=2))
    return 0
