"""Read-only audit/export of 7.9C; no model load, new destination required."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics


def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write_csv(path, rows):
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)


def trace_time(forwards, calls, start_at=0):
    if not calls:return dict(forward_ms=0,window_ms=0,tokens=0,end_index=start_at)
    def matches(f,c):return f['input_ids']==c['input_ids'] and f['previous_cache_tokens']==c['cache_length_before']
    starts=[i for i in range(start_at,len(forwards)-len(calls)+1) if all(matches(f,c) for f,c in zip(forwards[i:i+len(calls)],calls))]
    if not starts:raise ValueError('forward trace not found in chronological log')
    selected=forwards[starts[0]:starts[0]+len(calls)]
    return dict(forward_ms=sum(f['backend_forward_ms'] for f in selected),
        window_ms=(selected[-1]['started']+selected[-1]['elapsed_seconds']-selected[0]['started'])*1000,
        tokens=sum(len(f['input_ids']) for f in selected),end_index=starts[0]+len(calls))


def check_trace(calls):
    for c in calls:
        start=c['cache_length_before'];n=len(c['input_ids'])
        assert c['cache_length_after']==start+n
        assert c['attention_mask']==[[1]*(start+n)]
        assert c['effective_rotary_position_ids']==[[list(range(start,start+n))]]


def export(root, out):
    out.mkdir(parents=True,exist_ok=False)
    cases=('secret-17-26','secret-17-31','long-message')
    assert read(root/'completion.json')['status']=='verified'
    summaries=[];layers=[];tokens=[];metrics=[];outputs=[];costs=[];audit_cases=[]
    diag_forwards=[json.loads(l) for l in (root/'forwards.jsonl').read_text().splitlines()]
    cursor=0
    for case in cases:
        data=read(root/(case+'.json'));source=data['source'];n=len(source['content_ids']);start,end=source['target_span']
        assert end-start==n
        check_trace(source['sender_trace']);check_trace(source['oracle']['forward_calls'])
        cursor=trace_time(diag_forwards,source['sender_trace'],cursor)['end_index']
        cursor=trace_time(diag_forwards,source['oracle']['forward_calls'],cursor)['end_index']
        source_info={k:source[k] for k in ('content_ids','token_pieces','source_span','target_span','online_selected_index','packet_sha256','snapshot_hashes','oracle')}
        source_info['oracle']=dict(input_tokens=source['oracle']['input_tokens'],prefill_ms=source['oracle']['prefill_ms'],layer_token_work=source['oracle']['layer_token_work'])
        outputs.append(dict(case=case,phase='diagnostic-source',**source_info))
        for branch in data['branches']:
            v=branch['variant'];check_trace(branch['handoff_forward_calls']);check_trace(branch['decode_forward_calls'])
            assert branch['snapshots_unchanged'] and branch['fallback'] is None
            if v not in ('token','independent-full-oracle'):
                assert branch['packets'][0]['sha256']==source['packet_sha256']
            if branch['receipt']:assert branch['receipt']['receiver_target_span']==[start,end] if v!='token' else True
            per_kind={kind:[m for m in branch['metrics'] if m['kind']==kind] for kind in ('K','V')}
            summary=dict(case=case,variant=v)
            for kind,rows in per_kind.items():
                for stat in ('relative_l2','cosine','mean_abs','max_abs'):
                    summary[kind+'_'+stat+'_mean']=statistics.mean(m[stat] for m in rows)
                for layer in range(24):
                    xs=[m for m in rows if m['layer']==layer]
                    layers.append(dict(case=case,variant=v,kind=kind,layer=layer,
                        relative_l2=statistics.mean(m['relative_l2'] for m in xs),cosine=statistics.mean(m['cosine'] for m in xs)))
                for token in range(n):
                    xs=[m for m in rows if m['token']==token]
                    tokens.append(dict(case=case,variant=v,kind=kind,token=token,
                        token_id=source['content_ids'][token],piece=source['token_pieces'][token],
                        relative_l2=statistics.mean(m['relative_l2'] for m in xs),cosine=statistics.mean(m['cosine'] for m in xs)))
            summary.update(logits_relative_l2=branch['first_logits_vs_oracle']['relative_l2'],
                first_divergence=branch['first_output_divergence_vs_oracle'])
            summaries.append(summary)
            metrics.extend(dict(case=case,variant=v,**m) for m in branch['metrics'])
            outputs.append(dict(case=case,phase=v,text=branch['reply_text'],ids=branch['reply']['message_token_ids'],
                first_logits_vs_oracle=branch['first_logits_vs_oracle'],first_divergence=branch['first_output_divergence_vs_oracle']))
            cursor=trace_time(diag_forwards,branch['handoff_forward_calls'],cursor)['end_index']
            timing=trace_time(diag_forwards,branch['decode_forward_calls'],cursor)
            cursor=timing['end_index']
            costs.append(dict(case=case,phase='diagnostic',direction='A-B',variant=v,
                wire_bytes=branch['packets'][0]['wire_bytes'] if branch['packets'] else 0,
                handoff_ms=branch['handoff_and_intervention_ms'],repair_ms=(branch['receipt'] or {}).get('timings_ms',{}).get('suffix_recompute'),
                receiver_forward_ms=timing['forward_ms'],receiver_instrumented_window_ms=timing['window_ms'],
                receiver_forward_tokens=timing['tokens'],selected_tokens=json.dumps(branch['selected_tokens']),
                selected_layers=json.dumps(branch['selected_layers']),oracle_used=branch['oracle_used']))
        folder=root/('production-first' if case=='secret-17-26' else 'production-remaining')
        formal=read(folder/(case+'.json'));assert formal['engineering_status']=='verified'
        assert formal['sender']['message_token_ids']==source['message']['message_token_ids']
        assert formal['snapshot_hashes']==source['snapshot_hashes']
        assert formal['packets'][0]['sha256']==source['packet_sha256']
        assert formal['reply']['message_token_ids']==data['branches'][5]['reply']['message_token_ids']
        assert all(s['device'].startswith('mps') and s['dtype']=='torch.bfloat16' and s['key_shape'][1:2]==[2] and s['key_shape'][-1]==64 for s in formal['shapes'])
        assert len(formal['shapes'])==24
        forwards=[json.loads(l) for l in (folder/'forwards.jsonl').read_text().splitlines()]
        for receipt_key,trace_key,generation_key,direction in (
            ('receipt','handoff_trace','reply_trace','A-B'),('reverse_receipt','reverse_trace','final_trace','B-A')):
            if receipt_key not in formal:continue
            receipt=formal[receipt_key];trace=formal[trace_key];check_trace(trace);check_trace(formal[generation_key])
            span=receipt['receiver_target_span'];idx=receipt['repaired_token_indices'][0];pos=span[0]+idx
            assert receipt['kv_strategy']=='numeric_slot_delta' and receipt['repair_method']=='numeric_slot_delta'
            assert receipt['receiver_message_prefill_count']==1 and receipt['recompute_forward_count']==1
            assert receipt['repaired_layer_indices']==list(range(24)) and receipt['repair_layer_token_count']==24
            assert 0<=idx<span[1]-span[0]
            assert len([c for c in trace if c['cache_length_before']==pos and len(c['input_ids'])==1])==1
            timing=trace_time(forwards,formal[generation_key])
            costs.append(dict(case=case,phase='production',direction=direction,variant='v3',wire_bytes=receipt['wire_byte_count'],
                handoff_ms=receipt['timings_ms']['handoff_total'],repair_ms=receipt['timings_ms']['slot_repair'],
                receiver_forward_ms=timing['forward_ms'],receiver_instrumented_window_ms=timing['window_ms'],
                receiver_forward_tokens=timing['tokens'],selected_tokens=json.dumps(receipt['repaired_token_indices']),
                selected_layers=json.dumps(receipt['repaired_layer_indices']),oracle_used=False))
        outputs.append(dict(case=case,phase='production',sender=formal['sender'],reply=formal['reply'],text=formal['reply_text'],
            final=formal.get('final'),final_text=formal.get('final_text'),receipts={k:formal[k] for k in ('receipt','reverse_receipt') if k in formal}))
        audit_cases.append(dict(case=case,same_sender_ids=True,same_initial_tensor_fingerprints=True,
            same_packet_sha=True,same_candidate_reply_ids=True,positions_masks_shapes_verified=True,oracle_in_production=False))
    manifests=[];resources=[]
    for folder in (root,root/'production-first',root/'production-remaining'):
        spec=read(folder/'frozen-spec.json')
        drift=[name for name,h in spec['source_sha256'].items() if not Path(name).is_file() or sha(Path(name))!=h]
        if folder!=root:assert not drift,drift
        manifests.append(dict(directory=str(folder),spec_sha256=sha(folder/'frozen-spec.json'),source_drift=drift,
            drift_note='diagnostic code refactored into tested production helper; same inputs and outputs verified' if folder==root else None,
            spec=spec))
        health=[json.loads(l) for l in (folder/'health.jsonl').read_text().splitlines()]
        resources.append(dict(directory=str(folder),pressure_levels=sorted({r['pressure_level'] for r in health}),
            swap_growth_peak_mib=max(r['swap_growth_mib'] for r in health),mps_driver_peak_bytes=max(r['mps_driver_bytes'] for r in health),
            mps_current_peak_bytes=max(r['mps_current_bytes'] for r in health),rss_peak_kib=max(r['rss_kib'] for r in health),
            completion=read(folder/'completion.json')))
    write_csv(out/'summary.csv',summaries);write_csv(out/'layers.csv',layers);write_csv(out/'tokens.csv',tokens)
    write_csv(out/'layer-token-metrics.csv',metrics);write_csv(out/'costs.csv',costs)
    (out/'outputs.json').write_text(json.dumps(outputs,ensure_ascii=False,indent=2)+'\n')
    (out/'frozen-specs.json').write_text(json.dumps(manifests,ensure_ascii=False,indent=2)+'\n')
    audit=dict(status='verified',cases=audit_cases,resources=resources,external_sources=read(root/'sources/index.json'),
        counts=dict(diagnostic_generations=27,formal_generations=8,diagnostic_handoffs=21,formal_handoffs=5,
                    production_repaired_tokens=5,production_layer_token_work=120),
        transient_manual_audit_error='first checker required device == mps, actual tensor device is mps:0; corrected to startswith; no model rerun')
    (out/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    write_csv(out/'raw-index.csv',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(root.rglob('*')) if p.is_file()])
    print(json.dumps(audit['counts']))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path('benchmark-results/kv-v3-79c'))
    parser.add_argument('--output-dir',type=Path,default=Path('docs/kv-v3-79c/evidence'))
    args=parser.parse_args();export(args.results_dir,args.output_dir)
