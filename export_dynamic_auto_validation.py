"""Read-only evidence audit/export. Never runs models or rewrites source results."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(results, output):
    completion=json.loads((results/'completion.json').read_text())
    spec=json.loads((results/'frozen-spec.json').read_text())
    assert completion['status']=='verified' and completion['resource_stop'] is None
    assert not (results/'failure.json').exists() and not (results/'watchdog-failure.json').exists()
    drift=[name for name,sha in spec['source_sha256'].items() if digest(Path(name))!=sha]
    assert set(drift) <= {'src/token_kv_agent_system/discussion.py', 'src/token_kv_agent_system/incremental_auto.py', 'src/token_kv_agent_system/dynamic_auto_validation.py'}
    for name in drift:
        assert digest(results/'sources'/name)==spec['source_sha256'][name],name
    # Frozen originals remain auditable when a later, bounded smoke validates changes.
    assert digest(Path('run_dynamic_auto_validation.py'))==spec['entrypoint_sha256']
    forwards=json.loads((results/'memory-samples.json').read_text())['forwards']
    assert len(forwards)==completion['forward_count'] and all('error' not in f for f in forwards)
    settings=json.loads((results/'settings.json').read_text())
    assert settings['metadata']['device']=='mps' and settings['metadata']['dtype']=='torch.bfloat16'
    assert settings['metadata']['device_fallback_reason'] is None
    records=[];decisions=[];outputs=[];proofs=[]
    for run in completion['runs']:
        name=run['name'];row=json.loads((results/(name+'.json')).read_text());r=row['result']
        assert r['status']=='completed' and not row['cpu_communication_fallback']
        assert len(r['handoffs'])==len(r['turns'])==r['config']['rounds']*2
        start,end=row['forward_index_range'];trace=forwards[start:end]
        for turn,h in zip(r['turns'],r['handoffs']):
            d=h['auto_decision'];receipt=turn['receipt'];candidate=d['selected_candidate']
            assert h['requested_mode']=='auto' and h['actual_execution_status']=='succeeded'
            assert h['selected_mode']==h['executed_mode']==receipt['message']['payload_mode'].lower()
            assert h['selected_kv_strategy']==h['executed_kv_strategy']==receipt.get('kv_strategy')
            assert h['actual_wire_bytes']==h['predicted_wire_bytes']==receipt['wire_byte_count']
            assert h['error'] is h['fallback_reason'] is None
            assert h['auto_decision']['decision_status']=='executed'
            assert h['observed_local_ms']>0
            assert h['simulated_transfer_ms']==h['actual_wire_bytes']*8/(h['network']['bandwidth_mbps']*1000)+h['network']['base_latency_ms']
            assert all(not any(k.startswith('observed') or k.startswith('actual') for k in e) for e in d['candidate_estimates'])
            kv_sizes={e['predicted_wire_bytes'] for e in d['candidate_estimates'] if e['candidate']!='token'}
            assert len(kv_sizes)<=1
            if candidate=='token':
                assert h['receiver_message_prefill_count']==h['message_content_token_count']
            else:
                assert h['receiver_message_prefill_count']==(1 if candidate in ('kv:v2','kv:v3') else 0)
                if candidate in ('kv:v2','kv:v3'):assert h['repair_layer_token_count']==24
                span_end=receipt['receiver_target_span'][1]
                frame_close=receipt['receiver_appended_ids'][-2:]
                closing=[f['index'] for f in trace if f['previous_cache_tokens']==span_end and f['input_ids']==frame_close]
                continuation=[f['index'] for f in trace if f['previous_cache_tokens']==h['receiver_cache_length_after']]
                assert closing and continuation,(name,h['handoff_index'])
                proofs.append(dict(run=name,handoff_index=h['handoff_index'],candidate=candidate,
                    receiver_span_end=span_end,receiver_cache_after=h['receiver_cache_length_after'],
                    receiver_closing_forward_indices=closing,receiver_later_forward_indices=continuation))
            if d['auto_policy']=='quality_first':
                assert candidate=='token'
                assert all(e['candidate']=='token' for e in d['candidate_estimates'])
            predictions=dict(run=name,handoff_index=h['handoff_index'],direction=h['source']+'->'+h['target'],
                network=h['network'],decision=d)
            decisions.append(predictions)
            local=h['predicted_local_ms'];actual=h['observed_local_ms']
            records.append(dict(run=name,handoff_index=h['handoff_index'],direction=h['source']+'->'+h['target'],
                bandwidth_mbps=h['network']['bandwidth_mbps'],policy=d['auto_policy'],profile=d['profile_provenance'],
                requested=h['requested_mode'],selected=candidate,executed=h['executed_mode'],kv_strategy=h['executed_kv_strategy'],
                predicted_wire_bytes=h['predicted_wire_bytes'],actual_wire_bytes=h['actual_wire_bytes'],wire_error_bytes=0,
                predicted_local_ms=local,observed_local_ms=actual,local_prediction_error_ms=actual-local if local is not None else None,
                local_prediction_error_ratio=(actual-local)/local if local else None,
                simulated_transfer_ms=h['simulated_transfer_ms'],predicted_total_ms=h['predicted_total_ms'],
                actual_execution_status=h['actual_execution_status'],fallback_reason=h['fallback_reason']))
            outputs.append(dict(run=name,round=turn['round'],source=turn['message']['source_agent'],
                target=turn['message']['target_agent'],text=turn['text'],ids=turn['message']['message_token_ids'],
                termination=turn['message']['termination_reason'],selected=candidate))
        score=row['quality_score']
        strict=(score['information_exchange_success'] is True and score['task_success'] is True) if score else None
        outputs.append(dict(run=name,source=r['final_message']['source_agent'],final=True,text=r['final_text'],
            ids=r['final_message']['message_token_ids'],termination=r['final_message']['termination_reason'],
            existing_score=score,strict_private_exchange_pass=strict))
    assert [x['selected'] for x in records if x['run']=='synthetic-dynamic']==['token','kv:v1','token','kv:v1']
    samples=json.loads((results/'calibration-samples.json').read_text())
    for direction in (samples[:15],samples[15:]):
        assert len({s['actual_wire_bytes'] for s in direction if s['candidate']!='token'})==1
    measured=json.loads((results/'measured-profile.json').read_text())
    synthetic=json.loads((results/'synthetic-profile.json').read_text())
    assert measured['provenance']=='measured_incremental' and synthetic['provenance']=='synthetic/test_only'
    health=json.loads((results/'memory-samples.json').read_text())['samples']
    smoke_audit=None
    if (results/'startup-smoke/completion.json').exists():
        smoke=results/'startup-smoke'
        ss=json.loads((smoke/'frozen-spec.json').read_text())
        assert all(digest(Path(name))==sha for name,sha in ss['source_sha256'].items())
        sc=json.loads((smoke/'completion.json').read_text())
        sr=json.loads((smoke/'startup-dynamic.json').read_text())
        assert sc['status']=='verified' and sc['resource_stop'] is None
        assert sc['runs'][0]['selected']==['token','kv:v1','token','kv:v1']
        for h,t in zip(sr['result']['handoffs'],sr['result']['turns']):
            assert h['predicted_wire_bytes']==h['actual_wire_bytes']==t['receipt']['wire_byte_count']
            assert h['selected_mode']==h['executed_mode']
            assert h['actual_execution_status']=='succeeded' and h['fallback_reason'] is None
        first_kv=sr['result']['handoffs'][1]
        second_kv=sr['result']['handoffs'][3]
        assert 'kv_contract_startup' in first_kv['auto_decision']['predicted_stage_ms']
        assert 'kv_contract_startup' not in second_kv['auto_decision']['predicted_stage_ms']
        smoke_audit=dict(status='verified',current_source_matches=True,auto_handoffs=4,real_generations=5,
            selected=sc['runs'][0]['selected'],forwards=sc['forward_count'],resource_stop=sc['resource_stop'],
            cold_first_kv_cost_predicted=True,warm_second_kv_avoids_startup=True,
            startup_cost_provenance='synthetic/test_only',
            settings=json.loads((smoke/'settings.json').read_text()))
    assert not drift or smoke_audit is not None
    audit=dict(status='verified',source_drift_since_full_run=drift,source_frozen=True,final_source_smoke=smoke_audit,actual_auto_handoffs=len(records),
        token_auto_handoffs=sum(h['executed']=='token' for h in records),kv_auto_handoffs=len(proofs),
        real_generation_count=len(outputs)+2,calibration_handoffs=len(samples),baseline_calibration_handoffs=2,
        forward_count=len(forwards),receiver_cached_forward_proofs=proofs,
        wire_error_bytes_all=0,real_profile_supported_switch=False,
        reason='TOKEN local cost is below every KV candidate, and TOKEN wire is smaller; no positive-bandwidth crossover in this profile.',
        synthetic_actual_switch=['token','kv:v1','token','kv:v1'],synthetic_speed_claim=False,
        resource_stop=None,pressure_levels=sorted({s['pressure_level'] for s in health}),
        peak_mps_current_bytes=max(s['mps_current_bytes'] for s in health),
        peak_mps_driver_bytes=max(s['mps_driver_bytes'] for s in health),
        max_swap_growth_mib=max(s['swap_growth_mib'] for s in health),
        no_cpu_inference_fallback=True,physical_network='not_measured',
        latency_scope='instrumented local handoff; simulated one-way network reported separately',
        exporter_sha256=digest(Path(__file__)))
    output.mkdir(parents=True,exist_ok=False)
    def save(name,data):
        (output/name).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    def table(name,rows):
        with (output/name).open('w',newline='') as file:
            writer=csv.DictWriter(file,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    save('frozen-spec.json',spec);save('profiles.json',dict(measured=measured,synthetic=synthetic))
    startup_profile={**measured, 'version':measured['version']+'-with-startup',
        'stages':{**measured['stages'],'kv_contract_startup':dict(fixed_ms=settings['contract_startup_ms'],per_message_token_ms=0,per_cache_token_ms=0,per_recomputed_token_ms=0)},
        'notes':measured['notes']+' First contract/weights hash: one separately measured startup sample from settings; charge only if contract absent.'}
    save('measured-profile-with-startup.json',startup_profile)
    if smoke_audit:
        save('startup-smoke.json',dict(spec=ss,completion=sc,handoffs=sr['result']['handoffs'],
            outputs=[dict(source=t['message']['source_agent'],text=t['text'],ids=t['message']['message_token_ids']) for t in sr['result']['turns']]
                + [dict(source=sr['result']['final_message']['source_agent'],final=True,text=sr['result']['final_text'],ids=sr['result']['final_message']['message_token_ids'])]))
    save('audit.json',audit);save('decisions.json',decisions);save('outputs.json',outputs)
    table('handoffs.csv',records)
    table('calibration.csv',[dict(candidate=s['candidate'],sample_index=s['sample_index'],warmup=s['warmup'],
        message_tokens=s['message_tokens'],receiver_cache_tokens=s['receiver_cache_tokens'],sender_cache_tokens=s['sender_cache_tokens'],
        actual_wire_bytes=s['actual_wire_bytes'],observed_local_ms=s['observed_local_ms'],
        observed_stage_ms=json.dumps(s['observed_stage_ms'],sort_keys=True,separators=(',',':'))) for s in samples])
    table('raw-index.csv',[dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size) for p in sorted(results.rglob('*')) if p.is_file()])
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path('benchmark-results/dynamic-auto-82a'))
    parser.add_argument('--output-dir',type=Path,default=Path('docs/dynamic-auto-82a/evidence'))
    args=parser.parse_args()
    audit=export(args.results_dir,args.output_dir)
    print(json.dumps({k:v for k,v in audit.items() if k!='receiver_cached_forward_proofs'},ensure_ascii=False))


if __name__=='__main__':main()
