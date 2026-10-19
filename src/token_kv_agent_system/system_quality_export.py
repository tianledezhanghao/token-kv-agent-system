"""Read-only 8.3C evidence validation and aggregation; never loads a model."""
from collections import Counter,defaultdict
from dataclasses import asdict
import json
from pathlib import Path

from .local_qwen_comparison import file_hash
from .system_quality_tasks import PAIRS,VARIANTS,score_pair
from .system_quality_validation import write_once,digest


def verified_index(path):
    rows=json.loads((path/'sha256-index.json').read_text())
    for row in rows:
        target=path/row['path']
        if not target.resolve().is_relative_to(path.resolve()):raise ValueError('unsafe evidence path')
        if not target.is_file() or target.stat().st_size!=row['bytes'] or file_hash(target)!=row['sha256']:
            raise ValueError('raw SHA mismatch: '+str(target))
    return rows


def count_values(values):
    values=list(values)
    return dict(n=len(values),yes=sum(x is True for x in values),no=sum(x is False for x in values),unknown=sum(x is None for x in values))


def paired_statistics(rows,field='final'):
    groups=defaultdict(list)
    for row in rows:groups[row['category']].append(row)
    result={}
    for category,items in groups.items():
        by_task=defaultdict(dict)
        for row in items:by_task[row['task']][row['variant']]=row
        report={}
        for variant in VARIANTS:
            rs=[x[variant] for x in by_task.values()]
            subset=[x[variant] for x in by_task.values() if x['token']['score'][field] is True]
            report[variant]=dict(all=count_values(x['score'][field] for x in rs),
                token_success_subset=count_values(x['score'][field] for x in subset),
                kv_new_failures=[name for name,x in by_task.items() if x['token']['score'][field] is True and x[variant]['score'][field] is False],
                kv_unique_success=[name for name,x in by_task.items() if x['token']['score'][field] is False and x[variant]['score'][field] is True],
                unknown_with_token_success=[name for name,x in by_task.items() if x['token']['score'][field] is True and x[variant]['score'][field] is None])
        result[category]=report
    return result


def audit_handoff(row):
    if not row['engineering']:
        assert row.get('rollback_verified') is True
        return
    receipt=row['receipt'];message=row['message'];a=row['source'];b=row['target']
    ids=message['message_token_ids'];content=ids[:-1] if message['termination_reason']=='eos' else ids
    assert receipt['received_token_ids']==ids
    assert row['after'][a]['ledger'][slice(*message['sender_message_span'])]==ids
    start=receipt['message']['receiver_message_start']
    assert row['after'][b]['ledger'][start:start+len(content)]==content
    assert row['after'][b]['cached_tokens']==receipt['receiver_cached_count_after']
    assert row['packets'][0]['wire_bytes']==receipt['wire_byte_count']
    for call in row['forward_calls']:
        before=call['cache_length_before'];n=len(call['input_ids'])
        assert call['cache_length_after']==before+n
        assert call['attention_mask']==[[1]*(before+n)]
        assert call['effective_rotary_position_ids']==[[list(range(before,before+n))]]
    if row['variant']=='kv':
        n=len(content);indices=receipt['repaired_token_indices']
        if receipt['kv_strategy']=='suffix_recompute_delta':assert indices==[n-1]
        if receipt['kv_strategy'] in ('suffix_recompute_delta','numeric_slot_delta'):
            assert receipt['recompute_forward_count']==1 and receipt['receiver_message_prefill_count']==1
            pos=start+indices[0]
            assert any(c['cache_length_before']==pos and c['input_ids']==[content[indices[0]]] for c in row['forward_calls'])
        else:assert receipt['receiver_message_prefill_count']==0
        assert all(p['executed']=='packed' for p in row['transfer_paths'])


def export(root,output):
    output.mkdir(parents=True,exist_ok=False)
    phases=['pilot-v1','pilot-v2','formal-v2','discussion-v2','profile-v2','dashboard-v2']
    index={};counts={};generations=[];handoffs=[];drift={}
    for phase in phases:
        folder=root/phase
        index[phase]=verified_index(folder)
        gs=sorted(folder.glob('*-generation-*.json'));hs=sorted(folder.glob('*-handoff-*.json'))
        counts[phase]=dict(generations=len(gs),handoffs=len(hs),
            forwards=json.loads((folder/'completion.json').read_text())['forward_count'])
        spec=json.loads((folder/'frozen-spec.json').read_text())
        write_once(output/(phase+'-frozen-spec.json'),spec)
        current=Path(__file__).resolve().parents[2]
        drift[phase]={p:dict(frozen=sha,current=file_hash(current/p)) for p,sha in spec['source_sha256'].items()
            if (current/p).is_file() and file_hash(current/p)!=sha}
        for p in gs:
            r=json.loads(p.read_text());m=r.get('message')
            if not m:continue
            generations.append(dict(phase=phase,file=p.name,message=m,text=r['text'],input_ids=r['input_ids'],
                raw_sha256=file_hash(p),input_sha256=digest(r['input_ids'])))
        for p in hs:
            r=json.loads(p.read_text());audit_handoff(r)
            handoffs.append(dict(phase=phase,file=p.name,source=r['source'],target=r['target'],variant=r['variant'],
                engineering=r['engineering'],receipt=r.get('receipt'),error=r.get('error'),rollback=r.get('rollback_verified'),
                full_handoff_ms=r.get('full_handoff_ms'),packets=r['packets'],paths=r.get('transfer_paths'),raw_sha256=file_hash(p),
                before_lengths={n:x['cached_tokens'] for n,x in r['before'].items()}))
    formal=root/'formal-v2';pairs=[];sources=[]
    for task in PAIRS:
        source=json.loads((formal/(task.name+'-source.json')).read_text())
        sources.append(dict(task=asdict(task),sender_text=source['sender_text'],message=source['message'],
            source_packet_sha256=source['source_packet_sha256'],initial=source['initial'],
            raw_sha256=file_hash(formal/(task.name+'-source.json'))))
        reference_input=None
        for variant in VARIANTS:
            p=formal/(task.name+'-'+variant.replace(':','-')+'.json');row=json.loads(p.read_text())
            assert row['score']==score_pair(task,source['sender_text'],row.get('reply_text'),delivered=row['engineering'])
            if row['engineering']:
                ids=row['evidence']['generations'][-1]['input_ids']
                if reference_input is None:reference_input=ids
                assert ids==reference_input
                if variant!='token':assert row['evidence']['handoffs'][0]['packets'][0]['sha256']==source['source_packet_sha256']
            pairs.append({k:row[k] for k in ('task','category','variant','engineering','score','snapshot_hashes','snapshots_unchanged')}
                |dict(reply_text=row.get('reply_text'),reply=row.get('reply'),error=row.get('error'),raw_sha256=file_hash(p)))
    runtimes=[]
    for phase,pattern in [('formal-v2','secret-*.json'),('discussion-v2','discussion-*.json')]:
        for p in sorted((root/phase).glob(pattern)):
            if '-generation-' in p.name or '-handoff-' in p.name:continue
            row=json.loads(p.read_text())
            runtimes.append(dict(phase=phase,file=p.name,task=row['task'],variant=row['variant'],result=row['result'],
                historical_score=row.get('historical_score'),raw_sha256=file_hash(p)))
    summary=dict(counts=counts,pair_auto_final=paired_statistics(pairs),pair_auto_extraction=paired_statistics(pairs,'extraction'),
        engineering={phase:count_values(h['engineering'] for h in handoffs if h['phase']==phase) for phase in phases},
        termination=dict(Counter(g['message']['termination_reason'] for g in generations)))
    for name,value in [('raw-index',index),('source-drift',drift),('generations',generations),('handoffs',handoffs),
        ('paired',pairs),('sources',sources),('runtimes',runtimes),('summary',summary)]:write_once(output/(name+'.json'),value)
    manifest=[dict(path=p.name,bytes=p.stat().st_size,sha256=file_hash(p)) for p in sorted(output.iterdir()) if p.is_file()]
    write_once(output/'evidence-sha256.json',manifest)
    return summary


def main(argv=None):
    import argparse
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(argv);print(json.dumps(export(a.results_dir,a.output_dir),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
