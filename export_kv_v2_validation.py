"""Read-only post-hoc audit and compact Git evidence export; never load a model."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def export(results,docs):
    root=Path(__file__).resolve().parent
    spec=json.loads((results/'frozen-spec.json').read_text())
    for path,expected in spec['source_sha256'].items():
        assert digest(root/path)==expected, f'frozen source changed: {path}'
    completion=json.loads((results/'completion.json').read_text())
    assert completion['status']=='verified' and completion['resource_stop'] is None
    rows=[];generations=0;handoffs=0;recomputed=0;suffix_forwards=0;trace_count=0
    text=['# 7.9B 完整真实输出与核心 IDs','',f'原始目录：`{results}`。仅真实 Sender Decode；每个用例 A 只生成一次。所有正文为模型实际输出；来源/位置/完整 mask 在对应 JSON。','']
    for case in ('secret-17-26','secret-17-31','long-message'):
        d=json.loads((results/(case+'.json')).read_text());source=d['source'];generations+=1
        text += [f'## {case}', '', 'A 原文：', '```text',source['sender_text'],'```','',
                 'A 实际 generated IDs：', '```json',json.dumps(source['sender']['message_token_ids']), '```','',
                 '同输入 B generation ledger IDs（所有分支逐 ID 相同）：','```json',
                 json.dumps(d['modes'][0]['receiver_generation_context']['ledger_token_ids']),'```','']
        text += ['|variant|k / N|首 argmax|共同输出前缀 / TOKEN|首次分叉|','|---|---|---|---|---|']
        n=source['export']['sender_span_length']
        for row in d['modes']:
            receipt=row['receipt'];k=receipt.get('receiver_message_prefill_count',n)
            text.append(f"|{row['variant']}|{k}/{n}|{row['first_logits_top5']['ids'][0]}|{row['common_output_prefix_with_token']}|{row['first_output_divergence_with_token']}|")
        text.append('')
        for row in d['modes']:
            generations+=1+int('final' in row);handoffs+=1+int('reverse_receipt' in row)
            assert row['snapshots_unchanged'] and row['fallback'] is None
            assert row['receiver_generation_context']['template_ids_equal']
            if row['variant']!='token':assert row['packets'][0]['sha256']==source['source_packet_sha256']
            for receipt,trace in [(row['receipt'],row['handoff_forward_calls'])]+(
                [(row['reverse_receipt'],row['reverse_forward_calls'])] if 'reverse_receipt' in row else []):
                if receipt.get('kv_strategy')=='suffix_recompute_delta':
                    k=receipt['receiver_message_prefill_count'];recomputed+=k;suffix_forwards+=int(k>0)
                    assert receipt['recompute_forward_count']==int(k>0)
                    start,end=receipt['receiver_target_span'];ids=receipt['received_token_ids']
                    if receipt['message']['termination_reason']=='eos':ids=ids[:-1]
                    suffix=ids[len(ids)-k:] if k else []
                    # Framing is a different Token sequence, even at k=0's end position.
                    calls=[c for c in trace if c['cache_length_before']==end-k and c['input_ids']==suffix]
                    assert len(calls)==int(k>0),f"{case}/{row['variant']} suffix forward mismatch"
                    if k:assert calls[0]['cache_length_after']==end
                assert receipt['receiver_cached_count_after']==receipt['receiver_target_span'][1]+2 if 'receiver_target_span' in receipt else True
            traces=[source['sender_trace']] if row is d['modes'][0] else []
            traces += [row['handoff_forward_calls'],row['receiver_generation_forward_calls']]
            if 'reverse_forward_calls' in row:traces += [row['reverse_forward_calls'],row['final_forward_calls']]
            for trace in traces:
                for c in trace:
                    before=c['cache_length_before'];after=before+len(c['input_ids']);trace_count+=1
                    assert c['cache_length_after']==after
                    assert c['attention_mask']==[[1]*after]
                    assert c['effective_rotary_position_ids']==[[list(range(before,after))]]
            receipt=row['receipt'];cost=receipt.get('timings_ms') or {}
            rows.append(dict(case=case,variant=row['variant'],content_tokens=n,
                recomputed_tokens=receipt.get('receiver_message_prefill_count',0),wire_bytes=receipt['wire_byte_count'],
                **cost,handoff_with_trace_ms=row['handoff_with_trace_ms'],receiver_stage_and_decode_ms=row['receiver_stage_and_decode_ms']))
            text += [f"### {row['variant']}", '', 'B 原文：','```text',row['reply_text'],'```','',
                     'B generated IDs：','```json',json.dumps(row['reply']['message_token_ids']),'```','']
            if 'final' in row:
                text += ['A Final 原文：','```text',row['final_text'],'```','',
                         'A Final generated IDs：','```json',json.dumps(row['final']['message_token_ids']),'```','']
    health=[json.loads(s) for s in (results/'health.jsonl').read_text().splitlines()]
    assert all(h['stop_reason'] is None for h in health)
    audit=dict(real_generations=generations,real_handoffs=handoffs,recomputed_tokens=recomputed,
        suffix_forward_calls=suffix_forwards,audited_forward_traces=trace_count,
        guarded_forward_count=completion['forward_count'],no_fallback=True,no_resource_stop=True,
        peaks={k:max(h[k] for h in health) for k in ('mps_current_bytes','mps_driver_bytes','rss_kib','swap_used_mib','pressure_level')})
    docs.mkdir(parents=True,exist_ok=True)
    (docs/'kv-v2-79b-raw-outputs.md').write_text('\n'.join(text).rstrip()+'\n')
    (docs/'kv-v2-79b-frozen-spec.json').write_bytes((results/'frozen-spec.json').read_bytes())
    (docs/'kv-v2-79b-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (docs/'kv-v2-79b-costs.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys,restval='not_measured',lineterminator='\n');writer.writeheader();writer.writerows(rows)
    files=sorted(results.glob('*.json'))+sorted(results.glob('*.jsonl'))
    # Also retain the zero-forward initial startup failure without overwriting it.
    files+=sorted(results.parent.glob('*.json'))+sorted(results.parent.glob('*.jsonl'))
    with (docs/'kv-v2-79b-index.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['path','bytes','sha256'])
        for path in files:writer.writerow([str(path),path.stat().st_size,digest(path)])
    print(json.dumps(audit,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path('benchmark-results/kv-v2-79b/validated'))
    parser.add_argument('--docs-dir',type=Path,default=Path('docs'))
    args=parser.parse_args();export(args.results_dir,args.docs_dir)
