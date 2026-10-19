"""Read-only SHA/statistics/compatibility audit and small 8.3B evidence export."""
import argparse
import ast
import csv
from dataclasses import asdict
import difflib
import hashlib
import json
from pathlib import Path

from token_kv_agent_system.kv_transfer_validation import verify_index,analyze
from token_kv_agent_system.performance_validation import dump,summarize_group
from token_kv_agent_system.local_qwen_comparison import file_hash


def measured_adapter_unchanged(measured,final):
    """Only default dispatch/domain and docstrings may differ after measurement.

    Explicit legacy and packed execution branches must remain identical AST.
    """
    def normalized(text):
        tree=ast.parse(text)
        class Normalize(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant):
                    node.body=node.body[1:]
                if node.name=='_transfer_path':
                    # Scope-specific auto dispatch changed; explicit paths still checked below.
                    node.args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[])
                    node.body=[n for n in node.body if not (isinstance(n,ast.If) and ast.unparse(n.test)=='path == \'auto\'')]
                return self.generic_visit(node)
            def visit_Assign(self,node):
                if any(isinstance(t,ast.Name) and t.id in ('DEFAULT_TRANSFER_PATH','MAX_AUTO_PACKED_TRANSFER_BYTES') for t in node.targets):
                    return None
                return self.generic_visit(node)
            def visit_Call(self,node):
                if isinstance(node.func,ast.Name) and node.func.id=='_transfer_path':
                    node.args=node.args[:3];node.keywords=[]
                return self.generic_visit(node)
        return ast.dump(Normalize().visit(tree),include_attributes=False)
    if normalized(measured)!=normalized(final):
        raise ValueError('measured explicit adapter execution changed')
    return ''.join(difflib.unified_diff(measured.splitlines(True),final.splitlines(True),fromfile='measured',tofile='final'))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir',type=Path,default=Path('benchmark-results/kv-transfer-83b/validated'))
    parser.add_argument('--output-dir',type=Path,default=Path('docs/kv-transfer-83b/evidence'))
    args=parser.parse_args(argv);root=args.results_dir;out=args.output_dir
    raw_index=verify_index(root)
    completion=json.loads((root/'completion.json').read_text())
    assert completion['status']=='verified' and completion['failures']==0
    spec=json.loads((root/'frozen-spec.json').read_text())
    verify_index(Path(spec['input_dir']))
    assert all(file_hash(Path(p))==sha for p,sha in spec['protected_user_assets'].items())
    groups=json.loads((root/'groups.json').read_text());all_samples=[]
    for g in groups:
        c=g['config'];samples=[json.loads(line) for line in (root/(c['label']+'-samples.jsonl')).read_text().splitlines()]
        assert summarize_group(samples)==g['summary']
        assert len(samples)==g['successful_handoffs'] and g['failed_handoffs']==0
        for s in samples:
            assert s['local_ms']>0 and s['base_snapshot_unchanged']
            assert s['receipt']['received_token_ids']==c['message_ids']
            assert all(f['device'].startswith('mps') for f in s['forward_calls'])
            if s['mode']=='instrumented':
                inclusive=s['stages']['handoff_residual_validation_commit']['inclusive_ms']
                assert abs(sum(e['exclusive_ms'] for e in s['stage_events'])-inclusive)<1e-6
                assert s['local_ms']>=inclusive
            if s['candidate']!='token':
                assert s['packet_sha256']==g['packet_sha256']
                assert s['receipt']['message']['payload_mode']=='KV'
                from token_kv_agent_system.communication import HandoffChoice
                assert s['receipt']['kv_strategy']==HandoffChoice.parse(s['candidate']).kv_strategy.value
                if s['candidate'] in ('kv:v0','kv:v1'):
                    assert s['receipt']['receiver_message_prefill_count']==0
        for candidate in {s['candidate'] for s in samples}:
            subset=[s for s in samples if s['candidate']==candidate]
            assert all(s['committed_tensor_sha256']==subset[0]['committed_tensor_sha256'] for s in subset)
        g['_samples']=samples
        all_samples.extend(dict(group=c['label'],**s) for s in samples)
    settings=json.loads((root/'settings.json').read_text())
    paired,networks,profiles=analyze(groups,settings)
    for name,data in [('paired-deltas',paired),('network',networks),('profiles',profiles)]:
        assert json.loads(json.dumps(data))==json.loads((root/(name+'.json')).read_text())
    # Per-pair mathematical thresholds, including slow samples, no outlier removal.
    for n in networks:
        d=n['delta_wire_bytes']
        n['per_pair_break_even_mbps']=[-8*d/(1000*x) if x<0 else None for x in n['paired_local_kv_minus_token_ms']['values']]
        n['paired_total_kv_minus_token_ms']={str(b):[x+8*d/(1000*b) for x in n['paired_local_kv_minus_token_ms']['values']]
                                                for b in (20,100,1000,10000)}
    measured=(root/'sources/kv_adapter.py').read_text();current=Path('src/token_kv_agent_system/model/kv_adapter.py').read_text()
    adapter_diff=measured_adapter_unchanged(measured,current)
    drift={p:dict(measured=sha,current=file_hash(Path(p))) for p,sha in spec['source_sha256'].items() if file_hash(Path(p))!=sha}
    assert set(drift)<= {'src/token_kv_agent_system/model/kv_adapter.py'}
    out.mkdir(parents=True,exist_ok=False)
    for name in ('frozen-spec','settings','completion','pilot-gate','correctness','eos-2-buffer-probe','pending-512-buffer-probe'):
        dump(out/(name+'.json'),json.loads((root/(name+'.json')).read_text()))
    dump(out/'groups.json',[{k:v for k,v in g.items() if k!='_samples'} for g in groups])
    dump(out/'paired-deltas.json',paired);dump(out/'network.json',networks);dump(out/'profiles.json',profiles)
    minimal=[{k:v for k,v in s.items() if k not in ('receipt','stage_events','forward_calls')} for s in all_samples]
    dump(out/'samples.json',minimal)
    columns=['group','candidate_label','path','mode','index','warmup','local_ms','wire_bytes','packet_sha256',
             'host_export_ms','restore_ms','cpu_reconstruction_ms','upload_ms','cache_construction_ms',
             'mps_allocated_after_bytes','mps_driver_after_bytes']
    with (out/'samples.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,lineterminator='\n');w.writeheader()
        for s in all_samples:
            row={k:s[k] for k in columns if k in s};a=s['adapter_submetrics']
            for dest,key in [('host_export_ms','device_to_host_and_host_bytes_ms'),('restore_ms','total_restore_ms'),
                             ('cpu_reconstruction_ms','cpu_tensor_reconstruction_ms'),('upload_ms','host_to_device_ms'),
                             ('cache_construction_ms','cache_object_construction_ms')]:
                row[dest]=sum(a[key]) if key in a else ''
            w.writerow(row)
    resources=json.loads((root/'resource-samples.json').read_text())
    dump(out/'resources.json',[{k:v for k,v in r.items() if k not in ('vm_stat','swap')} for r in resources])
    (out/'measured-sources').mkdir()
    for p in (root/'sources').iterdir():(out/'measured-sources'/p.name).write_bytes(p.read_bytes())
    failure=root.parent/'failure.json'
    startup_failure=json.loads(failure.read_text()) if failure.is_file() else None
    if failure.is_file():verify_index(root.parent)
    dump(out/'raw-index.json',dict(success_directory=str(root),files=raw_index,
        startup_failure_directory=str(root.parent),startup_failure=startup_failure,
        startup_failure_sha256=file_hash(failure) if failure.is_file() else None))
    # Confirm final default selection for each measured MPS BF16 payload, no model reload.
    from token_kv_agent_system.model.kv_adapter import DEFAULT_TRANSFER_PATH,_transfer_path
    assert DEFAULT_TRANSFER_PATH=='auto'
    assert all(_transfer_path(None,'mps',g['config']['message_tokens']*12288,dtype='bfloat16')=='packed' for g in groups)
    dump(out/'audit.json',dict(status='verified',samples=len(all_samples),valid_samples=sum(not s['warmup'] for s in all_samples),
        handoffs=completion['core_handoffs']+completion['pilot_handoffs']+completion['compatibility_handoffs'],
        initial_failed_model_loads=1,successful_process_model_loads=1,total_model_loads=2,
        input83a_sha_verified=True,packet_bitwise_identity=True,committed_cache_bitwise_identity=True,
        exact_timing_stage_sums=True,no_cpu_forward=True,no_token_fallback=True,original_assets_unchanged=True,
        source_drift=drift,explicit_measured_branches_ast_identical=True,final_default_dispatch_for_measured_domain=True,
        adapter_final_diff=adapter_diff,source83a_original_stride='not_saved; replay layout recorded in configs',
        frozen_state_metadata='actual tensor/ledger/draft/pending/framing restored; nonsemantic version/local spans reconstructed from fresh role template',
        production_default='MPS BF16 homogeneous <=6MiB packed; others legacy; no exception retry',
        peak_memory_scope='observed live allocation boundaries and 2s system samples; allocator absolute highwater not measured'))
    dump(out/'evidence-sha256.json',[dict(path=str(p.relative_to(out)),bytes=p.stat().st_size,sha256=file_hash(p))
          for p in sorted(out.rglob('*')) if p.is_file() and p.name!='evidence-sha256.json'])
    print(json.dumps(dict(status='verified',samples=len(all_samples),output=str(out),files=len(list(out.rglob('*')))),ensure_ascii=False))
    return 0

if __name__=='__main__':raise SystemExit(main())
