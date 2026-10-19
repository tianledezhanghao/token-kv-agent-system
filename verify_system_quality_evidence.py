"""Read-only audit of the retained 8.3C study, no model loading or output rewriting."""
import ast,json,hashlib,subprocess
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.parse_args()
from pathlib import Path
from collections import Counter,defaultdict
base=Path('docs/system-quality-83c/evidence');data=base/'data';raw=Path('benchmark-results/system-quality-83c')
load=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manual=load(base/'manual-pairs.json');secrets=load(base/'manual-secrets.json');sessions=load(data/'runtimes.json')
def count(values):
 v=list(values);return dict(n=len(v),yes=sum(x is True for x in v),no=sum(x is False for x in v),unknown=sum(x is None for x in v))
summary={}
for category in sorted({r['category'] for r in manual}):
 summary[category]={}
 for variant in ('token','kv:v0','kv:v1','kv:v2','kv:v3'):
  rows=[r for r in manual if r['category']==category and r['variant']==variant]
  summary[category][variant]={field:count(r[field] for r in rows) for field in ('extraction','final_answer','information_use','strict_task')}
paired={}
for variant in ('token','kv:v0','kv:v1','kv:v2','kv:v3'):
 paired[variant]={}
 for field in ('extraction','final_answer','strict_task'):
  token={r['task']:r[field] for r in manual if r['variant']=='token'}
  rows=[r for r in manual if r['variant']==variant]
  paired[variant][field]=dict(all=count(r[field] for r in rows),
   token_success_subset=count(r[field] for r in rows if token[r['task']] is True),
   added_failures=[r['task'] for r in rows if token[r['task']] is True and r[field] is False],
   added_unknown=[r['task'] for r in rows if token[r['task']] is True and r[field] is None],
   kv_unique_success=[r['task'] for r in rows if token[r['task']] is False and r[field] is True])
# Human review data and frozen historical parser must agree on unambiguous numeric sums.
for row in sessions:
 if row['phase']!='formal-v2':continue
 a,b=row['task'];review=next(r for r in secrets if (r['secret_a'],r['secret_b'],r['variant'])==(a,b,row['variant']))
 if row['historical_score']['final_sum_correct'] is not None:
  assert review['final_sum_correct'] is row['historical_score']['final_sum_correct'],(a,b,row['variant'])
# Full final functions used for formal pair execution remain exactly those frozen before execution.
old=base/'measured-sources/formal-system_quality_validation.py'
expected=load(raw/'formal-v2/frozen-spec.json')['source_sha256']['src/token_kv_agent_system/system_quality_validation.py']
assert sha(old)==expected
current=Path('src/token_kv_agent_system/system_quality_validation.py')
funcs=lambda p:{n.name:ast.dump(n,include_attributes=False) for n in ast.parse(p.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
a,b=funcs(old),funcs(current)
changed=[n for n in a if a[n]!=b[n]]
assert changed==['profile_audit'],changed
protected=load(raw/'protected-assets.json')
assert all(sha(Path(p))==value for p,value in protected.items())
# Audit the closed 8.3C study commit, not later authorized production changes.
# Raw evidence/AST/SHA checks below still validate the retained study data.
baseline='71873f406403f55fe04c42ec23f7e68bf689f38c'
baseline_files=set(subprocess.check_output(['git','ls-tree','-r','--name-only',baseline,'--','src'],text=True).splitlines())
study_ref='dc084d14596d37799ebb3961302ceed02ddb0f8b'
modified=subprocess.check_output(['git','diff',baseline,study_ref,'--name-only','--','src'],text=True).splitlines()
assert set(modified) & baseline_files=={'src/token_kv_agent_system/discussion.py','src/token_kv_agent_system/incremental_auto.py'}
resources={}
for phase in ('pilot-v1','pilot-v2','formal-v2','discussion-v2','profile-v2','dashboard-v2'):
 health=[json.loads(x) for x in (raw/phase/'health.jsonl').read_text().splitlines()]
 resources[phase]=dict(pressure_levels=sorted(set(x['pressure_level'] for x in health)),
  swap_growth_peak_mib=max(x['swap_growth_mib'] for x in health),
  mps_current_peak_bytes=max(x['mps_current_bytes'] for x in health),
  mps_driver_peak_bytes=max(x['mps_driver_bytes'] for x in health),
  stop_reasons=sorted(set(x['stop_reason'] for x in health if x.get('stop_reason'))))
generations=load(data/'generations.json');handoffs=load(data/'handoffs.json')
assert len(generations)==279 and len(handoffs)==225 and sum(r['engineering'] for r in handoffs)==219
assert Counter(r['message']['termination_reason'] for r in generations)==Counter(eos=264,max_new_tokens=15)
# Core profile costs are bit-for-bit identical to old 8.3B numbers; exact guard narrows domain only.
oldprofiles=load(Path('docs/kv-transfer-83b/evidence/profiles.json'))
for label in ('eos-2','pending-512'):
 p=load(raw/'profile-v2'/f'{label}-strict-profile.json');oldp=next(r['profile'] for r in oldprofiles if r['label']==label)
 for field in ('stages','model_identity','sample_count','message_token_range','cache_token_range','recompute_token_range'):
  assert p[field]==oldp[field]
# Validate all small artifact manifests, including manual review raw SHA links.
pairsha={(r['task'],r['variant']):r['raw_sha256'] for r in load(data/'paired.json')}
assert all(pairsha[r['task'],r['variant']]==r['raw_sha256'] for r in manual)
for item in load(data/'evidence-sha256.json'):assert sha(data/item['path'])==item['sha256']
for name,value in [('manual-summary',summary),('manual-paired-differences',paired),('resources',resources)]:
 assert load(base/(name+'.json')) == value, name
audit=dict(status='verified',paired_tasks=15,paired_branches=75,secret_sessions=15,discussion_sessions=24,
 session_complete=22,discussion_failures=dict(v3_not_applicable=1,resource_abort=1),
 raw_indices_verified=6,completed_generations=279,generation_attempts=280,failed_generation_attempts=1,
 handoff_attempts=225,successful_handoffs=219,v3_explicit_rejections=6,
 protected_user_files=len(protected),model_loads=6,forward_count=12171,
 no_download=True,no_cpu_inference_fallback=True,no_tensor_weights_in_git=True,
 formal_function_source_change_only=changed,
 frozen_parser='unchanged; punctuation-related unknown retained; manual judgments separate',
 missing_experiments=['open AUTO final generation aborted by original resource protection; no retry or replacement'],
 frontend='source/build unchanged; one real 1440x900 session; historical matrices reused',
 tests='1063 passed / 1 existing skip / 2 existing warnings',note='Research execution complete with preserved model/task failures; not a semantic reliability certificate')
assert load(base/'audit.json') == audit
for item in load(base/'all-evidence-sha256.json'):
 assert sha(base/item['path'])==item['sha256'],item['path']
print(json.dumps(audit,ensure_ascii=False,indent=2))
