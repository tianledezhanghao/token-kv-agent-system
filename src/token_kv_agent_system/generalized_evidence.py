"""Read-only export of small calibration/holdout evidence, no model loading."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from statistics import mean

from .generalized_calibration import verify_output, write_json
from .performance_characterization import bandwidth_analysis


def export(source,destination,browser=None):
    audit=verify_output(source)
    destination.mkdir(parents=True,exist_ok=False)
    names=('spec.json','startup.json','samples.json','sources.json','holdout.json','report.json','complete.json',
           'measured-discussion.json','measured-ws.json','idealized-discussion.json','idealized-ws.json','dynamic-patch.json')
    for name in names:
        path=source/name
        if path.exists():shutil.copyfile(path,destination/name)
    shutil.copyfile(source/'profile.json',destination/'measured-profile.json')
    shutil.copyfile(source/'manifest.json',destination/'raw-index.json')
    frozen=source/'measured-sources'
    if frozen.exists():shutil.copytree(frozen,destination/'measured-sources')
    if browser:
        (destination/'screenshots').mkdir()
        for name in ('debug-final-1440.png','receipt-final-1440.png','config-v2.png','presentation-1440.png','presentation-wide.png'):
            shutil.copyfile(browser/name,destination/'screenshots'/name)
        for name in ('debug-1440.json','wide.json'):
            shutil.copyfile(browser/name,destination/name)
    report=json.loads((source/'report.json').read_text())
    stats={c:dict(n=len(values),mape_percent=100*mean(v['relative'] for v in values),
                  mae_ms=mean(v['absolute_ms'] for v in values),max_relative_percent=100*max(v['relative'] for v in values))
           for c in sorted({r['candidate'] for r in report['errors']})
           if (values:=[r for r in report['errors'] if r['candidate']==c])}
    points=json.loads((source/'profile.json').read_text())['points']
    curves=[]
    for token in (p for p in points if p['candidate']=='token'):
        kv=next(p for p in points if p['candidate']!='token' and all(p['features'][k]==v for k,v in token['features'].items() if k not in ('recomputed_tokens','numeric_slot_index')))
        curves.append(dict(features=token['features'],analysis=bandwidth_analysis(token['local_ms'],kv['local_ms'],token['wire_bytes'],kv['wire_bytes'])))
    held=json.loads((source/'holdout.json').read_text())
    summary=dict(audit=audit,holdout_statistics=stats,
        agreement=sum(c.get('agreement',False) for r in held for c in r['comparisons']),
        compared_decisions=sum('decision' in c for r in held for c in r['comparisons']),
        uncertainty_coverage=sum(e['within_envelope'] for e in report['errors']),
        errors=len(report['errors']),positive_training_crossings=sum(c['analysis']['break_even_mbps'] is not None for c in curves))
    write_json(destination/'statistics.json',summary)
    write_json(destination/'network-analysis.json',curves)
    drift={}
    for frozen_source in (destination/'measured-sources').glob('*.py'):
        current=Path(__file__).with_name(frozen_source.name)
        if current.read_bytes()!=frozen_source.read_bytes():
            drift[frozen_source.name]=dict(measured_sha256=hashlib.sha256(frozen_source.read_bytes()).hexdigest(),
                current_sha256=hashlib.sha256(current.read_bytes()).hexdigest(),
                explanation='Post-measurement validation, server entry or provenance labels; handoff algorithms/timing boundaries unchanged.')
    write_json(destination/'source-drift.json',drift)
    manifest={str(p.relative_to(destination)):hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.rglob('*') if p.is_file()}
    write_json(destination/'manifest.json',manifest)
    return summary


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--browser-dir',type=Path)
    args=p.parse_args(argv)
    print(json.dumps(export(args.results_dir,args.output_dir,args.browser_dir),indent=2))
    return 0
