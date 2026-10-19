"""Explicit tiny random CPU model browser harness. Not pretrained acceptance."""
import argparse
from pathlib import Path
import sys
import json
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from token_kv_agent_system.distributed_lab import DistributedLab
from token_kv_agent_system.dashboard.distributed_service import DistributedDashboardService
from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.metrics import MetricCollector
import uvicorn

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False)
    with DistributedLab(a.output_dir/'processes',worker_command=[sys.executable,str(Path(__file__).with_name('distributed_tiny_worker.py'))]) as lab:
        settings=dict(workers=lab.workers,routes=lab.routes,relays={n:lab.readiness[n] for n in ('relay_a','relay_b')},measurement_source='Mac TCP / loopback',timeout=30)
        (a.output_dir/'connection.json').write_text(json.dumps(settings,indent=2))
        collector=MetricCollector()
        def event(e):
            with (a.output_dir/'events.jsonl').open('a') as out:
                out.write(json.dumps(dict(type=e.event_type.value,metadata=e.metadata,timestamp=e.timestamp.isoformat()))+'\n')
        collector.subscribe(event)
        service=DistributedDashboardService(settings,collector=collector,history_path=a.output_dir/'history.json')
        uvicorn.run(create_app(collector=collector,distributed_service=service),host='127.0.0.1',port=9210)
