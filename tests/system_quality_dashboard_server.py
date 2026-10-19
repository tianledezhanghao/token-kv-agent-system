"""Opt-in 8.3C real Dashboard evidence server; no fixture and no new UI."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from token_kv_agent_system.system_quality_validation import (frozen_spec,write_once,record_session,EvidenceTransport)
from token_kv_agent_system.local_qwen_comparison import snapshot_path,memory
from token_kv_agent_system.local_qwen_resources import ResourceMonitor
from token_kv_agent_system.discussion import DiscussionRuntime
from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard import discussion_service as module
from token_kv_agent_system.metrics import MetricCollector


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--port',type=int,default=8013)
    args=p.parse_args();args.output_dir.mkdir(parents=True,exist_ok=False)
    write_once(args.output_dir/'frozen-spec.json',dict(**frozen_spec('dashboard'),
        browser_plan=dict(question='What is 23 + 48? Give the result and a brief check in English.',rounds=2,
            max_new_tokens=256,mode='auto',network_mode='trace',bandwidth_trace=[20,10000,50,1000],
            patch='After first completed handoff, PATCH 10000Mbps/2ms while Session active; use next unfrozen boundary',
            viewport=[1440,900],frontend='existing production build; no visual changes')))
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OTEL_SDK_DISABLED='true')
    import torch,uvicorn
    from token_kv_agent_system.model import HuggingFaceModelBackend
    monitor=ResourceMonitor(lambda:memory(torch),memory(torch),args.output_dir)
    monitor.start()
    class RecordedRuntime(DiscussionRuntime):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw);self.session.transport=EvidenceTransport()
        def run(self):
            label='browser-'+str(len(list(args.output_dir.glob('browser-result-*.json')))+1)
            with record_session(self.session,args.output_dir,label) as evidence:
                result=super().run()
            write_once(args.output_dir/(label.replace('browser-','browser-result-')+'.json'),dict(result=asdict(result),evidence=evidence))
            return result
    module.DiscussionRuntime=RecordedRuntime
    original=None;backend=None
    try:
        monitor.check();torch.manual_seed(77)
        backend=HuggingFaceModelBackend(str(snapshot_path('05b')),device='mps')
        assert backend.device=='mps' and backend.metadata.dtype=='torch.bfloat16'
        original=backend._timed_forward
        backend._timed_forward=lambda **kw:monitor.forward(original,**kw)
        collector=MetricCollector();service=module.DiscussionService(backend,collector=collector)
        def event(e):
            with (args.output_dir/'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(type=e.event_type.value,metadata=e.metadata,timestamp=e.timestamp.isoformat()))+'\n')
            if e.event_type.value in ('DISCUSSION_COMPLETED','DISCUSSION_FAILED'):
                sid=e.metadata['session_id']
                write_once(args.output_dir/(sid+'.json'),dict(session=service.get(sid),model=asdict(backend.metadata),
                    forwards=len(monitor.forwards),resource=monitor.sample(),resource_stop=monitor.stop_reason))
        collector.subscribe(event)
        uvicorn.run(create_app(collector=collector,discussion_service=service),host='127.0.0.1',port=args.port)
    finally:
        if original is not None:backend._timed_forward=original
        monitor.close()

if __name__=='__main__':main()
