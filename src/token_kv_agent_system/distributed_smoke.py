"""Opt-in cached 0.5B/BF16 two-process Mac acceptance, not a quality benchmark."""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import time

from .communication import HandoffChoice
from .discussion import DiscussionConfig
from .distributed_discussion import DistributedDiscussion
from .distributed_lab import DistributedLab
from .distributed_wire import unpack_delivery, sha
from .distributed_worker import sample_resources
from .incremental_auto import IncrementalAutoConfig
from .network.handoff_network import HandoffNetworkConfig


def verify_packets(root, results):
    """Match independent on-disk Sender/Receiver bytes and application receipts."""
    manifests=[]
    for result in results:
        for turn in result['turns']:
            delivery=turn['delivery']
            if not delivery or delivery['outcome']!='receiver_committed':
                continue
            r=delivery['receipt']
            files=[]
            for who,direction in ((turn['source'],'outgoing'),(turn['target'],'incoming')):
                paths=list((root/f'worker-{who.lower()}'/'packets').glob(f'{direction}-{r["session_id"]}-{r["handoff"]}-*.bin'))
                if len(paths)!=1:
                    raise AssertionError('missing/duplicate packet evidence')
                data=paths[0].read_bytes()
                assert sha(data)==r['packet_sha256']
                meta,protocol=unpack_delivery(data)
                assert sha(protocol)==r['protocol_sha256'] and len(protocol)==r['protocol_bytes']
                assert meta['message']['message_id']==r['message_id']
                assert protocol[:4]==(b'TKVT' if r['mode']=='token' else b'TKVI')
                files.append(dict(path=str(paths[0].relative_to(root)),sha256=sha(data),bytes=len(data),
                                  protocol_sha256=sha(protocol),protocol_bytes=len(protocol)))
            assert files[0]['sha256']==files[1]['sha256']
            manifests.append(dict(session_id=r['session_id'],message_id=r['message_id'],handoff=r['handoff'],files=files))
    return manifests


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--model',type=Path)
    args=parser.parse_args(argv)
    from .local_qwen_comparison import snapshot_path
    model=args.model or snapshot_path('05b')
    args.output_dir.mkdir(parents=True,exist_ok=False)
    initial=sample_resources(None)
    results=[]
    report=dict(scope='Mac loopback TCP software only; no wireless/CUDA/5090 measurements',
                pretrained=True,shared_weights=False,baseline=initial,results=results)
    def save():
        (args.output_dir/'result.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    save()
    try:
        with DistributedLab(args.output_dir/'processes',model_a=model,max_context_tokens=2048) as lab:
            report['readiness']=lab.readiness
            for mode in ('token','kv:v0','kv:v1','kv:v2','kv:v3','auto_quality','auto_idealized'):
                kwargs={}
                if mode=='auto_quality':
                    kwargs['auto']=IncrementalAutoConfig()
                elif mode=='auto_idealized':
                    kwargs['auto']=IncrementalAutoConfig(policy='performance_experiment',idealized=True)
                    kwargs['network']=HandoffNetworkConfig(bandwidth_mbps=100000)
                else:
                    kwargs['handoff_choice']=HandoffChoice.parse(mode)
                config=DiscussionConfig(question='Calculate 23 + 48. Give the numeric result and a brief check.',
                    rounds=2 if mode in ('token','kv:v0') else 1,max_new_tokens=24,max_context_tokens=2048,**kwargs)
                runtime=DistributedDiscussion(lab.workers,routes=lab.routes,config=config)
                commands=[]
                def event(kind,row):
                    if kind=='control_command':commands.append(row)
                    if mode=='auto_idealized' and kind=='turn_completed' and row['source']=='A':
                        runtime.network.update(HandoffNetworkConfig(bandwidth_mbps=.001,source='smoke_live_patch'))
                runtime.event_sink=event
                result=runtime.run()
                result.update(case=mode,control_commands=commands,resource_after=sample_resources(None))
                results.append(result)
                save()
                print(json.dumps(dict(case=mode,status=result['status'],turns=len(result['turns']))),flush=True)
                if result['status']!='completed':
                    raise RuntimeError(f'real smoke case {mode} did not complete; failure retained')
                if (result['resource_after']['pressure_level']>=4 or
                        result['resource_after']['swap_used_mib']-initial['swap_used_mib']>256):
                    raise RuntimeError('smoke resource guard: critical pressure or >256 MiB swap growth')
            report['packets']=verify_packets(lab.output_dir,results)
            report['worker_resources']={}
            for who in ('a','b'):
                folder=lab.output_dir/f'worker-{who}'
                samples=[json.loads(line) for line in (folder/'health.jsonl').read_text().splitlines()]
                forwards=[json.loads(line) for line in (folder/'forwards.jsonl').read_text().splitlines()]
                report['worker_resources'][who]=dict(samples=len(samples),forward_count=len(forwards),
                    max_pressure=max(s['pressure_level'] for s in samples),
                    max_swap_growth_mib=max(s['swap_growth_mib'] for s in samples),
                    max_mps_driver_bytes=max(s['mps_driver_bytes'] for s in samples),
                    max_rss_kib=max(s['rss_kib'] for s in samples))
            report['status']='completed'
            report['final_resource']=sample_resources(None)
            save()
        report['owned_processes_stopped']=all(p.poll() is not None for p in lab.processes)
        sources=list(Path(__file__).parent.glob('distributed_*.py'))+[Path(__file__).with_name('incremental_auto.py')]
        report['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        save()
        return 0
    except BaseException as exc:
        report.update(status='failed',error=f'{type(exc).__name__}: {exc}',resource_at_failure=sample_resources(None))
        save()
        raise


if __name__=='__main__':
    raise SystemExit(main())
