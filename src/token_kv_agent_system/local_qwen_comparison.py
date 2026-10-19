"""7.8A opt-in BF16 comparison; reuses frozen 7.7G and production TOKEN paths."""
import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import time

from .autogen_style_reference import (ReferenceHarness, READ_TASK, STAGES,
                                     generate, capture_token, token_comparison, ComparisonRuntime)
from .autogen_style_probe import FIXTURES, QUESTIONS
from .discussion import DiscussionConfig, DiscussionRuntime
from .message_diagnostics import ForwardTrace
from .secret_sum import SecretSumConfig
from .secret_sum_validation import run_scene, no_handoff
from .local_qwen_resources import ResourceConfig, ResourceMonitor, ResourcePolicy, classify_failure

REVISIONS = {'3b': ('Qwen/Qwen2.5-3B-Instruct', 'aa8e72537993ba99e69dfaafa59ed015b17504d1'),
             '05b': ('Qwen/Qwen2.5-0.5B-Instruct', '7ae557604adf67be50417f59c2c2f167def9a775')}
BOOT = ('你好，请用中文回答。', '10+10等于多少？', '3x=9，x是多少？',
        '请用一句话解释什么是人工智能。', '请判断：10+10=21是否正确？')
EXTRA = (('equation-correct', '3x=9，x=?', 'x=3'),
         ('equation-wrong', '3x=9，x=?', 'x=4'),
         ('university', '中山大学在哪里？', '中山大学位于广东省中山市。'))


def snapshot_path(model):
    repo, revision = REVISIONS[model]
    return Path.home()/'.cache/huggingface/hub'/('models--'+repo.replace('/', '--'))/'snapshots'/revision


def file_hash(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def memory(torch=None):
    def command(*args):
        return subprocess.check_output(args, text=True, timeout=5).strip()
    swap = command('sysctl', '-n', 'vm.swapusage')
    row = dict(monotonic=time.monotonic(), swap=swap,
               swap_used_mib=float(re.search(r'used = ([\d.]+)M', swap)[1]),
               vm_stat=command('vm_stat'),
               pressure_level=int(command('sysctl', '-n', 'kern.memorystatus_vm_pressure_level')),
               rss_kib=int(command('ps', '-o', 'rss=', '-p', str(os.getpid()))))
    page_size=int(re.search(r'page size of (\d+) bytes',row['vm_stat'])[1])
    pages=int(re.search(r'Pages occupied by compressor:\s+(\d+)',row['vm_stat'])[1])
    row['compressed_memory_bytes']=page_size*pages
    if torch is not None:
        row.update(mps_current_bytes=torch.mps.current_allocated_memory(),
                   mps_driver_bytes=torch.mps.driver_allocated_memory(),
                   mps_recommended_bytes=torch.mps.recommended_max_memory())
    return row


def check_memory(row, initial_swap):
    """Compatibility helper: a single sample can warn, never prove persistence."""
    return ResourcePolicy(dict(swap_used_mib=initial_swap)).observe(
        dict(row,monotonic=row.get('monotonic',0)))


def fixed_case(backend, name, question, fixture, structure, task, limit):
    h = ReferenceHarness(backend, question, structure=structure)
    if fixture is not None:
        h.deliver('A', 'B', fixture, origin='fixed_fixture')
    row = h.turn('B', READ_TASK if task == 'read' else STAGES['critic'].b_first, limit)
    row.update(case=name, question=question, fixture=fixture, task=task,
               input_origin='fixed_fixture' if fixture is not None else 'no_message')
    return row


def production(backend, question, limit, mode=None):
    from .model.cache_inspection import inspect_kv_cache
    from .model.state_cache import require_disjoint_caches
    runtime = ComparisonRuntime(backend,question,mode,limit) if mode else DiscussionRuntime(
        backend,config=DiscussionConfig(question=question,rounds=2,max_new_tokens=limit))
    timeline=[]
    def caches(kind, data=None):
        require_disjoint_caches(runtime.session.state('A').cache,runtime.session.state('B').cache)
        timeline.append(dict(event=kind,agents={n:asdict(inspect_kv_cache(
            runtime.session.state(n).cache,runtime.session.state(n).cached_token_count)) for n in ('A','B')}))
    caches('initial')
    runtime.event_sink=caches
    with ForwardTrace(backend) as trace:
        row = capture_token(runtime)
    caches('final')
    row['forward_calls'] = trace.calls
    row['wire_records'] = [asdict(r) for r in runtime.session.transport.records]
    row['cache_timeline']=timeline
    row['shared_backend']=runtime.session.state('A').backend is runtime.session.state('B').backend
    row['independent_caches_verified']=True
    row['runtime'] = 'unchanged_production_token' if mode is None else 'production_token_with_frozen_77g_'+mode
    if mode:row['mode']=mode
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=REVISIONS, default='3b')
    parser.add_argument('--phase', choices=('short','ramp','boot','fixed','verify','extra','token-core','token-extra','peer','reference','budget','dialogue','supplement','secret'), required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    for key,value in asdict(ResourceConfig()).items():
        parser.add_argument('--'+key.replace('_','-'),type=float,default=value)
    args = parser.parse_args(argv)
    resource_config=ResourceConfig(**{k:getattr(args,k) for k in asdict(ResourceConfig())})
    # Never overwrite old experiments, including previous partial runs.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    def emit(name, row):
        (args.output_dir/(name+'.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
        print(name, flush=True)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['OTEL_SDK_DISABLED'] = 'true'
    import torch, transformers
    from .model import HuggingFaceModelBackend
    backend = None
    original_forward = None
    start = time.perf_counter()
    baseline = memory(torch)
    emit('preload', baseline)
    monitor = ResourceMonitor(lambda:memory(torch), baseline, args.output_dir, resource_config)
    try:
        monitor.start()
        path = snapshot_path(args.model)
        # Explicit BF16 at construction avoids a 12 GB FP32 staging copy.
        tokenizer = transformers.AutoTokenizer.from_pretrained(path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                                 local_files_only=True)
        backend = HuggingFaceModelBackend(str(path), device='mps', tokenizer=tokenizer, model=model)
        del model, tokenizer
        torch.mps.synchronize()
        load_seconds = time.perf_counter()-start
        original_forward = backend._timed_forward
        def guarded_forward(**kwargs):
            return monitor.forward(original_forward,**kwargs)
        backend._timed_forward = guarded_forward
        c = backend.kv_model_config
        kv_per_token = 2*c.num_hidden_layers*c.num_key_value_heads*c.head_dim*2
        emit('settings', dict(model=asdict(backend.metadata), repo=REVISIONS[args.model][0],
            revision=REVISIONS[args.model][1], checkpoint=str(path), config=backend._model.config.to_dict(),
            kv_config=asdict(c), parameter_bytes=sum(p.numel()*p.element_size() for p in backend._model.parameters()),
            kv_bytes_per_agent_token=kv_per_token, two_agent_kv_at_4096=kv_per_token*4096*2,
            load_seconds=load_seconds, torch=torch.__version__, transformers=transformers.__version__,
            python=platform.python_version(), macos=platform.mac_ver()[0],
            eos_ids=backend.eos_token_ids, chat_template=backend.chat_tokenizer.chat_template,
            files_sha256={p.name:file_hash(p) for p in path.iterdir() if p.is_file()},
            do_sample=False, batch_size=1, shared_model_instances=1, phase=args.phase,
            production_changed=False, resource_config=asdict(resource_config),memory_after_load=monitor.sample()))
        def run(name, fn):
            gc.collect(); torch.mps.empty_cache()
            before = monitor.sample(); monitor.check()
            forward_start=len(monitor.forwards)
            started = time.perf_counter()
            row = fn()
            torch.mps.synchronize()
            row.update(elapsed_seconds=time.perf_counter()-started, memory_before=before,
                       memory_after=monitor.sample(), model_key=args.model,
                       forward_metrics=monitor.forwards[forward_start:])
            emit(name, row)
            result = row.get('discussion', row.get('result', {}))
            if result.get('status') == 'failed':
                raise RuntimeError(result.get('error', 'runtime execution failed'))
            return row
        if args.phase in ('short','ramp'):
            cases=[(16,'你好，请用中文简单介绍一下自己。')] if args.phase=='short' else [
                (32,'10+10等于多少？请用中文回答。'),
                (64,'3x=9，x是多少？请简要解释。'),
                (128,'请判断10+10=21是否正确，并说明理由。')]
            for limit,q in cases:
                run('single-'+str(limit),lambda q=q,limit=limit:generate(backend,[dict(role='user',content=q)],limit))
        elif args.phase == 'boot':
            for i, q in enumerate(BOOT):
                run('boot-'+str(i+1), lambda q=q: generate(backend,[dict(role='user',content=q)],128))
        elif args.phase == 'verify':
            for name,fixture in FIXTURES[:2]:
                for task in ('read','review'):
                    run(f'fixed-128-{name}-A-{task}',lambda n=name,f=fixture,t=task:
                        fixed_case(backend,n,'10 + 10 = ?',f,'A',t,128))
        elif args.phase == 'fixed':
            for name, fixture in FIXTURES:
                for structure in ('A','D'):
                    for task in ('read','review'):
                        run(f'fixed-128-{name}-{structure}-{task}',lambda n=name,f=fixture,s=structure,t=task:
                            fixed_case(backend,n,'10 + 10 = ?',f,s,t,128))
        elif args.phase == 'extra':
            for name,q,f in EXTRA:
                for structure in ('A','D'):
                    for task in ('read','review'):
                        run(f'fixed-128-{name}-{structure}-{task}',lambda n=name,q=q,f=f,s=structure,t=task:
                            fixed_case(backend,n,q,f,s,t,128))
        elif args.phase == 'supplement':
            for name,q in QUESTIONS:
                if name in ('equation','addition','chinese'):
                    run(f'production-default-128-{name}',lambda q=q:production(backend,q,128))
        elif args.phase in ('token-core','token-extra','peer','reference','budget'):
            for name,q in QUESTIONS:
                if args.phase=='token-core' and name in ('ten','equation'):
                    run(f'production-default-128-{name}',lambda q=q:production(backend,q,128))
                if args.phase=='token-extra' and name not in ('ten','equation'):
                    run(f'production-default-128-{name}',lambda q=q:production(backend,q,128))
                if args.phase=='peer' and name in ('ten','equation','chinese'):
                    run(f'token-128-{name}-peer',lambda q=q:production(backend,q,128,mode='peer'))
                if args.phase=='reference' and name in ('ten','equation'):
                    for mode in ('critic','peer'):
                        run(f'dialogue-128-{name}-{mode}-D',lambda q=q,m=mode:ReferenceHarness(backend,q,structure='D',mode=m).run(limit=128))
                if args.phase=='budget' and name in ('ten','multiply'):
                    run(f'production-default-64-{name}',lambda q=q:production(backend,q,64))
        elif args.phase == 'dialogue':
            for name,q in QUESTIONS:
                run(f'production-default-128-{name}',lambda q=q:production(backend,q,128))
            for name,q in QUESTIONS:
                if name in ('ten','equation','chinese'):
                    run(f'token-128-{name}-peer',lambda q=q:production(backend,q,128,mode='peer'))
                if name in ('ten','equation'):
                    for mode in ('critic','peer'):
                        run(f'dialogue-128-{name}-{mode}-D',lambda q=q,m=mode:
                            ReferenceHarness(backend,q,structure='D',mode=m).run(limit=128))
                if name in ('ten','multiply'):
                    run(f'production-default-64-{name}',lambda q=q:production(backend,q,64))
        else:
            for b in (26,31):
                run(f'secret-17-{b}',lambda b=b:run_scene(backend,SecretSumConfig(secret_b=b,max_new_tokens=96)))
                run(f'no-handoff-17-{b}',lambda b=b:no_handoff(backend,SecretSumConfig(secret_b=b,max_new_tokens=96)))
        emit('completion',dict(execution_completed=True,quality_requires_manual_review=True))
        return 0
    except Exception as exc:
        emit('failure',dict(type=type(exc).__name__,error=str(exc),failure_category=classify_failure(exc),
                           elapsed_seconds=time.perf_counter()-start, memory=memory(torch)))
        return 1
    finally:
        monitor.close()
        emit('memory-samples',dict(samples=monitor.samples,forwards=monitor.forwards))
        if backend is not None and original_forward is not None:
            backend._timed_forward = original_forward
        del backend
        gc.collect(); torch.mps.empty_cache()


if __name__ == '__main__':
    raise SystemExit(main())
