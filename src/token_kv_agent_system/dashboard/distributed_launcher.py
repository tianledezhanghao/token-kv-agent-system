"""Owned local launch or connect to prestarted server Workers. No Local model load."""
import argparse
import json
from pathlib import Path
import socket
import threading
import time
from contextlib import ExitStack

from ..distributed_lab import DistributedLab
from ..distributed_worker import sample_resources
from ..metrics import MetricCollector
from .app import create_app
from .distributed_service import DistributedDashboardService


def check_port(host, port):
    with socket.socket() as sock:
        sock.bind((host, port))  # Fail before any model starts, never kill an owner.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=9210)
    parser.add_argument('--history', type=Path, help='Existing metadata-only history file; never restores Cache')
    parser.add_argument('--connect', type=Path, help='Existing workers/routes/relays JSON; no models launched')
    parser.add_argument('--with-frontend', action='store_true', help='Reuse 5173 or start an owned Vite process with bounded readiness')
    args = parser.parse_args(argv)
    # One owned local lab across all output directories, even with different ports.
    import fcntl
    lock_path=Path('benchmark-results/.distributed-dashboard.lock')
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    launch_lock=lock_path.open('a')
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    check_port(args.host,args.port)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    baseline = sample_resources(None)
    (args.output_dir/'baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
    with ExitStack() as stack:
        if args.connect:
            settings = json.loads(args.connect.read_text())
        else:
            # Refuse a competing 8.6B smoke/manual lab before allocating weights.
            import subprocess
            process_rows=subprocess.check_output(['ps','-axo','pid,command'],text=True).splitlines()
            if any(('token_kv_agent_system.distributed_worker' in line or 'run_distributed_worker.py' in line) and int(line.split()[0]) != __import__('os').getpid() for line in process_rows):
                raise RuntimeError('Distributed Worker processes already exist; use --connect; no duplicate model load')
            # Budget includes live available memory with the existing Local worker
            # still resident. DistributedLab reserves BOTH models before loading.
            from urllib.request import urlopen
            try:
                with urlopen('http://127.0.0.1:8000/api/discussions/latest',timeout=2) as response:
                    local=json.load(response)
                if local.get('active_session_id') or local.get('model_switch',{}).get('loading'):
                    raise RuntimeError('Local model is active/loading; wait before starting two Workers')
                (args.output_dir/'local-before.json').write_text(json.dumps(local,indent=2)+'\n')
            except OSError:
                pass  # No existing Local service; never stop or replace one.
            from ..local_qwen_comparison import snapshot_path
            from ..distributed_worker import resource_budget
            try:
                budget=resource_budget(snapshot_path('05b'),copies=2,max_context_tokens=2048)
                (args.output_dir/'budget.json').write_text(json.dumps(budget,indent=2)+'\n')
                lab = stack.enter_context(DistributedLab(args.output_dir/'processes', model_a=snapshot_path('05b'), max_context_tokens=2048))
            except Exception as exc:
                (args.output_dir/'startup-failure.json').write_text(json.dumps(dict(error=str(exc),resource=sample_resources(None)),indent=2)+'\n')
                raise
            settings = dict(workers=lab.workers,routes=lab.routes,timeout=120,
                relays={k:lab.readiness[k] for k in ('relay_a','relay_b')}, measurement_source='Mac TCP / loopback')
        (args.output_dir/'connection.json').write_text(json.dumps(settings,indent=2)+'\n')
        if args.with_frontend:
            try:
                check_port('127.0.0.1',5173)
            except OSError:
                print('Port 5173 already owned; reusing existing frontend without stopping it',flush=True)
            else:
                import subprocess
                frontend_log=stack.enter_context((args.output_dir/'frontend.log').open('w'))
                frontend=subprocess.Popen(['npm','run','dev','--','--strictPort'],cwd=Path(__file__).resolve().parents[3]/'frontend',stdout=frontend_log,stderr=frontend_log,start_new_session=True)
                def stop_frontend():
                    if frontend.poll() is None:
                        import os,signal
                        os.killpg(frontend.pid,signal.SIGTERM)
                        try:frontend.wait(timeout=5)
                        except subprocess.TimeoutExpired:os.killpg(frontend.pid,signal.SIGKILL);frontend.wait(timeout=5)
                stack.callback(stop_frontend)
                deadline=time.monotonic()+20
                while True:
                    try:
                        with socket.create_connection(('127.0.0.1',5173),timeout=.5):break
                    except OSError:
                        if frontend.poll() is not None or time.monotonic()>deadline:
                            raise RuntimeError('Owned frontend startup failed; see frontend.log')
                        time.sleep(.1)
        collector=MetricCollector()
        service=DistributedDashboardService(settings,collector=collector,history_path=args.history or args.output_dir/'history.json')
        health=service.health(force=True)
        if not health['available']:
            print('Distributed services unavailable; serving diagnostics without Local fallback',flush=True)
        import uvicorn
        server=uvicorn.Server(uvicorn.Config(create_app(collector=collector,distributed_service=service),host=args.host,port=args.port))
        done=threading.Event()
        def monitor():
            while not done.wait(5):
                row=sample_resources(None)
                with (args.output_dir/'resources.jsonl').open('a') as out:
                    out.write(json.dumps(row)+'\n')
                if row['pressure_level']>=4 or row['swap_used_mib']-baseline['swap_used_mib']>256:
                    # Preserve in-flight receipt semantics through bounded ASGI
                    # shutdown, then terminate only this launcher's processes.
                    server.should_exit=True
                    return
        threading.Thread(target=monitor,daemon=True).start()
        print(json.dumps(dict(event='dashboard_ready',pid=__import__('os').getpid(),health=health)),flush=True)
        try:
            server.run()
        finally:
            done.set()
    return 0
