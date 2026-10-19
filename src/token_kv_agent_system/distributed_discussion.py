"""Coordinator policy over restricted Worker control; no business data delivery."""
from dataclasses import asdict
import argparse
import json
from pathlib import Path
from threading import Lock, Event
from uuid import uuid4

from .communication import CommunicationMode
from .discussion import DiscussionConfig
from .distributed_worker import control_call
from .network.handoff_network import HandoffNetwork, HandoffNetworkConfig


class DistributedDiscussion:
    def __init__(self, workers, *, routes=None, config=None, timeout=120, event_sink=None, session_id=None, network=None):
        if set(workers) != {'A','B'} or workers['A'] == workers['B']:
            raise ValueError('two distinct Worker control addresses required')
        self.workers, self.routes = dict(workers), routes
        self.config = config or DiscussionConfig()
        self.timeout = timeout
        self.network = network if network is not None else HandoffNetwork(self.config.network)
        self.session_id = session_id or uuid4().hex
        self.status, self.result = 'ready', None
        self.event_sink = event_sink or (lambda event, row:None)
        self.lock = Lock()
        self.stop_requested = Event()

    def call(self, name, operation, **arguments):
        # Only metadata, the original public task, and fixed-stage commands leave
        # this coordinator. No generated IDs, peer utterance, or KV is forwarded.
        self.event_sink('control_command', dict(agent=name, operation=operation,
                                              argument_keys=sorted(arguments)))
        return control_call(self.workers[name], operation, timeout=self.timeout, **arguments)

    def request_stop(self):
        """Stop at the next operation boundary; never interrupt/undo a delivery."""
        self.stop_requested.set()

    def _check_stop(self):
        if self.stop_requested.is_set():
            raise RuntimeError("Coordinator stopping at operation boundary; remote commits are not rolled back")

    def run(self):
        with self.lock:
            if self.status != 'ready':
                raise ValueError('Coordinator runs once; no implicit command replay')
            self.status = 'running'
        result = dict(session_id=self.session_id, status='running', turns=[], final=None,
                      architecture='independent_agent_workers_v1', workers={}, released={})
        started = []
        stage = 'configure'
        try:
            for name in ('A','B'):
                result['workers'][name] = self.call(name, 'status')
            a,b = (result['workers'][n] for n in ('A','B'))
            if a['pid'] == b['pid'] and a['host_id'] == b['host_id']:
                raise ValueError('Agents must have independent OS processes')
            if a['contract'] != b['contract'] or a['agent_id'] != 'A' or b['agent_id'] != 'B':
                raise ValueError('incompatible Worker model/tokenizer identity')
            if self.routes is not None:
                for name, peer in (('A','B'),('B','A')):
                    self.call(name, 'configure', route=list(self.routes[peer]), peer_control=self.workers[peer])
            for name in ('A','B'):
                # Never disclose a peer's private System Prompt to this Worker.
                config = {key:getattr(self.config,key) for key in ('question','rounds','max_new_tokens',
                    'summarizer','max_context_tokens')}
                field = f'agent_{name.lower()}_system_prompt'
                config[field] = getattr(self.config,field)
                self.call(name, 'start', session_id=self.session_id, config=config)
                started.append(name)
                self.event_sink('worker_prepared', dict(agent=name, state=self.call(name, 'status')))
            for round_number in range(1, self.config.rounds+1):
                for source, target in (('A','B'),('B','A')):
                    stage = 'stop_boundary'
                    self._check_stop()
                    index = len(result['turns'])+1
                    stage = f'generation:{index}:{source}'
                    self.event_sink('generation_started', dict(source=source, target=target, round=round_number, index=index))
                    generated = self.call(source, 'generate', session_id=self.session_id, stage=index, final=False)
                    self.event_sink('generation_completed', dict(source=source, target=target, round=round_number, index=index, generated=generated))
                    choice = self.config.choice_for(index-1)
                    snap = self.network.snapshot(index)
                    row = dict(round=round_number, source=source, target=target, generated=generated,
                               network=asdict(snap), delivery=None)
                    result['turns'].append(row)
                    import time
                    handoff_started = time.monotonic()
                    stage = f'decision:{index}:{source}'
                    selected = (self.call(source, 'decide', session_id=self.session_id, handoff=index,
                        choice=asdict(choice), auto=asdict(self.config.auto), network=asdict(snap))
                        if self.config.auto else dict(choice=asdict(choice), decision=None))
                    row['selection'] = selected
                    self.event_sink('decision_completed', row)
                    stage = f'delivery:{index}:{source}->{target}'
                    self.event_sink('delivery_started', row)
                    delivery = self.call(source, 'send', session_id=self.session_id, handoff=index,
                        choice=selected['choice'], auto=None, network=asdict(snap))
                    delivery['decision'] = selected['decision']
                    row['observed_handoff_wall_ms'] = (time.monotonic()-handoff_started)*1000
                    row['delivery'] = delivery
                    # Simulation covers the full application request separately
                    # from measured real socket I/O, import and application ACK.
                    row['simulated_application_transfer_ms'] = snap.transfer_ms(delivery['application_bytes'])
                    self.event_sink('delivery_completed', row)
                    if delivery['outcome'] != 'receiver_committed':
                        result['status'] = 'delivery_unknown' if delivery['outcome'] == 'delivery_unknown' else 'failed'
                        raise RuntimeError(f"handoff did not confirm Receiver commit: {delivery['outcome']}")
                    self.event_sink('turn_completed', row)
            stage = 'stop_boundary'
            self._check_stop()
            stage = 'final'
            self.event_sink('final_started', dict(source=self.config.summarizer))
            result['final'] = self.call(self.config.summarizer, 'generate', session_id=self.session_id,
                                       stage=self.config.rounds*2+1, final=True)
            self.event_sink('final_completed', result['final'])
            result['status'] = 'completed'
        except Exception as exc:
            if result['status'] == 'running':
                result['status'] = 'delivery_unknown' if stage.startswith('delivery:') else 'failed'
            result.update(error=f'{type(exc).__name__}: {exc}', failure_stage=stage)
        finally:
            # Preserve unknown transactions for explicit status/reconcile. Other
            # runs free session caches even after a confirmed rejection/failure.
            for name in started:
                try:
                    result.setdefault('final_states', {})[name] = self.call(name, 'audit', session_id=self.session_id)
                    if result['status'] == 'completed':
                        result['released'][name] = self.call(name, 'release', session_id=self.session_id)
                except Exception as exc:
                    result.setdefault('cleanup_errors', {})[name] = str(exc)
                    if result['status'] != 'delivery_unknown':
                        result['status'] = 'failed'
            self.result, self.status = result, result['status']
            self.event_sink('discussion_finished', result)
        return result


def config_from_dict(data):
    from .communication import HandoffChoice
    from .incremental_auto import IncrementalAutoConfig
    from .generalized_cost import load_cost_profile
    values = dict(data)
    for field in ('handoff_choice','handoff_plan'):
        if field in values:
            if field == 'handoff_choice':
                values[field] = HandoffChoice(**values[field])
            else:
                values[field] = tuple(HandoffChoice(**v) for v in values[field])
    if values.get('network'):
        net = dict(values['network'])
        if 'trace_mbps' in net:
            net['trace_mbps'] = tuple(net['trace_mbps'])
        values['network'] = HandoffNetworkConfig(**net)
    if values.get('auto'):
        auto = dict(values['auto'])
        if 'candidates' in auto:
            auto['candidates'] = tuple(auto['candidates'])
        if auto.get('profile'):
            auto['profile'] = load_cost_profile(auto['profile'])
        if auto.get('synthetic_assumptions') is not None:
            from .synthetic_cost import SyntheticCostAssumptions
            auto['synthetic_assumptions'] = SyntheticCostAssumptions.from_dict(auto['synthetic_assumptions'])
        values['auto'] = IncrementalAutoConfig(**auto)
    return DiscussionConfig(**values)


def create_app(settings, output_dir):
    """Separate opt-in REST entry; existing Dashboard and its defaults untouched."""
    from fastapi import FastAPI, HTTPException
    import threading
    app = FastAPI(title='Distributed Agent Coordinator')
    lock = Lock()
    state = {'runtime':None}

    def log(event, row):
        with (output_dir/'coordinator.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(event=event, **row), ensure_ascii=False)+'\n')

    @app.get('/state')
    def status():
        runtime = state['runtime']
        return dict(status=runtime.status, session_id=runtime.session_id, result=runtime.result,
                    network=runtime.network.current()) if runtime else dict(status='idle')

    @app.post('/discussion')
    def start(config:dict):
        with lock:
            previous = state['runtime']
            if previous and (previous.status in ('ready','running','delivery_unknown','failed')):
                raise HTTPException(409, 'previous run active/unresolved; inspect and release Workers explicitly')
            try:
                runtime = DistributedDiscussion(settings['workers'], routes=settings['routes'],
                    config=config_from_dict(config), timeout=settings.get('timeout',120), event_sink=log)
            except (TypeError, ValueError) as exc:
                raise HTTPException(422,str(exc)) from exc
            state['runtime'] = runtime
            # Mark admission before starting to prevent a concurrent duplicate POST.
            def run():
                runtime.run()
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            return dict(session_id=runtime.session_id, status='accepted')

    @app.patch('/network')
    def patch(config:dict):
        runtime = state['runtime']
        if runtime is None or runtime.status not in ('ready','running'):
            raise HTTPException(409,'no active discussion')
        try:
            net = dict(config)
            if 'trace_mbps' in net:
                net['trace_mbps'] = tuple(net['trace_mbps'])
            runtime.network.update(HandoffNetworkConfig(**net))
        except (TypeError,ValueError) as exc:
            raise HTTPException(422,str(exc)) from exc
        return runtime.network.current()
    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--serve', type=__import__('token_kv_agent_system.relay',fromlist=['address']).address)
    args = parser.parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    settings = json.loads(args.config.read_text())
    if args.serve:
        import uvicorn
        uvicorn.run(create_app(settings,args.output_dir),host=args.serve[0],port=args.serve[1])
        return 0
    runtime = DistributedDiscussion(settings['workers'], routes=settings['routes'],
        config=config_from_dict(settings.get('discussion',{})), timeout=settings.get('timeout',120))
    result = runtime.run()
    (args.output_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],session_id=result['session_id'])))
    return 0 if result['status']=='completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
