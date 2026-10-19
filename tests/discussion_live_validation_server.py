"""Opt-in 8.2C evidence server: offline 0.5B, unchanged resource rules, no fixture IDs.

Browser chooses/starts the two scenarios. No models are loaded by pytest.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from token_kv_agent_system.dashboard.app import create_app
from token_kv_agent_system.dashboard.discussion_service import DiscussionService
from token_kv_agent_system.metrics import MetricCollector
from token_kv_agent_system.local_qwen_comparison import memory, snapshot_path
from token_kv_agent_system.local_qwen_resources import ResourceMonitor
from token_kv_agent_system.model import HuggingFaceModelBackend
from token_kv_agent_system.stateful_token import StatefulTokenSession
from token_kv_agent_system.model.state_cache import require_disjoint_caches


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8010)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    os.environ['HF_HUB_OFFLINE'] = '1'
    import torch, uvicorn
    baseline = memory(torch)
    monitor = ResourceMonitor(lambda: memory(torch), baseline, args.output_dir)
    monitor.start()
    current = {'session_id': None, 'generations': [], 'deliveries': []}
    original_generate = StatefulTokenSession._generate_working
    def generated(self, *a, **kw):
        state = original_generate(self, *a, **kw)
        current['generations'].append(asdict(state.draft_message))
        return state
    StatefulTokenSession._generate_working = generated
    def track(original):
        def handoff(self, source, target, **kw):
            before = self.state(source)
            raw = before.draft_message.message_token_ids
            receipt = original(self, source, target, **kw)
            assert receipt.received_token_ids == raw
            content = raw[:-1] if receipt.message.termination_reason == 'eos' else raw
            s, t = self.state(source), self.state(target)
            s.validate(); t.validate(); require_disjoint_caches(s.cache, t.cache)
            start, end = receipt.message.sender_message_span
            assert s.token_ledger[start:end] == raw
            x = receipt.message.receiver_message_start
            assert t.token_ledger[x:x+len(content)] == content
            assert s.pending_token_id is None and t.pending_token_id is None
            current['deliveries'].append(dict(message_id=receipt.message.message_id, receipt=asdict(receipt),
                original_pending_id=before.pending_token_id, ids_exact=True, sender_span_exact=True,
                receiver_span_exact=True, closed_ledgers=True, independent_caches=True))
            return receipt
        return handoff
    StatefulTokenSession.handoff = track(StatefulTokenSession.handoff)
    StatefulTokenSession.handoff_kv = track(StatefulTokenSession.handoff_kv)
    try:
        monitor.check()
        backend = HuggingFaceModelBackend(str(snapshot_path('05b')), device='mps')
        assert backend.device == 'mps' and backend.metadata.dtype.removeprefix('torch.') == 'bfloat16'
        original_forward = backend._timed_forward
        backend._timed_forward = lambda **kw: monitor.forward(original_forward, **kw)
        collector = MetricCollector()
        service = DiscussionService(backend, collector=collector)
        def event(e):
            sid = e.metadata['session_id']
            if e.event_type.value == 'DISCUSSION_STARTED':
                current.update(session_id=sid, generations=[], deliveries=[], forward_start=len(monitor.forwards))
            with (args.output_dir / 'events.jsonl').open('a') as f:
                f.write(json.dumps(dict(type=e.event_type.value, timestamp=e.timestamp.isoformat(), metadata=e.metadata))+'\n')
            if e.event_type.value in ('DISCUSSION_COMPLETED', 'DISCUSSION_FAILED'):
                row = dict(session=service.get(sid), model=asdict(backend.metadata),
                    generations=current['generations'], deliveries=current['deliveries'],
                    forward_count=len(monitor.forwards)-current['forward_start'], resource_stop=monitor.stop_reason,
                    resource_sample=monitor.sample())
                (args.output_dir / (sid+'.json')).write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n')
                print('EVIDENCE', sid, row['session']['status'], len(row['generations']), flush=True)
        collector.subscribe(event)
        uvicorn.run(create_app(collector=collector, discussion_service=service), host='127.0.0.1', port=args.port)
    finally:
        monitor.close()

if __name__ == '__main__': main()
