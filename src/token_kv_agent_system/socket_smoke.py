"""Small opt-in 0.5B/MPS socket smoke; one weight load, no performance matrix."""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from .socket_lab import SocketLab
from .transport import SocketConfig


def smoke(backend, transport):
    from .agent import LLMAgent
    from .communication import HandoffChoice
    from .incremental_kv_protocol import inspect_model_contract
    from .model.state_cache import require_disjoint_caches
    from .stateful_token import StatefulTokenSession
    base = StatefulTokenSession([
        LLMAgent('A', backend, system_prompt='You are Agent A. Follow the instruction briefly.'),
        LLMAgent('B', backend, system_prompt='You are Agent B. Follow the instruction briefly.'),
    ], transport, max_context_tokens=1024)
    message = base.generate_message('A', 'B', local_prompt='Reply with only the digit 7.', max_new_tokens=4)
    # All five paths use the SAME actual Decode snapshot, no forced predictions.
    base._kv_contract = inspect_model_contract(backend)
    rows = []
    for mode in ('token', 'kv:v0', 'kv:v1', 'kv:v2', 'kv:v3'):
        session = copy.copy(base)
        session._states = dict(base._states)
        choice = HandoffChoice.parse(mode)
        receipt = (session.handoff('A', 'B') if mode == 'token' else
                   session.handoff_kv('A', 'B', strategy=choice.kv_strategy))
        assert receipt.received_token_ids == message.message_token_ids
        continued = session.generate_message('B', 'A', max_new_tokens=2,
                                             local_prompt='Reply with only the digit 8.')
        row = dict(mode=mode, source_ids=message.message_token_ids, receipt=asdict(receipt),
                   receiver_continuation_ids=continued.message_token_ids,
                   receiver_continuation_text=backend.decode(continued.message_token_ids, skip_special_tokens=True))
        if mode == 'token':
            reverse = session.handoff('B', 'A')
            after_reverse = session.generate_local_message('A', local_prompt='Reply briefly.', max_new_tokens=2)
            row.update(reverse_receipt=asdict(reverse), after_reverse_ids=after_reverse.message_token_ids)
        for name in ('A', 'B'):
            state = session.state(name)
            state.validate()
            assert state.cached_token_count == len(state.token_ledger) - int(state.pending_token_id is not None)
        require_disjoint_caches(session.state('A').cache, session.state('B').cache)
        row['cache_lengths'] = {n: session.state(n).cached_token_count for n in ('A', 'B')}
        rows.append(row)
    return dict(scope='Mac loopback TCP software smoke; no wireless/server measurement',
        transaction='remote byte verification, then existing local two-cache Session commit',
        model=asdict(backend.metadata), shared_weights=True, independent_caches=True,
        source_text=backend.decode(message.message_token_ids, skip_special_tokens=True),
        cases=rows, measurements=[asdict(m) for m in transport.measurements])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise ValueError('use a new output directory to preserve earlier evidence')
    args.output_dir.mkdir(parents=True)
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      PYTORCH_ENABLE_MPS_FALLBACK='0', HF_HUB_DISABLE_TELEMETRY='1')
    import torch
    from .local_qwen_comparison import memory, snapshot_path
    from .local_qwen_resources import ResourceMonitor
    from .model import HuggingFaceModelBackend
    from .model.huggingface import resolve_device
    resolve_device('mps', torch)
    monitor = ResourceMonitor(lambda: memory(torch), memory(torch), args.output_dir)
    monitor.start()
    try:
        backend = HuggingFaceModelBackend(str(snapshot_path('05b')), device='mps')
        if backend.metadata.device_fallback_reason or backend.metadata.dtype != 'torch.bfloat16':
            raise RuntimeError('smoke requires explicit MPS/BF16, without fallback')
        original = backend._timed_forward
        backend._timed_forward = lambda **kwargs: monitor.forward(original, **kwargs)
        with SocketLab(args.output_dir/'socket-logs', config=SocketConfig(chunk_bytes=4093)) as lab:
            result = smoke(backend, lab.transport)
            result['relay_endpoint_pids'] = [p.pid for p in lab.processes]
        sources = [Path(__file__), Path(__file__).with_name('relay.py'),
                   Path(__file__).with_name('socket_lab.py'), Path(__file__).with_name('stateful_token.py'),
                   Path(__file__).with_name('token_protocol.py'), Path(__file__).with_name('incremental_kv_protocol.py')]
        sources += list((Path(__file__).parent/'transport').glob('socket_*.py'))
        result['source_sha256'] = {str(p.relative_to(Path(__file__).parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sources}
        monitor.check()
        result.update(status='completed', forward_count=len(monitor.forwards))
        (args.output_dir/'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps(dict(status='completed', cases=len(result['cases']),
                              transfers=len(result['measurements']), forward_count=result['forward_count'])))
        return 0
    except Exception as exc:
        (args.output_dir/'failure.json').write_text(json.dumps(dict(error=f'{type(exc).__name__}: {exc}')))
        raise
    finally:
        monitor.close()


if __name__ == '__main__':
    raise SystemExit(main())
