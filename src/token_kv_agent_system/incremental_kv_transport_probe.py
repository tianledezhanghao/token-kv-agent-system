"""Capture four real discussion Decode spans over an independent KV byte channel."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

from .discussion import DiscussionConfig, DiscussionRuntime, error_text
from .discussion_demo import print_result
from .incremental_kv import IncrementalKVChannel
from .model import HuggingFaceModelBackend
from .model.config import DEFAULT_MODEL_NAME
from .transport import InMemoryByteTransport


def run_probe(runtime: DiscussionRuntime, channel: IncrementalKVChannel):
    """Wrap only this probe instance, after TOKEN delivery; no Runtime code change.

    TOKEN already closes/materializes its sender draft. Reuse those KV values,
    rather than executing a second supplemental forward for the same message.
    """
    original = runtime.session.handoff
    records = []

    def capture(source, target):
        generated = runtime.session.state(source)
        message = generated.draft_message
        receiver_before = runtime.session.state(target)
        receipt = original(source, target)
        sender = runtime.session.state(source)
        receiver = runtime.session.state(target)
        # The side channel never receives either Agent's state as its destination.
        decoded, diagnostics = channel.send(sender, receipt.message)
        if runtime.session.state(source) is not sender or runtime.session.state(target) is not receiver:
            raise RuntimeError('KV side channel changed discussion states')
        start = decoded.sender_span_start
        record = {
            'message_id': message.message_id, 'sequence': message.sequence,
            'source_agent': source, 'target_agent': target,
            'text': sender.backend.decode(message.message_token_ids, skip_special_tokens=True),
            'generation_cache_length': generated.cached_token_count,
            'generation_content_cached_count': min(len(decoded.message_token_ids),
                                                   generated.cached_token_count - start),
            'generation_pending_token_id': generated.pending_token_id,
            'token_path_materialized_content_count': max(0, start + decoded.message_token_count
                                                         - generated.cached_token_count),
            'receiver_cache_before_token': receiver_before.cached_token_count,
            'receiver_cache_after_token': receiver.cached_token_count,
            'kv_receiver_cache_appended_count': 0,
            'receiver_content_ids_exact': (
                receiver.token_ledger[receipt.message.receiver_message_start:
                                      receipt.message.receiver_message_start + decoded.message_token_count]
                == decoded.message_token_ids),
            'diagnostics': asdict(diagnostics),
        }
        if not record['receiver_content_ids_exact']:
            raise RuntimeError('TOKEN Receiver actual content and Delta IDs disagree')
        records.append(record)
        return receipt

    runtime.session.handoff = capture
    try:
        result = runtime.run()
    finally:
        runtime.session.handoff = original
    return result, records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default=DEFAULT_MODEL_NAME)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--question', default='23 + 48 = ?')
    parser.add_argument('--rounds', type=int, choices=(1, 2, 3), default=2)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--summarizer', choices=('A', 'B'), default='A')
    parser.add_argument('--max-context-tokens', type=int, default=4096)
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        config = DiscussionConfig(args.question, args.rounds, args.max_new_tokens,
                                  args.summarizer, args.max_context_tokens)
    except ValueError as exc:
        parser.error(str(exc))
    if not args.allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
    try:
        backend = HuggingFaceModelBackend(args.model, device=args.device)
        runtime = DiscussionRuntime(backend, config=config)
        transport = InMemoryByteTransport()
        channel = IncrementalKVChannel(backend, backend, transport)
        result, records = run_probe(runtime, channel)
        print_result(result)
        for row in records:
            d, e = row['diagnostics'], row['diagnostics']['export']
            print(f"\nIncremental KV {row['message_id']} / {row['source_agent']} -> {row['target_agent']}")
            print(f"Raw generated IDs: {e['raw_generated_token_ids']}")
            print(f"Exact content IDs: {e['message_token_ids']}")
            print(f"Generation cache length: {row['generation_cache_length']}; extraction cache length: {e['sender_cache_length']}")
            print(f"Sender Delta span: [{e['sender_span_start']}, {e['sender_span_start'] + e['sender_span_length']}); Delta seq_len: {e['sender_span_length']}")
            print(f"Model: {e['model']}; Device: {e['device']}; Dtype: {e['dtype']}; Layers: {e['layer_count']}")
            print(f"TOKEN wire: {d['token_protocol_wire_bytes']}; Delta raw: {e['delta_raw_bytes']}; KV v1 wire: {d['kv_protocol_wire_bytes']}; Envelope total: {d['envelope_total_bytes']}")
            print(f"Content cached/generated: {e['cached_message_token_count']}/{e['original_message_token_count']}; supplement: {e['supplemented_token_count']} / {e['pending_token_ms']:.3f} ms")
            print(f"Extraction: {e['tensor_extraction_ms']:.3f} ms; KV serialize: {d['kv_serialize_ms']:.3f} ms; envelope serialize: {d['envelope_serialize_ms']:.3f} ms; deserialize: {d['deserialize_ms']:.3f} ms")
            print(f"Bitwise exact: {d['bitwise_exact']}; actual Receiver content IDs exact: {row['receiver_content_ids_exact']}; KV append: 0")
        report = {'model': asdict(backend.metadata), 'model_contract': asdict(channel.contract),
                  'discussion': asdict(result), 'incremental_messages': records,
                  'kv_transport': [asdict(r) for r in transport.records],
                  'scope': 'Decode export, bytes, unpack only; no KV receive append or RoPE alignment'}
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
        return 0 if result.status == 'completed' and len(records) == 2 * config.rounds else 1
    except Exception as exc:
        print(f'Incremental KV probe FAILED: {error_text(exc)}')
        return 1
