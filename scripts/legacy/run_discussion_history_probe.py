"""Read-only cross-version runner; imports ONLY the selected checkout's code.

Run with the current venv and --repo pointing at a detached historical worktree.
No installed editable package or current production logic is used in that run.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

CASES = (('add', '23 + 48 = ?', 2), ('multiply', '12 * 13 = ?', 2),
         ('ten', '10 + 10 = ?', 2), ('chinese', '10+10=? 用中文讨论', 3))


def capture_run(runtime, backend):
    captures = []
    originals = {}
    for method in ('generate_message', 'generate_local_message'):
        original = getattr(runtime.session, method)
        originals[method] = original

        def wrap(source, *args, _original=original, **kwargs):
            state = runtime.session.state(source)
            stage = kwargs['local_prompt']
            ids = (state.token_ledger + state.framing.local_user_ids(stage)
                   + state.framing.assistant.prefix_ids)
            before = dict(agent=source, system=state.system_prompt, stage=stage,
                ledger_token_ids=ids, decoded_ledger=backend.decode(ids, skip_special_tokens=False),
                cached_count=state.cached_token_count, local_input_spans=state.local_input_spans,
                received=[asdict(m) for m in state.received_messages],
                sent=[asdict(m) for m in state.sent_messages])
            message = _original(source, *args, **kwargs)
            assert runtime.session.state(source).token_ledger[:message.sender_message_span[0]] == ids
            captures.append(dict(**before, output=asdict(message),
                text=backend.decode(message.message_token_ids, skip_special_tokens=True)))
            return message

        setattr(runtime.session, method, wrap)
    try:
        result = runtime.run()
    finally:
        for method, original in originals.items():
            setattr(runtime.session, method, original)
    return dict(discussion=asdict(result), generations=captures,
                final_ledgers={n: runtime.session.state(n).token_ledger for n in ('A', 'B')})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-0.5B-Instruct')
    parser.add_argument('--device', default='mps')
    args = parser.parse_args(argv)
    root, out = args.repo.resolve(), args.output_dir.resolve()
    if out.exists():
        parser.error('choose a new output directory to preserve previous evidence')
    sys.path.insert(0, str(root / 'src'))
    os.environ['HF_HUB_OFFLINE'] = '1'
    from token_kv_agent_system.model import HuggingFaceModelBackend
    from token_kv_agent_system.discussion import DiscussionRuntime, DiscussionConfig
    import token_kv_agent_system.discussion as module
    import torch
    import transformers
    from huggingface_hub import snapshot_download
    assert Path(module.__file__).is_relative_to(root), 'imported the wrong checkout'
    checkpoint = (Path(args.model).resolve() if Path(args.model).is_dir() else
                  Path(snapshot_download(args.model, local_files_only=True)))
    backend = HuggingFaceModelBackend(str(checkpoint), device=args.device)
    settings = dict(root=str(root), import_file=module.__file__,
        commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        model=asdict(backend.metadata), checkpoint=str(checkpoint), torch=torch.__version__,
        transformers=transformers.__version__, platform=platform.platform(),
        chat_template=backend.chat_tokenizer.chat_template, do_sample=False,
        max_new_tokens=64, summarizer='A', cases=CASES,
        files={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in checkpoint.iterdir() if p.is_file()})
    out.mkdir(parents=True)
    def write(name, record):
        (out / (name + '.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    write('settings', settings)
    for name, question, rounds in CASES:
        runtime = DiscussionRuntime(backend, config=DiscussionConfig(
            question=question, rounds=rounds, max_new_tokens=64, summarizer='A'))
        record = capture_run(runtime, backend)
        write(name, record)
        result = record['discussion']
        print(name, result['status'], json.dumps([t['text'] for t in result['turns']]
              + [result['final_text']], ensure_ascii=False), flush=True)
        if result['status'] != 'completed':
            raise RuntimeError(result['error'])
        del runtime, record
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
