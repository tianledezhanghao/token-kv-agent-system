"""Opt-in, read-only context inspection and TOKEN oracle (never a data plane).

Decoded text is display/reference-template data only. Exact ledger IDs remain the
input to every communication and continuation call. No oracle repairs a cache.
"""
from dataclasses import replace
import hashlib

from .agent.state import AgentMessage, append_working_ids, working_state


def cache_fingerprint(cache):
    import torch
    digest = hashlib.sha256()
    for layer in cache.layers:
        for tensor in (layer.keys, layer.values):
            digest.update(tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def cache_shapes(cache):
    return [dict(layer=i, key_shape=list(layer.keys.shape), value_shape=list(layer.values.shape),
                 dtype=str(layer.keys.dtype), device=str(layer.keys.device))
            for i, layer in enumerate(cache.layers)]


def content_ids(message, eos):
    ids = message.message_token_ids
    return ids[:-1] if message.termination_reason == 'eos' and ids[-1] in eos else ids


def token_injection(session, text, *, source='A', target='B'):
    """Artificial TOKEN diagnostic fixture, explicitly NOT Sender Decode output.

    Put exact fixed IDs into an isolated source draft, then require ordinary handoff.
    This helper has no KV option; the harness only exports KV from generate_message.
    """
    old, _ = session._pair(source, target)
    if old.draft_message is not None or old.assistant_open:
        raise ValueError('test injection requires a closed Sender with no draft')
    ids = tuple(old.backend.chat_tokenizer.encode(text, add_special_tokens=False))
    if not ids or any(i in old.backend.chat_tokenizer.all_special_ids for i in ids):
        raise ValueError('test injection must be nonempty ordinary content')
    prefix = old.framing.assistant.prefix_ids
    session._reserve(old, len(prefix) + len(ids) + len(old.framing.assistant.closing_ids))
    work = append_working_ids(working_state(old), prefix)
    start = work.cached_token_count
    if len(ids) > 1:
        work = append_working_ids(work, ids[:-1])
    message = AgentMessage(f'{source}:{len(old.sent_messages)+1}', len(old.sent_messages)+1,
                           source, target, ids, 'max_new_tokens', (start, start+len(ids)))
    session._validate_message(message, work)
    work = replace(work, token_ledger=work.token_ledger+(ids[-1],),
                   pending_token_id=ids[-1], assistant_open=True, draft_message=message,
                   version=old.version+1)
    work.validate()
    session._states[source] = work
    return message


def context_view(state, *, question=None, stage=None, generation_prefix=False):
    """Recover source spans from existing state metadata without forward/mutation."""
    state.validate()
    b, f = state.backend, state.framing
    decode = lambda ids: b.decode(ids, skip_special_tokens=False)
    ids = state.token_ledger
    parts = []
    logical = []

    def add(start, end, source, role, body=None, identity=None):
        parts.append(dict(start=start, end=end, source=source, role=role,
                          speaker=identity, token_ids=ids[start:end], text=decode(ids[start:end])))
        if body is not None:
            logical.append((start, dict(role=role, content=body)))

    add(0, len(f.initial_ids), 'system', 'system', state.system_prompt, state.agent_id)
    local_frame = f._frame(f.system, 'user', '')
    for start, end in state.local_input_spans:
        body = ids[start+len(local_frame.prefix_ids):end-len(local_frame.closing_ids)]
        add(start, start+len(local_frame.prefix_ids), 'local framing', 'user')
        add(start+len(local_frame.prefix_ids), end-len(local_frame.closing_ids),
            'user/stage', 'user', decode(body), state.agent_id)
        add(end-len(local_frame.closing_ids), end, 'local framing', 'user')
    for m in state.received_messages:
        start = m.receiver_message_start
        end = start + m.receiver_content_token_count
        frame = f.incoming(m.source_agent, state.agent_id)
        add(start-len(frame.prefix_ids), start, 'local framing', 'user', identity=m.source_agent)
        add(start, end, 'received', 'user',
            decode(frame.prefix_ids[len(local_frame.prefix_ids):])+decode(ids[start:end]), m.source_agent)
        add(end, end+len(frame.closing_ids), 'local framing', 'user')
    own = list(state.sent_messages) + list(state.local_messages)
    if state.draft_message:
        own.append(state.draft_message)
    for m in own:
        start, raw_end = m.sender_message_span
        body = content_ids(m, b.eos_token_ids)
        end = start + len(body)
        add(start-len(f.assistant.prefix_ids), start, 'assistant framing', 'assistant')
        add(start, end, 'assistant', 'assistant', decode(body), state.agent_id)
        add(end, min(len(ids), max(raw_end, end+len(f.assistant.closing_ids))),
            'assistant closing/control', 'assistant')
    if stage is not None:
        start = len(ids)
        local = f.local_user_ids(stage)
        ids += local
        add(start, start+len(local_frame.prefix_ids), 'local framing', 'user')
        add(start+len(local_frame.prefix_ids), len(ids)-len(local_frame.closing_ids),
            'stage', 'user', stage, state.agent_id)
        add(len(ids)-len(local_frame.closing_ids), len(ids), 'local framing', 'user')
    if generation_prefix:
        start = len(ids)
        ids += f.assistant.prefix_ids
        add(start, len(ids), 'assistant generation prefix', 'assistant', identity=state.agent_id)
    parts.sort(key=lambda p: p['start'])
    coverage = [0]*len(ids)
    for part in parts:
        for i in range(part['start'], part['end']):
            coverage[i] += 1
    messages = [m for _, m in sorted(logical, key=lambda x: x[0])]
    # Oracle ONLY: independently render expected logical template. Never feed
    # this decode/encode result back into a session, even when IDs differ.
    expected = f._render_ids(messages, generation=generation_prefix)
    first = next((i for i, (a,c) in enumerate(zip(ids, expected)) if a != c), None)
    if first is None and len(ids) != len(expected):
        first = min(len(ids), len(expected))
    recent = state.received_messages[-1] if state.received_messages else None
    return dict(agent=state.agent_id, question=question, system_prompt=state.system_prompt,
                latest_sender=recent.source_agent if recent else None,
                latest_received=decode(ids[recent.receiver_message_start:
                    recent.receiver_message_start+recent.receiver_content_token_count]) if recent else None,
                stage_prompt=stage, ledger_token_ids=ids, ledger_length=len(ids),
                effective_cache_length=state.cached_token_count, pending_token_id=state.pending_token_id,
                preview_extra_tokens=len(ids)-len(state.token_ledger),
                decoded_ledger=decode(ids), segments=parts, logical_messages=messages,
                exact_span_coverage=all(n == 1 for n in coverage),
                special_boundaries=[dict(position=i, token_id=t, text=decode((t,)))
                    for i,t in enumerate(ids) if t in b.chat_tokenizer.all_special_ids],
                template_token_ids=expected, template_ids_equal=ids == expected,
                first_input_difference=first, cache_layers=cache_shapes(state.cache))


def tensor_difference(reference, actual):
    r, a = reference.detach().float().cpu(), actual.detach().float().cpu()
    d = (r-a).abs()
    return dict(max_abs=d.max().item(), mean_abs=d.mean().item(),
                relative_l2=(d.norm()/r.norm().clamp_min(1e-12)).item())


def logits_difference(reference, actual):
    r, a = reference.detach().float().cpu(), actual.detach().float().cpu()
    return dict(**tensor_difference(r,a),
                reference_top1=int(r.argmax()), actual_top1=int(a.argmax()),
                top1_equal=int(r.argmax()) == int(a.argmax()))


class ForwardTrace:
    """Temporary diagnostic hooks, restored even on failure. Single CLI owner."""
    def __init__(self, backend, prefix_ids=()):
        self.b = backend
        self.prefix_ids = tuple(prefix_ids)
        self.calls = []
        self.positions = []
        self.prefix_result = None

    def __enter__(self):
        self.original = self.b._timed_forward
        self.continue_original = self.b.continue_from_cache
        rotary = getattr(getattr(self.b._model, 'model', None), 'rotary_emb', None)
        self.hook = None
        if rotary is not None:
            def positions(module, args, kwargs):
                p = kwargs.get('position_ids', args[1] if len(args)>1 else None)
                if p is not None:
                    self.positions.append(p.detach().cpu().tolist())
            self.hook = rotary.register_forward_pre_hook(positions, with_kwargs=True)
        def traced(**kwargs):
            before = len(self.positions)
            past = kwargs.get('past_key_values')
            count = int(past.get_seq_length()) if past is not None else 0
            row = dict(input_ids=kwargs['input_ids'].detach().cpu().tolist()[0],
                       cache_length_before=count,
                       attention_mask=kwargs['attention_mask'].detach().cpu().tolist(),
                       supplied_position_ids=None if kwargs.get('position_ids') is None
                           else kwargs['position_ids'].detach().cpu().tolist(),
                       supplied_cache_position=None if kwargs.get('cache_position') is None
                           else kwargs['cache_position'].detach().cpu().tolist())
            self.calls.append(row)
            out = self.original(**kwargs)
            row['effective_rotary_position_ids'] = self.positions[before:]
            row['cache_length_after'] = int(out[0].past_key_values.get_seq_length())
            return out
        def continuation(ids, **kwargs):
            out = self.continue_original(ids, **kwargs)
            # Capture only generation prefix. Clone before subsequent Decode.
            if self.prefix_ids and tuple(ids) == self.prefix_ids:
                self.prefix_result = (out.last_token_logits.detach().clone(),
                                      self.b.clone_cache(out.cache))
            return out
        self.b._timed_forward = traced
        self.b.continue_from_cache = continuation
        return self

    def __exit__(self, *exc):
        self.b._timed_forward = self.original
        self.b.continue_from_cache = self.continue_original
        if self.hook:
            self.hook.remove()


def greedy_from_prefill(backend, result, limit):
    cache = backend.clone_cache(result.cache)
    count = result.logical_cached_token_count
    ids = [result.first_token_id]
    while ids[-1] not in backend.eos_token_ids and len(ids) < limit:
        out = backend.continue_from_cache((ids[-1],), previous_cache=cache,
                                         logical_cached_token_count=count)
        cache, count = out.cache, out.logical_cached_token_count
        ids.append(out.predicted_next_token_id)
    return tuple(ids)


def token_oracle(backend, view, incremental_logits, incremental_cache, generated_ids, limit):
    """Same-ID clean Prefill vs production prefix result, numerical tolerance report."""
    if not view['template_ids_equal']:
        return dict(status='input_mismatch', first_difference=view['first_input_difference'])
    ids = tuple(view['ledger_token_ids'])
    with ForwardTrace(backend) as trace:
        full = backend.prefill(input_token_ids=ids)
    backend.validate_cache(incremental_cache, len(ids))
    layers = []
    for i,(r,a) in enumerate(zip(full.cache.layers, incremental_cache.layers)):
        layers.append(dict(layer=i, key=tensor_difference(r.keys, a.keys),
                           value=tensor_difference(r.values, a.values)))
    reference_ids = greedy_from_prefill(backend, full, limit)
    first = next((i for i,(r,a) in enumerate(zip(reference_ids, generated_ids)) if r != a),
                 min(len(reference_ids),len(generated_ids)) if len(reference_ids)!=len(generated_ids) else None)
    divergence = None
    if first is not None and first < min(len(reference_ids),len(generated_ids)):
        # Both caches see only the common generated prefix up to the FIRST fork.
        rc, ac = backend.clone_cache(full.cache), backend.clone_cache(incremental_cache)
        rl, al = full.last_token_logits, incremental_logits
        count = len(ids)
        for token in generated_ids[:first]:
            ro = backend.continue_from_cache((token,),previous_cache=rc,logical_cached_token_count=count)
            ao = backend.continue_from_cache((token,),previous_cache=ac,logical_cached_token_count=count)
            rc, ac, rl, al = ro.cache, ao.cache, ro.last_token_logits, ao.last_token_logits
            count += 1
        def top5(logits):
            values, choices = logits.detach().float().cpu().topk(5)
            return dict(ids=choices.tolist(), values=values.tolist(), margin=(values[0]-values[1]).item())
        divergence = dict(generation_index=first, context_position=count,
                          same_input_prefix=ids+tuple(generated_ids[:first]),
                          logits=logits_difference(rl,al),reference_top5=top5(rl),
                          incremental_top5=top5(al))
    return dict(status='compared', input_ids_equal=True, cache_length=len(ids),
                full_forward=trace.calls, reference_cache_layers=cache_shapes(full.cache),
                incremental_cache_layers=cache_shapes(incremental_cache), kv_differences=layers,
                logits=logits_difference(full.last_token_logits, incremental_logits),
                reference_generated_ids=reference_ids,
                reference_text=backend.decode(reference_ids, skip_special_tokens=True),
                generation_ids_equal=reference_ids == tuple(generated_ids),
                first_generation_difference=first, first_fork_same_input=divergence)
