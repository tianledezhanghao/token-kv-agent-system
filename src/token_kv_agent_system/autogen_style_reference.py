"""Independent 7.7G full-prefill reference; never a communication implementation.

A/B/C are controlled message grouping alternatives. D preserves AutoGen 0.7.5
model-message roles/sources, with an explicit local name-prefix adapter. Official
AgentChat execution is available separately, with optional isolated dependencies.
"""
from collections.abc import Mapping
from dataclasses import asdict, dataclass, replace
from difflib import SequenceMatcher

from .discussion import DiscussionConfig, DiscussionRuntime
from .discussion_prompts import AGENT_A_SYSTEM_PROMPT, AGENT_B_SYSTEM_PROMPT, StageInstructions
from .message_diagnostics import greedy_from_prefill

STRUCTURES = ('A', 'B', 'C', 'D')
MODES = ('critic', 'peer')
READ_TASK = ('说明Agent A实际说了什么，用自己的话解释其观点。不要把对方发言当作你自己的上一轮回答。'
             '如果没有收到A的消息，明确说明。')
STAGES = {
    'critic': StageInstructions(
        a_first='Answer the original question briefly.',
        b_first="Check Agent A's actual answer to the original question. Identify a specific error with a reason, or briefly confirm it.",
        a_followup="Use Agent B's actual feedback to correct or maintain your answer. Reply briefly.",
        b_followup="Check Agent A's updated answer. Identify a specific error, or briefly confirm it.",
        final='Give the final answer to the original question using the discussion. Be brief.'),
    'peer': StageInstructions(
        a_first='Answer the original question briefly.',
        b_first="Read Agent A's actual idea. Add a relevant fact, reason or approach. If already solved, briefly confirm it. Do not merely copy it.",
        a_followup="Use Agent B's actual contribution to continue answering the original question. Reply briefly.",
        b_followup="Read Agent A's updated idea. Add relevant information, or briefly confirm a solved question. Do not merely copy it.",
        final='Give the final answer to the original question using the discussion. Be brief.')}
PEER_B = ('You are Agent B, collaborating with Agent A on the user question. '
          'Understand A\'s idea and contribute a relevant reason, fact or approach. '
          'If the question is already solved, briefly confirm it. Do not merely copy A. '
          'Do not invent disagreements. Use the requested language.')


def systems(mode):
    if mode not in MODES:
        raise ValueError('unknown collaboration mode')
    return {'A': AGENT_A_SYSTEM_PROMPT, 'B': AGENT_B_SYSTEM_PROMPT if mode == 'critic' else PEER_B}


@dataclass(frozen=True)
class Event:
    source: str
    content: str
    origin: str  # user_task / fixed_fixture / sender_decode / stage_instruction


def model_message(role, content, sources):
    return dict(role=role, content=content, sources=[asdict(e) for e in sources])


def group_incoming(structure, target, events, stage):
    """Convert only NEW deliveries. Retained own replies are never converted to user.

    Stage is controller-local, not an Agent utterance. Missing messages stay absent.
    C combines only the new batch, so the original question is not reinserted later.
    """
    if structure not in STRUCTURES:
        raise ValueError('unknown message structure')
    events = list(events)
    task = Event('controller', stage, 'stage_instruction') if stage is not None else None
    if structure == 'D':
        # Equivalent to AutoGen model messages + add_name_prefixes=True for user.
        return [model_message('user', f'{e.source} said:\n{e.content}', [e])
                for e in events + ([task] if task else [])]
    if structure == 'A':
        rows = [model_message('user', e.content if e.source == 'user' else
                f'Message from {e.source} to {target}:\n{e.content}', [e]) for e in events]
        return rows + ([model_message('user', stage, [task])] if task else [])
    if structure == 'B':
        rows = [model_message('user', e.content, [e]) for e in events if e.source == 'user']
        incoming = [e for e in events if e.source != 'user']
        chunks = [f'这是来自 Agent {e.source} 的消息：\n<{e.source}_MESSAGE>\n{e.content}\n</{e.source}_MESSAGE>' for e in incoming]
        if task:
            chunks.append(stage)
        if chunks:
            rows.append(model_message('user', '\n\n'.join(chunks), incoming + ([task] if task else [])))
        return rows
    chunks = [f'Original question:\n{e.content}' if e.source == 'user' else
              f'Message source: Agent {e.source}\nMessage content:\n{e.content}' for e in events]
    if task:
        chunks.append(f'Task for Agent {target}:\n{stage}')
    return [model_message('user', '\n\n'.join(chunks), events + ([task] if task else []))] if chunks else []


def template_ids(backend, messages):
    # sources are retained in trace; only standard role/content reaches HF template.
    rendered = backend.chat_tokenizer.apply_chat_template(
        [dict(role=m['role'], content=m['content']) for m in messages],
        tokenize=True, add_generation_prompt=True)
    if isinstance(rendered, Mapping):
        rendered = rendered['input_ids']
    return tuple(int(i) for i in rendered)


def generate(backend, messages, limit):
    if type(limit) is not int or limit <= 0:
        raise ValueError('token limit must be positive')
    ids = template_ids(backend, messages)
    output = greedy_from_prefill(backend, backend.prefill(input_token_ids=ids), limit)
    stop = 'eos' if output[-1] in backend.eos_token_ids else 'max_new_tokens'
    return dict(messages=messages, input_ids=ids, output_ids=output,
                text=backend.decode(output, skip_special_tokens=True), stop_reason=stop,
                truncated=stop == 'max_new_tokens', max_new_tokens=limit,
                do_sample=False, eos_ids=tuple(backend.eos_token_ids), prefill='complete_history')


class ReferenceHarness:
    """Two independent logical contexts sharing ONE local backend, full Prefill/turn.

    No persistent cross-turn KV, TOKEN bytes, KV injection or production changes.
    """
    def __init__(self, backend, question, *, structure='D', mode='critic', prompts=None):
        if not question.strip() or structure not in STRUCTURES:
            raise ValueError('nonempty question and known structure required')
        self.backend, self.structure, self.mode = backend, structure, mode
        prompts = prompts or systems(mode)
        self.contexts = {n: [model_message('system', prompts[n], [])] for n in ('A', 'B')}
        self.pending = {n: [Event('user', question, 'user_task')] for n in ('A', 'B')}
        self.deliveries = []
        self.trace = []

    def deliver(self, source, target, content, *, origin):
        if source not in self.contexts or target not in self.contexts or source == target:
            raise ValueError('delivery requires distinct A/B')
        if origin not in ('fixed_fixture', 'sender_decode') or not isinstance(content, str) or not content:
            raise ValueError('explicit origin and nonempty content required')
        event = Event(source, content, origin)
        self.pending[target].append(event)
        self.deliveries.append(dict(target=target, **asdict(event)))

    def turn(self, agent, stage, limit):
        messages = self.contexts[agent] + group_incoming(self.structure, agent, self.pending[agent], stage)
        row = generate(self.backend, messages, limit)
        row.update(agent=agent, stage=stage, structure=self.structure, mode=self.mode,
                   origin='real_local_decode', deliveries=tuple(self.deliveries))
        self.contexts[agent] = messages + [model_message('assistant', row['text'],
                                                     [Event(agent, row['text'], 'sender_decode')])]
        self.pending[agent].clear()
        self.trace.append(row)
        return row

    def run(self, *, limit=128, rounds=2, stages=None):
        if type(rounds) is not int or not 1 <= rounds <= 2:
            raise ValueError('reference supports one or two rounds')
        stages = stages or STAGES[self.mode]
        for round_number in range(1, rounds + 1):
            for agent, target in (('A', 'B'), ('B', 'A')):
                row = self.turn(agent, stages.for_turn(agent, round_number), limit)
                self.deliver(agent, target, row['text'], origin='sender_decode')
        self.turn('A', stages.final, limit)
        return dict(runtime='lightweight_autogen_style_not_official', structure=self.structure,
                    mode=self.mode, turns=self.trace, deliveries=self.deliveries,
                    final_contexts=self.contexts, repetition=repetition_report(self.trace))


def overlap(left, right):
    """Descriptive only: short identical answers can be valid confirmation."""
    a, b = left.strip(), right.strip()
    return dict(exact_match=bool(a) and a == b,
                character_overlap=SequenceMatcher(None, a, b, autojunk=False).ratio(),
                long_exact_copy=bool(a) and a == b and len(a) >= 80)


def repetition_report(turns):
    comparisons = [dict(left=i, right=i+1, **overlap(a['text'], b['text']))
                   for i, (a, b) in enumerate(zip(turns, turns[1:]))]
    same_agent = [dict(left=i, right=j, **overlap(turns[i]['text'], turns[j]['text']))
                  for i in range(len(turns)) for j in range(i+1, len(turns))
                  if turns[i]['agent'] == turns[j]['agent']]
    return dict(adjacent=comparisons, same_agent=same_agent,
                adjacent_exact_rate=sum(x['exact_match'] for x in comparisons)/len(comparisons) if comparisons else 0,
                semantic_judgment='requires_content_and_source_review')


def experimental_stop(turns, *, max_turns, content_verified=False, review_signal=None):
    """Independent policy prototype; never used to shorten experimental traces.

    A reviewer signal alone is not factual validation. Specific correction calls
    for another A turn. Limit/repetition/truncation are safety stops, not success.
    """
    if max_turns <= 0:
        raise ValueError('positive max_turns required')
    if len(turns) >= max_turns:
        return dict(stop=True, reason='max_turns', success=False)
    if turns and turns[-1]['stop_reason'] == 'max_new_tokens':
        return dict(stop=True, reason='truncated', success=False)
    if len(turns) >= 3 and all(overlap(a['text'], b['text'])['long_exact_copy']
                               for a, b in zip(turns[-3:], turns[-2:])):
        return dict(stop=True, reason='long_repetition', success=False)
    if turns and turns[-1]['agent'] == 'B' and review_signal == 'approve' and content_verified:
        return dict(stop=True, reason='externally_verified_approval', success=True)
    return dict(stop=False, reason='specific_correction' if review_signal == 'correction' else 'continue', success=False)


class ComparisonRuntime(DiscussionRuntime):
    """Harness-local stage overrides; production defaults remain unchanged."""
    def __init__(self, backend, question, mode, limit):
        prompts = systems(mode)
        super().__init__(backend, config=DiscussionConfig(question=question, rounds=2,
            max_new_tokens=limit, agent_a_system_prompt=prompts['A'], agent_b_system_prompt=prompts['B']))
        self._prompt_config = replace(self.prompt_config, stages=STAGES[mode], version='77g-'+mode)


def capture_token(runtime):
    # Inspect actual IDs, not an assumption that decoding generated IDs and
    # re-encoding their text is canonical. 128-token replies expose that mismatch.
    from .message_diagnostics import context_view
    contexts=[]
    originals={}
    for method in ('generate_message','generate_local_message'):
        original=getattr(runtime.session,method)
        originals[method]=original
        def wrapper(source,*args,_original=original,**kwargs):
            view=context_view(runtime.session.state(source),question=runtime.config.question,
                              stage=kwargs['local_prompt'],generation_prefix=True)
            if not view['exact_span_coverage']:
                raise ValueError('actual ledger span coverage is incomplete')
            message=_original(source,*args,**kwargs)
            if runtime.session.state(source).token_ledger[:message.sender_message_span[0]] != tuple(view['ledger_token_ids']):
                raise ValueError('captured input differs from actual generation ledger')
            contexts.append(dict(context=view,output=asdict(message)))
            return message
        setattr(runtime.session,method,wrapper)
    try:
        result=runtime.run()
    finally:
        for method,original in originals.items():
            setattr(runtime.session,method,original)
    return dict(discussion=asdict(result),contexts=contexts,
                final_contexts={n:context_view(runtime.session.state(n)) for n in ('A','B')})


def token_comparison(backend, question, mode, limit):
    record = capture_token(ComparisonRuntime(backend, question, mode, limit))
    turns = [dict(agent=t['message']['source_agent'], text=t['text'],
                  stop_reason=t['message']['termination_reason']) for t in record['discussion']['turns']]
    final = record['discussion']['final_message']
    turns.append(dict(agent='A', text=record['discussion']['final_text'], stop_reason=final['termination_reason']))
    record.update(runtime='production_token_with_harness_local_prompts', mode=mode,
                  repetition=repetition_report(turns))
    return record
