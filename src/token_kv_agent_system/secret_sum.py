"""Asymmetric private-input scene over the unchanged discussion/session engine.

The controller's expected values enter initialization for their OWNER only.
Scoring consumes completed outputs; it never supplies an inference input.
"""
from dataclasses import dataclass, replace
import re

from .communication import HandoffChoice
from .discussion import DiscussionConfig, DiscussionRuntime
from .incremental_auto import IncrementalAutoConfig
from .network.handoff_network import HandoffNetworkConfig
from .discussion_prompts import StageInstructions, resolve_prompts

PUBLIC_TASK = (
    'You each have a private number. Exchange your numbers, add them together, '
    'and give the final sum. Answer briefly in English. '
    'If the other number has not been received, say it is unknown; do not guess.'
)
STAGES = StageInstructions(
    a_first='Tell Agent B your private number and ask for its number. Do not calculate yet.',
    b_first=("Read the received message from Agent A. State Agent A's number and your own "
             'number with clear ownership, then write their addition as an equation. '
             'If A did not provide a number, say it is unknown.'),
    a_followup='Unused in this single exchange scene.',
    b_followup='Unused in this single exchange scene.',
    final=("Use the received message from Agent B and your own private number. "
           "State Agent A's number and Agent B's number with clear ownership, "
           'then write their addition as an equation. If B did not provide a number, '
           'say it is unknown. Do not guess.'),
)


def private_prompt(agent: str, number: int) -> str:
    return (f'You are Agent {agent}, collaborating with the other agent. '
            f'Your private number is {number}. '
            'You initially do not know the other agent\'s number. '
            'Learn it only from received messages. Do not invent missing information.')


@dataclass(frozen=True)
class SecretSumConfig:
    secret_a: int = 17
    secret_b: int = 26
    max_new_tokens: int = 96
    max_context_tokens: int = 4096
    handoff_choice: HandoffChoice = HandoffChoice()
    handoff_plan: tuple[HandoffChoice, ...] = ()

    auto: IncrementalAutoConfig | None = None
    network: HandoffNetworkConfig = HandoffNetworkConfig()

    def __post_init__(self):
        for value in (self.secret_a, self.secret_b):
            if type(value) is not int or not 1 <= value <= 99:
                raise ValueError('private numbers must be integers in 1–99')
        self.runtime_config()  # Reuse budget and routing validation.

    @property
    def rounds(self):
        return 1

    def runtime_config(self):
        return DiscussionConfig(question=PUBLIC_TASK, rounds=1, summarizer='A',
            max_new_tokens=self.max_new_tokens, max_context_tokens=self.max_context_tokens,
            handoff_choice=self.handoff_choice, handoff_plan=self.handoff_plan,
            auto=self.auto, network=self.network,
            agent_a_system_prompt=private_prompt('A', self.secret_a),
            agent_b_system_prompt=private_prompt('B', self.secret_b))

    def prompts(self, *, control_tokens=()):
        return replace(resolve_prompts(private_prompt('A', self.secret_a),
            private_prompt('B', self.secret_b), control_tokens=control_tokens),
            version='secret-sum-v1', stages=STAGES)


class SecretSumRuntime(DiscussionRuntime):
    """One A→B→A exchange, using production generation, bytes and cache checks.

    No receiver-local message content injection or changes to framing/protocol.
    Ordinary DiscussionRuntime and its prompt definitions remain unchanged.
    """
    def __init__(self, backend_a, backend_b=None, *, config=None, **kwargs):
        self.secret_config = config or SecretSumConfig()
        self.initial_token_ledgers = {}
        super().__init__(backend_a, backend_b, config=self.secret_config.runtime_config(), **kwargs)
        self._prompt_config = replace(self.prompt_config, version='secret-sum-v1', stages=STAGES)

    def _verify(self):
        super()._verify()
        if not self.initial_token_ledgers:
            self.initial_token_ledgers = {n: self.session.state(n).token_ledger for n in ('A', 'B')}

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        if hasattr(self, '_config'):
            raise ValueError('secret_sum configuration is fixed; create a new Session')
        self._config = value


# Narrow evidence parser: ambiguous/conflicting/unsupported prose is undecidable.
# Explicit ownership is required; mere occurrences of the expected numbers do
# not establish receipt. Equations support arithmetic scoring independently.
def output_evidence(text: str | None, speaker: str, *, allow_bare_own=False):
    text = text or ''
    owners = {}
    for agent in ('A', 'B'):
        patterns = [rf"\b(?:Agent\s+{agent}(?:'s|’s)?|{agent}['’]s)\s+(?:private\s+)?number\s*(?:is|=|:)\s*(\d+)\b(?![./]\d)"]
        patterns.append(rf"\bAgent\s+{agent}\s*:\s*(\d+)\b(?![./]\d)")
        if agent == speaker:
            patterns.append(r'\bMy\s+(?:private\s+)?number\s*(?:is|=|:)\s*(\d+)\b(?![./]\d)')
        values = {int(m) for pattern in patterns for m in re.findall(pattern, text, re.I)}
        owners[agent] = next(iter(values)) if len(values) == 1 else None
    if allow_bare_own and re.fullmatch(r'\s*\d+\s*[.!]?\s*', text):
        owners[speaker] = int(re.search(r'\d+', text)[0])
    equations = {tuple(map(int, m)) for m in re.findall(r'(?<![\d.\-])(\d+)\s*\+\s*(\d+)\s*=\s*(\d+)(?!\d|[./]\d)', text)}
    sums = {eq[2] for eq in equations}
    sums.update(int(m) for m in re.findall(r'\b(?:The\s+)?sum\s+(?:of\s+(?:our|their|the)\s+(?:numbers|private numbers)\s+)?(?:is|=|:)\s*(\d+)(?!\d|[./]\d)', text, re.I))
    return dict(numbers=owners, equations=sorted(equations),
                stated_sum=next(iter(sums)) if len(sums)==1 else None,
                parse_note='Explicit ownership, bare sender number, integer equation or explicit sum statement; ambiguity is unknown.')


def _and(*values):
    return False if False in values else None if None in values else True


def score_outputs(config, a_text, b_text, final_text, *, engineering_success, delivered=(True, True)):
    """Post-hoc evidence, not proof of causality from a single correct answer."""
    first = output_evidence(a_text, 'A', allow_bare_own=True)
    reply = output_evidence(b_text, 'B', allow_bare_own=True)
    final = output_evidence(final_text, 'A')
    def equal(value, expected):
        return value == expected if value is not None else None
    a_sent = equal(first['numbers']['A'], config.secret_a)
    b_sent = equal(reply['numbers']['B'], config.secret_b)
    b_read = _and(delivered[0], a_sent, equal(reply['numbers']['A'], config.secret_a))
    a_read = _and(delivered[1], b_sent, equal(final['numbers']['B'], config.secret_b))
    expected = config.secret_a + config.secret_b  # Evaluation ONLY, after Decode.
    equations = final['equations']
    sum_correct = equal(final['stated_sum'], expected)
    operands_correct = (sorted(equations[0][:2]) == sorted((config.secret_a, config.secret_b))) if len(equations) == 1 else None
    if not equations and all(final['numbers'][n] is not None for n in ('A','B')):
        operands_correct = sorted(final['numbers'].values()) == sorted((config.secret_a, config.secret_b))
    return dict(engineering_success=engineering_success, a_sent_own_secret=a_sent,
        b_sent_own_secret=b_sent, b_received_a=b_read, a_received_b=a_read,
        information_exchange_success=_and(b_read, a_read), final_sum_correct=sum_correct,
        task_success=_and(engineering_success, sum_correct, operands_correct),
        expected_sum=expected, final_operands_correct=operands_correct,
        evidence=dict(a_first=first, b_reply=reply, a_final=final),
        note='Post-hoc evidence only. Paired secret changes and absent-message controls are needed for causal assessment.')


def score_result(config, result):
    texts = {t.message.source_agent: t.text for t in result.turns}
    delivered = tuple(any(h.source == n and h.append_succeeded for h in result.handoffs) for n in ('A', 'B'))
    return score_outputs(config, texts.get('A'), texts.get('B'), result.final_text,
                         engineering_success=result.status == 'completed', delivered=delivered)
