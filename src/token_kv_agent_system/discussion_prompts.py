"""Authoritative discussion prompts and immutable per-session resolution."""
from dataclasses import dataclass, asdict

MAX_SYSTEM_PROMPT_CHARS = 4000
PROMPT_VERSION = 'general-v1'

AGENT_A_SYSTEM_PROMPT = (
    "You are Agent A. Answer the user's question with a brief reason. "
    "Stay on topic and use the language requested by the user. "
    "Say when you are unsure. Read Agent B's feedback and fix errors if needed. "
    "Avoid repeating confirmations; give a useful answer."
)
AGENT_B_SYSTEM_PROMPT = (
    "You are Agent B. Independently review Agent A's answer to the user's question. "
    "Check relevance, facts, logic and any calculations. "
    "Point out specific errors with a reason. If correct, give a brief check or reason. "
    "If unsure, say so. Do not blindly agree or invent objections. "
    "Use the language requested by the user."
)


@dataclass(frozen=True)
class StageInstructions:
    a_first: str = "Answer the original user question with a brief reason. Use the requested language."
    b_first: str = (
        "Review Agent A's answer above against the original user question. "
        "Check it yourself. Explain any specific error or your check; say if unsure. "
        "Use the requested language."
    )
    a_followup: str = (
        "Read Agent B's actual feedback above. Correct your answer if needed. "
        "If there is no new issue, give a concise useful answer to the original question "
        "without repeating agreement. Use the requested language."
    )
    b_followup: str = (
        "Check Agent A's updated answer above against the original question and prior feedback. "
        "Does it resolve the issues? Give a specific check or remaining error; say if unsure. "
        "Avoid repeating agreement. Use the requested language."
    )
    final: str = (
        "Answer the original user question using your discussion history above. "
        "Give the answer and a brief reason, including any remaining uncertainty. "
        "Use the requested language."
    )

    def for_turn(self, agent: str, round_number: int) -> str:
        return getattr(self, f'{agent.lower()}_' + ('first' if round_number == 1 else 'followup'))


STAGE_INSTRUCTIONS = StageInstructions()


@dataclass(frozen=True)
class EffectiveSystemPrompt:
    text: str
    source: str


@dataclass(frozen=True)
class PromptConfiguration:
    agent_a: EffectiveSystemPrompt
    agent_b: EffectiveSystemPrompt
    version: str = PROMPT_VERSION
    stages: StageInstructions = STAGE_INSTRUCTIONS

    def to_dict(self):
        return asdict(self)


def validate_override(value: str | None, *, control_tokens=()) -> None:
    if value is None:
        return
    if not isinstance(value, str):
        raise ValueError('System Prompt must be a string or null')
    if len(value) > MAX_SYSTEM_PROMPT_CHARS:
        raise ValueError(f'System Prompt must be at most {MAX_SYSTEM_PROMPT_CHARS} characters')
    if any(ord(c) < 32 and c not in '\n\r\t' or 0xD800 <= ord(c) <= 0xDFFF for c in value):
        raise ValueError('System Prompt contains invalid control characters')
    if any(token and token in value for token in control_tokens):
        raise ValueError('System Prompt must not contain chat control tokens')


def resolve_prompts(agent_a: str | None = None, agent_b: str | None = None,
                    *, control_tokens=()) -> PromptConfiguration:
    def resolve(value, default):
        validate_override(value, control_tokens=control_tokens)
        # Preserve valid custom text exactly for reproducible cache initialization.
        return EffectiveSystemPrompt(value, 'custom') if value and value.strip() else EffectiveSystemPrompt(default, 'default')
    return PromptConfiguration(resolve(agent_a, AGENT_A_SYSTEM_PROMPT),
                               resolve(agent_b, AGENT_B_SYSTEM_PROMPT))
