"""Manual discussion routing, independent of the canonical Selector."""
from dataclasses import dataclass
from enum import Enum
import math


class CommunicationMode(str, Enum):
    TOKEN = 'token'
    KV = 'kv'


class KVStrategy(str, Enum):
    RAW_DELTA = 'raw_delta'
    ROPE_ALIGNED_DELTA = 'rope_aligned_delta'
    SUFFIX_RECOMPUTE_DELTA = 'suffix_recompute_delta'
    NUMERIC_SLOT_DELTA = 'numeric_slot_delta'


@dataclass(frozen=True)
class HandoffChoice:
    mode: CommunicationMode = CommunicationMode.TOKEN
    kv_strategy: KVStrategy | None = None

    recompute_tokens: int | None = None
    recompute_ratio: float | None = None

    def __post_init__(self):
        object.__setattr__(self, 'mode', CommunicationMode(self.mode))
        if self.kv_strategy is not None:
            object.__setattr__(self, 'kv_strategy', KVStrategy(self.kv_strategy))
        if (self.mode == CommunicationMode.KV) != (self.kv_strategy is not None):
            raise ValueError('KV requires v0/v1/v2/v3 strategy; TOKEN cannot have a KV strategy')

        validate_recompute_policy(self.recompute_tokens, self.recompute_ratio)
        if self.kv_strategy != KVStrategy.SUFFIX_RECOMPUTE_DELTA and (
                self.recompute_tokens is not None or self.recompute_ratio is not None):
            raise ValueError('recompute configuration requires kv:v2')

    @classmethod
    def parse(cls, value: str, *, recompute_tokens=None, recompute_ratio=None):
        if value == 'token':
            return cls(recompute_tokens=recompute_tokens, recompute_ratio=recompute_ratio)
        strategies = {'kv:v0': KVStrategy.RAW_DELTA, 'kv:v1': KVStrategy.ROPE_ALIGNED_DELTA,
                      'kv:v2': KVStrategy.SUFFIX_RECOMPUTE_DELTA,
                      'kv:v3': KVStrategy.NUMERIC_SLOT_DELTA}
        if value not in strategies:
            raise ValueError(f'unsupported handoff {value!r}; use token, kv:v0, kv:v1, kv:v2 or kv:v3')
        return cls(CommunicationMode.KV, strategies[value], recompute_tokens, recompute_ratio)


def validate_recompute_policy(tokens=None, ratio=None):
    if tokens is not None and (type(tokens) is not int or tokens < 0):
        raise ValueError('recompute_tokens must be a nonnegative integer')
    if ratio is not None and (type(ratio) not in (int, float) or not math.isfinite(ratio)
                              or not 0 <= ratio <= 1):
        raise ValueError('recompute_ratio must be finite and in [0, 1]')
    if tokens is not None and ratio is not None:
        raise ValueError('choose recompute_tokens OR recompute_ratio')


def suffix_recompute_count(message_count, *, tokens=None, ratio=None):
    """Nonempty content; explicit saturation at N, default k=1."""
    validate_recompute_policy(tokens, ratio)
    if type(message_count) is not int or message_count <= 0:
        raise ValueError('empty message content cannot be recomputed')
    requested = math.ceil(message_count * ratio) if ratio is not None else (1 if tokens is None else tokens)
    return min(requested, message_count)
