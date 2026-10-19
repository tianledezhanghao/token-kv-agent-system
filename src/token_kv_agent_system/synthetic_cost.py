"""Versioned, immutable assumptions; never a measured cost profile."""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class SyntheticCostAssumptions:
    schema: str = 'synthetic_compute_v1'
    source: str = 'synthetic_assumption'
    units: str = 'ms'
    a: float = 20
    b: float = 2
    c: float = 0.002
    d: float = 5
    e: float = 0.05
    f: float = 0.2

    def __post_init__(self):
        if (self.schema, self.source, self.units) != ('synthetic_compute_v1', 'synthetic_assumption', 'ms'):
            raise ValueError('Synthetic assumptions require synthetic_compute_v1, synthetic_assumption and ms')
        for name in 'abcdef':
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1e9:
                raise ValueError(f'Synthetic {name} must be a finite number in [0, 1e9]')

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) - set(cls.__dataclass_fields__):
            raise ValueError('Unknown Synthetic assumption fields')
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    def predict(self, features):
        n, h, k = (features[x] for x in ('message_tokens', 'receiver_history_tokens', 'recomputed_tokens'))
        for value in (n, h, k):
            if type(value) is not int or value < 0:
                raise ValueError('Synthetic N/H/k must be nonnegative integers')
        result = self.a + self.b*n + self.c*n*h, self.d + self.e*n + self.f*k
        if any(not math.isfinite(v) for v in result):
            raise ValueError('Nonfinite Synthetic cost')
        return result
