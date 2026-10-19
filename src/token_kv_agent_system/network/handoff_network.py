"""Replayable, handoff-indexed simulation; never sleeps or limits physical IO."""
from dataclasses import dataclass, asdict
import hashlib
import json
import math
import random
from threading import RLock
from .shannon import ShannonChannelConfig, channel_snapshot


def finite_number(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')


@dataclass(frozen=True)
class HandoffNetworkConfig:
    mode: str = 'manual'
    bandwidth_mbps: float = 20.0
    base_latency_ms: float = 0.0  # One-way, charged once per directional handoff.
    trace_mbps: tuple[float, ...] = ()
    source: str = 'session_config'
    exhaustion: str = 'hold_last'
    channel: ShannonChannelConfig | None = None

    def __post_init__(self):
        if self.mode not in ('manual', 'trace', 'shannon') or self.exhaustion != 'hold_last':
            raise ValueError('network requires manual/trace/shannon mode and hold_last exhaustion')
        if (self.mode == 'shannon' and not isinstance(self.channel, ShannonChannelConfig)) or (
                self.mode != 'shannon' and self.channel is not None):
            raise ValueError('shannon mode requires a ShannonChannelConfig; legacy modes cannot carry one')
        finite_number(self.bandwidth_mbps, 'bandwidth_mbps', positive=True)
        finite_number(self.base_latency_ms, 'base_latency_ms')
        if not isinstance(self.trace_mbps, tuple):
            raise ValueError('trace_mbps must be a tuple')
        for value in self.trace_mbps:
            finite_number(value, 'trace bandwidth', positive=True)
        if (self.mode == 'trace') != bool(self.trace_mbps):
            raise ValueError('trace mode requires a nonempty trace; manual mode cannot carry a trace')
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError('network source must be nonempty')


@dataclass(frozen=True)
class HandoffNetworkSnapshot:
    handoff_index: int
    bandwidth_mbps: float
    base_latency_ms: float
    network_mode: str
    trace_id: str
    source: str
    config_revision: int
    simulation_step: int
    trace_exhausted: bool
    is_simulated: bool = True
    advancement: str = 'handoff_index'
    latency_semantics: str = 'one_way_once'
    channel: dict | None = None

    def transfer_ms(self, wire_bytes: int) -> float:
        if type(wire_bytes) is not int or wire_bytes < 0:
            raise ValueError('wire bytes must be a nonnegative integer')
        try:
            result = wire_bytes * 8 / (self.bandwidth_mbps * 1000) + self.base_latency_ms
        except OverflowError as exc:
            raise ValueError('simulated transfer prediction exceeds finite numeric range') from exc
        finite_number(result, 'simulated transfer prediction')
        return result


class HandoffNetwork:
    """Snapshot before an attempt. Retries reuse it; updates affect future indices.

    No implicit retry or advancement on success/failure. A mode switch starts a
    new trace at its first state for the next unseen index. Session-level REST
    configuration and live control share the same revision boundary.
    """
    def __init__(self, config=HandoffNetworkConfig(), *, lock=None):
        self._lock = lock if lock is not None else RLock()
        self._snapshots = {}
        self._revision = 0
        self._start = 1
        self._config = config
        self._rng = random.Random(config.channel.seed if config.channel else 0)
        self._ar_state = None

    def update(self, config):
        if not isinstance(config, HandoffNetworkConfig):
            raise ValueError('network config required')
        with self._lock:
            if config.channel and (not self._config.channel or config.channel.seed != self._config.channel.seed):
                self._rng = random.Random(config.channel.seed)
                self._ar_state = None
            self._config = config
            self._revision += 1
            self._start = len(self._snapshots) + 1

    def current(self):
        with self._lock:
            return dict(config=asdict(self._config), config_revision=self._revision,
                        next_unfrozen_handoff_index=len(self._snapshots) + 1,
                        is_simulated=True, effective_rule='next_unfrozen_handoff')

    def snapshot(self, handoff_index):
        if type(handoff_index) is not int or handoff_index < 1:
            raise ValueError('handoff_index must be a positive integer')
        with self._lock:
            if handoff_index in self._snapshots:
                return self._snapshots[handoff_index]
            if handoff_index != len(self._snapshots) + 1:
                raise ValueError('new snapshots must advance one index at a time')
            c = self._config
            step = handoff_index - self._start
            channel = None
            bandwidth = c.trace_mbps[min(step, len(c.trace_mbps)-1)] if c.mode == 'trace' else c.bandwidth_mbps
            if c.channel:
                innovation = self._rng.gauss(0,1)
                self._ar_state = innovation if self._ar_state is None else (
                    c.channel.correlation*self._ar_state + math.sqrt(1-c.channel.correlation**2)*innovation)
                channel = channel_snapshot(c.channel,self._ar_state)
                bandwidth = channel['effective_rate_mbps']
            trace_id = hashlib.sha256(json.dumps(asdict(c), sort_keys=True).encode()).hexdigest()[:16]
            snap = HandoffNetworkSnapshot(handoff_index, bandwidth, c.base_latency_ms,
                c.mode, trace_id, c.source, self._revision, step,
                c.mode == 'trace' and step >= len(c.trace_mbps), channel=channel)
            self._snapshots[handoff_index] = snap
            return snap
