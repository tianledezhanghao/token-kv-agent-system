"""Experiment-only macOS resource policy; never changes OS or MPS limits."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import threading
import time


class ResourceStop(RuntimeError):
    pass


@dataclass(frozen=True)
class ResourceConfig:
    sample_seconds: float = 2
    swap_warning_mib: float = 1024
    critical_seconds: float = 10
    trend_seconds: float = 30
    rapid_swap_mib_per_minute: float = 512
    pressure_seconds: float = 30
    stalled_forward_seconds: float = 60
    stop_grace_seconds: float = 10

    def __post_init__(self):
        import math
        if any(not math.isfinite(v) or v <= 0 for v in asdict(self).values()):
            raise ValueError('resource thresholds must be finite and positive')
        if self.sample_seconds > min(self.critical_seconds, self.trend_seconds):
            raise ValueError('sampling interval must not exceed decision windows')


class ResourcePolicy:
    """Normal=1, warning=2, critical=4; swap alone never stops a run."""
    def __init__(self, baseline, config=None):
        self.config = config or ResourceConfig()
        self.baseline = baseline
        self.history = []
        self.critical_since = self.pressure_since = None

    def observe(self, row, *, active_forward_since=None):
        now = row['monotonic']
        c = self.config
        self.history.append(row)
        # Keep one sample before the window boundary for an honest rate estimate.
        while len(self.history)>2 and self.history[1]['monotonic'] <= now-c.trend_seconds:
            self.history.pop(0)
        level = row['pressure_level']
        if level >= 4:
            if self.critical_since is None:self.critical_since=now
        else:self.critical_since=None
        if level >= 2:
            if self.pressure_since is None:self.pressure_since=now
        else:self.pressure_since=None
        critical_duration = now-self.critical_since if self.critical_since is not None else 0
        pressure_duration = now-self.pressure_since if self.pressure_since is not None else 0
        elapsed = now-self.history[0]['monotonic']
        rate = (row['swap_used_mib']-self.history[0]['swap_used_mib'])*60/elapsed if elapsed else 0
        growth = row['swap_used_mib']-self.baseline['swap_used_mib']
        reason = None
        if critical_duration >= c.critical_seconds:
            reason = 'sustained_critical_pressure'
        elif (elapsed >= c.trend_seconds and pressure_duration >= c.pressure_seconds
              and rate >= c.rapid_swap_mib_per_minute):
            reason = 'sustained_pressure_and_rapid_swap'
        elif (active_forward_since is not None and now-active_forward_since >= c.stalled_forward_seconds
              and pressure_duration >= c.pressure_seconds):
            reason = 'stalled_forward_with_sustained_pressure'
        return dict(swap_growth_mib=growth, swap_rate_mib_per_minute=rate,
                    rate_window_seconds=elapsed, critical_duration_seconds=critical_duration,
                    pressure_duration_seconds=pressure_duration,
                    warning=growth > c.swap_warning_mib or level >= 2,
                    stop_reason=reason)


def classify_failure(exc):
    chain=[]
    for _ in range(4):
        chain.append(str(exc))
        if exc.__cause__ is None:break
        exc=exc.__cause__
    text=' '.join(chain).lower()
    if 'out of memory' in text or 'oom' in text:return 'oom'
    if isinstance(exc,ResourceStop) or 'resourcestop' in text or any(x in text for x in ('sustained_critical_pressure','sustained_pressure_and_rapid_swap','stalled_forward_with_sustained_pressure')):return 'resource_policy'
    return 'execution_exception'


class ResourceMonitor:
    """Samples during load/forward. Stop at boundary, or terminate OWN process
    after grace if unsafe work cannot return. Persist failure before termination.
    """
    def __init__(self, sampler, baseline, output_dir, config=None):
        self.sampler=sampler
        self.policy=ResourcePolicy(baseline,config)
        self.output_dir=Path(output_dir)
        self.samples=[]
        self.forwards=[]
        self.active_forward_since=None
        self.stop_reason=None
        self.stop_at=None
        self.done=threading.Event()
        self.lock=threading.RLock()
        self.thread=None

    def sample(self):
        row=self.sampler()
        row.setdefault('monotonic',time.monotonic())
        with self.lock:
            decision=self.policy.observe(row,active_forward_since=self.active_forward_since)
            row.update(decision)
            row["compressed_memory_change_bytes"]=row.get("compressed_memory_bytes",0)-self.policy.baseline.get("compressed_memory_bytes",0)
            self.samples.append(row)
            with (self.output_dir/'health.jsonl').open('a') as f:
                f.write(json.dumps(row,ensure_ascii=False)+'\n')
            if decision['stop_reason'] and self.stop_reason is None:
                self.stop_reason=decision['stop_reason'];self.stop_at=row['monotonic']
            return row

    def check(self):
        if self.stop_reason:raise ResourceStop(self.stop_reason)

    def start(self):
        self.sample()
        def watch():
            while not self.done.wait(self.policy.config.sample_seconds):
                try:self.sample()
                except Exception as exc:
                    self.stop_reason='resource_sampling_exception: '+str(exc)
                    self.stop_at=self.stop_at or time.monotonic()
                if self.stop_at is not None and time.monotonic()-self.stop_at >= self.policy.config.stop_grace_seconds:
                    row=dict(failure_category='resource_policy',error=self.stop_reason,
                             monitor_terminated_own_process=True, samples=self.samples[-3:])
                    (self.output_dir/'watchdog-failure.json').write_text(json.dumps(row,indent=2))
                    os._exit(2)
        self.thread=threading.Thread(target=watch,name='qwen-resource-monitor',daemon=True)
        self.thread.start()

    def forward(self, original, **kwargs):
        self.check()
        started=time.monotonic()
        self.active_forward_since=started
        row=dict(index=len(self.forwards),started=started,
                 input_ids=kwargs['input_ids'].detach().cpu().tolist()[0],
                 previous_cache_tokens=int(kwargs['past_key_values'].get_seq_length()) if kwargs.get('past_key_values') is not None else 0)
        self.forwards.append(row)
        try:
            result=original(**kwargs)
            row.update(elapsed_seconds=time.monotonic()-started, backend_forward_ms=result[1],
                       next_id=int(result[0].logits[0,-1,:].argmax().item()))
            # Persist progress even if a later turn fails or the watchdog stops.
            with (self.output_dir/'forwards.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            self.check()
            return result
        except Exception as exc:
            row.update(elapsed_seconds=time.monotonic()-started,error=str(exc),failure_category=classify_failure(exc))
            (self.output_dir/'forward-failure.json').write_text(json.dumps(row,indent=2))
            raise
        finally:self.active_forward_since=None

    def close(self):
        self.done.set()
        if self.thread is not None:self.thread.join(timeout=5)
