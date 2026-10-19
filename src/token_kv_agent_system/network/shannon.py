"""Shannon upper-bound channel and stationary, handoff-indexed AR(1) environment.

MHz, dB, Mbps, bytes and milliseconds are explicit. No timers, IO or sleeps.
"""
from dataclasses import dataclass, asdict, fields
import math


@dataclass(frozen=True)
class ShannonChannelConfig:
    bandwidth_mhz: float = 20.0
    snr_mean_db: float = 10.0
    snr_sigma_db: float = 4.0
    correlation: float = 0.85
    seed: int = 42
    efficiency: float = 0.8
    snr_min_db: float | None = None
    snr_max_db: float | None = None

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data)-{f.name for f in fields(cls)}:
            raise ValueError('unknown or malformed Shannon channel parameters')
        return cls(**data)

    def __post_init__(self):
        bounds = {'bandwidth_mhz':(.001,100000), 'snr_mean_db':(-120,120),
                  'snr_sigma_db':(0,40), 'correlation':(0,1), 'efficiency':(0,1)}
        for name,(low,high) in bounds.items():
            v = getattr(self,name)
            if type(v) not in (int,float) or not math.isfinite(v) or not low <= v <= high:
                raise ValueError(f'{name} must be finite and in [{low}, {high}]')
        if self.correlation == 1 or self.efficiency == 0:
            raise ValueError('correlation must be <1; efficiency must be >0')
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32-1:
            raise ValueError('seed must be a uint32 integer')
        for v in (self.snr_min_db,self.snr_max_db):
            if v is not None and (type(v) not in (int,float) or not math.isfinite(v) or not -120 <= v <= 120):
                raise ValueError('SNR clipping bounds must be in [-120,120] dB')
        if self.snr_min_db is not None and self.snr_max_db is not None and self.snr_min_db > self.snr_max_db:
            raise ValueError('SNR minimum must not exceed maximum')
        if self.efficiency*self.bandwidth_mhz*math.log1p(10**(-120/10))/math.log(2) <= 0:
            raise ValueError('efficiency is too small to represent a positive effective rate')


def capacity_mbps(bandwidth_mhz, snr_db):
    if (type(bandwidth_mhz) not in (int,float) or type(snr_db) not in (int,float)
            or not math.isfinite(bandwidth_mhz) or not .001 <= bandwidth_mhz <= 100000
            or not math.isfinite(snr_db) or not -120 <= snr_db <= 120):
        raise ValueError('invalid finite MHz/dB channel values')
    return bandwidth_mhz * math.log1p(10**(snr_db/10)) / math.log(2)


def channel_snapshot(config, standard_normal_state):
    if not math.isfinite(standard_normal_state):
        raise ValueError('AR state must be finite')
    raw = config.snr_mean_db + config.snr_sigma_db*standard_normal_state
    low = config.snr_min_db if config.snr_min_db is not None else -120
    high = config.snr_max_db if config.snr_max_db is not None else 120
    snr = max(low,min(high,raw))
    capacity = capacity_mbps(config.bandwidth_mhz,snr)
    return dict(**asdict(config),snr_db=snr,unclipped_snr_db=raw,clipped=raw!=snr,
                capacity_mbps=capacity,effective_rate_mbps=config.efficiency*capacity,
                rate_source='Shannon theoretical capacity × assumed efficiency',
                sampling='stationary_AR1_once_per_handoff',sample_time_unit='handoff_index')


def crossover_analysis(estimates, channel=None):
    """Use ONLY already eligible same-workload costs; never invent a Profile."""
    token = next((e for e in estimates if e['candidate']=='token'),None)
    kv = next((e for e in estimates if e['candidate']!='token'),None)
    if not token or not kv or any(e.get('predicted_local_ms') is None for e in (token,kv)):
        return dict(status='unavailable',reason='Two eligible calibrated costs (or explicit synthetic assumptions) required')
    delta_p = token['predicted_local_ms'] - kv['predicted_local_ms']
    delta_d = kv['predicted_wire_bytes'] - token['predicted_wire_bytes']
    source = 'synthetic_assumption' if any(e.get('cost_source')=='synthetic' for e in (token,kv)) else 'profile_prediction'
    result = dict(cost_source=source,kv_candidate=kv['candidate'],message_condition='current_handoff_only',
                  token_local_ms=token['predicted_local_ms'],kv_local_ms=kv['predicted_local_ms'],
                  token_bytes=token['predicted_wire_bytes'],kv_bytes=kv['predicted_wire_bytes'])
    if delta_p == 0 or delta_d == 0 or (delta_p > 0) != (delta_d > 0):
        winner = ('equal' if delta_p==0 and delta_d==0 else
                  'token' if delta_p<=0 and delta_d>=0 else kv['candidate'])
        return dict(result,status='no_positive_crossover',preferred=winner,
                    reason='No positive finite crossover for these eligible costs and bytes')
    rate = 8*delta_d/(delta_p*1000)  # bytes / milliseconds -> Mbps
    if not math.isfinite(rate) or rate <= 0:
        return dict(result,status='unavailable',reason='Crossover outside finite numeric range')
    result.update(status='positive_crossover',effective_rate_mbps=rate,
                  below='token' if delta_d>0 else kv['candidate'],above=kv['candidate'] if delta_d>0 else 'token')
    if channel:
        y = rate/(channel['efficiency']*channel['bandwidth_mhz'])*math.log(2)
        if y <= 0 or not math.isfinite(y):
            return dict(result,snr_threshold_status='outside_finite_numeric_range')
        # Stable log(expm1(y)); no exponential overflow for huge thresholds.
        snr = 10/math.log(10)*(y+math.log1p(-math.exp(-y)) if y>50 else math.log(math.expm1(y)))
        if math.isfinite(snr):
            result['snr_threshold_db'] = snr
            low = channel.get('snr_min_db')
            high = channel.get('snr_max_db')
            result['within_configured_clip_bounds'] = (low is None or low<=snr) and (high is None or snr<=high)
            result['within_mean_plus_minus_2sigma'] = abs(snr-channel['snr_mean_db']) <= 2*channel['snr_sigma_db']
    return result
