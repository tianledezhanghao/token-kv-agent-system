"""Online incremental selector. No Canonical profile or candidate execution."""
from __future__ import annotations

from .synthetic_cost import SyntheticCostAssumptions

from typing import TYPE_CHECKING
from dataclasses import asdict, dataclass, fields
import hashlib
import json

from .communication import HandoffChoice, suffix_recompute_count, validate_recompute_policy
from .network.handoff_network import finite_number
from .incremental_kv_protocol import IncrementalKVMessage, ModelContract, canonical_json, DIGEST_BYTES
from .kv_protocol import KVModelCompatibility, estimate_kv_wire_size

if TYPE_CHECKING:
    from .generalized_cost import GeneralizedCostProfile

CANDIDATES = ('token', 'kv:v0', 'kv:v1', 'kv:v2', 'kv:v3')
# Stable tie-break: TOKEN, then least invasive KV; independent of caller order.
STAGES = {
    'token': ('token_handoff',),
    'kv:v0': ('kv_base',),
    'kv:v1': ('kv_base', 'rope'),
    'kv:v2': ('kv_base', 'rope', 'suffix'),
    'kv:v3': ('kv_base', 'rope', 'selection', 'slot'),
}


def workload_identity(sender, receiver, message):
    """Conservative exact workload guard; metadata only, no forward/KV export.

    This is deliberately a replay domain, not a claim of content generalization.
    Loaded model identity is checked separately. Callers must not mutate models
    or caches outside the Session ownership contract.
    """
    digest = lambda value: hashlib.sha256(canonical_json(value)).hexdigest()
    frame = receiver.framing.incoming(message.source_agent, message.target_agent)
    return dict(sender_cache_tokens=sender.cached_token_count,
        receiver_cache_tokens=receiver.cached_token_count,
        sender_span_start=message.sender_message_span[0],
        termination_reason=message.termination_reason,
        sender_pending_token_id=sender.pending_token_id,
        receiver_pending_token_id=receiver.pending_token_id,
        sender_ledger_sha256=digest(sender.token_ledger),
        receiver_ledger_sha256=(receiver.selection_ledger_sha256 if hasattr(receiver, 'selection_ledger_sha256') else digest(receiver.token_ledger)),
        message_sha256=digest(message.message_token_ids),
        framing_sha256=(receiver.selection_framing_sha256 if hasattr(receiver, 'selection_framing_sha256') else digest(asdict(frame))))


def validate_workload_domain(domain):
    integers = ('sender_cache_tokens', 'receiver_cache_tokens', 'sender_span_start')
    pending = ('sender_pending_token_id', 'receiver_pending_token_id')
    hashes = ('sender_ledger_sha256', 'receiver_ledger_sha256', 'message_sha256', 'framing_sha256')
    if (not isinstance(domain, dict) or set(domain) != set((*integers, *pending, *hashes, 'termination_reason'))
            or any(type(domain[k]) is not int or domain[k] < 0 for k in integers)
            or any(domain[k] is not None and (type(domain[k]) is not int or domain[k] < 0) for k in pending)
            or domain['termination_reason'] not in ('eos', 'max_new_tokens', 'context_limit')
            or any(not isinstance(domain[k], str) or len(domain[k]) != 64 or
                   any(c not in '0123456789abcdef' for c in domain[k]) for k in hashes)):
        raise ValueError('invalid exact workload domain')


@dataclass(frozen=True)
class StageCost:
    fixed_ms: float
    per_message_token_ms: float = 0.0
    per_cache_token_ms: float = 0.0
    per_recomputed_token_ms: float = 0.0

    def __post_init__(self):
        for name, value in asdict(self).items():
            finite_number(value, name)

    def predict(self, message_tokens, cache_tokens, recomputed_tokens):
        return (self.fixed_ms + self.per_message_token_ms * message_tokens
                + self.per_cache_token_ms * cache_tokens
                + self.per_recomputed_token_ms * recomputed_tokens)


@dataclass(frozen=True)
class IncrementalCostProfile:
    version: str
    provenance: str  # measured_incremental or synthetic/test_only
    model_identity: dict
    stages: dict[str, StageCost]
    sample_count: int
    message_token_range: tuple[int, int]
    cache_token_range: tuple[int, int]
    notes: str
    semantics: str = 'decode_incremental_handoff_v1'
    recompute_token_range: tuple[int, int] = (0, 4096)
    workload_domain: dict | None = None

    def __post_init__(self):
        if self.workload_domain is not None:
            validate_workload_domain(self.workload_domain)
        if self.semantics != 'decode_incremental_handoff_v1':
            raise ValueError('Canonical profiles cannot estimate incremental handoffs')
        if self.provenance not in ('measured_incremental', 'synthetic/test_only'):
            raise ValueError('invalid incremental profile provenance')
        if not isinstance(self.version, str) or not self.version or not isinstance(self.notes, str) or not self.notes or not isinstance(self.model_identity, dict) or not self.model_identity:
            raise ValueError('profile requires version, identity and uncertainty notes')
        if type(self.sample_count) is not int or self.sample_count < 0 or (
                self.provenance == 'measured_incremental' and self.sample_count < 1):
            raise ValueError('measured profile requires samples')
        for bounds in (self.message_token_range, self.cache_token_range, self.recompute_token_range):
            if len(bounds) != 2 or any(type(x) is not int or x < 0 for x in bounds) or bounds[0] > bounds[1]:
                raise ValueError('invalid profile domain')
        if any(k not in set(sum(STAGES.values(), ())) | {'kv_contract_startup'} or not isinstance(v, StageCost) for k, v in self.stages.items()):
            raise ValueError('unsupported profile stages')

    @classmethod
    def from_dict(cls, data):
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError('invalid incremental profile fields')
        try:
            return cls(**{**data, 'stages': {k: StageCost(**v) for k, v in data['stages'].items()},
                'message_token_range': tuple(data['message_token_range']),
                'cache_token_range': tuple(data['cache_token_range']),
                'recompute_token_range': tuple(data.get('recompute_token_range', (0,4096)))})
        except (KeyError, TypeError, AttributeError) as exc:
            raise ValueError('malformed incremental profile') from exc


def backend_cost_identity(backend):
    """No weights read. Loaded revision/config + tokenizer identity + placement.

    Fingerprints cached once; profile never reused across different endpoints.
    """
    if not hasattr(backend, '_incremental_cost_identity'):
        from .model.backend_pair import inspect_backend_identity
        identity = asdict(inspect_backend_identity(backend))
        backend._incremental_cost_identity = {
            'backend_identity_sha256': hashlib.sha256(canonical_json(identity)).hexdigest(),
            'model_id': backend.model_name,
            'revision': getattr(backend._model.config, '_commit_hash', None),
            'config_sha256': hashlib.sha256(canonical_json(backend._model.config.to_dict())).hexdigest(),
            'dtype': backend.metadata.dtype, 'device': backend.device,
        }
    return dict(backend._incremental_cost_identity)


def estimate_incremental_wire_bytes(state, message):
    """Exact dense batch1 TKVI/TKVK length; no KV extraction/copy/encoding.

    Hash fields have fixed 64-character lengths. The neutral span source_device
    is cpu by the existing adapter contract, irrespective of model placement.
    """
    b = state.backend
    structure = b.kv_model_config
    content = message.message_token_ids[:-1] if message.termination_reason == 'eos' else message.message_token_ids
    if not content:
        raise ValueError('empty KV content')
    dtype = b.metadata.dtype.removeprefix('torch.')
    contract = ModelContract(b.model_name, '0'*64, '0'*64, '0'*64, b.vocab_size,
        structure.num_hidden_layers, structure.num_key_value_heads, structure.head_dim,
        dtype, tuple(b.eos_token_ids), tuple(sorted(set(b.chat_tokenizer.all_special_ids))))
    envelope = IncrementalKVMessage(message.message_id, message.sequence, message.source_agent,
        message.target_agent, content, len(content), message.termination_reason,
        message.message_token_ids[-1] if message.termination_reason == 'eos' else None,
        message.sender_message_span[0], len(content), contract, None)
    kv = estimate_kv_wire_size(len(content), KVModelCompatibility(b.model_name,
        structure.num_hidden_layers, structure.num_key_value_heads, structure.head_dim),
        dtype=dtype, source_device='cpu')
    return 17 + DIGEST_BYTES + len(canonical_json(envelope.metadata())) + 10 + 4*len(content) + kv.total_wire_bytes


@dataclass(frozen=True)
class IncrementalAutoConfig:
    policy: str = 'quality_first'
    candidates: tuple[str, ...] = ('token', 'kv:v0')
    allow_experimental_v3: bool = False
    profile: IncrementalCostProfile | GeneralizedCostProfile | None = None
    recompute_tokens: int | None = None
    recompute_ratio: float | None = None
    idealized: bool = False
    synthetic_assumptions: SyntheticCostAssumptions | None = None

    @property
    def selected_kv(self):
        return next(c for c in self.candidates if c != 'token')

    def __post_init__(self):
        from .generalized_cost import GeneralizedCostProfile
        if self.policy not in ('quality_first', 'performance_experiment'):
            raise ValueError('invalid auto policy')
        if not isinstance(self.candidates, tuple) or not self.candidates or len(set(self.candidates)) != len(self.candidates):
            raise ValueError('nonempty unique candidate whitelist required')
        if any(c not in CANDIDATES for c in self.candidates):
            raise ValueError('unsupported_strategy')
        kv = [c for c in self.candidates if c != 'token']
        if len(kv) != 1 or len(self.candidates) != 2 or 'token' not in self.candidates:
            raise ValueError('AUTO requires TOKEN and exactly one selected KV strategy')
        object.__setattr__(self, 'candidates', ('token', kv[0]))
        if type(self.idealized) is not bool:
            raise ValueError('idealized must be boolean')
        if self.idealized and (self.policy != 'performance_experiment' or self.profile is not None):
            raise ValueError('idealized requires explicit performance_experiment and no measured/test profile')
        if self.synthetic_assumptions is not None and (not self.idealized or not isinstance(self.synthetic_assumptions, SyntheticCostAssumptions)):
            raise ValueError('Synthetic assumptions require idealized mode')
        if self.idealized and self.synthetic_assumptions is None:
            object.__setattr__(self, 'synthetic_assumptions', SyntheticCostAssumptions())
        if type(self.allow_experimental_v3) is not bool:
            raise ValueError('experimental V3 permission must be boolean')
        if self.profile is not None and not isinstance(self.profile, (IncrementalCostProfile, GeneralizedCostProfile)):
            raise ValueError('incremental cost profile required')
        validate_recompute_policy(self.recompute_tokens, self.recompute_ratio)
        if self.selected_kv == 'kv:v2' and self.recompute_tokens is None and self.recompute_ratio is None:
            object.__setattr__(self, 'recompute_tokens', 1)
        if (self.recompute_tokens is not None or self.recompute_ratio is not None) and 'kv:v2' not in self.candidates:
            raise ValueError('recompute configuration requires kv:v2 candidate')
        if 'kv:v3' in self.candidates and not self.allow_experimental_v3 and self.policy == 'performance_experiment':
            raise ValueError('V3 requires explicit experimental permission')


class IncrementalSelector:
    def __init__(self, config):
        self.config = config

    def select(self, sender, receiver, message, network, *, kv_contract_ready=False,
               require_workload_domain=False, wire_overhead=None):
        # Production Runtime requires the complete domain. The historical
        # metadata-only offline analysis API keeps its old reproducible meaning.
        from .generalized_cost import (GeneralizedCostProfile, environment_identity,
            workload_features, idealized_prediction)
        c = self.config
        n = message.token_count - int(message.termination_reason == 'eos')
        cache_count = sender.cached_token_count + receiver.cached_token_count
        estimates, exclusions = [], []
        profile = c.profile
        generalized = isinstance(profile, GeneralizedCostProfile)
        identity = None
        environment = None
        if profile is not None:
            identity = {'sender': backend_cost_identity(sender.backend), 'receiver': backend_cost_identity(receiver.backend)}
        if generalized:
            environment = {'sender': environment_identity(sender.backend), 'receiver': environment_identity(receiver.backend)}
        workload_reason = None
        if profile and not generalized and profile.provenance == 'measured_incremental':
            if profile.workload_domain is None and require_workload_domain:
                workload_reason = 'profile_workload_domain_unverified'
            elif profile.workload_domain is not None and workload_identity(sender, receiver, message) != profile.workload_domain:
                workload_reason = 'profile_out_of_domain'
        kv_bytes = None
        for candidate in CANDIDATES:
            if candidate not in c.candidates:
                exclusions.append(dict(candidate=candidate, reason='not_whitelisted'))
                continue
            reason = None
            recomputed = suffix_recompute_count(n, tokens=c.recompute_tokens, ratio=c.recompute_ratio) if candidate == 'kv:v2' and n else int(candidate == 'kv:v3')
            if c.policy == 'quality_first' and candidate != 'token':
                reason = 'known_quality_regression' if candidate == 'kv:v3' else 'quality_not_certified'
            elif profile is None and not c.idealized and candidate != 'token':
                reason = 'missing_cost_profile'
            elif profile and identity != profile.model_identity:
                reason = 'model_profile_mismatch'
            elif profile and not generalized and not (profile.message_token_range[0] <= n <= profile.message_token_range[1]
                and profile.cache_token_range[0] <= cache_count <= profile.cache_token_range[1]):
                reason = 'profile_out_of_domain'
            elif profile and not generalized and not profile.recompute_token_range[0] <= recomputed <= profile.recompute_token_range[1]:
                reason = 'profile_recompute_out_of_domain'
            elif workload_reason:
                reason = workload_reason
            elif profile and not generalized and candidate != 'token' and not kv_contract_ready and 'kv_contract_startup' not in profile.stages:
                reason = 'missing_startup_cost_profile'
            elif profile and not generalized and any(s not in profile.stages for s in STAGES[candidate]):
                reason = 'missing_cost_profile'
            if candidate != 'token' and n == 0:
                reason = reason or 'unsupported_empty_kv'
            prediction = {}
            if reason is None and candidate == 'kv:v3' and hasattr(message, 'message_token_ids'):
                from .model.token_slot_repair import first_numeric
                content = message.message_token_ids[:n]
                if first_numeric(content, sender.backend.decode) is None:
                    reason = 'v3_no_numeric_slot'
            if reason is None and (generalized or c.idealized):
                try:
                    features = workload_features(sender, receiver, message, candidate,
                        contract_ready=kv_contract_ready, recomputed_tokens=recomputed)
                    if generalized:
                        prediction = profile.predict(candidate, features, identity, environment)
                    else:
                        token_cost, kv_cost = idealized_prediction(features, c.synthetic_assumptions)
                        local_cost = token_cost if candidate == 'token' else kv_cost
                        prediction = dict(predicted_local_ms=local_cost, uncertainty_ms=None,
                            predicted_stage_ms={'assumed_full_handoff':local_cost}, cost_source='synthetic',
                            prediction_method='idealized_reuse_formula_v1', reliability='assumption_only',
                            workload_features=features)
                except ValueError as exc:
                    reason = str(exc)
            uncalibrated_domain = reason in ('profile_out_of_domain', 'profile_workload_domain_unverified',
                'profile_condition_unverified', 'profile_sparse_cell') and candidate == 'token' and c.policy == 'quality_first'
            domain_status = reason
            if uncalibrated_domain:
                reason = None  # Quality reference stays safe; no short-message extrapolation.
            if reason:
                exclusions.append(dict(candidate=candidate, reason=reason))
                continue
            try:
                if candidate != 'token' and kv_bytes is None:
                    kv_bytes = estimate_incremental_wire_bytes(sender, message)
            except (ValueError, AttributeError, TypeError) as exc:
                exclusions.append(dict(candidate=candidate, reason='unsupported_strategy', error=str(exc)))
                continue
            wire = 10 + 4*message.token_count if candidate == 'token' else kv_bytes
            if wire_overhead is not None:
                extra = wire_overhead(candidate)
                if type(extra) is not int or extra < 0:
                    raise ValueError('application wire overhead must be a nonnegative integer')
                wire += extra
            if candidate != 'token':
                from .incremental_kv_protocol import IncrementalLimits
                limits = IncrementalLimits()
                if (n > limits.max_message_tokens or wire > limits.max_packet_bytes or
                        getattr(message, 'sender_message_span', (0, 0))[0] + n > limits.max_sender_positions):
                    exclusions.append(dict(candidate=candidate, reason='protocol_capacity'))
                    continue
            stages = prediction.get('predicted_stage_ms', {}) if generalized or c.idealized else ({s: profile.stages[s].predict(n, cache_count, recomputed) for s in STAGES[candidate]} if profile and not uncalibrated_domain else {})
            if profile and not generalized and candidate != 'token' and not kv_contract_ready:
                stages['kv_contract_startup'] = profile.stages['kv_contract_startup'].predict(n, cache_count, recomputed)
            for value in stages.values():
                finite_number(value, 'predicted stage cost')
            local = sum(stages.values()) if (profile or c.idealized) and not uncalibrated_domain else None
            if local is not None:
                finite_number(local, 'predicted local cost')
            if local is not None and local <= 0 and not c.idealized:
                exclusions.append(dict(candidate=candidate, reason='missing_cost_profile'))
                continue
            if not prediction:
                prediction = dict(cost_source='synthetic' if profile and profile.provenance == 'synthetic/test_only' else 'predicted' if local is not None else 'unavailable',
                    calibration_source='measured' if profile and profile.provenance == 'measured_incremental' else None)
            transfer = network.transfer_ms(wire)
            if local is not None:
                finite_number(local + transfer, 'predicted total cost')
            estimates.append(dict(**{k:v for k,v in prediction.items() if k not in ('predicted_local_ms','predicted_stage_ms')}, candidate=candidate, predicted_wire_bytes=wire,
                wire_estimate_method='protocol_exact_dense_batch1_plus_application_envelope' if wire_overhead else 'protocol_exact_dense_batch1', predicted_stage_ms=stages,
                predicted_local_ms=local, simulated_transfer_ms=transfer,
                predicted_total_ms=local + transfer if local is not None else None,
                recomputed_tokens=recomputed, cost_status=domain_status if uncalibrated_domain else 'synthetic_assumption' if c.idealized else 'estimated' if profile else 'uncalibrated'))
        if profile and c.policy == 'performance_experiment' and estimates and not any(e['candidate']=='token' for e in estimates):
            exclusions.extend(dict(candidate=e['candidate'], reason='token_reference_unavailable') for e in estimates)
            estimates = []  # A predicted KV cost cannot replace a missing TOKEN baseline.
        if not estimates:
            raise ValueError('no eligible AUTO candidates: ' + json.dumps(exclusions))
        ranked = sorted(estimates, key=lambda e: (e['predicted_total_ms'] if e['predicted_total_ms'] is not None else float('inf'), CANDIDATES.index(e['candidate'])))
        selected = ranked[0]
        selection_reason = 'minimum_predicted_pair_total_prefer_token_on_tie' if selected['predicted_total_ms'] is not None else 'sole_safe_TOKEN_uncalibrated'
        token_estimate = next((e for e in estimates if e['candidate']=='token'), None)
        if selected['candidate'] != 'token' and token_estimate and generalized:
            gap = token_estimate['predicted_total_ms'] - selected['predicted_total_ms']
            if gap <= token_estimate.get('uncertainty_ms',0) + selected.get('uncertainty_ms',0):
                selected = token_estimate
                selection_reason = 'uncertainty_overlap_prefer_token'
        runner = next((e for e in ranked if e is not selected), None)
        from .network.shannon import crossover_analysis
        decision = dict(crossover=crossover_analysis(estimates, getattr(network,'channel',None)),
            synthetic_assumptions=c.synthetic_assumptions.to_dict() if c.idealized else None,
            auto_policy=c.policy, network=asdict(network), candidate_set=list(c.candidates),
            candidate_estimates=estimates, excluded_candidates=exclusions, selected_candidate=selected['candidate'],
            selection_reason=selection_reason, selected_kv=c.selected_kv, idealized=c.idealized,
            runner_up=runner['candidate'] if runner else None,
            cost_gap_ms=runner['predicted_total_ms'] - selected['predicted_total_ms'] if runner else None,
            prediction_config_version=profile.version if profile else 'idealized-reuse-v1' if c.idealized else 'safe-token-no-profile-v1',
            profile_evidence_sha256=getattr(profile, 'evidence_sha256', None),
            profile_calibrated_at=getattr(profile, 'calibrated_at', None),
            profile_sample_count=profile.sample_count if profile else None,
            profile_provenance=profile.provenance if profile else 'synthetic' if c.idealized else 'not_measured',
            profile_notes=profile.notes if profile else 'Synthetic compute assumptions: TOKEN=a+bN+cNH ms; KV=d+eN+fk ms. See frozen synthetic_assumptions and workload_features; not measured.' if c.idealized else 'No local cost calibration; no latency comparison claimed.',
            quality_risk=c.policy == 'performance_experiment',
            v3_semantic_risk=selected['candidate'] == 'kv:v3', decision_status='selected', **selected)
        return HandoffChoice.parse(selected['candidate'], **(dict(recompute_tokens=c.recompute_tokens,
            recompute_ratio=c.recompute_ratio) if selected['candidate']=='kv:v2' else {})), decision
