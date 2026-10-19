"""Bounded joint workload interpolation, separate from exact SHA replay profiles.

No tensors, fitting framework or model forwards in online prediction. A complete
rectilinear cell is mandatory: sparse grids never become bounding-box evidence.
"""
from dataclasses import asdict, dataclass, fields
from functools import lru_cache
import hashlib
from importlib.metadata import version
from itertools import product
from pathlib import Path
import platform

from .incremental_kv_protocol import canonical_json
from .network.handoff_network import finite_number

SCHEMA = 'generalized_incremental_cost_v1'
AXES = ('message_tokens', 'receiver_history_tokens', 'sender_history_tokens')
CONDITIONS = ('termination_reason', 'sender_pending', 'receiver_pending',
              'sender_close_tokens', 'receiver_close_tokens', 'prefix_tokens',
              'closing_tokens', 'transfer_path', 'contract_ready', 'recomputed_tokens',
              'numeric_slot_index')
MEASUREMENT_SCOPE = 'runtime_instrumented_full_session_call_v1'


def digest(value):
    return hashlib.sha256(canonical_json(value)).hexdigest()


@lru_cache(maxsize=1)
def implementation_identity():
    root = Path(__file__).parent
    paths = ('stateful_token.py', 'incremental_kv.py', 'incremental_kv_receive.py',
             'incremental_kv_protocol.py', 'token_protocol.py', 'kv_protocol.py',
             'agent/state.py', 'agent/chat_framing.py', 'model/kv_adapter.py',
             'model/delta_repair.py', 'model/token_slot_repair.py', 'model/huggingface.py')
    return dict(algorithm_protocol_sha256=digest({p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}),
                torch=version('torch'), transformers=version('transformers'),
                measurement_scope=MEASUREMENT_SCOPE)


def environment_identity(backend):
    """Hardware adapter, software and model identity; never reuse MPS on CUDA."""
    from .incremental_auto import backend_cost_identity
    if not hasattr(backend, '_generalized_cost_environment'):
        device = backend.device
        torch = backend._torch
        if device.startswith('cuda'):
            props = torch.cuda.get_device_properties(device)
            hardware = dict(name=props.name, capability=list(torch.cuda.get_device_capability(device)),
                            total_memory=props.total_memory, cuda=torch.version.cuda)
        elif device == 'mps':
            # Mac model/SoC identity, no subprocess for each handoff.
            import subprocess
            hardware = dict(name=subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
                            machine=platform.machine(), os=platform.mac_ver()[0])
        else:
            hardware = dict(name=platform.processor(), machine=platform.machine(), os=platform.platform())
        backend._generalized_cost_environment = dict(model=backend_cost_identity(backend),
            hardware=hardware, implementation=implementation_identity())
    return backend._generalized_cost_environment


def workload_features(sender, receiver, message, candidate, *, contract_ready, recomputed_tokens):
    from .model.kv_adapter import _transfer_path
    from .model.token_slot_repair import first_numeric
    content = message.message_token_ids[:-1] if message.termination_reason == 'eos' else message.message_token_ids
    frame = receiver.framing.incoming(message.source_agent, message.target_agent)
    b = sender.backend
    structure = b.kv_model_config
    size = len(content)*2*structure.num_hidden_layers*structure.num_key_value_heads*structure.head_dim*{
        'bfloat16':2,'float16':2,'float32':4}[b.metadata.dtype.removeprefix('torch.')]
    def close_count(state):
        if not state.assistant_open:
            return 0
        closing = state.framing.assistant.closing_ids
        return 1 + len(closing) - int(state.pending_token_id == closing[0])
    return dict(message_tokens=len(content), receiver_history_tokens=receiver.cached_token_count,
        sender_history_tokens=message.sender_message_span[0], termination_reason=message.termination_reason,
        sender_pending=sender.pending_token_id is not None, receiver_pending=receiver.pending_token_id is not None,
        sender_close_tokens=close_count(sender), receiver_close_tokens=close_count(receiver),
        prefix_tokens=len(frame.prefix_ids), closing_tokens=len(frame.closing_ids),
        transfer_path=_transfer_path(None,b.device,size,dtype=b.metadata.dtype.removeprefix('torch.')),
        contract_ready=contract_ready, recomputed_tokens=recomputed_tokens,
        numeric_slot_index=first_numeric(content,b.decode) if candidate == 'kv:v3' else None)


def validate_features(features):
    if not isinstance(features, dict) or set(features) != set(AXES + CONDITIONS):
        raise ValueError('missing or unknown workload features')
    for key in (*AXES, 'sender_close_tokens', 'receiver_close_tokens', 'prefix_tokens', 'closing_tokens', 'recomputed_tokens'):
        if type(features[key]) is not int or features[key] < 0:
            raise ValueError('invalid workload count: '+key)
    for key in ('sender_pending','receiver_pending','contract_ready'):
        if type(features[key]) is not bool:
            raise ValueError('invalid workload condition: '+key)
    if features['termination_reason'] not in ('eos','max_new_tokens','context_limit') or features['transfer_path'] not in ('packed','legacy'):
        raise ValueError('invalid workload condition')
    slot = features['numeric_slot_index']
    if slot is not None and (type(slot) is not int or not 0 <= slot < features['message_tokens']):
        raise ValueError('invalid numeric slot')


@dataclass(frozen=True)
class GeneralizedCostProfile:
    version: str
    model_identity: dict
    environment: dict
    points: list[dict]
    sample_count: int
    calibrated_at: str
    evidence_sha256: str
    notes: str
    schema: str = SCHEMA
    provenance: str = 'measured'
    units: str = 'ms'
    reliability: str = 'experimental'
    relative_error_floor: float = 0.25
    startup_cost_ms: float | None = None
    startup_uncertainty_ms: float | None = None

    def __post_init__(self):
        if self.schema != SCHEMA or self.provenance != 'measured' or self.units != 'ms':
            raise ValueError('generalized profile must be measured ms with supported schema')
        if self.reliability not in ('experimental', 'holdout_validated'):
            raise ValueError('invalid reliability')
        if any(not isinstance(x,str) or not x.strip() for x in (self.version,self.calibrated_at,self.notes)):
            raise ValueError('profile needs version, calibration time and notes')
        from datetime import datetime
        if datetime.fromisoformat(self.calibrated_at).tzinfo is None:
            raise ValueError('calibration timestamp needs timezone')
        if not isinstance(self.model_identity,dict) or set(self.model_identity) != {'sender','receiver'}:
            raise ValueError('profile needs both model identities')
        if not isinstance(self.environment,dict) or set(self.environment) != {'sender','receiver'}:
            raise ValueError('profile needs both environments')
        if any(not isinstance(v,dict) or not all(k in v for k in ('model','hardware','implementation')) for v in self.environment.values()):
            raise ValueError('profile needs hardware and implementation identities')
        if any(self.environment[k]['model'] != self.model_identity[k] for k in ('sender','receiver')):
            raise ValueError('profile environment/model mismatch')
        if (self.startup_cost_ms is None) != (self.startup_uncertainty_ms is None):
            raise ValueError('startup cost and uncertainty must appear together')
        if self.startup_cost_ms is not None:
            finite_number(self.startup_cost_ms, 'startup cost', positive=True)
            finite_number(self.startup_uncertainty_ms, 'startup uncertainty')
        finite_number(self.relative_error_floor,'relative error floor',positive=True)
        if self.relative_error_floor < 0.1:
            raise ValueError('uncertainty floor must be at least 10 percent')
        if not isinstance(self.points,list) or not self.points or len(self.points)>8192:
            raise ValueError('profile requires bounded calibration points')
        count=0; seen=set()
        for p in self.points:
            if not isinstance(p,dict) or set(p) != {'candidate','features','local_ms','spread_ms','samples','wire_bytes'}:
                raise ValueError('invalid calibration point')
            if p['candidate'] not in ('token','kv:v0','kv:v1','kv:v2','kv:v3'):
                raise ValueError('invalid point strategy')
            validate_features(p['features'])
            if p['candidate'] != 'token' and p['features']['message_tokens'] == 0:
                raise ValueError('KV calibration requires nonempty content')
            if p['candidate']=='kv:v3' and p['features']['numeric_slot_index'] is None:
                raise ValueError('V3 point requires legal slot')
            finite_number(p['local_ms'],'local_ms',positive=True)
            finite_number(p['spread_ms'],'spread_ms')
            if type(p['samples']) is not int or p['samples']<1 or type(p['wire_bytes']) is not int or p['wire_bytes']<=0:
                raise ValueError('invalid point samples/bytes')
            key=(p['candidate'],digest(p['features']))
            if key in seen:
                raise ValueError('duplicate calibration point')
            seen.add(key);count+=p['samples']
        if type(self.sample_count) is not int or count != self.sample_count:
            raise ValueError('profile sample count mismatch')
        if digest(self.points) != self.evidence_sha256:
            raise ValueError('calibration points SHA mismatch')

    @classmethod
    def from_dict(cls,data):
        if not isinstance(data,dict) or set(data)-{f.name for f in fields(cls)}:
            raise ValueError('invalid generalized profile fields')
        try:
            return cls(**data)
        except (TypeError,KeyError,AttributeError) as exc:
            raise ValueError('malformed generalized profile') from exc

    def predict(self,candidate,features,identity,environment):
        validate_features(features)
        if identity != self.model_identity:
            raise ValueError('model_profile_mismatch')
        if environment != self.environment:
            raise ValueError('environment_profile_mismatch')
        startup = 0.0
        startup_uncertainty = 0.0
        original_features = features
        if not features['contract_ready']:
            if candidate != 'token':
                if self.startup_cost_ms is None:
                    raise ValueError('missing_startup_cost_profile')
                startup = self.startup_cost_ms
                startup_uncertainty = self.startup_uncertainty_ms
            features = {**features, 'contract_ready':True}
        rows=[p for p in self.points if p['candidate']==candidate and all(p['features'][k]==features[k] for k in CONDITIONS)]
        if not rows:
            raise ValueError('profile_condition_unverified')
        brackets=[]
        for key in AXES:
            values=sorted({p['features'][key] for p in rows}); value=features[key]
            lo=[x for x in values if x<=value]; hi=[x for x in values if x>=value]
            if not lo or not hi:
                raise ValueError('profile_out_of_domain')
            brackets.append((max(lo),min(hi)))
        lookup={tuple(p['features'][k] for k in AXES):p for p in rows}
        weighted=[]
        for corner in product(*(sorted(set(b)) for b in brackets)):
            if corner not in lookup:
                raise ValueError('profile_sparse_cell')
            weight=1.0
            for key,x,(lo,hi) in zip(AXES,corner,brackets):
                if lo!=hi:
                    weight *= (features[key]-lo)/(hi-lo) if x==hi else (hi-features[key])/(hi-lo)
            weighted.append((weight,lookup[corner]))
        local=sum(w*p['local_ms'] for w,p in weighted)
        # Empirical envelope, NOT a statistical confidence interval. Interior
        # distance adds conservatism; no negative/monotonicity assumptions.
        distance=sum(4*(features[k]-lo)*(hi-features[k])/(hi-lo)**2 if hi!=lo else 0
                     for k,(lo,hi) in zip(AXES,brackets))/len(AXES)
        variation=sum(w*p['spread_ms'] for w,p in weighted)
        uncertainty=max(variation,local*self.relative_error_floor)*(1+distance)+startup_uncertainty
        local += startup
        finite_number(local,'predicted local',positive=True)
        finite_number(uncertainty,'prediction uncertainty')
        return dict(predicted_local_ms=local,uncertainty_ms=uncertainty,
            cost_source='predicted',calibration_source='measured',
            prediction_method='calibrated_point' if len(weighted)==1 else 'bounded_joint_multilinear',
            reliability=self.reliability, training_sample_count=self.sample_count,
            support_points=len(weighted),normalized_interior_distance=distance,
            predicted_stage_ms={'full_ready_handoff':local-startup, 'contract_startup':startup},workload_features=original_features)


def load_cost_profile(data):
    if not isinstance(data, dict):
        raise ValueError('cost profile must be a JSON object')
    if data.get('schema') == SCHEMA:
        return GeneralizedCostProfile.from_dict(data)
    from .incremental_auto import IncrementalCostProfile
    return IncrementalCostProfile.from_dict(data)


def idealized_prediction(features, assumptions=None):
    """Synthetic local compute only; real timings and wire costs stay untouched."""
    from .synthetic_cost import SyntheticCostAssumptions
    return (assumptions or SyntheticCostAssumptions()).predict(features)
