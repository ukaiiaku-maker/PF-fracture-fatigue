"""Default-off private residual/commitment race for near-complete companions.

Retains the existing action, threshold, ordinal and threshold RNG identity.
Hazard segments are supplied frozen or from a separately qualified subgrid
state provider. This module neither advances global time nor evolves emission.
It is not registered in the production driver or owner-transition machinery.
"""
from dataclasses import dataclass
import hashlib
import json
import math

from .conditional_branch_mark_v13 import BranchMark, MarkedCleavageResult, BOUNDARY

MODEL_ID='v13.inherited-residual-versus-commitment-arrest/1'


@dataclass(frozen=True)
class InheritedCompanionClock:
    candidate_id: str
    action: float
    threshold: float
    event_ordinal: int
    threshold_rng_identity: str

    def __post_init__(self):
        if (not self.candidate_id or not self.threshold_rng_identity or self.event_ordinal<1 or
                not all(math.isfinite(x) for x in (self.action,self.threshold)) or
                not 0<=self.action<=self.threshold):
            raise ValueError('invalid inherited clock')


@dataclass(frozen=True)
class ResidualRaceParameters:
    enabled: bool=False
    commitment_rate_per_s: float | None=None
    arrest_rate_per_s: float | None=None
    branch_seed: int=0

    def validate(self):
        if self.enabled:
            values=(self.commitment_rate_per_s,self.arrest_rate_per_s)
            if any(x is None or not math.isfinite(x) or x<0 for x in values) or max(values)<=0:
                raise ValueError('explicit finite commitment/arrest rates are required')


def residual_completion_time(clock, hazard_segments):
    remaining=clock.threshold-clock.action
    elapsed=0.
    for duration,rate in hazard_segments:
        if not all(math.isfinite(v) and v>=0 for v in (duration,rate)):
            raise ValueError('invalid residual hazard segment')
        if remaining==0:
            return elapsed
        if rate>0 and remaining/rate<=duration:
            return elapsed+remaining/rate
        remaining-=rate*duration
        elapsed+=duration
    return elapsed if remaining==0 else math.inf


def apply_inherited_residual_race(*,parent,baseline_single_state,clock,hazard_segments,parameters,
                                exact_pair_trial,observation_accepted_state_id=None,
                                observation_process_sha256=None,opportunity_ordinal=0):
    if parent is None or not parameters.enabled:
        return MarkedCleavageResult(baseline_single_state,BranchMark(None,None,()),'BRANCH_DISABLED')
    parameters.validate()
    if (clock.candidate_id==parent.primary_candidate_id or opportunity_ordinal<0 or
        parent.primary_event_id not in baseline_single_state.competition.consumed_event_ids):
        raise ValueError('inherited race requires a distinct companion and accepted primary')
    if (observation_accepted_state_id!=parent.accepted_state_id or
        observation_process_sha256!=parent.process_state_sha256):
        raise ValueError('stale inherited-clock observation')
    completion=residual_completion_time(clock,hazard_segments)
    key={'model_id':MODEL_ID,'seed':parameters.branch_seed,'lineage':parent.front_lineage,
        'event':parent.primary_event_id,'ordinal':parent.primary_event_ordinal,
        'companion':clock.candidate_id,'inherited_ordinal':clock.event_ordinal,
        'threshold_rng_identity':clock.threshold_rng_identity,'opportunity_ordinal':opportunity_ordinal}
    identities=[]
    stopping=[]
    for name,rate in (('commitment',parameters.commitment_rate_per_s),('arrest',parameters.arrest_rate_per_s)):
        identity=hashlib.sha256(json.dumps(dict(key,branch_only_process=name),sort_keys=True).encode()).hexdigest()
        u=((int(identity[:16],16)>>12)+1)/(2**52+1)
        stopping.append(-math.log(u)/rate if rate else math.inf)
        identities.append((name,identity))
    if not completion<min(stopping):
        return MarkedCleavageResult(baseline_single_state,BranchMark(None,None,tuple(identities)),
            'SINGLE_COMMITMENT_OR_EMBRYO_ARREST')
    mark=BranchMark(clock.candidate_id,completion,tuple(identities))
    result=exact_pair_trial(parent,mark)
    if not result.accepted:
        if not result.rejection_reason:
            raise RuntimeError('exact pair veto requires its reason')
        return MarkedCleavageResult(baseline_single_state,mark,'SINGLE_COMPANION_REJECTED',result.rejection_reason)
    for name in ('competition','rng_state','tip_process_state','event_counters'):
        if getattr(result.state,name) is not getattr(baseline_single_state,name):
            raise RuntimeError('inherited overlay changed canonical '+name)
    return MarkedCleavageResult(result.state,mark,'PAIR_ACCEPTED')
