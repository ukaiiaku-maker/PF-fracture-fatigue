"""Default-off conditional pair transition, not a second parent cleavage law.

All kinetic scales are explicit and uncalibrated. No automatic parameter
selection, production registration, clock update, or continuum energy-to-
activation-work conversion is provided. Embryo arrest retains the accepted
canonical single arm; it does not undo parent fracture.
"""
from dataclasses import dataclass
import hashlib
import json
import math

from .conditional_branch_mark_v13 import BOUNDARY, BranchMark, MarkedCleavageResult

MODEL_ID = "v13.cooperative-pair-versus-primary-commitment/1"
KB = 1.380649e-23


@dataclass(frozen=True)
class CooperativePairParameters:
    enabled: bool = False
    combined_pair_barrier_J: float | None = None
    pair_attempt_rate_per_s: float | None = None
    primary_commitment_rate_per_s: float | None = None
    embryo_arrest_rate_per_s: float | None = None
    branch_seed: int = 0

    def validate(self):
        if not self.enabled:
            return self
        for name in ("combined_pair_barrier_J", "pair_attempt_rate_per_s",
                     "primary_commitment_rate_per_s", "embryo_arrest_rate_per_s"):
            value = getattr(self, name)
            if value is None or not math.isfinite(value) or value < 0:
                raise ValueError(f"enabled pair transition requires explicit nonnegative finite {name}")
        if max(self.primary_commitment_rate_per_s,self.embryo_arrest_rate_per_s) <= 0:
            raise ValueError("commitment/arrest competition needs a positive total rate")
        return self


def frozen_pair_probability(parameters, *, temperature_K, rate_balance):
    parameters.validate()
    if not parameters.enabled:
        return {"P_pair_embryo":0., "pair_rate_per_s":0., "enabled":False}
    if not math.isfinite(temperature_K) or temperature_K <= 0 or not 0 <= rate_balance <= 1:
        raise ValueError("invalid temperature or co-criticality balance")
    k = parameters.pair_attempt_rate_per_s*rate_balance*math.exp(-parameters.combined_pair_barrier_J/(KB*temperature_K))
    scale=max(k,parameters.primary_commitment_rate_per_s,parameters.embryo_arrest_rate_per_s)
    probability=(k/scale)/(k/scale+parameters.primary_commitment_rate_per_s/scale+parameters.embryo_arrest_rate_per_s/scale)
    return {"P_pair_embryo":probability, "pair_rate_per_s":k, "enabled":True,
        "boundary":BOUNDARY, "model_id":MODEL_ID,
        "barrier_semantics":"one combined pair activation barrier; no independent junction/overlap fit"}


def apply_cooperative_pair_transition(*, parent, baseline_single_state, companion_id,
                                      parameters, temperature_K, rate_balance,
                                      exact_pair_trial, opportunity_ordinal=0):
    if parent is None or not parameters.enabled:
        return MarkedCleavageResult(baseline_single_state,BranchMark(None,None,()),
            "BRANCH_DISABLED" if not parameters.enabled else "NO_CLEAVAGE_EVENT")
    if companion_id == parent.primary_candidate_id or opportunity_ordinal < 0:
        raise ValueError("invalid companion/opportunity identity")
    probability=frozen_pair_probability(parameters,temperature_K=temperature_K,rate_balance=rate_balance)
    key={"model_id":MODEL_ID,"seed":parameters.branch_seed,"lineage":parent.front_lineage,
        "event_id":parent.primary_event_id,"ordinal":parent.primary_event_ordinal,
        "companion_id":companion_id,"opportunity_ordinal":opportunity_ordinal}
    identity=hashlib.sha256(json.dumps(key,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    u=((int(identity[:16],16)>>12)+1)/(2**52+1)
    if u >= probability["P_pair_embryo"]:
        return MarkedCleavageResult(baseline_single_state,BranchMark(None,None,((companion_id,identity),)),
            "SINGLE_COMMITMENT_OR_EMBRYO_ARREST")
    mark=BranchMark(companion_id,None,((companion_id,identity),))
    result=exact_pair_trial(parent,mark)
    if not result.accepted:
        if not result.rejection_reason:
            raise RuntimeError("pair veto must report an exact reason")
        return MarkedCleavageResult(baseline_single_state,mark,"SINGLE_COMPANION_REJECTED",result.rejection_reason)
    for name in ("competition","rng_state","tip_process_state","event_counters"):
        if getattr(result.state,name) is not getattr(baseline_single_state,name):
            raise RuntimeError(f"cooperative pair changed canonical parent {name}")
    return MarkedCleavageResult(result.state,mark,"PAIR_ACCEPTED")
