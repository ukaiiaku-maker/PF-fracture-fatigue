#!/usr/bin/env python3
"""Mechanism-specific, material-frame developed-fatigue campaign for V3.6."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from arrhenius_fracture.mechanism_specific_feedback_v10230 import MechanismSpecificFeedback
from arrhenius_fracture.operator_consistent_projection_v10230 import MaterialSlipPatch, OperatorConsistentProjection
from arrhenius_fracture.swept_slip_ledger_v10230 import SweptSlipLedger
from scripts import select_v2_3_exact_row_cycle_hazard_loads_v10230 as exact_row
from scripts.v3_3_support_v10230 import load_v3_1_candidate

LOADS = (4.781762498950186, 8.598738351065925, 15.462474735549586)
CHECKPOINT_SCHEMA = "v10.2.30_v3_6_material_frame_checkpoint_v2"


class MaterialHistory:
    def __init__(self, n_bins: int):
        self.archived: list[MaterialSlipPatch] = []
        self.active_anchor_m = 0.0
        self.active_id = 0
        self.ledger = SweptSlipLedger.zeros(2, n_bins)

    def active_gamma(self):
        return tuple(np.mean(self.ledger.plastic_shear, axis=1).tolist())

    def patches(self):
        return self.archived + [MaterialSlipPatch(self.active_id, self.active_anchor_m, self.active_gamma())]

    def commit_advance(self, old_tip_m: float, new_tip_m: float):
        before = float(sum(np.sum(abs(np.asarray(x.gamma))) for x in self.archived) + np.sum(abs(np.asarray(self.active_gamma()))))
        self.archived.append(MaterialSlipPatch(self.active_id, self.active_anchor_m, self.active_gamma()))
        self.active_id += 1; self.active_anchor_m = float(new_tip_m)
        self.ledger = SweptSlipLedger.zeros(2, self.ledger.plastic_shear.shape[1])
        after = float(sum(np.sum(abs(np.asarray(x.gamma))) for x in self.archived))
        return {"material_history_abs_gamma_before": before, "material_history_abs_gamma_after": after,
                "coordinate_motion_plastic_increment": after - before,
                "old_tip_m": float(old_tip_m), "new_tip_m": float(new_tip_m)}


def _history_to_checkpoint(history):
    return {
        "material_patches": [x.__dict__ for x in history.archived],
        "active_anchor_m": history.active_anchor_m,
        "active_id": history.active_id,
        "active_ledger": {
            "plastic_shear": history.ledger.plastic_shear.tolist(),
            "cumulative_abs_swept_area_per_volume":
                history.ledger.cumulative_abs_swept_area_per_volume.tolist(),
            "nucleation_patch_shear": history.ledger.nucleation_patch_shear.tolist(),
        },
    }


def _history_from_checkpoint(physics, saved):
    history = MaterialHistory(physics.n_bins)
    history.archived = [MaterialSlipPatch(
        int(row["patch_id"]), float(row["material_anchor_m"]), tuple(row["gamma"])
    ) for row in saved["material_patches"]]
    history.active_anchor_m = float(saved["active_anchor_m"])
    history.active_id = int(saved["active_id"])
    history.ledger.plastic_shear = np.asarray(
        saved["active_ledger"]["plastic_shear"], float)
    history.ledger.cumulative_abs_swept_area_per_volume = np.asarray(
        saved["active_ledger"]["cumulative_abs_swept_area_per_volume"], float)
    history.ledger.nucleation_patch_shear = np.asarray(
        saved["active_ledger"]["nucleation_patch_shear"], float)
    return history


def _recent_to_checkpoint(recent):
    return [{
        "action": float(action),
        "state_delta": {key: value.tolist() for key, value in state_delta.items()},
        "ledger_delta": [value.tolist() for value in ledger_delta],
    } for action, state_delta, ledger_delta in recent]


def _recent_from_checkpoint(saved):
    return [(float(row["action"]),
             {key: np.asarray(value, float) for key, value in row["state_delta"].items()},
             tuple(np.asarray(value, float) for value in row["ledger_delta"]))
            for row in saved]


def _checkpoint_digest(payload):
    return hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=True,
    ).encode()).hexdigest()


def _case_identity(candidate, physics, projection, Kmax, seed, phase_steps):
    return {
        "Kmax_MPa_sqrt_m": float(Kmax), "seed": int(seed),
        "phase_steps": int(phase_steps), "candidate_id": candidate.candidate_id,
        "candidate_physics_sha256": hashlib.sha256(
            repr((candidate, physics)).encode()).hexdigest(),
        "projection_nodes_sha256": hashlib.sha256(
            np.ascontiguousarray(projection.mesh.nodes).tobytes()).hexdigest(),
        "projection_elements_sha256": hashlib.sha256(
            np.ascontiguousarray(projection.mesh.elems).tobytes()).hexdigest(),
    }


def _write_checkpoint(directory, payload):
    directory.mkdir(parents=True, exist_ok=True)
    complete = dict(payload)
    complete["integrity_sha256"] = _checkpoint_digest(payload)
    identity = hashlib.sha256(json.dumps({
        "K": payload["case"]["Kmax_MPa_sqrt_m"],
        "phase_intervals": payload["phase_intervals"],
        "state": complete["integrity_sha256"],
    }, sort_keys=True).encode()).hexdigest()[:20]
    generation = directory / f"v3_6_generation_{identity}.json"
    if not generation.exists():
        generation.write_text(json.dumps(complete, indent=2, sort_keys=True)+"\n")
    manifest = {"schema": CHECKPOINT_SCHEMA, "generation": generation.name,
                "generation_sha256": hashlib.sha256(generation.read_bytes()).hexdigest()}
    with tempfile.NamedTemporaryFile("w", dir=directory, delete=False) as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True); stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, directory/"v3_6_restart.json")
    return directory/"v3_6_restart.json"


def _load_checkpoint(path):
    source = Path(path)
    manifest_path = source/"v3_6_restart.json" if source.is_dir() else source
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("incompatible V3.6 restart checkpoint schema")
    generation = manifest_path.parent/manifest["generation"]
    if hashlib.sha256(generation.read_bytes()).hexdigest() != manifest["generation_sha256"]:
        raise RuntimeError("V3.6 restart checkpoint generation hash mismatch")
    payload = json.loads(generation.read_text())
    integrity = payload.pop("integrity_sha256", None)
    if integrity != _checkpoint_digest(payload):
        raise RuntimeError("V3.6 restart checkpoint integrity mismatch")
    return payload


def strip_adapter(state):
    for name in ("local_rates", "cleavage_rate_s", "_substep", "tip_radius_m", "K_shield_MPa_sqrt_m"):
        state.__dict__.pop(name, None)


def install(state, projection, history):
    adapter = MechanismSpecificFeedback(projection, history.patches, lambda: state.extension_m)
    adapter.install(state)
    return adapter


def gamma_rate(state, rates):
    return float(state.c.b_m) * np.sum(np.maximum(state.mobile_m2, 0.0), axis=1) * np.asarray(rates["velocity_m_s"])


def advance_basic(state, history, projection, load, p0, p1):
    """Advance with native accepted kinetics and converged trapezoid flux."""
    duration = (p1 - p0) * load.period_s; elapsed = 0.0; action = 0.0; steps = 0
    while elapsed < duration - 1e-18:
        f0 = p0 + (p1 - p0) * elapsed / duration
        adapter = install(state, projection, history)
        K0 = load.K_at_phase(f0); rates0 = state.local_rates(K0, load.temperature_K)
        h = min(float(state._substep(rates0, duration - elapsed)), duration - elapsed)
        f1 = p0 + (p1 - p0) * (elapsed + h) / duration; fm = .5 * (f0 + f1)
        r0 = state.cleavage_rate_s(K0, load.temperature_K); g0 = gamma_rate(state, rates0)
        inc = state._advance_spatial_step(h, load.K_at_phase(fm), load.temperature_K); state.time_s += h
        rates1 = state.local_rates(load.K_at_phase(f1), load.temperature_K); g1 = gamma_rate(state, rates1)
        history.ledger.plastic_shear += .5 * (g0 + g1) * h
        history.ledger.cumulative_abs_swept_area_per_volume += .5 * (abs(g0) + abs(g1)) * h
        r1 = state.cleavage_rate_s(load.K_at_phase(f1), load.temperature_K)
        action += .5 * (r0 + r1) * h; elapsed += h; steps += 1
        strip_adapter(state)
    return action, steps


def intensive(state):
    return np.concatenate((state.mobile_m2.ravel(), state.retained_m2.ravel(), state.accumulated_slip_m2.ravel()))


def converged_interval(state, history, projection, load, p0, p1):
    """Compare one full interval with two halves; accept the two-half state."""
    sf, hf = copy.deepcopy(state), copy.deepcopy(history); af, _ = advance_basic(sf, hf, projection, load, p0, p1)
    sh, hh = copy.deepcopy(state), copy.deepcopy(history); pm = .5 * (p0 + p1)
    ah0, n0 = advance_basic(sh, hh, projection, load, p0, pm); ah1, n1 = advance_basic(sh, hh, projection, load, pm, p1)
    scale = np.maximum(np.maximum(abs(intensive(sf)), abs(intensive(sh))), max(float(state.c.rho_forest_floor_m2), 1.0))
    state_error = float(np.max(abs(intensive(sf) - intensive(sh)) / scale))
    gamma_error = float(np.max(abs(hf.ledger.plastic_shear - hh.ledger.plastic_shear)))
    action_error = float(abs(af - ah0 - ah1) / max(abs(ah0 + ah1), 1e-300))
    rf = projection.project(hh.patches(), crack_extension_m=sh.extension_m, K_MPa_sqrt_m=load.K_at_phase(p1), compute_kinematics=False)
    r1 = projection.project(hf.patches(), crack_extension_m=sf.extension_m, K_MPa_sqrt_m=load.K_at_phase(p1), compute_kinematics=False)
    stress_error = float(max(abs(np.asarray(rf["resolved_shear_Pa"]) - np.asarray(r1["resolved_shear_Pa"]))))
    return sh, hh, ah0 + ah1, {"state_relative_error": state_error, "gamma_absolute_error": gamma_error,
        "action_relative_error": action_error, "stress_absolute_error_Pa": stress_error,
        "accepted_native_steps": n0 + n1}


def domain_metrics(projection, history, state, K):
    r = projection.project(history.patches(), crack_extension_m=state.extension_m, K_MPa_sqrt_m=K)
    c = projection.centroids
    mask = (c[:, 0] >= projection.geom.a0 + 2e-6) & (abs(c[:, 1]) <= 20e-6)
    return projection.diagnostics(r, mask), r


def first_crossing(hazard_fraction, domain_fraction):
    candidates = [(hazard_fraction, "hazard"), (domain_fraction, "domain")]
    candidates = [(float(x), name) for x, name in candidates if x is not None and 0.0 <= float(x) <= 1.0]
    return min(candidates, default=(None, None), key=lambda x: x[0])


def affine_state_delta(before, after):
    result = {}
    for name, value in before.__dict__.items():
        other = after.__dict__.get(name)
        if isinstance(value, np.ndarray) and isinstance(other, np.ndarray) and value.shape == other.shape:
            result[name] = other - value
    return result


def apply_affine_projection(state, history, state_delta, ledger_delta, cycles):
    for name, delta in state_delta.items():
        value = getattr(state, name)
        projected = value + float(cycles) * delta
        if name in ("mobile_m2", "retained_m2", "accumulated_slip_m2", "source_available_m2", "source_capacity_m2"):
            projected = np.maximum(projected, 0.0)
        setattr(state, name, projected)
    history.ledger.plastic_shear += float(cycles) * ledger_delta[0]
    history.ledger.cumulative_abs_swept_area_per_volume += float(cycles) * ledger_delta[1]
    state.time_s += float(cycles) * 1e-3


def run_load(fatigue, candidate, physics, projection, Kmax, seed, maximum_cycles,
             target_events, checkpoint_dir, phase_steps=64, restart=None,
             stop_after_phase_intervals=None):
    load = fatigue.FatigueLoading(deltaK_MPa_sqrt_m=.9*Kmax, R=.1, frequency_Hz=1000., temperature_K=300., phase_steps=int(phase_steps))
    if restart is None:
        state = fatigue.EmergentGNDState(candidate, physics); history = MaterialHistory(physics.n_bins)
        rng = np.random.default_rng(seed); threshold = float(rng.exponential()); hazard = 0.0
        cycles = 0.0; events = []; convergence = []; acceleration = []; orbit = []
        recent = []; cycle_snapshot_state = copy.deepcopy(state); cycle_snapshot_history = copy.deepcopy(history); cycle_action_start = 0.0
        phase_intervals = 0
        last_metrics = {"projected_beta_p_max_abs":0.0,"projected_engineering_eigenstrain_max_abs":0.0,
          "projected_displacement_gradient_max_abs":0.0,"projected_rotation_max_abs":0.0,"projected_elastic_strain_max_abs":0.0}
    else:
        expected = _case_identity(candidate, physics, projection, Kmax, seed, phase_steps)
        mismatch = {key: (restart["case"].get(key), value) for key, value in expected.items()
                    if restart["case"].get(key) != value}
        if mismatch:
            raise RuntimeError(f"V3.6 restart case incompatibility: {mismatch}")
        state = fatigue._state_from_checkpoint(candidate, physics, restart["state"])
        history = _history_from_checkpoint(physics, restart["history"])
        rng = np.random.default_rng(); rng.bit_generator.state = restart["rng_state"]
        threshold = float(restart["pending_threshold"]); hazard = float(restart["hazard"])
        cycles = float(restart["cycles"]); events = list(restart["events"])
        convergence = list(restart["convergence"]); acceleration = list(restart["acceleration"]); orbit = list(restart["orbit"])
        recent = _recent_from_checkpoint(restart["recent"])
        cycle_snapshot_state = fatigue._state_from_checkpoint(candidate, physics, restart["cycle_snapshot_state"])
        cycle_snapshot_history = _history_from_checkpoint(physics, restart["cycle_snapshot_history"])
        cycle_action_start = float(restart["cycle_action_start"])
        phase_intervals = int(restart["phase_intervals"])
        last_metrics = dict(restart["last_metrics"])
    status = "requested_horizon_censor"
    call_phase_intervals = 0

    def commit_checkpoint(reason):
        return _write_checkpoint(checkpoint_dir, {
            "schema": CHECKPOINT_SCHEMA,
            "case": _case_identity(candidate, physics, projection, Kmax, seed, phase_steps),
            "reason": reason, "state": fatigue._state_to_checkpoint(state),
            "history": _history_to_checkpoint(history), "cycles": cycles,
            "hazard": hazard, "pending_threshold": threshold,
            "rng_state": rng.bit_generator.state, "events": events,
            "convergence": convergence, "acceleration": acceleration,
            "orbit": orbit, "recent": _recent_to_checkpoint(recent),
            "cycle_snapshot_state": fatigue._state_to_checkpoint(cycle_snapshot_state),
            "cycle_snapshot_history": _history_to_checkpoint(cycle_snapshot_history),
            "cycle_action_start": cycle_action_start,
            "phase_intervals": phase_intervals, "last_metrics": last_metrics,
        })

    commit_checkpoint("initial_committed_state" if restart is None else "restored_committed_state")
    while cycles < maximum_cycles and len(events) < target_events:
        if (stop_after_phase_intervals is not None
                and call_phase_intervals >= int(stop_after_phase_intervals)):
            status = "requested_phase_interval_stop"; break
        cycle_index = int(math.floor(cycles + 1e-12)); phase_index = int(round((cycles-cycle_index)*load.phase_steps))
        if phase_index >= load.phase_steps: cycle_index += 1; phase_index = 0
        p0 = phase_index/load.phase_steps; p1 = (phase_index+1)/load.phase_steps
        if phase_index == 0:
            cycle_snapshot_state = copy.deepcopy(state); cycle_snapshot_history = copy.deepcopy(history); cycle_action_start = hazard
        before_state, before_history, before_hazard = copy.deepcopy(state), copy.deepcopy(history), hazard
        if not convergence:
            qualified_state, qualified_history, qualified_da, err = converged_interval(state, history, projection, load, p0, p1)
            state, history, da = qualified_state, qualified_history, qualified_da
        else:
            da, native_steps = advance_basic(state, history, projection, load, p0, p1)
            err = {"state_relative_error":math.nan,"gamma_absolute_error":math.nan,"action_relative_error":math.nan,
                   "stress_absolute_error_Pa":math.nan,"accepted_native_steps":native_steps}
        hazard += da; cycles = cycle_index + p1
        phase_intervals += 1; call_phase_intervals += 1
        fast_field = projection.project(history.patches(), crack_extension_m=state.extension_m,
                                        K_MPa_sqrt_m=load.K_at_phase(p1), compute_kinematics=False)
        fast_beta = float(np.max(abs(fast_field["beta_p"])))
        if phase_index == load.phase_steps-1 or fast_beta >= .05:
            metrics, field = domain_metrics(projection, history, state, load.K_at_phase(p1))
        else:
            metrics = {"projected_beta_p_max_abs":fast_beta,
              "projected_engineering_eigenstrain_max_abs":float(np.max(abs(fast_field["engineering_eigenstrain"]))),
              "projected_displacement_gradient_max_abs":0.0,"projected_rotation_max_abs":0.0,"projected_elastic_strain_max_abs":0.0}
        domain_ratio = max(metrics["projected_beta_p_max_abs"]/.05, metrics["projected_displacement_gradient_max_abs"]/.10,
                           metrics["projected_rotation_max_abs"]/.05, metrics["projected_elastic_strain_max_abs"]/.05)
        last_metrics = dict(metrics)
        candidate_roots = []
        if hazard >= threshold: candidate_roots.append("hazard")
        if domain_ratio >= 1.0: candidate_roots.append("domain")
        fraction, root = (None, None)
        localized = {}
        for candidate_root in candidate_roots:
            lo, hi = 0.0, 1.0; chosen_candidate = None
            for _ in range(18):
                mid = .5*(lo+hi); ts, th = copy.deepcopy(before_state), copy.deepcopy(before_history)
                td, _ = advance_basic(ts, th, projection, load, p0, p0+mid*(p1-p0))
                if candidate_root == "hazard":
                    crossed = td >= threshold-before_hazard; tm = None
                else:
                    tm, _ = domain_metrics(projection, th, ts, load.K_at_phase(p0+mid*(p1-p0)))
                    tr = max(tm["projected_beta_p_max_abs"]/.05, tm["projected_displacement_gradient_max_abs"]/.10,
                             tm["projected_rotation_max_abs"]/.05, tm["projected_elastic_strain_max_abs"]/.05)
                    crossed = tr >= 1.0
                if crossed: hi = mid; chosen_candidate = (ts, th, td, tm)
                else: lo = mid
            localized[candidate_root] = (hi, chosen_candidate)
        if localized:
            fraction, root = min((v[0], k) for k, v in localized.items())
        if root is not None:
            chosen = localized[root][1]; hi = localized[root][0]
            if chosen is None: chosen = (state, history, da, metrics)
            state, history, used, metrics = chosen; hazard = before_hazard + used; cycles = cycle_index+p0+hi*(p1-p0)
            if metrics is None: metrics, _ = domain_metrics(projection, history, state, load.K_at_phase(p0+hi*(p1-p0)))
            last_metrics = dict(metrics)
            if root == "domain":
                status = "operator_consistent_kinematic_domain_exit"
                commit_checkpoint("accepted_domain_exit_state"); break
            factor = fatigue._event_length_factor(threshold, fatigue.FatigueNumerics(base_event_length_m=5e-6, event_length_factor_min=.5, event_length_factor_max=4.))
            advance = 5e-6*factor; old_tip = float(state.extension_m); state.translate_tip(advance)
            transaction = history.commit_advance(old_tip, float(state.extension_m))
            event = {"event_index": len(events), "Kmax_MPa_sqrt_m": Kmax, "seed": seed, "cycles": cycles,
                     "phase": cycles-math.floor(cycles), "threshold": threshold, "hazard_action": hazard,
                     "advance_m": advance, "extension_m": state.extension_m, **metrics, **transaction}
            events.append(event); hazard = 0.0; threshold = float(rng.exponential())
            recent.clear(); cycle_snapshot_state = copy.deepcopy(state); cycle_snapshot_history = copy.deepcopy(history); cycle_action_start = 0.0
            commit_checkpoint("accepted_event")
            continue
        if not convergence or all(math.isfinite(float(err[k])) for k in ("state_relative_error","action_relative_error")):
            convergence.append({"Kmax_MPa_sqrt_m":Kmax,"cycles":cycles,"phase":p1,**err})
        if abs(cycles-round(cycles)) < 1e-10:
            action_cycle = hazard-cycle_action_start
            orbit.append({"Kmax_MPa_sqrt_m":Kmax,"cycles":cycles,"event_index":len(events),"action_cycle":action_cycle,
                          "hazard_action":hazard,**metrics})
            # One-cycle affine map with a curvature bound; cumulative rather
            # than last-step error limits the proposed block.
            sd = affine_state_delta(cycle_snapshot_state, state)
            ld = (history.ledger.plastic_shear-cycle_snapshot_history.ledger.plastic_shear,
                  history.ledger.cumulative_abs_swept_area_per_volume-cycle_snapshot_history.ledger.cumulative_abs_swept_area_per_volume)
            recent.append((action_cycle, sd, ld)); recent = recent[-3:]
            if len(recent) == 3:
                action_cycle = float(recent[-1][0])
                keys = set(recent[-1][1]) & set(recent[-2][1])
                curvature = max((float(np.max(abs(recent[-1][1][k]-recent[-2][1][k]))) for k in keys), default=0.0)
                scale = max((float(np.max(abs(getattr(state,k)))) for k in keys), default=1.0)
                remaining = max(threshold-hazard,0.0)
                proposed = min(int(max(math.floor(.8*remaining/max(action_cycle,1e-300)),0)), 4096, int(maximum_cycles-cycles))
                if curvature > 0.0:
                    curvature_cap = int(max((math.sqrt(1.0+8.0e-3*max(scale,1.0)/curvature)-1.0)/2.0,0.0))
                    proposed = min(proposed, curvature_cap)
                bound = .5*curvature*proposed*(proposed+1)/max(scale,1.0)
                mech_per_cycle = float(np.max(abs(recent[-1][2][0])))
                if proposed > 0 and bound < 1e-3 and mech_per_cycle*proposed < 1e-7:
                    apply_affine_projection(state,history,recent[-1][1],recent[-1][2],proposed)
                    cycles += proposed; hazard += proposed*action_cycle
                    acceleration.append({"Kmax_MPa_sqrt_m":Kmax,"event_index":len(events),"cycles_projected":proposed,
                        "cumulative_state_curvature_bound":bound,"projected_gamma_bound":mech_per_cycle*proposed,
                        "action_per_cycle":action_cycle,"pending_threshold":threshold})
                    recent.clear()
        commit_checkpoint("accepted_phase_interval")
    else:
        status = "target_events_completed" if len(events) >= target_events else "requested_horizon_censor"
    return {"events":events,"convergence":convergence,"acceleration":acceleration,"orbit":orbit,
            "summary":{"Kmax_MPa_sqrt_m":Kmax,"seed":seed,"status":status,"events":len(events),"final_cycles":cycles,
                       "final_extension_m":state.extension_m,"pending_hazard":hazard,"pending_threshold":threshold,
                       "phase_intervals":phase_intervals,"checkpoint":str(checkpoint_dir/"v3_6_restart.json"),**last_metrics}}


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--registry",type=Path,required=True); p.add_argument("--out-root",type=Path,required=True)
    p.add_argument("--parent-registry",type=Path,default=None)
    p.add_argument("--canonical-registry",type=Path,default=None)
    p.add_argument("--canonical-manifest",type=Path,default=None)
    p.add_argument("--seed",type=int,default=1720); p.add_argument("--maximum-cycles",type=float,default=1e6); p.add_argument("--target-events",type=int,default=12)
    p.add_argument("--loads",default=','.join(str(x) for x in LOADS))
    p.add_argument("--phase-steps",type=int,default=64)
    p.add_argument("--restart-checkpoint",type=Path,default=None)
    p.add_argument("--checkpoint-dir",type=Path,default=None)
    p.add_argument("--stop-after-phase-intervals",type=int,default=None)
    a=p.parse_args(argv); a.out_root.mkdir(parents=True,exist_ok=True)
    ck=(a.checkpoint_dir or (a.out_root/"checkpoint")).resolve(); ck.mkdir(parents=True,exist_ok=True)
    reg=json.loads(a.registry.read_text()); row=next(x for x in reg["rows"] if x["candidate_id"]=="V3_ceramic_V2_P25_007__V3_1_TRUST1")
    fatigue=exact_row.load_external_fatigue()
    loader_options = {
        key: value for key, value in {
            "registry_path": a.parent_registry,
            "canonical_registry": a.canonical_registry,
            "canonical_manifest": a.canonical_manifest,
        }.items() if value is not None
    }
    candidate,physics,_,parent=load_v3_1_candidate(fatigue,row,**loader_options); projection=OperatorConsistentProjection(orientation_signs=physics.emission_signs)
    all_events=[]; all_conv=[]; all_acc=[]; all_orbit=[]; summaries=[]
    selected_loads=tuple(float(x) for x in a.loads.split(',') if x.strip())
    restart = None if a.restart_checkpoint is None else _load_checkpoint(a.restart_checkpoint)
    if restart is not None and len(selected_loads) != 1:
        raise SystemExit("V3.6 restart continuation requires exactly one selected load")
    for K in selected_loads:
        result=run_load(fatigue,candidate,physics,projection,K,a.seed,a.maximum_cycles,a.target_events,ck,a.phase_steps,
                        restart=restart,stop_after_phase_intervals=a.stop_after_phase_intervals)
        all_events+=result["events"]; all_conv+=result["convergence"]; all_acc+=result["acceleration"]; all_orbit+=result["orbit"]; summaries.append(result["summary"])
    pd.DataFrame(all_events).to_csv(a.out_root/"developed_events.csv",index=False); pd.DataFrame(all_conv).to_csv(a.out_root/"coupled_step_convergence.csv",index=False)
    pd.DataFrame(all_acc).to_csv(a.out_root/"validated_cycle_acceleration.csv",index=False); pd.DataFrame(all_orbit).to_csv(a.out_root/"cycle_orbit_history.csv",index=False)
    pd.DataFrame(summaries).to_csv(a.out_root/"load_summary.csv",index=False)
    parent_registry = a.parent_registry or Path(__file__).resolve().parents[1]/"analysis_outputs/v3_fatigue_informed_joint_barrier_search/v3_complete_candidate_registry.json"
    canonical_registry = a.canonical_registry or exact_row.ROOT/"analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer/production_candidate_rows.csv"
    canonical_manifest = a.canonical_manifest or exact_row.ROOT/"analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer/production_candidate_manifest.json"
    input_paths = {"optimized": a.registry, "parent": parent_registry,
                   "canonical": canonical_registry, "canonical_manifest": canonical_manifest}
    contract={"schema":"v10.2.30_v3_6_operator_consistent_developed_fatigue_v1","candidate_id":row["candidate_id"],"parent_row_sha256":parent["complete_v3_row_sha256"],
      "loads_MPa_sqrt_m":list(selected_loads),"maximum_cycles":a.maximum_cycles,"target_events":a.target_events,"support_width_m":2e-6,"patch_length_m":5e-6,
      "observation_weight":"fixed physical element-area measure","equal_element_role":"historical audit only","cleavage_feedback":"area-weighted opening stress",
      "slip_feedback":"system-resolved FEM shear on active physical support","unresolved_internal_feedback":"native tau_gnd and source backstress retained separately",
      "initial_nucleus_swept_area":"UNSPECIFIED; zero in conditional glide-only lane, not a physical proof","barriers_prefactors_q_tau_event_lengths_changed":False,
      "root_policy":"all candidate roots from same pre-step state; earliest physical phase commits","high_load_is_gate":False,
      "input_registries":{name:{"path":str(path.resolve()),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}
      for name,path in input_paths.items()}}
    contract["phase_steps"]=a.phase_steps
    (a.out_root/"mechanical_contract.json").write_text(json.dumps(contract,indent=2,sort_keys=True)+"\n")
    return 0
if __name__=="__main__": raise SystemExit(main())
