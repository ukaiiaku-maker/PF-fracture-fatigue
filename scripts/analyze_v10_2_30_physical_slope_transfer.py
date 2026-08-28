#!/usr/bin/env python3
"""Build and freeze the analysis-only physical slope-transfer audit."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arrhenius_fracture.physical_slope_transfer_v10230 import (
    MODEL_ID,
    aggregate_stage_points,
    cross_target_fit,
    extract_event_history,
    seed_check_predictions,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "runs/inverse_fatigue_barrier_design_v1"
OUT = ROOT / "runs/physical_slope_transfer_v1"
FIG = OUT / "figures"
BRANCH = "codex/v10.2.30-physical-slope-transfer"


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def resolution_audit() -> pd.DataFrame:
    rows = [
        ("prescribed_K_signed", "cycle/phase", "DERIVABLE_EXACT", "fixed-DeltaK control plus archived waveform equation"),
        ("opening_clipped_K", "cycle/phase", "DERIVABLE_EXACT", "max(K_signed,0)"),
        ("front_specific_K_used_by_cleavage", "event", "ARCHIVED", "hazard-energy gate event_K and probe_K"),
        ("front_specific_K_used_by_cleavage", "cycle/phase", "NOT_ARCHIVED", "no phase front-probe series"),
        ("K_shield", "event pre/post", "ARCHIVED", "kinetic audit and atomic first-passage checkpoint"),
        ("K_shield", "cycle/phase", "NOT_ARCHIVED", "steps CSV is a zero placeholder in this high-cycle path"),
        ("r_eff", "event pre/post", "ARCHIVED", "committed-state history and kinetic event audit"),
        ("r_eff", "cycle/phase", "NOT_ARCHIVED", "no accepted phase-state sequence"),
        ("cleavage_stress", "event peak", "ARCHIVED", "atomic first-passage checkpoint sigma_tip_Pa"),
        ("cleavage_stress", "cycle/phase", "NOT_ARCHIVED", "no accepted phase-state sequence"),
        ("raw_cleavage_barrier", "event", "DERIVED_NOT_ARCHIVED", "exact immutable EXP-floor row evaluated at archived event stress; CSV value is zero placeholder"),
        ("raw_cleavage_rate", "event", "DERIVED_NOT_ARCHIVED", "nu0 exp(-G/kBT) from derived barrier"),
        ("gamma_renewal_rate", "event endpoint", "ARCHIVED_AND_DERIVABLE", "checkpoint/steps lambda and exact gamma transform"),
        ("cleavage_action_increment", "event interval", "ARCHIVED", "physical_hazard_action_block"),
        ("cleavage_action_increment", "cycle/phase", "NOT_ARCHIVED", "no accepted per-phase action series"),
        ("residual_cleavage_action", "event interval start", "ARCHIVED", "atomic committed-state stochastic action"),
        ("first_passage_threshold", "event", "ARCHIVED", "geometry/event audit and RNG history"),
        ("proposed_event_length", "event", "ARCHIVED", "threshold-scaled stochastic proposal"),
        ("energy_admissible_event_length", "event", "ARCHIVED", "hazard-energy gate audit"),
        ("accepted_projected_event_length", "event", "ARCHIVED", "developed event and geometry transaction"),
        ("event_phase", "event", "NOT_ARCHIVED", "localizer records cycle fraction but not terminal waveform phase"),
        ("event_cycle", "event", "ARCHIVED", "developed event measurement"),
        ("pre_event_state", "event", "ARCHIVED", "atomic committed-state 3202-coordinate snapshot"),
        ("post_event_state", "event", "ARCHIVED", "kinetic audit 3202-coordinate snapshot and scalar diagnostics"),
        ("transaction_and_geometry_rebuild", "event", "ARCHIVED", "energy, geometry, first-passage invalidation audits"),
    ]
    return pd.DataFrame(rows, columns=["quantity", "resolution", "availability", "authoritative_source"])


def lineage_markdown() -> str:
    return """# Equation and state lineage for physical slope transfer

## Immutable production path

The fixed loading is `K_signed(phi)=Kmax[(1+R)/2+(1-R)cos(phi)/2]` and cleavage receives
`K_open=max(K_signed,0)`.  The production mechanical operator maps the opening load and
the evolving signed MPZ state to peak cleavage stress.  Cleavage then uses the frozen
EXP-floor barrier, raw Arrhenius rate, three-hit gamma renewal, integrated stochastic
action, threshold first passage, threshold-correlated event proposal, post-first-passage
energy gate, and checked sharp-wake transaction.  Geometry commit translates the MPZ
and invalidates the geometry-specific high-cycle cache.

## Reduced closures

- `A0_OPENING_ONLY`: exact opening-waveform quadrature with the frozen scalar barrier,
  fixed reference radius, and no event-conditioned MPZ history.
- `A1_CONTINUOUS_MEAN_BLUNTING`: stationary continuous mean-state correction used only
  in the earlier analytical comparison.
- `A2_PT_STATE_ONLY`: has `g_A2 = g_A1` by construction in that closure and is not an
  independent physical PT ablation.
- `B1_EVENT_CONDITIONED_EMISSION`: reduced event-conditioned emission/blunting map
  validated only against archived rates; it is not the production state.
- `PRODUCTION`: the 80-bin signed mobile/retained/accumulated/returned and wake fields,
  event-resolved stochastic clock, energy gate, moving-frame transaction, and mechanical
  kernel used by the completed trajectories.

## Evidence boundary

The archive contains complete event-boundary active-state vectors and event transactions,
but not the accepted state at every waveform phase.  Therefore T0 and the T4 archived
identity are exact.  A held-event-state phase quadrature is reported as `D1/D2`, not as an
exact T1/T2/T3 physical replay.  Missing phase histories are never replaced by terminal
means or zero-filled CSV placeholders.
"""


def replay_status() -> pd.DataFrame:
    rows = [
        ("T0", "exact A0 opening quadrature", "EXACT", "analytical immutable-barrier path"),
        ("T1", "opening hazard on full archived K/r_eff/K_shield history", "NOT_EXACTLY_IDENTIFIABLE", "phase-resolved r_eff and K_shield not archived"),
        ("T2", "T1 plus residual action and first passage", "NOT_EXACTLY_IDENTIFIABLE", "T1 phase history missing; event thresholds/actions archived"),
        ("T3", "T2 plus production event/energy transaction", "NOT_EXACTLY_IDENTIFIABLE", "transaction is archived but exact T2 denominator is unavailable"),
        ("T4", "full archived event-conditioned response", "EXACT_ARCHIVED_IDENTITY", "accepted advance divided by archived developed cycles"),
        ("D1", "held-event-state opening quadrature", "REDUCED_DIAGNOSTIC", "exact quadrature of explicitly reduced event-endpoint mapping"),
        ("D2", "D1 with archived thresholds and event rewards", "REDUCED_DIAGNOSTIC", "does not claim missing phase-state evolution"),
    ]
    return pd.DataFrame(rows, columns=["replay", "definition", "status", "reason"])


def summary_payload(events: pd.DataFrame, points: pd.DataFrame, fit: pd.DataFrame) -> dict:
    radius_identity = 1.0 - 0.5 * points.m_cycle_weighted_r_eff_m
    heldout_rmse = float(np.sqrt(np.mean(fit.M4_heldout_residual**2)))
    threshold_sets = events.groupby(["M_target", "Kmax_MPa_sqrt_m"]).first_passage_threshold.apply(tuple)
    proposal_sets = events.groupby(["M_target", "Kmax_MPa_sqrt_m"]).proposed_event_length_m.apply(tuple)
    return {
        "schema": "v10.2.30_physical_slope_transfer_operator_v1",
        "model_id": MODEL_ID,
        "classification": "UNIVERSAL_REDUCED_TRANSFER_WITH_PHASE_ARCHIVE_LIMITATION",
        "physics_modified": False,
        "diagnostic_fit_is_constitutive_law": False,
        "trajectory_count": 21,
        "event_count": int(len(events)),
        "all_events_geometry_committed": bool(events.geometry_commit_inserted.all()),
        "all_first_passage_actions_close_to_threshold": bool(events.first_passage_action_closes_to_threshold.all()),
        "all_energy_gates_untruncated": bool(events.proposal_matches_gate.all()),
        "all_energy_admissible_path_lengths_committed": bool(events.gate_matches_path_commit.all()),
        "common_threshold_stream_across_all_targets_and_loads": len(set(threshold_sets)) == 1,
        "common_event_reward_stream_across_all_targets_and_loads": len(set(proposal_sets)) == 1,
        "maximum_A_phys_spread_across_M": float(fit.A_phys_spread.max()),
        "heldout_M4_RMSE": heldout_rmse,
        "heldout_M4_maximum_absolute_residual": float(fit.M4_heldout_residual.abs().max()),
        "universal_acceptance": bool(fit.A_phys_spread.max() <= 0.03 and heldout_rmse <= 0.05),
        "maximum_radius_mapping_identity_residual": float(np.max(np.abs(points.m_K_to_sigma-radius_identity))),
        "maximum_absolute_shield_fraction": float(np.max(np.abs(events.K_shield_post_Pa_sqrt_m/events.K_signed_max_Pa_sqrt_m))),
        "phase_resolved_state_history_archived": False,
        "exact_T1_T2_T3_replay_available": False,
        "exact_T0_available": True,
        "exact_T4_archived_identity_available": True,
        "transfer_fit_formula": "m_physical=A(K)*m_A0+B(K); diagnostic only",
    }


def figures(events: pd.DataFrame, points: pd.DataFrame, fit: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 180})

    def finish(name, title, x, y, legend=True):
        plt.title(title); plt.xlabel(x); plt.ylabel(y)
        if legend: plt.legend(frameon=False)
        plt.tight_layout(); plt.savefig(FIG/name, bbox_inches="tight"); plt.close()

    plt.figure(figsize=(6.4,4.2))
    for M,g in points.groupby("M_target"): plt.plot(g.Kmax_MPa_sqrt_m,g.A_phys,"o-",label=f"M={M}")
    finish("A_PHYS_BY_K_AND_M.png","Physical slope transmission",r"$K_{max}$",r"$A_{phys}=m_{physical}/m_{A0}$")

    plt.figure(figsize=(6.4,4.2)); g=points[points.M_target==4]
    plt.plot(g.Kmax_MPa_sqrt_m,g.m_T0_exact_A0_da_dN,"o-",label="A0")
    plt.plot(g.Kmax_MPa_sqrt_m,g.m_K_to_sigma,"o-",label="K→sigma")
    plt.plot(g.Kmax_MPa_sqrt_m,g.m_T4_archived_identity_da_dN,"o-",label="physical")
    finish("M4_STAGEWISE_SLOPE_TRANSMISSION.png","M=4 stagewise local slopes",r"$K_{max}$","local log slope")

    plt.figure(figsize=(6.4,4.2));
    for M,g in points.groupby("M_target"):
        radius=1-.5*g.m_cycle_weighted_r_eff_m
        plt.plot(g.Kmax_MPa_sqrt_m,g.m_K_to_sigma,"o-",label=f"stress M={M}")
        plt.plot(g.Kmax_MPa_sqrt_m,radius,"--",color=plt.gca().lines[-1].get_color())
    finish("RADIUS_DERIVATIVE_VS_STRESS_TRANSMISSION.png","Stress transmission and 1-0.5 dln(r)/dln(K)",r"$K_{max}$","mapping slope")

    plt.figure(figsize=(6.4,4.2))
    plt.plot(fit.Kmax_MPa_sqrt_m,fit.M4_heldout_physical_slope,"ko-",label="M=4 physical")
    plt.plot(fit.Kmax_MPa_sqrt_m,fit.M4_heldout_predicted_slope,"s--",label="M=2/6 prediction")
    finish("M4_HELDOUT_TRANSFER_TEST.png","Held-out cross-target transfer test",r"$K_{max}$","local slope")

    plt.figure(figsize=(6.4,4.2))
    for M,g in points.groupby("M_target"): plt.plot(g.Kmax_MPa_sqrt_m,1e6*g.cycle_weighted_r_eff_m,"o-",label=f"M={M}")
    finish("EVENT_CONDITIONED_RADIUS_BY_K.png","Cycle-weighted event radius",r"$K_{max}$",r"$r_{eff}$ (µm)")

    plt.figure(figsize=(6.4,4.2))
    for M,g in points.groupby("M_target"): plt.semilogy(g.Kmax_MPa_sqrt_m,np.maximum(abs(g.mean_K_shield_Pa_sqrt_m)/(g.Kmax_MPa_sqrt_m*1e6),1e-30),"o-",label=f"M={M}")
    finish("ACTUAL_CLEAVAGE_SHIELDING_FRACTION.png","Archived signed shielding magnitude",r"$K_{max}$",r"$|K_{shield}|/K_{max}$")

    plt.figure(figsize=(6.4,4.2)); g=points[points.M_target==4]
    for name,label in [("m_T4_archived_identity_da_dN","physical"),("m_diagnostic_freeze_r_eff_da_dN","freeze radius"),("m_diagnostic_zero_K_shield_da_dN","zero shielding"),("m_diagnostic_remove_energy_truncation_da_dN","remove energy truncation")]:
        plt.plot(g.Kmax_MPa_sqrt_m,g[name],"o-",label=label)
    finish("M4_CONTROLLED_COUNTERFACTUAL_SLOPES.png","Reduced/event-ledger counterfactuals",r"$K_{max}$","local slope")

    plt.figure(figsize=(6.4,4.2));
    for M,g in points.groupby("M_target"): plt.plot(g.Kmax_MPa_sqrt_m,g.A_reduced_peak_state,"o-",label=f"M={M}")
    finish("REDUCED_REPLAY_TRANSMISSION.png","Held-event-state reduced replay",r"$K_{max}$",r"$m_{D1}/m_{A0}$")


def build(require_clean: bool) -> None:
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit("wrong branch for slope-transfer analysis")
    if require_clean and git("status", "--porcelain"):
        raise SystemExit("transfer freeze requires a clean worktree")
    OUT.mkdir(parents=True, exist_ok=True)
    events = extract_event_history(SOURCE)
    prospective = pd.read_csv(SOURCE / "prospective_physical_predictions.csv")
    points = aggregate_stage_points(events, prospective)
    fit = cross_target_fit(points)
    predictions = seed_check_predictions(points)
    audit = resolution_audit()
    status = replay_status()
    summary = summary_payload(events, points, fit)

    events.to_parquet(OUT / "archived_event_stage_history.parquet", index=False)
    points.to_csv(OUT / "stagewise_slope_transmission.csv", index=False)
    fit.to_csv(OUT / "cross_target_transfer_fit.csv", index=False)
    predictions.to_csv(OUT / "second_seed_prospective_predictions.csv", index=False)
    audit.to_csv(OUT / "archive_resolution_audit.csv", index=False)
    status.to_csv(OUT / "counterfactual_replay_status.csv", index=False)
    (OUT / "equation_and_state_lineage.md").write_text(lineage_markdown())
    write_json(OUT / "physical_transfer_operator.json", summary)

    radius = points[["option_key","M_target","Kmax_MPa_sqrt_m","cycle_weighted_r_eff_m","event_mean_r_eff_m","mean_K_shield_Pa_sqrt_m","m_cycle_weighted_r_eff_m","m_K_to_sigma"]].copy()
    radius["radius_summary_was_initial_value"] = True
    radius.to_csv(OUT / "r_eff_and_shielding_audit.csv", index=False)
    figures(events, points, fit)

    artifacts = ["archived_event_stage_history.parquet","stagewise_slope_transmission.csv",
                 "cross_target_transfer_fit.csv","second_seed_prospective_predictions.csv",
                 "archive_resolution_audit.csv","counterfactual_replay_status.csv",
                 "equation_and_state_lineage.md","physical_transfer_operator.json",
                 "r_eff_and_shielding_audit.csv"]
    freeze = {
        "schema":"v10.2.30_physical_slope_transfer_freeze_v1", "frozen_utc":now(),
        "branch":BRANCH, "git_head":git("rev-parse","HEAD"), "physics_launch_utc":None,
        "source_inverse_design_verification_sha256":sha(SOURCE/"inverse_design_verification.json"),
        "source_candidate_registry_sha256":sha(SOURCE/"inverse_design_candidate_registry.csv"),
        "source_solver_sha256":json.loads((SOURCE/"inverse_design_verification.json").read_text())["solver_sha256"],
        "analysis_artifact_hashes":{name:sha(OUT/name) for name in artifacts},
        "seed_check_acceptance":{
            "maximum_absolute_A_interval_difference":0.15,
            "refit_after_second_seed":False,
            "loads_MPa_sqrt_m":[12.0,18.0,24.3],"candidate":"INV_OPENING_M4_V1","seed":1001723,
        },
        "new_physics_runs_before_freeze":0,
    }
    write_json(OUT / "transfer_freeze.json", freeze)
    print(json.dumps({"result":"PASS","events":len(events),"points":len(points),
                      "universal":summary["universal_acceptance"],"head":freeze["git_head"]}))


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("command",choices=["build","freeze"]); args=parser.parse_args()
    build(require_clean=args.command=="freeze"); return 0


if __name__ == "__main__": raise SystemExit(main())
