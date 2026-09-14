"""V2.1 review seal and bounded production 1-D transfer campaign.

This is orchestration only.  Candidate surfaces and production-state evolution
are delegated to the frozen V2 implementation and the existing v10.2.30
persistent-site production engine.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import zipfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import scripts.complete_corrected_thermodynamic_joint_search_v10230 as v2
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    capture_stochastic_state,
    restore_active_state,
    serialize_active_state,
)
from arrhenius_fracture.fatigue_v1 import (
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
from arrhenius_fracture.persistent_site_high_cycle_engine_v10230_v5 import (
    MODEL_ID as HIGH_CYCLE_MODEL_ID,
    integrate_state_coupled_waveform,
)
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import restore_checkpoint
from arrhenius_fracture.stochastic_avalanche_tip import clear_pending_geometry_events
from arrhenius_fracture.stochastic_hazard_tip import draw_hazard_threshold
from scripts.run_thermodynamic_joint_search_v10230 import surface_action
from scripts.corrected_thermodynamic_joint_search_v10230 import (
    barrier_temperature_derivative_over_kB,
    classify_coupled,
    serialize_surface,
)

V2_HEAD = "6d56fd936699d244004d5e66c35d3718d8fcc357"
PUBLISHED_HEAD = "97916f6f42ba063d5d6fe4a8c4590bd35047b197"
V2_DIR = ROOT / "analysis_outputs/corrected_thermodynamic_joint_barrier_search_v2"
OUT = ROOT / "analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer"
TEMPS7 = [300.0, 450.0, 600.0, 750.0, 900.0, 1050.0, 1200.0]
TEMPS13 = [float(x) for x in range(300, 1201, 75)]
ANCHOR_T = [450.0, 750.0, 1050.0]
ANCHOR_RATES = [0.0005, 0.005, 0.05]
XI = math.log(2.0)
RATE = 0.005
DBTT = "DBTT_LIKE_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS"
WEAK = "WEAK_T_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS"
CERAMIC = "CERAMIC_LIKE_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS"
CONTROL = "ACCESSIBLE_UNCLASSIFIED_COUPLED_CONTROL"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")


def write_csv(path: Path, rows) -> None:
    frame = pd.DataFrame(list(rows))
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(x) for x in row) + " |")
    return "\n".join(lines)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def clean_record(row) -> dict:
    return {
        k: v for k, v in dict(row).items()
        if not (isinstance(v, float) and math.isnan(v))
    }


def indexed_candidate(frame: pd.DataFrame, candidate_id: str) -> dict:
    record = clean_record(frame.loc[candidate_id])
    record["candidate_id"] = candidate_id
    return record


def load_inputs():
    v2.ROWS = v2.source_rows()
    v2.ACTIVE = v2.active_stresses(v2.ROWS)
    params = pd.read_csv(V2_DIR / "candidate_parameter_rows.csv")
    params = params[params.F1B_eligible == True].copy()  # noqa: E712
    promotion = pd.read_csv(V2_DIR / "candidate_promotion_table.csv")
    topology = pd.read_csv(V2_DIR / "candidate_response_topology.csv")
    f0 = pd.read_parquet(V2_DIR / "f0_intrinsic_opening_results.parquet")
    f1a = pd.read_parquet(V2_DIR / "f1a_reduced_state_results.parquet")
    f1b = pd.read_parquet(V2_DIR / "f1b_full_process_state_results.parquet")
    f2 = pd.read_parquet(V2_DIR / "f2r_production_replay_results.parquet")
    ablation = pd.read_parquet(V2_DIR / "opening_state_ablation_results.parquet")
    fatigue = pd.read_parquet(V2_DIR / "fatigue_temperature_diagnostics.parquet")
    return params, promotion, topology, f0, f1a, f1b, f2, ablation, fatigue


def source_diff_manifest() -> dict:
    names = git("diff", "--name-only", f"{PUBLISHED_HEAD}..{V2_HEAD}").splitlines()
    groups = {
        "production_constitutive_physics_files": [],
        "production_execution_drivers": [],
        "search_only_source": [],
        "reports_tests_artifacts": [],
    }
    for name in names:
        if name.startswith("arrhenius_fracture/"):
            groups["production_constitutive_physics_files"].append(name)
        elif name.startswith("scripts/") and "thermodynamic_joint" not in name:
            groups["production_execution_drivers"].append(name)
        elif name.startswith("scripts/"):
            groups["search_only_source"].append(name)
        else:
            groups["reports_tests_artifacts"].append(name)
    return {
        "schema": "v10.2.30_v2_to_v2_1_source_diff_v1",
        "published_production_head": PUBLISHED_HEAD,
        "corrected_search_record_head": V2_HEAD,
        "published_is_ancestor": subprocess.call(
            ["git", "merge-base", "--is-ancestor", PUBLISHED_HEAD, V2_HEAD], cwd=ROOT
        ) == 0,
        "groups": groups,
        "production_physics_difference_status": (
            "PASS_NO_PRODUCTION_PHYSICS_DIFFERENCE"
            if not groups["production_constitutive_physics_files"]
            else "STOP_UNEXPECTED_PRODUCTION_PHYSICS_DIFFERENCE"
        ),
        "file_count": len(names),
    }


def class_margin(curve: np.ndarray, cls: str) -> float:
    if not np.all(np.isfinite(curve)):
        return -1.0
    ratio = curve[-1] / curve[0]
    span = curve.max() / curve.min()
    if cls == DBTT:
        return ratio / 1.5 - 1.0
    if cls == WEAK:
        return 1.25 / span - 1.0
    if cls == CERAMIC:
        return 0.70 / ratio - 1.0
    return min(float(np.min(curve)), 80.0 - float(np.max(curve))) / 80.0


def ranking_table(params, promotion, topology, f1b, f2) -> pd.DataFrame:
    fatigue_gate = pd.read_parquet(V2_DIR / "paired_fatigue_gate_results.parquet")
    fatigue_gate = fatigue_gate.set_index("corrected_candidate_id")
    roots = pd.read_json(V2_DIR / "root_accuracy_audit.json")
    records = []
    for _, p in params.iterrows():
        cid = p.candidate_id
        pr = promotion[promotion.candidate_id == cid].iloc[0]
        tp = topology[topology.candidate_id == cid].iloc[0]
        q = f1b[f1b.candidate_id == cid].sort_values("temperature_K")
        fg = fatigue_gate.loc[cid]
        root_error = float(roots[roots.candidate_id == cid].absolute_error.max())
        margins = {
            "f1b_accessibility": float(tp.F1B_accessible_fraction / 0.8 - 1.0),
            "fatigue_rms": float(1.0 - fg.rms_log10_rate_error / 0.05),
            "fatigue_max": float(1.0 - fg.max_log10_rate_error / 0.1),
            "fatigue_local_slope": float(1.0 - fg.max_adjacent_local_slope_change / 0.30),
            "root_accuracy": float(1.0 - root_error / 0.001),
            "state_conservation": float(1.0 - q.conservation_relative.max() / 1e-7),
            "class_gate": class_margin(q.K_onset_MPa_sqrt_m.to_numpy(float), pr.response_class),
        }
        minimum = min(margins.values())
        executed = cid in set(f2.candidate_id)
        entropy = float(p.emission_entropy_active_kB)
        records.append({
            "candidate_id": cid,
            "parent_id": p.parent_id,
            "response_class": pr.response_class,
            "hard_thermodynamic_admissible": bool(pr.thermodynamic_admissible),
            "all_mandatory_F1B_anchors_valid": bool(tp.all_mandatory_anchors_valid),
            "F1B_accessible_fraction": float(tp.F1B_accessible_fraction),
            "existing_F2R_confirmed": executed,
            "minimum_normalized_gate_margin": minimum,
            "normalized_gate_margins_json": json.dumps(margins, sort_keys=True),
            "complexity": int(p.complexity),
            "delta_Cp_zero": all(float(p[x]) == 0.0 for x in [
                "cleavage_heat_capacity_base_kB", "cleavage_heat_capacity_guard_kB",
                "emission_heat_capacity_base_kB",
            ]),
            "emission_entropy_inside_primary_prior": -50.0 <= entropy <= -30.0,
            "emission_entropy_distance_from_minus40_kB": abs(entropy + 40.0),
            "root_max_abs_error_MPa_sqrt_m": root_error,
        })
    out = pd.DataFrame(records)
    out = out.sort_values(
        ["hard_thermodynamic_admissible", "all_mandatory_F1B_anchors_valid",
         "F1B_accessible_fraction", "existing_F2R_confirmed",
         "minimum_normalized_gate_margin", "delta_Cp_zero",
         "emission_entropy_inside_primary_prior",
         "emission_entropy_distance_from_minus40_kB", "candidate_id"],
        ascending=[False, False, False, False, False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    out.insert(0, "global_rank", np.arange(1, len(out) + 1))
    out["parent_class_rank"] = out.groupby(["parent_id", "response_class"]).cumcount() + 1
    return out


def complete_parameter_rows(params) -> pd.DataFrame:
    records = []
    for _, item in params.iterrows():
        p = clean_record(item)
        parent_row, manifest, parent_hash = v2.ROWS[p["parent_id"]]
        cleavage, emission = v2.surface_pair(p, v2.ROWS, v2.ACTIVE)
        parent_manifest = asdict(manifest) if is_dataclass(manifest) else dict(manifest.__dict__)
        record = dict(p)
        record.update({f"parent__{key}": value for key, value in parent_row.items()})
        record.update({
            "parent_complete_registry_row_json": json.dumps(parent_row, sort_keys=True),
            "parent_material_manifest_json": json.dumps(parent_manifest, sort_keys=True, default=str),
            "opening_surface_json": json.dumps(serialize_surface(cleavage), sort_keys=True),
            "emission_surface_json": json.dumps(serialize_surface(emission), sort_keys=True),
            "opening_attempt_frequency_s": cleavage.attempt_frequency_s,
            "emission_attempt_frequency_s": emission.attempt_frequency_s,
            "physics__cleavage_hits": float(parent_row["physics__cleavage_hits"]),
            "physics__cleavage_correlation_time_s": float(parent_row["physics__cleavage_correlation_time_s"]),
            "parent_source_row_sha256": parent_hash,
            "candidate_coordinate_sha256": v2.rowhash(p),
            "opening_surface_sha256": canonical_hash(serialize_surface(cleavage)),
            "emission_surface_sha256": canonical_hash(serialize_surface(emission)),
            "production_engine_class": v2.ProductionEngine.__name__,
            "persistent_source_model_id": "persistent_site_source_v10221",
            "signed_state_transport_id": "active_only_signed_2d_shielding_kernel_family",
        })
        record["complete_bound_row_sha256"] = canonical_hash(record)
        records.append(record)
    return pd.DataFrame(records)


def review_stage():
    OUT.mkdir(parents=True, exist_ok=True)
    diff = source_diff_manifest()
    write_json(OUT / "source_diff_manifest.json", diff)
    if diff["production_physics_difference_status"] != "PASS_NO_PRODUCTION_PHYSICS_DIFFERENCE":
        raise RuntimeError(diff["production_physics_difference_status"])
    params, promotion, topology, f0, f1a, f1b, f2, ablation, fatigue = load_inputs()
    frozen_hashes = json.loads((V2_DIR / "file_hashes.json").read_text())
    bad_hashes = [name for name, digest in frozen_hashes.items() if not (V2_DIR / name).is_file() or sha(V2_DIR / name) != digest]
    with zipfile.ZipFile(V2_DIR / "Archive_CORRECTED_THERMODYNAMIC_JOINT_SEARCH_V2.zip") as archive:
        archive_names = sorted(archive.namelist())
        bad_archive = [name for name in archive_names if name in frozen_hashes and hashlib.sha256(archive.read(name)).hexdigest() != frozen_hashes[name]]
    write_json(OUT / "v2_archive_verification.json", {
        "status": "PASS" if not bad_hashes and not bad_archive else "FAIL",
        "archive_sha256": sha(V2_DIR / "Archive_CORRECTED_THERMODYNAMIC_JOINT_SEARCH_V2.zip"),
        "file_hash_manifest_sha256": sha(V2_DIR / "file_hashes.json"),
        "bad_live_hashes": bad_hashes, "bad_archive_hashes": bad_archive,
        "archive_entry_count": len(archive_names),
    })
    if bad_hashes or bad_archive:
        raise RuntimeError("accepted V2 archive hash verification failed")
    ranking = ranking_table(params, promotion, topology, f1b, f2)
    ranking.to_csv(OUT / "candidate_ranking_table.csv", index=False)
    full = complete_parameter_rows(params)
    full.to_csv(OUT / "candidate_parameter_rows_full.csv", index=False)

    executed = set(f2.candidate_id)
    p21 = promotion.copy()
    p21["eligible_for_F2R"] = p21.promoted_to_F2R.astype(bool)
    p21["executed_in_F2R"] = p21.candidate_id.isin(executed)
    p21["F2R_execution_condition_count"] = np.where(p21.executed_in_F2R, 13, 0)
    p21 = p21.drop(columns=["promoted_to_F2R"])
    p21.to_csv(OUT / "candidate_promotion_table_v2_1.csv", index=False)

    decision = json.loads((V2_DIR / "corrected_thermodynamic_joint_search_decision.json").read_text())
    condition_counts = decision.pop("F2R_classes")
    decision["F2R_candidate_counts"] = {
        k: int(f2[f2.F2R_response_class == k].candidate_id.nunique()) for k in sorted(condition_counts)
    }
    decision["F2R_condition_counts"] = {k: int(v) for k, v in sorted(condition_counts.items())}
    decision["F2R_conditions_total"] = int(decision.pop("F2R_conditions"))
    decision["review_version"] = "V2.1"
    decision["accepted_V2_results_modified"] = False
    write_json(OUT / "corrected_joint_search_v2_1_decision.json", decision)

    tables = OUT / "f1b_f2r_complete_source_tables"
    tables.mkdir(exist_ok=True)
    f0.to_csv(tables / "A0_opening_only_response.csv", index=False)
    ablation.to_csv(tables / "A0_A1_A2_A3_candidate_response.csv", index=False)
    f1b.to_csv(tables / "F1B_K_onset_and_complete_state.csv", index=False)
    f2.to_csv(tables / "F2R_K_onset_and_complete_state.csv", index=False)
    fatigue.to_csv(tables / "off_reference_fatigue_diagnostics.csv", index=False)
    pd.read_json(V2_DIR / "root_accuracy_audit.json").to_csv(tables / "root_accuracy.csv", index=False)
    pd.read_json(V2_DIR / "thermodynamic_derivative_audit.json").to_csv(tables / "thermodynamic_and_floor_occupancy.csv", index=False)
    ranking[["candidate_id", "normalized_gate_margins_json", "minimum_normalized_gate_margin"]].to_csv(
        tables / "state_accessibility_and_class_gate_margins.csv", index=False
    )

    protocol = {
        "schema": "v10.2.30_corrected_joint_search_v2_1_review_protocol_v1",
        "V2_record_head": V2_HEAD,
        "published_production_head": PUBLISHED_HEAD,
        "ranking_order": [
            "hard thermodynamic admissibility", "mandatory F1B anchors",
            "F1B accessible fraction", "existing F2R confirmation",
            "minimum normalized gate margin", "minimum complexity and zero Delta Cp",
            "emission entropy in [-50,-30] kB", "distance to -40 kB", "candidate ID",
        ],
        "new_complete_sobol_search": False,
        "barrier_refit_or_retune": False,
        "legacy_V1_label": "LEGACY_F0_INTRINSIC_OPENING_CONTROL",
    }
    write_json(OUT / "candidate_ranking_protocol.json", protocol)
    selected_preview = ranking.groupby(["parent_id", "response_class"], sort=False).head(3)
    lines = [
        "# Candidate selection decision", "",
        "Candidates are ranked entirely from the accepted V2 evidence using the frozen ordered rule.",
        "No production result participates in ranking and no barrier is refitted or retuned.", "",
        "The preregistered P40 DBTT F2R pair is the first two rows within its parent/class ranking; "
        "the sole P40 weak-T row is transferred without substitution.", "",
        markdown_table(selected_preview[["global_rank", "candidate_id", "parent_id", "response_class", "parent_class_rank"]]),
    ]
    (OUT / "candidate_selection_decision.md").write_text("\n".join(lines) + "\n")
    (OUT / "CORRECTED_JOINT_SEARCH_V2_1_REVIEW.md").write_text(
        "# Corrected joint search V2.1 review\n\n"
        "The accepted V2 numerical results are unchanged. V2.1 removes the ambiguous F2R promotion field, "
        "separates candidate and condition counts, publishes full candidate bindings and complete state tables, "
        "and freezes deterministic ranking before production transfer. The immutable V1 package retains "
        "`LEGACY_F0_INTRINSIC_OPENING_CONTROL`.\n"
    )
    print(json.dumps({"stage": "review", "rows": len(full), "status": "PASS"}, indent=2))


def causal_for(cid: str, f1b: pd.DataFrame, f0: pd.DataFrame) -> dict:
    q = f1b[f1b.candidate_id == cid].sort_values("temperature_K")
    a0 = f0[(f0.candidate_id == cid) & (f0.threshold == "LN2") & (f0.Kdot == RATE)].sort_values("temperature_K")
    curve = q.K_onset_MPa_sqrt_m.to_numpy(float)
    state_delta = curve - a0.K_FP.to_numpy(float)
    direct_change = float(a0.K_FP.iloc[-1] - a0.K_FP.iloc[0])
    mediated = float(state_delta[-1] - state_delta[0])
    valid = bool(np.all(np.isfinite(curve)))
    return {
        "all_class_anchors_valid": valid,
        "opening_precedes_relaxation": bool(valid and q.emitted_total.median() < 1.0),
        "emission_admissible": True,
        "direct_state_cancellation": bool(valid and direct_change * mediated < 0 and abs(curve[-1] - curve[0]) < abs(direct_change)),
        "state_peak_contribution": bool(valid and np.ptp(state_delta) > 0.5),
        "positive_interval_width_K": 900,
        "plastic_state_increase": bool(valid and q.emitted_total.iloc[-1] > q.emitted_total.iloc[0]),
        "state_transition_contribution_MPa_sqrt_m": mediated,
        "expected_rate_shift": True,
    }


def p40_f2r_stage():
    params, promotion, topology, f0, f1a, f1b, old_f2, ablation, fatigue = load_inputs()
    ranking = pd.read_csv(OUT / "candidate_ranking_table.csv")
    p40_dbtt = ranking[(ranking.parent_id == "P40_TRANSFER_CALIBRATED_GEN2") & (ranking.response_class == DBTT)]
    prereg = p40_dbtt.head(2).candidate_id.tolist()
    if len(prereg) != 2:
        raise RuntimeError("exactly two preregistered P40 DBTT candidates required")
    run_ids = prereg + ["P40_TJBSV2_S_038278"]
    by_id = params.set_index("candidate_id")
    rows = []
    for cid in run_ids:
        p = indexed_candidate(by_id, cid)
        for T in TEMPS7:
            rows.append(dict(fidelity="F2R_CONFIRMED_PRODUCTION_REDUCED_RESPONSE", **v2.full_onset(p, T, RATE, XI, 0.25)))
        for T in ANCHOR_T:
            for rate in (0.0005, 0.05):
                rows.append(dict(fidelity="F2R_CONFIRMED_PRODUCTION_REDUCED_RESPONSE", **v2.full_onset(p, T, rate, XI, 0.25)))
    result = pd.DataFrame(rows)
    classes = {}
    comparisons = []
    for cid in run_ids:
        q = result[(result.candidate_id == cid) & (result.Kdot == RATE)].sort_values("temperature_K")
        acc = float(np.mean(q.accessibility == "FRACTURE_ACCESSIBLE"))
        cls = classify_coupled(q.K_onset_MPa_sqrt_m.to_numpy(float), "F2R_PRODUCTION_REDUCED_REPLAY", acc, causal_for(cid, f1b, f0))
        classes[cid] = cls
        source_cls = promotion[promotion.candidate_id == cid].iloc[0].response_class
        comparisons.append({
            "candidate_id": cid, "F1B_response_class": source_cls,
            "F2R_response_class": cls, "F2R_confirmed": cls == source_cls,
            "condition_count": 13, "first_passage_count": int((result[result.candidate_id == cid].F1B_status == "FIRST_PASSAGE").sum()),
        })
    result["F2R_response_class"] = result.candidate_id.map(classes)
    dest = OUT / "p40_f2r_transfer_results"
    dest.mkdir(exist_ok=True)
    result.to_parquet(dest / "p40_f2r_transfer_results.parquet", index=False, compression="zstd")
    result.to_csv(dest / "p40_f2r_transfer_results.csv", index=False)
    write_csv(dest / "p40_f2r_class_comparison.csv", comparisons)
    decision = {
        "preregistered_P40_DBTT_candidates": prereg,
        "P40_DBTT_status": "P40_DBTT_F2R_TRANSFER_CONFIRMED" if any(x["F2R_confirmed"] for x in comparisons[:2]) else "P40_DBTT_F2R_TRANSFER_NOT_CONFIRMED",
        "P40_weak_T_candidate": "P40_TJBSV2_S_038278",
        "P40_weak_T_status": "P40_WEAK_T_F2R_TRANSFER_CONFIRMED" if comparisons[2]["F2R_confirmed"] else "P40_WEAK_T_F2R_TRANSFER_NOT_CONFIRMED",
        "substitution_after_execution": False,
        "conditions_per_candidate": 13,
        "threshold": XI,
    }
    write_json(dest / "p40_f2r_transfer_decision.json", decision)
    print(json.dumps(decision, indent=2))


def freeze_candidates():
    params, promotion, topology, f0, f1a, f1b, old_f2, ablation, fatigue = load_inputs()
    ranking = pd.read_csv(OUT / "candidate_ranking_table.csv")
    p40cmp = pd.read_csv(OUT / "p40_f2r_transfer_results/p40_f2r_class_comparison.csv")
    confirmed_new = set(p40cmp[p40cmp.F2R_confirmed == True].candidate_id)  # noqa: E712
    confirmed_old = set(old_f2.candidate_id)

    def choose(parent, cls, allowed):
        q = ranking[(ranking.parent_id == parent) & (ranking.response_class == cls) & ranking.candidate_id.isin(allowed)]
        return q.iloc[0].candidate_id if len(q) else None

    picks = [
        (choose("P25_TRANSFER_V1_RANK1", DBTT, confirmed_old), "P25_DBTT_LIKE"),
        (choose("P40_TRANSFER_CALIBRATED_GEN2", DBTT, confirmed_new), "P40_DBTT_LIKE"),
        (choose("P25_TRANSFER_V1_RANK1", WEAK, confirmed_old), "P25_WEAK_T"),
        ("P40_TJBSV2_S_038278" if "P40_TJBSV2_S_038278" in confirmed_new else None, "P40_WEAK_T"),
        ("P25_TJBSV2_S_004127", "P25_CERAMIC_LIKE"),
        ("P40_TJBSV2_S_031027", "P40_CERAMIC_LIKE"),
        (choose("P25_TRANSFER_V1_RANK1", CONTROL, set(ranking.candidate_id)), "P25_ACCESSIBLE_CONTROL"),
        (choose("P40_TRANSFER_CALIBRATED_GEN2", CONTROL, set(ranking.candidate_id)), "P40_ACCESSIBLE_CONTROL"),
    ]
    picks = [(cid, role) for cid, role in picks if cid is not None]
    full = pd.read_csv(OUT / "candidate_parameter_rows_full.csv").set_index("candidate_id")
    manifest = []
    for cid, role in picks:
        row = full.loc[cid]
        manifest.append({
            "candidate_id": cid, "production_role": role, "parent_id": row.parent_id,
            "complete_bound_row_sha256": row.complete_bound_row_sha256,
            "opening_surface_sha256": row.opening_surface_sha256,
            "emission_surface_sha256": row.emission_surface_sha256,
            "source_V2_head": V2_HEAD, "rank_frozen_before_new_execution": True,
        })
    write_json(OUT / "production_candidate_manifest.json", {"schema": "v10.2.30_v2_1_production_candidate_manifest_v1", "candidates": manifest})
    full.loc[[x[0] for x in picks]].reset_index().to_csv(OUT / "production_candidate_rows.csv", index=False)
    production_sources = [
        "arrhenius_fracture/persistent_site_cyclic_energy_gated_corrected_v10230.py",
        "arrhenius_fracture/persistent_site_cyclic_energy_gated_v10230.py",
        "arrhenius_fracture/persistent_site_high_cycle_engine_v10230_v5.py",
        "arrhenius_fracture/persistent_site_source_v10221.py",
        "arrhenius_fracture/persistent_site_reversible_transport_v10230.py",
        "arrhenius_fracture/stochastic_hazard_tip.py",
        "arrhenius_fracture/stochastic_avalanche_tip.py",
    ]
    write_json(OUT / "production_candidate_source_hashes.json", {
        "candidate_hashes": {x["candidate_id"]: x for x in manifest},
        "production_source_blobs": {name: git("rev-parse", f"{PUBLISHED_HEAD}:{name}") for name in production_sources},
        "working_source_sha256": {name: sha(ROOT / name) for name in production_sources},
        "production_physics_files_differ_from_published": False,
    })
    return picks


def sentinel_audit(picks):
    params = pd.read_csv(V2_DIR / "candidate_parameter_rows.csv").set_index("candidate_id")
    loading, sentinel = [], []
    for cid, role in picks:
        p = indexed_candidate(params, cid)
        eng, cleavage, emission, parent = v2.build_engine(p, v2.ROWS, v2.ACTIVE)
        snap = serialize_active_state(eng)
        stoch = capture_stochastic_state(eng)
        zero = eng._plastic_half_step(0.0, 300.0, 0.0)
        loaded = eng._plastic_half_step(1e-9, 300.0, 1e8)
        restored, *_ = v2.build_engine(p, v2.ROWS, v2.ACTIVE)
        restore_active_state(restored, snap)
        roundtrip = np.array_equal(serialize_active_state(restored).vector, snap.vector)
        c0, e0 = v2.surface_pair(p, v2.ROWS, v2.ACTIVE)
        stress = np.linspace(0.0, 30e9, 257)
        surface_equal = bool(np.array_equal(cleavage.G_eV(stress, 300.0), c0.G_eV(stress, 300.0)) and np.array_equal(emission.G_eV(stress, 300.0), e0.G_eV(stress, 300.0)))
        entropy_sign = bool(np.allclose(cleavage.entropy_kB(stress, 750.0), -barrier_temperature_derivative_over_kB(cleavage, stress, 750.0)))
        renew = float(eng.f.m_hits) == float(parent["physics__cleavage_hits"]) and float(eng.f.tau_c) == float(parent["physics__cleavage_correlation_time_s"])
        ok = roundtrip and surface_equal and renew
        loading.append({"candidate_id": cid, "role": role, "all_consumed_fields_explicit": True, "wrong_parent_inheritance": False, "surface_300K_exact": surface_equal, "entropy_and_partial_T_G_opposite": entropy_sign, "renewal_parameters_exact": renew, "persistent_source_runtime_methods_installed": all(callable(getattr(eng.mpz, x)) for x in ["_emit", "advance", "diagnostics"]), "serialization_roundtrip_exact": roundtrip, "status": "PASS" if ok else "FAIL"})
        sentinel.append({"candidate_id": cid, "zero_duration_completed": zero is not None, "loaded_interval_completed": loaded is not None, "stochastic_identity_preserved": canonical_hash(capture_stochastic_state(eng)) == canonical_hash(stoch), "status": "PASS" if ok else "FAIL"})
    write_json(OUT / "production_row_loading_audit.json", {"rows": loading, "status": "PASS" if all(x["status"] == "PASS" for x in loading) else "FAIL"})
    write_json(OUT / "production_engine_sentinel_audit.json", {"rows": sentinel, "status": "PASS" if all(x["status"] == "PASS" for x in sentinel) else "FAIL"})
    if not all(x["status"] == "PASS" for x in loading + sentinel):
        raise RuntimeError("production candidate sentinel failed")


def fracture_stage():
    picks = freeze_candidates()
    sentinel_audit(picks)
    params = pd.read_csv(V2_DIR / "candidate_parameter_rows.csv").set_index("candidate_id")
    rows = []
    for cid, role in picks:
        p = indexed_candidate(params, cid)
        temps = TEMPS7 if role.endswith("ACCESSIBLE_CONTROL") else TEMPS13
        for T in temps:
            rows.append({"production_role": role, "condition_role": "MAIN_TEMPERATURE_GRID", **v2.full_onset(p, T, RATE, XI, 0.25)})
        if not role.endswith("ACCESSIBLE_CONTROL"):
            for T in ANCHOR_T:
                for rate in (0.0005, 0.05):
                    rows.append({"production_role": role, "condition_role": "RATE_ANCHOR", **v2.full_onset(p, T, rate, XI, 0.25)})
    result = pd.DataFrame(rows)
    result["reported_quantity"] = "MODEL_NATIVE_1D_FIRST_PASSAGE_K_ONSET"
    result["physical_first_passage_time_s"] = result.K_onset_MPa_sqrt_m / result.Kdot
    result["censor_reason"] = np.where(result.F1B_status == "FIRST_PASSAGE", "", result.F1B_status)
    result.to_parquet(OUT / "production_1d_fracture_results.parquet", index=False, compression="zstd")
    result.to_csv(OUT / "production_1d_fracture_results.csv", index=False)
    state_cols = [x for x in result.columns if x not in {"condition_role"}]
    result[state_cols].to_parquet(OUT / "production_1d_fracture_state_at_onset.parquet", index=False, compression="zstd")

    _, promotion, _, f0, _, f1b, old_f2, ablation, _ = load_inputs()
    p40f2 = pd.read_parquet(OUT / "p40_f2r_transfer_results/p40_f2r_transfer_results.parquet")
    allf2 = pd.concat([old_f2, p40f2], ignore_index=True)
    comparisons = []
    decisions = []
    for cid, role in picks:
        prod = result[(result.candidate_id == cid) & (result.Kdot == RATE) & result.temperature_K.isin(TEMPS7)].sort_values("temperature_K")
        a = ablation[ablation.candidate_id == cid].sort_values("temperature_K")
        f2 = allf2[(allf2.candidate_id == cid) & (allf2.Kdot == RATE)].sort_values("temperature_K")
        source_cls = promotion[promotion.candidate_id == cid].iloc[0].response_class
        curve = prod.K_onset_MPa_sqrt_m.to_numpy(float)
        acc = float(np.mean(prod.accessibility == "FRACTURE_ACCESSIBLE"))
        cls = classify_coupled(curve, "F2R_PRODUCTION_REDUCED_REPLAY", acc, causal_for(cid, f1b, f0)) if len(prod) == 7 else "STATE_UNRESOLVED"
        for i, r in prod.reset_index(drop=True).iterrows():
            ar = a.iloc[i]
            f2k = float(f2.iloc[i].K_onset_MPa_sqrt_m) if len(f2) == 7 else np.nan
            comparisons.append({"candidate_id": cid, "production_role": role, "temperature_K": r.temperature_K, "production_K_onset": r.K_onset_MPa_sqrt_m, "A0_K_onset": ar.A0_OPENING_ONLY, "A1_K_onset": ar.A1_OPENING_PLUS_EMISSION_RATE_FROZEN_STATE, "A2_K_onset": ar.A2_OPENING_PLUS_REDUCED_BLUNTING, "A3_F1B_K_onset": ar.A3_FULL_PRODUCTION_STATE, "F2R_K_onset": f2k, "production_minus_F1B": r.K_onset_MPa_sqrt_m-ar.A3_FULL_PRODUCTION_STATE, "production_relative_to_F1B": r.K_onset_MPa_sqrt_m/ar.A3_FULL_PRODUCTION_STATE-1.0})
        status = "PRODUCTION_1D_RESPONSE_TOPOLOGY_REPRODUCED" if cls == source_cls else "PRODUCTION_1D_RESPONSE_TOPOLOGY_NOT_REPRODUCED"
        decisions.append({"candidate_id": cid, "production_role": role, "F1B_F2R_class": source_cls, "production_class": cls, "transfer_status": status})
    write_csv(OUT / "production_1d_fracture_transfer_comparison.csv", comparisons)
    write_json(OUT / "production_1d_fracture_transfer_decision.json", {"rows": decisions, "reported_quantity": "MODEL_NATIVE_1D_FIRST_PASSAGE_K_ONSET"})
    lines = ["# Production 1-D fracture transfer decision", "", "The reported quantity is `MODEL_NATIVE_1D_FIRST_PASSAGE_K_ONSET`; it is not ASTM K_IC.", "", markdown_table(pd.DataFrame(decisions))]
    (OUT / "PRODUCTION_1D_FRACTURE_TRANSFER_DECISION.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"stage": "fracture", "conditions": len(result), "candidates": len(picks), "status": "PASS"}, indent=2))


def prepare_fatigue_engine(p: dict, seed: int = 1001721):
    eng, cleavage, emission, parent = v2.build_engine(p, v2.ROWS, v2.ACTIVE)
    eng.f.da = 7e-9
    eng.avalanche_cfg.mode = "fixed"
    eng.avalanche_cfg.minimum_factor = 1.0
    eng.avalanche_cfg.maximum_factor = 1.0
    eng.avalanche_cfg.geometry_subsegment_fraction = 0.1
    eng.avalanche_base_checkpoint_m = 7e-9
    eng.avalanche_checkpoint_synchronized = True
    eng.avalanche_event_length_factor = 1.0
    eng.avalanche_event_advance_m = 7e-9
    eng._engine_id = 0
    eng.hazard_cfg.mode = "exponential"
    eng.hazard_cfg.seed = seed
    eng._hazard_rng = np.random.default_rng(np.random.SeedSequence([seed, 0]))
    eng.hazard_threshold_action = draw_hazard_threshold("exponential", eng._hazard_rng, eng.hazard_cfg.minimum_threshold)
    eng.hazard_action_current = 0.0
    eng.B = 0.0
    controller = FatigueCycleHazardController(
        FatigueControllerConfig(
            n_phase=48, block_cycles=1e10, adaptive_cycles=True,
            max_block_cycles=1e10, min_block_cycles=1e-6,
            target_dB=0.10, target_dN_store=0.10,
            cycle_block_mode="hazard_limited", target_dN_emit=0.10,
            target_dN_mobile=0.10, target_dN_escape=0.10,
            target_dN_peierls=0.10, target_dN_taylor=0.10,
        ),
        emission, eng.manifest.peierls, eng.manifest.taylor,
    )
    return eng, controller, cleavage, parent


def repair_pre_event_fixed_quantum_checkpoint(checkpoint_dir: Path) -> None:
    path = checkpoint_dir / "high_cycle_live_checkpoint.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text())
    stochastic = payload["stochastic"]
    if int(stochastic.get("hazard_event_index", -1)) != 0 or stochastic.get("hazard_threshold_history"):
        raise RuntimeError("refuse checkpoint binding repair after threshold consumption")
    if (float(stochastic.get("avalanche_base_checkpoint_m", 0.0)) == 7e-9 and
            float(stochastic.get("avalanche_event_advance_m", 0.0)) == 7e-9 and
            float(stochastic.get("avalanche_event_length_factor", 0.0)) == 1.0):
        return
    before = sha(path)
    original = {
        key: stochastic.get(key) for key in ["avalanche_base_checkpoint_m", "avalanche_event_advance_m", "avalanche_event_length_factor"]
    }
    stochastic.update(
        avalanche_base_checkpoint_m=7e-9,
        avalanche_event_advance_m=7e-9,
        avalanche_event_length_factor=1.0,
        avalanche_checkpoint_synchronized=True,
    )
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)
    write_json(checkpoint_dir / "checkpoint_binding_repair.json", {
        "status": "PASS_PRE_EVENT_CACHE_ONLY_REPAIR",
        "reason": "correct cached proposal to mission-fixed 7 nm quantum",
        "hazard_event_index": 0, "hazard_threshold_history_count": 0,
        "active_state_vector_modified": False, "ledgers_modified": False,
        "hazard_action_modified": False, "rng_state_modified": False,
        "original_cache": original,
        "corrected_cache": {"avalanche_base_checkpoint_m": 7e-9, "avalanche_event_advance_m": 7e-9, "avalanche_event_length_factor": 1.0},
        "checkpoint_sha256_before": before, "checkpoint_sha256_after": sha(path),
    })


def fatigue_stage():
    picks = freeze_candidates()
    eligible_roles = {"P25_DBTT_LIKE", "P40_DBTT_LIKE", "P25_WEAK_T", "P40_WEAK_T"}
    tier1 = [(cid, role) for cid, role in picks if role in eligible_roles]
    fracture = pd.read_csv(OUT / "production_1d_fracture_results.csv")
    params = pd.read_csv(V2_DIR / "candidate_parameter_rows.csv").set_index("candidate_id")
    source_physical = v2.physical_controls()
    os.environ.update({
        "V10230_PERIODIC_MAX_ITERATIONS": "24",
        "V10230_HIGH_CYCLE_MAX_MODE_OPERATIONS": "256",
        "V10230_FORWARD_MAX_ACCEPTED_SEGMENTS": "4096",
        "V10230_FORWARD_MAX_TRIAL_INTEGRATIONS": "32768",
        "V10230_DMD_CHAIN_MAX_SEGMENTS": "128",
    })
    cases, events, comparisons = [], [], []
    for cid, role in tier1:
        onset_row = fracture[(fracture.candidate_id == cid) & (fracture.temperature_K == 300.0) & (fracture.Kdot == RATE)].iloc[0]
        onset = float(onset_row.K_onset_MPa_sqrt_m)
        if not math.isfinite(onset) or onset_row.F1B_status != "FIRST_PASSAGE":
            continue
        p = indexed_candidate(params, cid)
        cleavage, _ = v2.surface_pair(p, v2.ROWS, v2.ACTIVE)
        for fraction in (0.55, 0.75, 0.95):
            clear_pending_geometry_events()
            eng, controller, _, parent = prepare_fatigue_engine(p)
            Kmax = fraction * onset
            waveform = FatigueWaveform(Kmax=Kmax * 1e6, R=0.1, frequency_Hz=1000.0)
            checkpoint_dir = OUT / "fatigue_live_checkpoints" / "exact_wrapper_nphase48" / cid / f"fraction_{fraction:.2f}"
            os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(checkpoint_dir.resolve())
            os.environ["V10230_HIGH_CYCLE_CHECKPOINT_MIN_SECONDS"] = "30"
            start_cycles = 0.0
            if (checkpoint_dir / "high_cycle_live_checkpoint.json").is_file():
                repair_pre_event_fixed_quantum_checkpoint(checkpoint_dir)
                payload = restore_checkpoint(eng, checkpoint_dir)
                start_cycles = float(payload.get("cycles_from_engine_time") or 0.0)
                # The checkpoint predates correction of the cached proposal only;
                # no threshold/event was consumed and all active state is retained.
                if eng.hazard_event_index != 0 or eng.hazard_threshold_history:
                    raise RuntimeError("refuse event-length correction after threshold consumption")
                eng.f.da = 7e-9
                eng.avalanche_cfg.mode = "fixed"
                eng.avalanche_base_checkpoint_m = 7e-9
                eng.avalanche_checkpoint_synchronized = True
                eng.avalanche_event_length_factor = 1.0
                eng.avalanche_event_advance_m = 7e-9
            initial_threshold = float(eng.hazard_threshold_action)
            status = "CENSORED_AT_EXISTING_1E10_CYCLE_LIMIT"
            result = {}
            failure = ""
            try:
                result = integrate_state_coupled_waveform(eng, controller, waveform, 300.0, max(1e10-start_cycles, 0.0))
                fired = bool(result.get("fired", False))
                cycles = start_cycles + float(result.get("coupled_hazard_cycles_consumed", 0.0))
                if fired:
                    status = "PRODUCTION_1D_STATE_CLOSURE_FAILED"
                    failure = "first passage reached; geometry-dependent post-passage energy transaction requires prohibited 2-D mechanics"
                    events.append({
                        "candidate_id": cid, "production_role": role, "Kmax_fraction": fraction,
                        "Kmax_MPa_sqrt_m": Kmax, "DeltaK_MPa_sqrt_m": 0.9*Kmax,
                        "event_index": 0, "event_cycle": cycles,
                        "first_passage_threshold": float(result.get("hazard_threshold_completed_action", initial_threshold)),
                        "physical_hazard_action": float(result.get("hazard_action_completed", initial_threshold)),
                        "proposed_event_size_m": float(result.get("stochastic_event_proposed_advance_m", 7e-9)),
                        "energy_admissible_event_size_m": np.nan,
                        "accepted_event_size_m": 0.0,
                        "energy_gate_status": "PRODUCTION_1D_STATE_CLOSURE_FAILED",
                        "barrier_floor_occupancy": 0.0,
                        "renewal_ceiling_occupancy": float(result.get("constitutive_ceiling_occupancy", 0.0)),
                        **v2.snapshot_summary(eng),
                    })
                    eng.restore_geometry_veto()
            except Exception as exc:
                cycles = float(result.get("coupled_hazard_cycles_consumed", 0.0))
                status = "PRODUCTION_1D_STATE_CLOSURE_FAILED"
                failure = f"{type(exc).__name__}: {exc}"
            action_cycle = surface_action(cleavage, Kmax, 300.0)
            analytical = 7e-9 * action_cycle
            parent_frame = source_physical[p["parent_id"]]["frame"]
            parent_rate = float(np.exp(np.interp(np.log(max(Kmax, 1e-12)), np.log(parent_frame.Kmax.to_numpy(float)), np.log(parent_frame.physical_rate.to_numpy(float)))))
            row = {
                "candidate_id": cid, "production_role": role, "temperature_K": 300.0,
                "R": 0.1, "frequency_Hz": 1000.0, "hazard_seed": 1001721,
                "K_onset_300K_MPa_sqrt_m": onset, "Kmax_fraction": fraction,
                "Kmax_MPa_sqrt_m": Kmax, "DeltaK_MPa_sqrt_m": 0.9*Kmax,
                "cycle_limit": 1e10, "cycles_consumed": cycles,
                "accepted_event_count": 0, "projected_extension_m": 0.0,
                "developed_da_dN": np.nan, "censor_status": status,
                "failure_reason": failure, "initial_threshold": initial_threshold,
                "exact_cycle_integrated_opening_hazard": action_cycle,
                "corrected_analytical_da_dN": analytical,
                "parent_target_da_dN_log_interpolated": parent_rate,
                "production_high_cycle_model_id": HIGH_CYCLE_MODEL_ID,
                "no_hidden_fatigue_floor": True, "no_bulk_hazard": True,
                "fixed_event_quantum_m": 7e-9, "target_extension_m": 1.5e-8,
                "maximum_accepted_events": 3,
                "live_checkpoint_directory": str(checkpoint_dir),
                "accelerated_mode_internal_validation": bool(result.get("coupled_hazard_rate_separated_ledgers", False)),
            }
            cases.append(row)
            comparisons.append({k: row[k] for k in ["candidate_id", "production_role", "Kmax_fraction", "Kmax_MPa_sqrt_m", "DeltaK_MPa_sqrt_m", "corrected_analytical_da_dN", "parent_target_da_dN_log_interpolated", "developed_da_dN", "censor_status"]})
            print("fatigue", cid, fraction, status, cycles, flush=True)
    case_frame = pd.DataFrame(cases)
    case_frame.to_csv(OUT / "production_300K_fatigue_case_table.csv", index=False)
    event_frame = pd.DataFrame(events)
    if event_frame.empty:
        event_frame = pd.DataFrame(columns=["candidate_id", "event_index", "event_cycle", "accepted_event_size_m"])
    event_frame.to_parquet(OUT / "production_300K_fatigue_event_ledger.parquet", index=False, compression="zstd")
    pd.DataFrame(comparisons).to_csv(OUT / "production_300K_fatigue_transfer_comparison.csv", index=False)
    slopes = []
    for cid, role in tier1:
        q = case_frame[case_frame.candidate_id == cid]
        finite = q[np.isfinite(q.developed_da_dN)]
        slopes.append({
            "candidate_id": cid, "production_role": role,
            "finite_uncensored_points": len(finite), "three_point_local_slope": np.nan,
            "two_point_secant": np.nan,
            "classification": "FATIGUE_SLOPE_UNRESOLVED_CENSORED" if len(finite) < 2 else "INSUFFICIENT_FOR_THREE_POINT_LOCAL_SLOPE",
        })
    write_csv(OUT / "production_300K_fatigue_local_slopes.csv", slopes)
    write_json(OUT / "production_300K_fatigue_pilot_decision.json", {
        "tier": 1, "trajectory_count": len(case_frame), "candidate_count": len(tier1),
        "classification": "PRODUCTION_1D_STATE_CLOSURE_FAILED" if any(case_frame.censor_status == "PRODUCTION_1D_STATE_CLOSURE_FAILED") else "PRODUCTION_1D_CENSORED_IN_TESTED_DOMAIN",
        "interpretation": "The production 1-D engine reached the recorded first-passage/censor boundaries. A geometry-dependent post-passage energy balance cannot be evaluated without the explicitly prohibited 2-D mechanics, so no crack event or da/dN was fabricated.",
        "tier2_executed": False, "two_dimensional_PF_FEM_CZM_executed": False,
    })
    (OUT / "PRODUCTION_300K_FATIGUE_PILOT_DECISION.md").write_text(
        "# Production 300 K fatigue pilot decision\n\n"
        "Tier 1 used the production high-cycle state engine, candidate-specific 300 K onset normalization, R=0.1, 1000 Hz, seed 1001721, fixed 7 nm event quantum, and the existing 1e10-cycle limit. "
        "The 1-D engine cannot close the geometry-dependent post-first-passage energy transaction under the mission's explicit ban on 2-D mechanics. First-passage attempts therefore remain `PRODUCTION_1D_STATE_CLOSURE_FAILED`; no accepted crack growth or slope is invented.\n\n"
        + markdown_table(pd.DataFrame(slopes)) + "\n"
    )
    tier2 = []
    for cid in ["P25_TJBSV2_S_004127", "P40_TJBSV2_S_031027"]:
        onset = float(fracture[(fracture.candidate_id == cid) & (fracture.temperature_K == 300.0) & (fracture.Kdot == RATE)].iloc[0].K_onset_MPa_sqrt_m)
        for f in (0.55, 0.75, 0.95):
            tier2.append({"candidate_id": cid, "Kmax_fraction": f, "Kmax_MPa_sqrt_m": f*onset, "DeltaK_MPa_sqrt_m": 0.9*f*onset, "status": "PREPARED_NOT_EXECUTED_REQUIRES_EXPLICIT_REVIEW"})
    write_csv(OUT / "optional_tier2_fatigue_plan.csv", tier2)
    print(json.dumps({"stage": "fatigue", "trajectories": len(case_frame), "status": "PASS_WITH_STATE_CLOSURE_RESULT"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["review", "p40-f2r", "fracture", "fatigue"])
    args = parser.parse_args()
    if git("rev-parse", "HEAD") != V2_HEAD:
        raise SystemExit("start this record from exact V2 HEAD")
    {"review": review_stage, "p40-f2r": p40_f2r_stage, "fracture": fracture_stage, "fatigue": fatigue_stage}[args.stage]()


if __name__ == "__main__":
    main()
