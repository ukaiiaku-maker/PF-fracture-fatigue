#!/usr/bin/env python3
"""Strict verifier for the bounded V2.2 reduced 1-D response mission."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from arrhenius_fracture.canonical_v2_registry_v10230 import load_rows


OUT = ROOT / "analysis_outputs/v2_2_1d_rising_resistance_and_fatigue"
PARENT = "7bb4a341b1c5370c826a3a6bc0253c042fd2e6fc"
PARENT_TREE = "b44418fd428e13c993ce8f5954e036411c3dc660"
PROTOCOL_COMMIT = "37f41c4e"
XI = math.log(2.0)
M_VALUE = 3.2732414351776242
TAU_S = 6.992153587194454e-7
LIMIT = "NOT_EXPERIMENTALLY_CALIBRATED_NOT_ASTMR_CURVE"
ROOT_ARTIFACTS = (
    "ONE_D_RISING_RESISTANCE_PROTOCOL.md",
    "one_d_rising_resistance_protocol.json",
    "fatigue_entrypoint_capability_audit.csv",
)
REQUIRED = (
    "preflight_and_source_diff_inventory.json", "parent_case_reproduction.json",
    "screen_terminal_records.json", "one_d_100um_continuation.json",
    "one_d_reload_separated_resistance_points.csv",
    "one_d_reload_separated_state_table.parquet",
    "one_d_rising_resistance_classification.csv",
    "one_d_full_vs_opening_only_ablation.csv", "ONE_D_RISING_RESISTANCE_DECISION.md",
    "fatigue_K_coordinate_audit.csv", "FATIGUE_K_COORDINATE_DECISION.md",
    "high_stress_fatigue_case_table.csv", "high_stress_fatigue_event_ledger.parquet",
    "high_stress_fatigue_rates_and_slopes.csv", "HIGH_STRESS_FATIGUE_DECISION.md",
    "canonical_candidate_promotion_table.csv", "canonical_parameter_registry_additions.json",
    "cross_code_registry_parity.json", "V2_2_1D_RISING_RESISTANCE_AND_FATIGUE_DECISION.md",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_archive() -> None:
    manifest_path = OUT / "SHA256_MANIFEST.json"
    archive_path = OUT / "Archive_V2_2_1D_RISING_RESISTANCE_AND_FATIGUE.zip"
    assert manifest_path.is_file() and archive_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "v10.2.30_v2_2_complete_artifact_manifest_v1"
    files = manifest["files"]
    assert files and "figures/01_reload_K_all_rows.png" in files
    for name, expected in files.items():
        local = ROOT / name if name in ROOT_ARTIFACTS else OUT / name
        assert local.is_file() and sha(local) == expected, name
    with zipfile.ZipFile(archive_path) as archive:
        assert set(files) | {manifest_path.name} == set(archive.namelist())
        for name, expected in files.items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
        assert archive.read(manifest_path.name) == manifest_path.read_bytes()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-unsealed", action="store_true")
    args = parser.parse_args()
    missing = [name for name in REQUIRED if not (OUT / name).is_file()]
    missing += [name for name in ROOT_ARTIFACTS if not (ROOT / name).is_file()]
    assert not missing, f"missing outputs: {missing}"

    assert git("rev-parse", f"{PARENT}^{{tree}}") == PARENT_TREE
    assert subprocess.run(["git", "merge-base", "--is-ancestor", PARENT, "HEAD"], cwd=ROOT).returncode == 0
    assert subprocess.run(["git", "merge-base", "--is-ancestor", PROTOCOL_COMMIT, "HEAD"], cwd=ROOT).returncode == 0
    protocol = json.loads((ROOT / "one_d_rising_resistance_protocol.json").read_text())
    assert protocol["immutable_parent_head"] == PARENT
    assert protocol["protocol_frozen_before_physical_execution"]
    assert protocol["observable"] == "1D_RELOAD_SEPARATED_EFFECTIVE_RESISTANCE"
    assert protocol["temperature_K"] == 900.0 and protocol["Kdot_MPa_sqrt_m_s"] == 0.005
    assert protocol["threshold_action"] == XI and protocol["event_advance_m"] == 5e-6
    assert not protocol["spatial_execution_authorized"] and not protocol["barrier_refit_or_retune"]

    preflight = json.loads((OUT / "preflight_and_source_diff_inventory.json").read_text())
    p = preflight["preflight"]
    assert preflight["immutable_parent_head"] == PARENT and preflight["immutable_parent_tree"] == PARENT_TREE
    assert p["initial_worktree_clean"] and p["related_workers"] == 0
    assert p["strict_v2_1_verifier"] == p["strict_inherited_v2_verifier"] == "PASS"
    assert p["v2_1_archive_sha256"] == "99deb87263fb888b4512f8041f496b86b95d9ef52ca938521eae88d92aa670ae"
    assert p["frozen_rows_loaded"] == 8 and p["frozen_rows_byte_fields_round_trip_exact"]
    reproduction = json.loads((OUT / "parent_case_reproduction.json").read_text())
    assert reproduction["status"] == "PASS"
    assert reproduction["monotonic"]["comparison"] == "PASS_ORIGINAL_SENTINEL_TOLERANCES"
    assert reproduction["fatigue"]["comparison"] == "PASS_EXACT"
    assert reproduction["fatigue"]["scientific_and_state_field_difference_count"] == 0

    rows = load_rows()
    assert len(rows) == 8
    terminals = json.loads((OUT / "screen_terminal_records.json").read_text())["rows"]
    assert len(terminals) == 16
    assert {(x["candidate_id"], x["mode"]) for x in terminals} == {
        (cid, mode) for cid in rows for mode in protocol["modes"]}
    assert all(x["status"] in {"COMPLETE", "K_CENSOR_REACHED"} for x in terminals)
    assert all(abs(x["exact_renewal_m"] - M_VALUE) < 1e-14 for x in terminals)
    assert all(abs(x["exact_renewal_tau_s"] - TAU_S) < 1e-20 for x in terminals)
    assert max(x["maximum_conservation_relative"] for x in terminals) < 1e-7
    censored = [x for x in terminals if x["status"] == "K_CENSOR_REACHED"]
    assert len(censored) == 1 and censored[0]["candidate_id"] == "P40_TJBSV2_S_038503"
    assert censored[0]["mode"] == "FULL_PRODUCTION_STATE" and censored[0]["events"] == 1

    points = pd.read_csv(OUT / "one_d_reload_separated_resistance_points.csv")
    required_state = {"emitted_total", "mobile", "retained", "wake_mobile", "wake_retained",
                      "escaped", "recovered", "radius_m", "backstress_Pa",
                      "active_shield_Pa_sqrt_m", "wake_shield_Pa_sqrt_m",
                      "total_shield_Pa_sqrt_m", "source_multiplicity", "physical_hazard_action",
                      "loading_time_s", "constitutive_ceiling_occupied", "conservation_relative"}
    assert required_state <= set(points)
    assert len(points) == 191 and points.candidate_id.nunique() == 8
    assert np.isfinite(points.K_reinit_MPa_sqrt_m).all()
    assert np.allclose(points.threshold_action, XI, rtol=0, atol=1e-14)
    assert np.allclose(points.physical_hazard_action, XI, rtol=0, atol=2e-12)
    assert (points.conservation_relative < 1e-7).all()
    assert not points[["constitutive_ceiling_occupied", "sigma_cap_active", "dN_cap_active", "N_sat_active"]].any().any()
    states = pd.read_parquet(OUT / "one_d_reload_separated_state_table.parquet")
    assert len(states) == 382 and set(states.state_position) == {"PRE_RELOAD", "POST_EVENT"}
    assert required_state - {"physical_hazard_action", "loading_time_s", "constitutive_ceiling_occupied", "conservation_relative"} <= set(states)
    assert np.isfinite(states[["mobile", "retained", "escaped", "recovered", "radius_m", "backstress_Pa"]]).all().all()

    classes = pd.read_csv(OUT / "one_d_rising_resistance_classification.csv")
    assert len(classes) == 8 and classes.candidate_id.nunique() == 8
    strong = classes[classes.classification == "1D_STRONG_RISING_RESISTANCE"]
    assert list(strong.candidate_id) == ["P25_TJBSV2_S_002987"]
    s = strong.iloc[0]
    assert s.finite_points == 10 and s.spearman_rho >= 0.8 and s.nonnegative_successive_increments >= 7
    assert s.normalized_rise >= 0.2 and s.state_mediated_fraction >= 0.5
    assert s.theil_sen_slope_MPa_sqrt_m_per_um > 0 and not bool(s.numerical_or_constitutive_pathology)
    assert (classes.classification == "1D_EFFECTIVE_RESISTANCE_FLAT").sum() == 6
    assert (classes.classification == "1D_EFFECTIVE_RESISTANCE_UNRESOLVED").sum() == 1
    continuation = json.loads((OUT / "one_d_100um_continuation.json").read_text())
    assert continuation["status"] == "PASS" and continuation["checkpoint_restart_parity"] == "PASS_EXACT"
    assert continuation["selected_by_frozen_50um_metrics"] == ["P25_TJBSV2_S_002987", "P40_TJBSV2_S_043821"]
    for cid in continuation["selected_by_frozen_50um_metrics"]:
        assert set(points[(points.candidate_id == cid) & (points.cumulative_advance_um == 100)]["mode"]) == set(protocol["modes"])
    ablation = pd.read_csv(OUT / "one_d_full_vs_opening_only_ablation.csv")
    assert len(ablation) == 71 and np.isfinite(ablation.state_mediated_K_MPa_sqrt_m).all()

    coordinate = pd.read_csv(OUT / "fatigue_K_coordinate_audit.csv")
    assert len(coordinate) == 3 and (coordinate.qualification == "COMMON_FATIGUE_AND_MONOTONIC_K_COORDINATE_QUALIFIED").all()
    fatigue_cases = pd.read_csv(OUT / "high_stress_fatigue_case_table.csv")
    assert len(fatigue_cases) == 1
    f = fatigue_cases.iloc[0]
    assert f.candidate_id == "P25_TJBSV2_S_002987"
    assert np.isclose(f.K_upper_MPa_sqrt_m, 0.9 * f.K_fracture_300_MPa_sqrt_m)
    assert f.K_prior_high_MPa_sqrt_m >= f.K_upper_MPa_sqrt_m
    assert f.status == "NO_UNSAMPLED_SUBFRACTURE_FATIGUE_INTERVAL" and not bool(f.new_physical_trajectory_launched)
    fatigue_ledger = pd.read_parquet(OUT / "high_stress_fatigue_event_ledger.parquet")
    assert len(fatigue_ledger) == 0
    rates = pd.read_csv(OUT / "high_stress_fatigue_rates_and_slopes.csv")
    assert len(rates) >= 3 and not rates.same_exact_candidate_row.any()
    assert (rates.point_role == "PRIOR_QUALIFIED_PARENT_ANCHOR").all()
    assert (rates.high_K_classification == "HIGH_K_FATIGUE_RESPONSE_UNRESOLVED_CENSORED").all()

    promotion = pd.read_csv(OUT / "canonical_candidate_promotion_table.csv")
    assert len(promotion) == 8 and not promotion.two_finite_exact_row_high_stress_fatigue_points.any()
    assert (promotion.promotion_status == "NOT_PROMOTED_MISSING_EXACT_ROW_HIGH_STRESS_FATIGUE").all()
    assert (promotion.permanent_qualification == LIMIT).all()
    additions = json.loads((OUT / "canonical_parameter_registry_additions.json").read_text())
    assert additions["additions"] == [] and not additions["existing_canonical_rows_replaced"]
    parity = json.loads((OUT / "cross_code_registry_parity.json").read_text())
    assert parity["status"] == "PASS" and len(parity["rows"]) == 8
    assert not parity["generic_defaults_fill_canonical_fields"]
    assert all(row["status"] == "PASS" and row["unknown_aliases_fail_closed"] for row in parity["rows"])

    assert len(list((OUT / "figures").glob("*.png"))) == 10
    assert len(list((OUT / "figure_source_data").glob("*.csv"))) == 10
    decision = (OUT / "V2_2_1D_RISING_RESISTANCE_AND_FATIGUE_DECISION.md").read_text()
    assert "NOT_EXPERIMENTALLY_CALIBRATED_NOT_ASTMR_CURVE" in decision
    assert "No two-dimensional or spatial energy-gated calculation was launched" in decision
    if not args.allow_unsealed:
        verify_archive()
    result = {"status": "PASS", "verifier": "STRICT_V2_2_1D_RISING_RESISTANCE_AND_FATIGUE",
              "canonical_rows": 8, "screen_terminal_records": 16,
              "strong_rising_rows": 1, "new_fatigue_trajectories": 0,
              "registry_additions": 0, "figures": 10, "two_dimensional_runs": 0,
              "sealed": not args.allow_unsealed}
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
