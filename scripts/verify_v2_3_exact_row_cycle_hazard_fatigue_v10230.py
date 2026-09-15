#!/usr/bin/env python3
"""Strict verifier for the V2.3 exact-row cycle-hazard fatigue mission."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from arrhenius_fracture.canonical_v2_registry_v10230 import load_row


OUT = ROOT / "analysis_outputs/v2_3_exact_row_cycle_hazard_fatigue"
PARENT = "f3ae18503c243b2afa6f5af7f323c69c6aa681ca"
SELECTION = "0b986300bffb12bb2aa020541e05dc1c8ecde66a"
EXECUTION = "40a9e37dc75497597ac178445e77f71ad236e688"
EXECUTION_TAG = "v10230-v2-3-exact-row-execution"
CANDIDATE = "P25_TJBSV2_S_002987"
ROW_SHA = "8a16415705c47c7708ec35385cde6c9a53bb6b9de6edf98b902a1f40626a9f72"
LIMIT = "NOT_EXPERIMENTALLY_CALIBRATED_NOT_ASTMR_CURVE"
ROOT_ARTIFACTS = (
    "V2_2_ACCEPTED_INTERPRETATION_ADDENDUM.md",
    "V2_3_EXACT_ROW_CYCLE_HAZARD_FATIGUE_PROTOCOL.md",
    "v2_3_exact_row_cycle_hazard_fatigue_protocol.json",
)
REQUIRED = (
    "CYCLE_HAZARD_LOAD_SELECTION_DECISION.md",
    "selected_exact_row_fatigue_loads.json",
    "cycle_hazard_load_selection_grid.csv",
    "EXACT_ROW_300K_FATIGUE_DECISION.md",
    "V2_3_FINAL_DECISION.md",
    "exact_row_300K_fatigue_case_table.csv",
    "exact_row_300K_fatigue_event_ledger.parquet",
    "exact_row_300K_fatigue_local_slopes.csv",
    "accelerated_explicit_overlap_parity.csv",
    "accelerated_explicit_overlap_decision.json",
    "raw_run_source_manifest.json",
    "rising_resistance_mechanism_attribution.csv",
    "rising_resistance_mechanism_decision.json",
    "prospective_cross_code_registry_additions.json",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def verify_archive() -> None:
    manifest_path = OUT / "SHA256_MANIFEST.json"
    archive_path = OUT / "Archive_V2_3_EXACT_ROW_CYCLE_HAZARD_FATIGUE.zip"
    assert manifest_path.is_file() and archive_path.is_file()
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema"] == "v10.2.30_v2_3_complete_artifact_manifest_v1"
    files = manifest["files"]
    assert files and "figures/05_accelerated_explicit_parity.png" in files
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

    for commit in (PARENT, SELECTION, EXECUTION):
        assert subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"], cwd=ROOT
        ).returncode == 0
    assert git("rev-parse", EXECUTION_TAG) == EXECUTION

    protocol = json.loads((ROOT / ROOT_ARTIFACTS[2]).read_text())
    assert protocol["immutable_parent_head"] == PARENT
    assert protocol["candidate_id"] == CANDIDATE
    assert protocol["temperature_K"] == 300.0 and protocol["R"] == 0.1
    assert protocol["frequency_Hz"] == 1000.0
    assert protocol["physical_trajectory"]["seed"] == 1720
    assert protocol["physical_trajectory"]["target_extension_m"] == 25e-6
    assert protocol["protocol_frozen_before_physical_execution"]
    assert not protocol["barrier_refit_or_retune"]
    assert not protocol["canonical_production_promotion_authorized"]
    assert protocol["additive_prospective_registry_authorized"]
    assert not protocol["spatial_or_two_dimensional_execution_authorized"]

    selection = json.loads((OUT / "selected_exact_row_fatigue_loads.json").read_text())
    assert selection["candidate_id"] == CANDIDATE
    assert selection["complete_bound_row_sha256"] == ROW_SHA
    assert not selection["barrier_or_renewal_retuned"]
    assert not selection["monotonic_ramp_cap_applied"]
    loads = selection["selected_loads_in_ascending_Kmax"]
    assert [x["Kmax_MPa_sqrt_m"] for x in loads] == [3.75, 6.0, 12.75]
    assert [x["target_median_first_passage_cycles"] for x in loads] == [10000.0, 1000.0, 100.0]
    assert np.allclose(
        [x["projected_median_first_passage_cycles"] for x in loads],
        [6662.204680547703, 983.2837387406177, 102.69435730638095], rtol=0, atol=1e-12,
    )

    raw = json.loads((OUT / "raw_run_source_manifest.json").read_text())
    assert raw["schema"] == "v10.2.30_v2_3_raw_run_source_manifest_v1"
    assert not raw["raw_run_outputs_added_to_git"] and len(raw["files"]) == 4
    for record in raw["files"]:
        path = ROOT / record["path"]
        checkpoint = ROOT / record["checkpoint_path"]
        assert path.is_file() and checkpoint.is_file()
        assert sha(path) == record["sha256"]
        assert sha(checkpoint) == record["checkpoint_sha256"]

    cases = pd.read_csv(OUT / "exact_row_300K_fatigue_case_table.csv")
    assert len(cases) == 4 and set(cases.candidate_id) == {CANDIDATE}
    assert (cases.status == "growth_target_reached").all()
    assert (cases.accepted_event_count == 6).all()
    assert np.allclose(cases.final_extension_m, 28.888107103674025e-6, rtol=0, atol=1e-18)
    assert (cases.developed_rate_status == "UNRESOLVED_ONLY_ONE_POINT_BEYOND_20UM").all()
    assert cases.developed_da_dN_m_per_cycle.isna().all()
    production = cases[cases.integration_role == "QUALIFIED_ACCELERATED_OR_EXACT"].sort_values("Kmax_MPa_sqrt_m")
    explicit = cases[cases.integration_role == "EXPLICIT_CYCLE_OVERLAP"]
    assert len(production) == 3 and len(explicit) == 1
    assert production.Kmax_MPa_sqrt_m.tolist() == [3.75, 6.0, 12.75]
    assert np.allclose(production.final_cycles, [48219.066543949215, 7134.826281993995, 1392.3046484622487], rtol=0, atol=1e-9)
    expected_rates = [5.991013342687585e-10, 4.048887241526582e-9, 2.074840957801866e-8]
    assert np.allclose(production.cumulative_short_growth_da_dN_m_per_cycle, expected_rates, rtol=1e-14, atol=0)
    assert np.all(np.diff(production.cumulative_short_growth_da_dN_m_per_cycle) > 0)
    assert np.isclose(explicit.iloc[0].final_cycles, 7140.415348786115, rtol=0, atol=1e-9)
    assert explicit.iloc[0].dmd_accepted_segments == 0

    events = pd.read_parquet(OUT / "exact_row_300K_fatigue_event_ledger.parquet")
    assert len(events) == 24 and set(events.event_index) == set(range(6))
    assert np.allclose(events.threshold_action, events.physical_hazard_action, rtol=0, atol=2e-12)
    reference = events[(events.Kmax_MPa_sqrt_m == 3.75)].sort_values("event_index")
    for _, group in events.groupby(["Kmax_MPa_sqrt_m", "integration_role"]):
        group = group.sort_values("event_index")
        assert np.array_equal(group.threshold_action.to_numpy(), reference.threshold_action.to_numpy())
        assert np.array_equal(group.committed_advance_m.to_numpy(), reference.committed_advance_m.to_numpy())
        assert np.all(np.diff(group.cycles) > 0) and np.all(np.diff(group.cumulative_extension_m) > 0)

    slopes = pd.read_csv(OUT / "exact_row_300K_fatigue_local_slopes.csv")
    assert len(slopes) == 2 and slopes.not_a_broad_Paris_exponent.all()
    assert np.allclose(slopes.local_two_point_log_slope, [4.06542948195295, 2.167801320714687], rtol=1e-12)
    assert abs(float(np.diff(slopes.local_two_point_log_slope)[0])) > 1.0

    parity = json.loads((OUT / "accelerated_explicit_overlap_decision.json").read_text())
    assert parity["status"] == "PASS_INHERITED_1E3_RELATIVE_CYCLE_TOLERANCE"
    assert parity["threshold_sequence_exact"] and parity["event_size_sequence_exact"]
    assert parity["terminal_cycle_relative_difference"] <= 1e-3
    parity_table = pd.read_csv(OUT / "accelerated_explicit_overlap_parity.csv")
    assert len(parity_table) == 6 and parity_table.threshold_exact.all() and parity_table.event_size_exact.all()

    mechanism = json.loads((OUT / "rising_resistance_mechanism_decision.json").read_text())
    assert mechanism["candidate_id"] == CANDIDATE
    assert mechanism["classification"] == "PROGRESSIVE_EFFECTIVE_RADIUS_BLUNTING_TRANSFER_DOMINANT"
    assert mechanism["K_ratio_5um_to_100um"] > 1.7
    assert mechanism["radius_ratio_5um_to_100um"] > 3.0
    assert abs(mechanism["residual_K_over_sqrt_radius_ratio"] - 1.0) < 0.02
    assert not mechanism["ASTM_or_conventional_R_curve_established"]

    registry = json.loads((OUT / "prospective_cross_code_registry_additions.json").read_text())
    assert registry["schema"] == "v10.2.30_additive_prospective_cross_code_registry_v1"
    assert not registry["existing_canonical_rows_replaced"]
    assert registry["canonical_production_rows_added"] == 0 and len(registry["additions"]) == 1
    addition = registry["additions"][0]
    assert addition["status"] == "PROSPECTIVE_CROSS_CODE_REGISTRY_ROW"
    assert addition["canonical_production_promotion"] == "WITHHELD"
    assert addition["parameter_row_sha256"] == ROW_SHA
    assert addition["full_parameter_row"] == load_row(CANDIDATE)
    assert addition["qualification"] == LIMIT
    assert addition["fatigue_metrics"]["developed_rate_status"] == "UNRESOLVED_ONLY_ONE_POINT_BEYOND_20UM"

    assert len(list((OUT / "figures").glob("*.png"))) == 5
    assert len(list((OUT / "figure_source_data").glob("*.csv"))) == 5
    decision = (OUT / "V2_3_FINAL_DECISION.md").read_text()
    for phrase in ("progressive effective-radius/blunting transfer", "P40 remains unresolved",
                   "Canonical production promotion remains withheld", "No spatial or two-dimensional"):
        assert phrase in decision
    if not args.allow_unsealed:
        verify_archive()
    print(json.dumps({
        "status": "PASS",
        "verifier": "STRICT_V2_3_EXACT_ROW_CYCLE_HAZARD_FATIGUE",
        "physical_trajectories": 4,
        "production_loads": 3,
        "complete_event_records": 24,
        "developed_rates_resolved": 0,
        "prospective_registry_additions": 1,
        "canonical_production_rows_added": 0,
        "two_dimensional_runs": 0,
        "sealed": not args.allow_unsealed,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
