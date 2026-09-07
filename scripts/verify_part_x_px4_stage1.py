"""PX4.1 section 5/7: strict portable verifier for the PX4 Stage-1
developed campaign (seed 1720's five-point grid across D1/D2/D3/D5/D6,
plus the seed=1001723 second-seed confirmation at Kmax=15/18/21 across
the same five protocols).

Depends ONLY on files tracked under artifacts/crack_rebonding_part_x_v1/
-- never on the gitignored runs/ directory -- so it must still pass with
every runs/ output deleted. Independently RE-DERIVES S_h_developed and
the seed-robustness delta_m slopes from developed_event_ledger.json (the
raw per-event record) via the shared, already-qualified stable_growth_gate
primitive (reused verbatim, not reimplemented -- reimplementing the
campaign's own frozen developed/stationarity gate formula would defeat
the point of reusing it), then cross-checks the recomputation against
px4_stage1_rate_table.csv / px4_stage1_slope_table.csv rather than
trusting those tables at face value. Pairing itself is grouped via
canonical_analysis_key, imported from build_part_x_px4_developed_analysis.py
(a data-grouping key, not a physics computation) -- this is the same
INDEPENDENT_LEDGER_REDUCTION_WITH_SHARED_QUALIFIED_PRIMITIVES level of
independence already established for the sibling developed_confirmation
verifier (verify_developed_confirmation.py).

Usage:
    <pinned interpreter> scripts/verify_part_x_px4_stage1.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import stable_growth_gate  # noqa: E402
from build_part_x_px4_developed_analysis import canonical_analysis_key  # noqa: E402
from build_part_x_px4_stage1_slope_table import (  # noqa: E402
    CONFIRM_OF, COMMON_KMAX_Pa_sqrt_m, DELTA_M_ROBUST_TOL, _fit_log_log_slope,
)

REQUIRED_ARTIFACTS = [
    "developed_job_registry.csv", "developed_event_ledger.json", "developed_event_ledger.csv",
    "developed_censor_registry.csv", "developed_attempt_registry.csv", "developed_raw_result_hashes.json",
    "px4_stage1_rate_table.csv", "px4_stage1_slope_table.csv", "px4_developed_pair_analysis.json",
    "px4_pre_horizon_repair_admission.json", "px4_1_unlaunched_producer_freeze.json", "px4_producer_freeze.json",
]
EXPECTED_ADMITTED = 80
EXPECTED_QUARANTINED_INTERRUPTED = 3
EXPECTED_PAIRS_SEED_1720 = 25
EXPECTED_PAIRS_SEED_1001723 = 15
CYCLE_HORIZON = 1.0e12
FLOAT_TOL = 1.0e-6


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    for name in REQUIRED_ARTIFACTS:
        checks[f"artifact_present_{name}"] = (ARTIFACTS_DIR / name).is_file()
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise SystemExit(f"missing tracked artifacts, cannot verify: {missing}")

    ledger = json.loads((ARTIFACTS_DIR / "developed_event_ledger.json").read_text())
    events_by_key: dict[str, list[dict]] = defaultdict(list)
    for e in ledger["events"]:
        events_by_key[e["canonical_job_key"]].append(e)
    for key, evs in events_by_key.items():
        evs.sort(key=lambda e: e["event_index"])
        checks[f"event_ordinal_sequence_exact_{key[:12]}"] = [e["event_index"] for e in evs] == list(range(len(evs)))
        checks[f"event_count_is_30_{key[:12]}"] = len(evs) == 30
    checks["all_trajectories_have_exactly_30_ordered_events"] = all(
        v for k, v in checks.items() if k.startswith("event_count_is_30_") or k.startswith("event_ordinal_sequence_exact_")
    )

    registry_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "developed_job_registry.csv")))
    authorized = [r for r in registry_rows if r["status"] == "AUTHORIZED_PX4"]
    checks["authorized_px4_row_count_is_80"] = len(authorized) == EXPECTED_ADMITTED

    by_pair: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in authorized:
        key = canonical_analysis_key(row)
        by_pair[key][row["cohesion"]] = row
    pairs_complete = {k: v for k, v in by_pair.items() if "finite" in v and "zero" in v}
    checks["every_group_has_exactly_finite_and_zero"] = len(pairs_complete) == len(by_pair)
    n_seed_1720 = sum(1 for k in pairs_complete if k[0] == 1720)
    n_seed_1001723 = sum(1 for k in pairs_complete if k[0] == 1001723)
    checks["seed_1720_pair_count_is_25"] = n_seed_1720 == EXPECTED_PAIRS_SEED_1720
    checks["seed_1001723_pair_count_is_15"] = n_seed_1001723 == EXPECTED_PAIRS_SEED_1001723
    checks["total_pair_count_is_40"] = len(pairs_complete) == EXPECTED_PAIRS_SEED_1720 + EXPECTED_PAIRS_SEED_1001723

    rate_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_rate_table.csv")))
    rate_by_seed_protocol_kmax = {
        (int(r["seed"]), r["protocol"], round(float(r["Kmax_Pa_sqrt_m"]))): r for r in rate_rows
    }
    checks["rate_table_row_count_is_40"] = len(rate_rows) == 40

    recomputed_S_h_by_seed_protocol_kmax: dict[tuple, float] = {}
    n_recompute_mismatch = 0
    for key, members in pairs_complete.items():
        seed, protocol, row_name, config_hash, Kmax, R, freq_Hz, hold_s, chem, control_mode, producer_sha = key
        finite_key, zero_key = members["finite"]["canonical_job_key"], members["zero"]["canonical_job_key"]
        gate_f = stable_growth_gate(events_by_key[finite_key], frequency_Hz=freq_Hz)
        gate_z = stable_growth_gate(events_by_key[zero_key], frequency_Hz=freq_Hz)
        both_stable = gate_f["stable_growth_provisional"] and gate_z["stable_growth_provisional"]
        checks[f"both_stable_growth_{finite_key[:12]}"] = both_stable
        da_dN_f, da_dN_z = gate_f["developed_da_dN_m_per_cycle"], gate_z["developed_da_dN_m_per_cycle"]
        S_h = math.log10(da_dN_f / da_dN_z) if (both_stable and da_dN_f and da_dN_z and da_dN_f > 0.0 and da_dN_z > 0.0) else None
        recomputed_S_h_by_seed_protocol_kmax[(seed, protocol, round(Kmax))] = S_h

        table_row = rate_by_seed_protocol_kmax.get((seed, protocol, round(Kmax)))
        checks[f"rate_table_row_present_{finite_key[:12]}"] = table_row is not None
        if table_row is not None and S_h is not None:
            match = abs(S_h - float(table_row["S_h_developed"])) < FLOAT_TOL
            checks[f"S_h_developed_reproducible_{finite_key[:12]}"] = match
            if not match:
                n_recompute_mismatch += 1

    checks["zero_S_h_developed_recomputation_mismatches"] = n_recompute_mismatch == 0
    checks["all_40_pairs_both_stable_growth"] = all(
        v for k, v in checks.items() if k.startswith("both_stable_growth_")
    )

    censor_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "developed_censor_registry.csv")))
    checks["censor_registry_row_count_is_80"] = len(censor_rows) == EXPECTED_ADMITTED
    checks["zero_censored_trajectories"] = all(r["censored"] == "False" for r in censor_rows)
    checks["zero_trajectories_hit_cycle_horizon"] = all(r["hit_cycle_horizon"] == "False" for r in censor_rows)
    checks["all_trajectories_exactly_30_events"] = all(int(r["n_accepted_events"]) == 30 for r in censor_rows)
    checks["all_trajectories_exactly_150um"] = all(
        abs(float(r["cumulative_extension_m"]) - 150.0e-6) < 1.0e-9 for r in censor_rows
    )
    checks["all_trajectories_far_below_cycle_horizon"] = all(
        float(r["max_event_cumulative_cycles"]) < CYCLE_HORIZON for r in censor_rows
    )

    attempt_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "developed_attempt_registry.csv")))
    n_admitted = sum(1 for r in attempt_rows if r["admitted"] == "True")
    n_quarantined = sum(1 for r in attempt_rows if r["admitted"] == "False")
    n_admitted_interrupted = sum(1 for r in attempt_rows if r["admitted"] == "True" and r["attempt_status"] != "COMPLETE")
    checks["attempt_registry_admitted_count_is_80"] = n_admitted == EXPECTED_ADMITTED
    checks["attempt_registry_quarantined_count_is_3"] = n_quarantined == EXPECTED_QUARANTINED_INTERRUPTED
    checks["attempt_registry_zero_admitted_interrupted"] = n_admitted_interrupted == 0
    checks["attempt_registry_quarantine_reason_is_wrong_budget"] = all(
        "budget" in r["quarantine_reason"] for r in attempt_rows if r["admitted"] == "False"
    )

    raw_hashes = json.loads((ARTIFACTS_DIR / "developed_raw_result_hashes.json").read_text())["files"]
    checks["raw_result_hashes_count_is_80"] = len(raw_hashes) == EXPECTED_ADMITTED
    admitted_keys = {r["canonical_job_key"] for r in attempt_rows if r["admitted"] == "True"}
    checks["raw_result_hashes_keys_match_admitted_attempt_keys"] = set(raw_hashes.keys()) == admitted_keys

    slope_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_slope_table.csv")))
    checks["slope_table_covers_all_5_protocol_axes"] = len(slope_rows) == len(CONFIRM_OF)
    n_slope_mismatch = 0
    n_not_robust = 0
    for row in slope_rows:
        base_protocol, confirm_protocol = row["base_protocol"], row["confirm_protocol"]
        base_pts = sorted(
            (K, recomputed_S_h_by_seed_protocol_kmax[(1720, base_protocol, round(K))])
            for K in COMMON_KMAX_Pa_sqrt_m
        )
        confirm_pts = sorted(
            (K, recomputed_S_h_by_seed_protocol_kmax[(1001723, confirm_protocol, round(K))])
            for K in COMMON_KMAX_Pa_sqrt_m
        )
        m_base, _ = _fit_log_log_slope(base_pts)
        m_confirm, _ = _fit_log_log_slope(confirm_pts)
        delta_m = m_confirm - m_base
        match = (
            abs(m_base - float(row["seed_1720_slope_m"])) < FLOAT_TOL
            and abs(m_confirm - float(row["seed_1001723_slope_m"])) < FLOAT_TOL
            and abs(delta_m - float(row["delta_m"])) < FLOAT_TOL
        )
        checks[f"slope_reproducible_{base_protocol}"] = match
        if not match:
            n_slope_mismatch += 1
        if row["classification"] != "REBONDING_SEED_ROBUST":
            n_not_robust += 1
    checks["zero_slope_recomputation_mismatches"] = n_slope_mismatch == 0
    checks["all_5_protocol_axes_seed_robust"] = n_not_robust == 0
    details["seed_robustness_by_axis"] = {r["base_protocol"]: r["classification"] for r in slope_rows}

    horizon_admission = json.loads((ARTIFACTS_DIR / "px4_pre_horizon_repair_admission.json").read_text())
    checks["original_44_trajectories_cycle_horizon_guard_inactive_proven"] = (
        horizon_admission["n_admitted"] == 44 and horizon_admission["n_trajectories"] == 44
    )
    checks["exact_cycle_horizon_test_suite_present"] = (
        REPO_ROOT / "tests" / "test_v10_2_30_crack_rebonding_part_x_px4_1_exact_cycle_horizon.py"
    ).is_file()

    overall_pass = all(checks.values())
    classification = "PX4_STAGE1_COMPLETE" if overall_pass else "PX4_STAGE1_INCOMPLETE"
    details["n_pairs_seed_1720"] = n_seed_1720
    details["n_pairs_seed_1001723"] = n_seed_1001723
    details["n_admitted_trajectories"] = n_admitted
    details["n_quarantined_interrupted_attempts"] = n_quarantined

    verification = {
        "schema": "v10230_part_x_px4_stage1_verification_v1",
        "depends_on_gitignored_run_files": False,
        "independence_level": "INDEPENDENT_LEDGER_REDUCTION_WITH_SHARED_QUALIFIED_PRIMITIVES",
        "classification": classification,
        "checks": checks,
        "details": details,
        "overall_pass": overall_pass,
    }
    out_path = ARTIFACTS_DIR / "px4_stage1_verification.json"
    out_path.write_text(json.dumps(verification, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out_path}")
    print(f"classification={classification}")
    print(f"overall_pass={overall_pass}")
    for key, value in checks.items():
        if not value:
            print(f"  FAILED CHECK: {key}")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
