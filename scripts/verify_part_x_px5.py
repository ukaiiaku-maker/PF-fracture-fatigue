"""PX5: strict portable verifier for the static-shield-vs-dynamic-
rebonding campaign (D2/D5 x ceiling_static/orbit_matched_static x
five-point seed=1720 Kmax grid).

Depends ONLY on files tracked under artifacts/crack_rebonding_part_x_v1/
-- never on the gitignored runs/ directory -- so it must still pass with
every runs/ output deleted. Independently RE-DERIVES every S_h_static,
S_h_dynamic, and reproduction fraction from the raw event ledgers
(px5_event_ledger.json for the static trajectories, developed_event_
ledger.json for PX4's own zero/dynamic baselines) via the shared,
already-qualified stable_growth_gate primitive, then cross-checks the
recomputation against px5_static_shield_analysis.csv and the
classification thresholds, rather than trusting those tables at face
value.

Usage:
    <pinned interpreter> scripts/verify_part_x_px5.py
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
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import stable_growth_gate  # noqa: E402

REQUIRED_ARTIFACTS = [
    "px5_static_shield_job_registry.csv", "px5_static_shield_wall_budget_retry_registry.csv",
    "px5_static_shield_predictions.csv", "px5_static_shield_analysis.csv", "px5_static_shield_analysis.json",
    "px5_scientific_decision.json", "px5_event_ledger.json", "px5_event_ledger.csv",
    "px5_censor_registry.csv", "px5_attempt_registry.csv", "px5_raw_result_hashes.json",
    "developed_event_ledger.json", "developed_job_registry.csv",
]
EXPECTED_ADMITTED = 20
EXPECTED_SUPERSEDED = 2
EXPECTED_POINTS = 10
KMAX_GRID_Pa_sqrt_m = [12.0e6, 15.0e6, 18.0e6, 21.0e6, 24.3e6]
REPRODUCTION_FRACTION_DOMINANT = 0.80
REPRODUCTION_FRACTION_HISTORY_REQUIRED = 0.30
FLOAT_TOL = 1.0e-6


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    for name in REQUIRED_ARTIFACTS:
        checks[f"artifact_present_{name}"] = (ARTIFACTS_DIR / name).is_file()
    if not all(checks.values()):
        missing = [k for k, v in checks.items() if not v]
        raise SystemExit(f"missing tracked artifacts, cannot verify: {missing}")

    px5_ledger = json.loads((ARTIFACTS_DIR / "px5_event_ledger.json").read_text())
    px5_events_by_key: dict[str, list[dict]] = defaultdict(list)
    for e in px5_ledger["events"]:
        px5_events_by_key[e["canonical_job_key"]].append(e)
    for key, evs in px5_events_by_key.items():
        evs.sort(key=lambda e: e["event_index"])
        checks[f"px5_event_count_is_30_{key[:12]}"] = len(evs) == 30

    dev_ledger = json.loads((ARTIFACTS_DIR / "developed_event_ledger.json").read_text())
    dev_events_by_key: dict[str, list[dict]] = defaultdict(list)
    for e in dev_ledger["events"]:
        dev_events_by_key[e["canonical_job_key"]].append(e)
    for evs in dev_events_by_key.values():
        evs.sort(key=lambda e: e["event_index"])

    main_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_job_registry.csv")))
    retry_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_wall_budget_retry_registry.csv")))
    admitted = [r for r in main_rows if r["status"] == "AUTHORIZED_PX5"] + [r for r in retry_rows if r["status"] == "AUTHORIZED_PX5"]
    superseded = [r for r in main_rows if r["status"] == "SUPERSEDED_WALL_BUDGET_TOO_SMALL"]
    checks["admitted_row_count_is_20"] = len(admitted) == EXPECTED_ADMITTED
    checks["superseded_row_count_is_2"] = len(superseded) == EXPECTED_SUPERSEDED
    checks["every_admitted_row_has_30_events_in_ledger"] = all(
        len(px5_events_by_key.get(r["canonical_job_key"], [])) == 30 for r in admitted
    )
    checks["no_admitted_row_overlaps_superseded_keys"] = not (
        {r["canonical_job_key"] for r in admitted} & {r["canonical_job_key"] for r in superseded}
    )

    px4_registry = list(csv.DictReader(open(ARTIFACTS_DIR / "developed_job_registry.csv")))
    px4_authorized = [r for r in px4_registry if r["status"] == "AUTHORIZED_PX4"]
    px4_key_by = {(r["protocol"], round(float(r["Kmax_Pa_sqrt_m"])), r["cohesion"]): r["canonical_job_key"] for r in px4_authorized}

    analysis_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_analysis.csv")))
    checks["analysis_row_count_is_10"] = len(analysis_rows) == EXPECTED_POINTS

    def _gate(events: list[dict], freq: float) -> dict:
        return stable_growth_gate(events, frequency_Hz=freq)

    n_mismatch = 0
    n_classification_mismatch = 0
    for protocol in ("D2", "D5"):
        for Kmax in KMAX_GRID_Pa_sqrt_m:
            zero_key = px4_key_by[(protocol, round(Kmax), "zero")]
            dyn_key = px4_key_by[(protocol, round(Kmax), "finite")]
            gate_zero = _gate(dev_events_by_key[zero_key], 1000.0)
            gate_dyn = _gate(dev_events_by_key[dyn_key], 1000.0)
            da_zero, da_dyn = gate_zero["developed_da_dN_m_per_cycle"], gate_dyn["developed_da_dN_m_per_cycle"]
            S_h_dyn = math.log10(da_dyn / da_zero)

            table_row = next(r for r in analysis_rows if r["protocol"] == protocol and round(float(r["Kmax_Pa_sqrt_m"])) == round(Kmax))
            checks[f"S_h_dynamic_reproducible_{protocol}_{round(Kmax/1e6)}"] = abs(S_h_dyn - float(table_row["S_h_dynamic"])) < FLOAT_TOL

            fractions = []
            for control_mode in ("ceiling_static", "orbit_matched_static"):
                match = [r for r in admitted if r["protocol"] == protocol and round(float(r["Kmax_Pa_sqrt_m"])) == round(Kmax) and r["cohesion"] == control_mode]
                if len(match) != 1:
                    raise RuntimeError(f"expected exactly 1 admitted row for {protocol}/{Kmax}/{control_mode}, found {len(match)}")
                key = match[0]["canonical_job_key"]
                gate_static = _gate(px5_events_by_key[key], 1000.0)
                da_static = gate_static["developed_da_dN_m_per_cycle"]
                S_h_static = math.log10(da_static / da_zero) if (da_static and da_static > 0) else None
                frac = S_h_static / S_h_dyn if (S_h_static is not None and S_h_dyn != 0.0) else None
                fractions.append(frac)

                col = f"{control_mode}_S_h"
                match_ok = S_h_static is not None and abs(S_h_static - float(table_row[col])) < FLOAT_TOL
                checks[f"{col}_reproducible_{protocol}_{round(Kmax/1e6)}"] = match_ok
                if not match_ok:
                    n_mismatch += 1
                frac_col = f"{control_mode}_reproduction_fraction"
                frac_ok = frac is not None and abs(frac - float(table_row[frac_col])) < FLOAT_TOL
                checks[f"{frac_col}_reproducible_{protocol}_{round(Kmax/1e6)}"] = frac_ok
                if not frac_ok:
                    n_mismatch += 1

            if all(f >= REPRODUCTION_FRACTION_DOMINANT for f in fractions):
                expected_class = "FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT"
            elif all(f <= REPRODUCTION_FRACTION_HISTORY_REQUIRED for f in fractions):
                expected_class = "DYNAMIC_REBONDING_HISTORY_REQUIRED"
            else:
                expected_class = "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY"
            class_ok = table_row["classification"] == expected_class
            checks[f"classification_reproducible_{protocol}_{round(Kmax/1e6)}"] = class_ok
            if not class_ok:
                n_classification_mismatch += 1

    checks["zero_S_h_or_fraction_recomputation_mismatches"] = n_mismatch == 0
    checks["zero_classification_recomputation_mismatches"] = n_classification_mismatch == 0
    checks["all_10_points_classified_mixed"] = all(r["classification"] == "MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY" for r in analysis_rows)

    censor_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_censor_registry.csv")))
    checks["censor_registry_row_count_is_20"] = len(censor_rows) == EXPECTED_ADMITTED
    checks["zero_censored_admitted_trajectories"] = all(r["censored"] == "False" for r in censor_rows)
    checks["all_admitted_exactly_150um"] = all(abs(float(r["cumulative_extension_m"]) - 150.0e-6) < 1.0e-9 for r in censor_rows)

    attempt_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_attempt_registry.csv")))
    n_admitted = sum(1 for r in attempt_rows if r["admitted"] == "True")
    n_not_admitted = sum(1 for r in attempt_rows if r["admitted"] == "False")
    checks["attempt_registry_admitted_count_is_20"] = n_admitted == EXPECTED_ADMITTED
    checks["attempt_registry_superseded_count_is_2"] = n_not_admitted == EXPECTED_SUPERSEDED

    raw_hashes = json.loads((ARTIFACTS_DIR / "px5_raw_result_hashes.json").read_text())["files"]
    checks["raw_result_hashes_count_is_20"] = len(raw_hashes) == EXPECTED_ADMITTED

    overall_pass = all(checks.values())
    classification = "PX5_COMPLETE" if overall_pass else "PX5_INCOMPLETE"
    details["n_admitted"] = n_admitted
    details["n_superseded"] = n_not_admitted

    verification = {
        "schema": "v10230_part_x_px5_verification_v1",
        "depends_on_gitignored_run_files": False,
        "independence_level": "INDEPENDENT_LEDGER_REDUCTION_WITH_SHARED_QUALIFIED_PRIMITIVES",
        "classification": classification,
        "checks": checks,
        "details": details,
        "overall_pass": overall_pass,
    }
    out_path = ARTIFACTS_DIR / "px5_verification.json"
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
