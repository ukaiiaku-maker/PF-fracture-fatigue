#!/usr/bin/env python3
"""Update the R10 Weak-T acquisition after the first cross-background ramp."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.design_r10_joint_level_iteration import frozen_onset
from scripts.design_r10_thermal_accessibility_iteration import (
    TEMPERATURES, canonical_hash, domain_audit, exp_surface, parity_300,
    set_exp_linear_coordinates,
)

PEAK = "R10T_PEAK_P40_G2600_S0P25GPA_P4_LOW_TAYLOR_V1"
FIRST = "R10C_WEAKT_PEAK300_G1988_S6P0GPA_P4_EP055_PP010_V1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--first-candidate", type=Path, required=True)
    parser.add_argument("--first-panel", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(args.registry.open()))
    peak_record = next(x for x in rows if x["candidate_id"] == PEAK)
    peak = json.loads(peak_record["complete_parameter_vector"])
    first_record = next(csv.DictReader(args.first_candidate.open()))
    first = json.loads(first_record["complete_parameter_vector"])
    first_surface = exp_surface(first, "cleave", 1.0e12)
    measured_correction = []
    for temperature in TEMPERATURES:
        result = json.loads((args.first_panel / f"T_{int(temperature):04d}K.json").read_text())
        a0 = frozen_onset(first_surface, 3.0, 1.0e-6, temperature, dK=0.01)
        measured_correction.append(float(result["K_init_MPa_sqrt_m"]) - a0)

    # The first row measured a localized EP055/PP010 hump.  The updated batch
    # returns to the retained Peak LOW_TAYLOR P4 state and uses that row's
    # already measured correction, rather than transferring the failed hump.
    peak_complete = [5.218191431656145, 5.725655455165621, 5.789876287820182,
                     5.764876631186712, 5.491698413183014, 5.132756391190972,
                     4.721988199665786]
    peak_surface = exp_surface(peak, "cleave", 1.0e12)
    low_taylor_correction = [
        full - frozen_onset(peak_surface, 3.0, 1.0e-6, temperature, dK=0.01)
        for full, temperature in zip(peak_complete, TEMPERATURES)
    ]

    scan = []
    for gT in np.linspace(0.0018, 0.0022, 81):
        for sigc_1200 in np.linspace(0.25, 6.0, 24):
            child = set_exp_linear_coordinates(
                peak, "cleave", gT=float(gT), sigc_1200_GPa=float(sigc_1200),
            )
            surface = exp_surface(child, "cleave", 1.0e12)
            roots = np.asarray([
                frozen_onset(surface, 3.0, 1.0e-6, temperature, dK=0.01)
                for temperature in TEMPERATURES
            ])
            predicted = roots + np.asarray(low_taylor_correction)
            scan.append({
                "cleave_gT_eV_per_K": float(gT),
                "cleave_sigc_1200_GPa": float(sigc_1200),
                "A0_roots_json": json.dumps(roots.tolist()),
                "A0_span": float(roots.max() / roots.min()),
                "predicted_complete_roots_json": json.dumps(predicted.tolist()),
                "predicted_complete_span": float(predicted.max() / predicted.min()),
                "mechanical_compensation_required": float(roots.max() / roots.min()) > 1.25,
                "row": child,
            })
    eligible = [x for x in scan if x["mechanical_compensation_required"]]
    selected = min(eligible, key=lambda x: (
        max(x["predicted_complete_span"] - 1.25, 0.0),
        x["predicted_complete_span"],
        abs(x["cleave_gT_eV_per_K"] - 0.0019625),
    ))
    child = selected["row"]
    cid = "R10C_WEAKT_PEAK300_G2050_S3P75GPA_P4_LOW_TAYLOR_V2"
    child["candidate_id"] = cid
    child["material_class"] = "weakT"
    child["role"] = "R10 updated compensated Weak-T continuation on retained Peak 300 K background"
    child["mechanism_summary"] = "P3 intrinsic decline with measured retained low-Taylor state compensation"
    audit = domain_audit(child); parity = parity_300(peak, child)
    if not audit["all_processes_valid"] or parity["maximum_abs_barrier_delta_eV"] > 2e-14:
        raise RuntimeError("updated acquisition failed domain or 300 K parity")
    changed = {key: {"parent": peak.get(key), "offspring": value}
               for key, value in child.items() if peak.get(key) != value}
    record = {
        "candidate_id": cid, "parent_candidate_id": PEAK,
        "family": "A_ORIGINAL_SINGLE_EXP_FLOOR", "target": "P40",
        "cleavage_hits": child["physics__cleavage_hits"],
        "cleavage_tau_s": child["physics__cleavage_correlation_time_s"],
        "complete_row_sha256": canonical_hash(child),
        "changed_fields_json": json.dumps(changed, sort_keys=True),
        "complete_parameter_vector": json.dumps(child, sort_keys=True),
        "selection_metrics_json": json.dumps({k: v for k, v in selected.items() if k != "row"}, sort_keys=True),
        "p4_parameters_json": json.dumps({"retained_variant": "P4_LOW_TAYLOR"}),
        "raw_domain_audit_json": json.dumps(audit, sort_keys=True),
        "reference_300K_parity_json": json.dumps(parity, sort_keys=True),
        "realization": "existing P1/P2 Peak 300 K background + updated P3 opening + retained P4 low-Taylor state",
    }
    with (args.out_root / "candidate_update.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(record), lineterminator="\n"); writer.writeheader(); writer.writerow(record)
    decision = {
        "schema": "r10.weakt-joint-continuation-acquisition-update/1",
        "first_candidate": FIRST,
        "first_candidate_measured_correction_json": json.dumps(measured_correction),
        "first_candidate_disposition": "NEAR_MISS_P4_HOT_MECHANICAL_HUMP",
        "selected_candidate_id": cid,
        "complete_row_sha256": record["complete_row_sha256"],
        "predictor": "retained Peak LOW_TAYLOR complete-minus-A0 correction on exact 300 K background",
        "selected_metrics": {k: v for k, v in selected.items() if k != "row"},
        "reference_300K_parity": parity,
        "raw_domain_audit": audit,
    }
    (args.out_root / "acquisition_update_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
