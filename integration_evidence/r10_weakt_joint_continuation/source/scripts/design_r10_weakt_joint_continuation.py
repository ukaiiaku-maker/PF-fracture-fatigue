#!/usr/bin/env python3
"""Acquire a compensated Weak-T row on an already-tested 300 K background.

This is an R10 continuation, not a new constitutive family.  It keeps the
selected Peak row's complete 300 K process surfaces (and hence its accepted
fatigue evidence), applies the inherited P3 opening continuation, and uses an
existing P4 emission/Peierls/Taylor endpoint.  Candidate ranking uses the
measured temperature-dependent state correction of the failed Weak-T row as
an acquisition predictor; promotion still requires a complete-state ramp.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.design_r10_joint_level_iteration import frozen_onset
from scripts.design_r10_thermal_accessibility_iteration import (
    TEMPERATURES, canonical_hash, domain_audit, exp_surface, parity_300,
    raw_coordinates, set_exp_linear_coordinates, set_transport_endpoint,
)

PEAK = "R10T_PEAK_P40_G2600_S0P25GPA_P4_LOW_TAYLOR_V1"
FAILED_WEAK = "R10T_WEAKT_P40_G1962_S4P5GPA_P4_EP055_PP010_V1"


def _row(registry: list[dict], candidate_id: str) -> dict:
    record = next(x for x in registry if x["candidate_id"] == candidate_id)
    return json.loads(record["complete_parameter_vector"])


def _p4_ep055_pp010(row: dict) -> dict:
    """Use the already-authorized P4 endpoint, preserving every 300 K value."""
    child = set_transport_endpoint(row, "taylor", 0.03)
    child = set_exp_linear_coordinates(child, "emit", sigc_1200_GPa=1.5)
    emit_300, _ = raw_coordinates(exp_surface(child, "emit", 1.0e11), 300.0)
    child = set_exp_linear_coordinates(
        child, "emit", gT=(0.55 - emit_300) / 900.0,
        sigc_1200_GPa=1.5,
    )
    return set_transport_endpoint(child, "peierls", 0.10)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--full-state-table", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)

    registry = list(csv.DictReader(args.registry.open()))
    peak = _row(registry, PEAK)
    failed = _row(registry, FAILED_WEAK)

    # The measured correction is operational K_complete-K_A0, not shielding or
    # a work partition.  The 300 K value uses the exact-contract result already
    # carried by the R10 analysis; the remaining values are fresh full-state.
    full = list(csv.DictReader(args.full_state_table.open()))
    measured = {float(x["temperature_K"]): float(x["K_init_MPa_sqrt_m"])
                for x in full if x["candidate_id"] == FAILED_WEAK}
    measured[300.0] = 5.2373966579
    failed_surface = exp_surface(failed, "cleave", 1.0e12)
    correction = []
    for temperature in TEMPERATURES:
        a0 = frozen_onset(failed_surface, 3.0, 1.0e-6, temperature, dK=0.01)
        correction.append(measured[temperature] - a0)

    scan = []
    g_values = np.unique(np.r_[np.linspace(0.0018, 0.0022, 33), 0.0019625])
    for gT in g_values:
        for sigc_1200 in (3.0, 4.5, 6.0):
            p3 = set_exp_linear_coordinates(
                peak, "cleave", gT=float(gT), sigc_1200_GPa=sigc_1200,
            )
            surface = exp_surface(p3, "cleave", 1.0e12)
            roots = np.asarray([
                frozen_onset(surface, 3.0, 1.0e-6, temperature, dK=0.01)
                for temperature in TEMPERATURES
            ], dtype=float)
            predicted = roots + np.asarray(correction)
            a0_span = float(roots.max() / roots.min())
            predicted_span = float(predicted.max() / predicted.min())
            scan.append({
                "background_candidate_id": PEAK,
                "cleave_gT_eV_per_K": float(gT),
                "cleave_sigc_1200_GPa": sigc_1200,
                "A0_roots_json": json.dumps(roots.tolist()),
                "measured_state_correction_json": json.dumps(correction),
                "predicted_complete_roots_json": json.dumps(predicted.tolist()),
                "A0_span": a0_span,
                "predicted_complete_span": predicted_span,
                "mechanical_compensation_required": a0_span > 1.25,
                "row": p3,
            })

    eligible = [x for x in scan if x["mechanical_compensation_required"]]
    selected = min(eligible, key=lambda x: (
        max(x["predicted_complete_span"] - 1.25, 0.0),
        x["predicted_complete_span"],
        abs(x["cleave_gT_eV_per_K"] - 0.0019625),
    ))
    child = _p4_ep055_pp010(selected["row"])
    cid = ("R10C_WEAKT_PEAK300_G1988_S6P0GPA_"
           "P4_EP055_PP010_V1")
    child["candidate_id"] = cid
    child["material_class"] = "weakT"
    child["role"] = "R10 compensated Weak-T continuation on retained Peak 300 K background"
    child["mechanism_summary"] = (
        "P3 intrinsic decline plus P4 current-state compensation; exact retained Peak 300 K background"
    )
    audit = domain_audit(child)
    parity = parity_300(peak, child)
    if not audit["all_processes_valid"]:
        raise RuntimeError("selected row is outside the inherited raw process domain")
    if parity["maximum_abs_barrier_delta_eV"] > 2.0e-14:
        raise RuntimeError(f"300 K background parity failed: {parity}")

    changed = {key: {"parent": peak.get(key), "offspring": value}
               for key, value in child.items() if peak.get(key) != value}
    record = {
        "candidate_id": cid,
        "parent_candidate_id": PEAK,
        "family": "A_ORIGINAL_SINGLE_EXP_FLOOR",
        "target": "P40",
        "cleavage_hits": child["physics__cleavage_hits"],
        "cleavage_tau_s": child["physics__cleavage_correlation_time_s"],
        "complete_row_sha256": canonical_hash(child),
        "changed_fields_json": json.dumps(changed, sort_keys=True),
        "complete_parameter_vector": json.dumps(child, sort_keys=True),
        "selection_metrics_json": json.dumps(
            {k: v for k, v in selected.items() if k != "row"}, sort_keys=True
        ),
        "p4_parameters_json": json.dumps({
            "emission_G0_1200_eV": 0.55,
            "peierls_G0_1200_eV": 0.10,
            "shared_sigc_1200_GPa": 1.5,
            "taylor_G0_1200_eV": 0.03,
        }, sort_keys=True),
        "raw_domain_audit_json": json.dumps(audit, sort_keys=True),
        "reference_300K_parity_json": json.dumps(parity, sort_keys=True),
        "realization": "existing P1/P2 Peak 300 K background + P3 opening + P4 transport endpoints",
    }
    fields = list(record)
    with (args.out_root / "candidate.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerow(record)
    scan_fields = [k for k in scan[0] if k != "row"]
    with (args.out_root / "analytical_scan.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=scan_fields, lineterminator="\n")
        writer.writeheader(); writer.writerows({k: x[k] for k in scan_fields} for x in scan)
    decision = {
        "schema": "r10.weakt-joint-continuation-acquisition/1",
        "selected_candidate_id": cid,
        "complete_row_sha256": record["complete_row_sha256"],
        "source_background": PEAK,
        "failed_row_used_as_information": FAILED_WEAK,
        "predictor": "A0 plus measured event-owning complete-minus-A0 temperature correction",
        "predictor_is_promotion": False,
        "selected_metrics": {k: v for k, v in selected.items() if k != "row"},
        "reference_300K_parity": parity,
        "raw_domain_audit": audit,
        "selection_reason": (
            "different retained 300 K background with passed developed fatigue; "
            "predicted bounded response still requires nonzero mechanical compensation"
        ),
    }
    (args.out_root / "acquisition_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
