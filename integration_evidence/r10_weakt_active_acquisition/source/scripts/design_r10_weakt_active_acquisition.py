#!/usr/bin/env python3
"""Acquire active Weak-T rows by moving the permitted P4 emission knee.

The parent is the executed active Weak-T row.  Only the emission reference
characteristic stress and its linear thermal continuation are moved; the
1200 K endpoint is retained.  This preserves all source, geometry, return,
attempt-frequency, event/reward and mechanics fields.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.design_r10_thermal_accessibility_iteration import (
    TEMPERATURES, canonical_hash, domain_audit, exp_surface, raw_coordinates,
    transport_surface,
)


PARENT = "R10T_WEAKT_P40_G1962_S4P5GPA_P4_EP055_PP010_V1"
STRESS_GRID_PA = np.linspace(0.0, 30.0e9, 301)


def surface_audit(row: dict) -> dict:
    surfaces = {
        "opening": exp_surface(row, "cleave", 1.0e12),
        "emission": exp_surface(row, "emit", 1.0e11),
        "peierls": transport_surface(row, "peierls"),
        "taylor": transport_surface(row, "taylor"),
    }
    result = {}
    for name, surface in surfaces.items():
        cases = []
        for temperature in TEMPERATURES:
            G0, sigc = raw_coordinates(surface, temperature)
            values = np.asarray(surface.values_eV(STRESS_GRID_PA, temperature), float)
            rates = np.asarray(surface.rate(STRESS_GRID_PA, temperature), float)
            raw_floor = max(surface.floor_min_eV, surface.floor_fraction * G0)
            effective_floor = min(surface.floor_max_fraction * G0, raw_floor)
            cases.append({
                "temperature_K": temperature,
                "raw_G0_eV": G0,
                "raw_sigc_GPa": sigc / 1.0e9,
                "raw_domain_valid": G0 > 0.0 and sigc > 0.0,
                "effective_floor_eV": effective_floor,
                "floor_min_guard_active": raw_floor == surface.floor_min_eV,
                "floor_max_guard_active": effective_floor == surface.floor_max_fraction * G0,
                "barrier_min_eV": float(values.min()),
                "barrier_max_eV": float(values.max()),
                "rate_min_s-1": float(rates.min()),
                "rate_max_s-1": float(rates.max()),
            })
        result[name] = cases
    result["all_raw_domains_valid"] = all(
        case["raw_domain_valid"] for cases in result.values()
        if isinstance(cases, list) for case in cases
    )
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)

    records = list(csv.DictReader(args.registry.open()))
    parent_record = next(x for x in records if x["candidate_id"] == PARENT)
    parent = json.loads(parent_record["complete_parameter_vector"])
    parent_surface = exp_surface(parent, "emit", 1.0e11)
    G300, sigc300 = raw_coordinates(parent_surface, 300.0)
    _, sigc1200 = raw_coordinates(parent_surface, 1200.0)

    # Broad analytical bracket.  The five serialized rows are logarithmically
    # informative about the transition from the measured blunting state to a
    # still-active, less-saturated emission response.  Full-state work is
    # selected after the compact surface and monotonic checks.
    # 3.0 GPa is the single sequential refinement added after the executed
    # 1.2/1.5/2.0 GPa developed slopes (2.620/3.120/3.246) bracketed but did
    # not reach the inherited 3.4 lower bound.  At the visited 9--10 GPa
    # high-load stresses it retains 6.16e6--2.24e7 emission events/s, so it is
    # an active-state test rather than an emission-off control.
    knee_values = (0.90, 1.20, 1.50, 2.00, 3.00)
    output = []
    analytical = []
    for knee_GPa in knee_values:
        child = dict(parent)
        new_sT = (sigc1200 / 1.0e9 - knee_GPa) / 900.0
        new_sigc0 = knee_GPa - new_sT * (300.0 - float(parent["Tref_K"]))
        child["emit_sigc0_GPa"] = new_sigc0
        child["emit_sT_GPa_per_K"] = new_sT
        tag = str(knee_GPa).replace(".", "P")
        cid = f"R10A_WEAKT_ACTIVE_EMIT_S{tag}GPA_V1"
        child["candidate_id"] = cid
        child["role"] = "R10 active Weak-T emission-knee acquisition"
        child["mechanism_summary"] = (
            "retained active Weak-T opening/transport row with permitted P4 "
            "300 K emission characteristic-stress leverage"
        )
        audit = surface_audit(child)
        endpoint = domain_audit(child)
        if not audit["all_raw_domains_valid"] or not endpoint["all_processes_valid"]:
            raise RuntimeError(f"candidate left the raw process domain: {cid}")
        emission = exp_surface(child, "emit", 1.0e11)
        probes = {}
        for stress_GPa in (3.02, 4.19, 4.97, 5.06):
            base = float(parent_surface.rate(stress_GPa * 1.0e9, 300.0))
            rate = float(emission.rate(stress_GPa * 1.0e9, 300.0))
            probes[str(stress_GPa)] = {
                "rate_s-1": rate,
                "relative_to_parent": rate / base,
            }
        changed = {
            key: {"parent": parent.get(key), "offspring": value}
            for key, value in child.items() if parent.get(key) != value
        }
        row_hash = canonical_hash(child)
        record = {
            "candidate_id": cid,
            "parent_candidate_id": PARENT,
            "family": parent_record["family"],
            "target": parent_record["target"],
            "cleavage_hits": parent_record["cleavage_hits"],
            "cleavage_tau_s": parent_record["cleavage_tau_s"],
            "complete_row_sha256": row_hash,
            "changed_fields_json": json.dumps(changed, sort_keys=True),
            "complete_parameter_vector": json.dumps(child, sort_keys=True),
            "selection_metrics_json": json.dumps({
                "parent_emit_G0_300_eV": G300,
                "parent_emit_sigc_300_GPa": sigc300 / 1.0e9,
                "retained_emit_sigc_1200_GPa": sigc1200 / 1.0e9,
                "offspring_emit_sigc_300_GPa": knee_GPa,
                "saved_state_emission_rate_probes": probes,
                "predictor_scope": "fixed saved-state acquisition only; not a coupled trajectory prediction",
            }, sort_keys=True),
            "p4_parameters_json": json.dumps({
                "coordinate": "emission_characteristic_stress_300K",
                "emit_sigc_300_GPa": knee_GPa,
                "emit_sigc_1200_GPa": sigc1200 / 1.0e9,
            }, sort_keys=True),
            "raw_domain_audit_json": json.dumps(audit, sort_keys=True),
            "endpoint_domain_audit_json": json.dumps(endpoint, sort_keys=True),
            "realization": "existing Family-A EXP_floor P4 reference-state emission knee",
        }
        output.append(record)
        analytical.append({
            "candidate_id": cid,
            "complete_row_sha256": row_hash,
            "emit_sigc_300_GPa": knee_GPa,
            "emit_sigc_1200_GPa": sigc1200 / 1.0e9,
            "saved_state_emission_rate_probes_json": json.dumps(probes, sort_keys=True),
            "changed_fields_json": json.dumps(changed, sort_keys=True),
        })

    with (args.out_root / "active_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    with (args.out_root / "analytical_acquisition.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(analytical[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(analytical)
    decision = {
        "schema": "r10.active-weakt-reference-state-acquisition/1",
        "parent_candidate_id": PARENT,
        "parent_complete_row_sha256": parent_record["complete_row_sha256"],
        "candidate_count": len(output),
        "candidates": analytical,
        "scientific_decision": (
            "retain the opening-dominated positive control; resume the active Weak-T "
            "lane by bracketing the permitted cold emission knee that directly controls "
            "the measured high-load blunting and loss of stress transmission"
        ),
        "unchanged_contract": [
            "opening surface", "emission G0/alpha/exponent/floor and attempt frequency",
            "Peierls and Taylor surfaces", "source density and multiplicity",
            "geometry-to-slip normalization", "return and retention laws",
            "blunting law", "event/reward/completion law", "mechanical kernel",
        ],
        "expensive_selection": (
            "run the moderate and strong bracket endpoints through same-row fracture; "
            "then use developed native fatigue at the measured 15/22 DeltaK discrepancy"
        ),
    }
    (args.out_root / "acquisition_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
