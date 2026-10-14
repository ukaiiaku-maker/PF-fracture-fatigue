#!/usr/bin/env python3
"""Continue R10 in the existing P3 thermal and P4 state coordinates.

P3 is an exact reparameterization of the committed linear EXP-floor law at
fixed 300 K surface.  P4 preserves the 300 K transport surfaces while moving
the already-studied Taylor endpoint and, where required, the shared raw
plastic characteristic-stress endpoint into its declared positive domain.
No basis, prefactor, clock, event law, or mechanics term is added.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.material_manifest import (
    ExpFloorBarrier, KB_EV_PER_K, TransportBarrier,
)
from arrhenius_fracture.thermal_accessibility_r10 import (
    maximum_floor_for_mark, opening_floor_bound, raw_coordinates,
)
from scripts.design_r10_joint_level_iteration import frozen_onset


TEMPERATURES = (300.0, 450.0, 600.0, 750.0, 900.0, 1050.0, 1200.0)
T0 = 300.0
TMAX = 1200.0


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def exp_surface(row, prefix: str, attempt: float) -> ExpFloorBarrier:
    return ExpFloorBarrier(
        G00_eV=float(row[f"{prefix}_G00_eV"]),
        gT_eV_per_K=float(row[f"{prefix}_gT_eV_per_K"]),
        sigc0_Pa=1.0e9 * float(row[f"{prefix}_sigc0_GPa"]),
        sT_Pa_per_K=1.0e9 * float(row[f"{prefix}_sT_GPa_per_K"]),
        alpha=float(row[f"{prefix}_exp_a"]),
        exponent=float(row[f"{prefix}_exp_n"]),
        floor_fraction=float(row[f"{prefix}_floor_frac"]),
        Tref_K=float(row["Tref_K"]), attempt_frequency_s=attempt,
    )


def set_exp_linear_coordinates(
    row: dict, prefix: str, *, gT: float | None = None,
    sigc_1200_GPa: float | None = None,
) -> dict:
    out = dict(row); old = exp_surface(row, prefix, 1.0)
    G300, sigc300_Pa = raw_coordinates(old, T0)
    new_gT = old.gT_eV_per_K if gT is None else float(gT)
    new_sT_GPa = old.sT_Pa_per_K / 1.0e9
    if sigc_1200_GPa is not None:
        new_sT_GPa = (float(sigc_1200_GPa) - sigc300_Pa / 1.0e9) / (TMAX - T0)
    out[f"{prefix}_gT_eV_per_K"] = new_gT
    out[f"{prefix}_G00_eV"] = G300 - new_gT * (T0 - old.Tref_K)
    out[f"{prefix}_sT_GPa_per_K"] = new_sT_GPa
    out[f"{prefix}_sigc0_GPa"] = sigc300_Pa / 1.0e9 - new_sT_GPa * (T0 - old.Tref_K)
    return out


def set_transport_endpoint(row: dict, prefix: str, endpoint_eV: float) -> dict:
    out = dict(row); Tref = float(row["Tref_K"])
    old_gT = -float(row[f"{prefix}_activation_entropy_kB"]) * KB_EV_PER_K
    G300 = float(row[f"{prefix}_H0_eV"]) + old_gT * (T0 - Tref)
    new_gT = (float(endpoint_eV) - G300) / (TMAX - T0)
    out[f"{prefix}_activation_entropy_kB"] = -new_gT / KB_EV_PER_K
    out[f"{prefix}_H0_eV"] = G300 - new_gT * (T0 - Tref)
    return out


def transport_surface(row: dict, prefix: str) -> ExpFloorBarrier:
    emission = exp_surface(row, "emit", 1.0e11)
    barrier = TransportBarrier(
        H0_eV=float(row[f"{prefix}_H0_eV"]),
        activation_entropy_kB=float(row[f"{prefix}_activation_entropy_kB"]),
        alpha=float(row[f"{prefix}_exp_a"]),
        exponent=float(row[f"{prefix}_exp_n"]),
        attempt_frequency_s=float(row[f"{prefix}_nu0_s"]),
    )
    return barrier.as_surface(emission)


def domain_audit(row: dict) -> dict:
    surfaces = {
        "opening": exp_surface(row, "cleave", 1.0e12),
        "emission": exp_surface(row, "emit", 1.0e11),
        "peierls": transport_surface(row, "peierls"),
        "taylor": transport_surface(row, "taylor"),
    }
    result = {}
    for name, surface in surfaces.items():
        endpoints = [raw_coordinates(surface, temperature) for temperature in (T0, TMAX)]
        result[name] = {
            "G0_300_eV": endpoints[0][0], "G0_1200_eV": endpoints[1][0],
            "sigc_300_GPa": endpoints[0][1] / 1.0e9,
            "sigc_1200_GPa": endpoints[1][1] / 1.0e9,
            "raw_domain_valid": min(x[0] for x in endpoints) > 0.0 and min(x[1] for x in endpoints) > 0.0,
        }
    result["all_processes_valid"] = all(x["raw_domain_valid"] for x in result.values())
    return result


def topology(roots: list[float | None]) -> tuple[str, float, float]:
    if any(x is None for x in roots):
        return "OPENING_CENSORED", math.inf, 0.0
    values = np.asarray(roots, dtype=float); span = float(values.max() / values.min())
    imax = int(np.argmax(values))
    prominence = float(values[imax] / max(values[0], values[-1]) - 1.0)
    if 0 < imax < len(values) - 1 and prominence >= 0.03:
        label = "PEAK_INTERIOR"
    elif values[-1] / values[0] >= 1.5:
        label = "DBTT_RISING_OPENING_ONLY"
    elif span <= 1.25:
        label = "WEAK_T_OPENING"
    else:
        label = "CERAMIC_FALLING"
    return label, span, prominence


def scan_p3(row: dict, base_id: str, class_name: str) -> list[dict]:
    records = []
    # These endpoint values and the broad gT interval are recovered P3/V4.2
    # coordinates.  The dense analytical product is inexpensive; selection is
    # by class objective, not by a proposal quota.
    g_values = np.unique(np.r_[np.linspace(0.0015, 0.00335, 75), 0.0019625])
    sigc_endpoints = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.5, 6.0)
    for gT in g_values:
        for sigc_endpoint in sigc_endpoints:
            candidate = set_exp_linear_coordinates(
                row, "cleave", gT=float(gT), sigc_1200_GPa=sigc_endpoint,
            )
            surface = exp_surface(candidate, "cleave", 1.0e12)
            q = float(candidate["physics__cleavage_hits"])
            tau = float(candidate["physics__cleavage_correlation_time_s"])
            roots = [frozen_onset(surface, q, tau, T, dK=0.01) for T in TEMPERATURES]
            label, span, prominence = topology(roots)
            hot_bound = opening_floor_bound(
                surface, temperature_K=TMAX, hits=q, tau_s=tau,
                exposure_s=16000.0, pending_mark=1.0,
            )
            records.append({
                "base_candidate_id": base_id, "class": class_name,
                "cleave_gT_eV_per_K": float(gT),
                "cleave_sigc_1200_GPa": float(sigc_endpoint),
                "opening_topology": label, "opening_span": span,
                "peak_prominence": prominence,
                "A0_roots_json": json.dumps(roots),
                "floor_1200_eV": hot_bound.evaluated_floor_eV,
                "floor_action_ceiling_1200": hot_bound.action_ceiling,
                "floor_necessary_accessible_1200": not hot_bound.necessarily_inaccessible,
                "row": candidate,
            })
    return records


def choose_p3(records: list[dict], class_name: str) -> dict:
    finite = [x for x in records if x["opening_topology"] != "OPENING_CENSORED"]
    if class_name == "PEAK":
        pool = [x for x in finite if x["opening_topology"] == "PEAK_INTERIOR"]
        return min(pool, key=lambda x: (abs(x["opening_span"] - 1.25), -x["peak_prominence"]))
    if class_name == "DBTT":
        # The flat-opening P4 bracket measured a nonmonotone state contribution.
        # Return to the retained rising P3 coordinate and combine it with the
        # newly bracketed moderate emission endpoints; promotion still requires
        # a resolved nonzero coupled-minus-opening contribution.
        return min(finite, key=lambda x: (
            abs(x["cleave_gT_eV_per_K"] - 0.003075),
            abs(x["cleave_sigc_1200_GPa"] - 0.25),
        ))
    # The required weak-T lane is compensated, not merely intrinsically flat.
    # Select a modest direct falling trend so the measured positive state
    # resistance has an opposing contribution to cancel.  The earlier
    # intrinsically weak P3 point remains in the scan and executed ledger.
    pool = [x for x in finite
            if json.loads(x["A0_roots_json"])[-1] < json.loads(x["A0_roots_json"])[0]]
    # The first two compensated batches measured their largest positive state
    # shift at 1050 K.  Applying the paired EP055/PP010 discrepancy to the
    # retained P3 grid is locally refined at gT=0.0019625,
    # sigc(1200)=4.5 GPa: its direct decline
    # supplies the opposing contribution without leaving the accessible domain.
    return min(pool, key=lambda x: (
        abs(x["cleave_gT_eV_per_K"] - 0.0019625),
        abs(x["cleave_sigc_1200_GPa"] - 4.5),
    ))


def p4_variants(row: dict, class_name: str) -> list[tuple[str, dict, dict]]:
    # Taylor endpoint levels and shared-scale endpoints are existing P4/V4.2
    # coordinates.  The paired variants measure state leverage at an identical
    # opening surface; they are not class promotions.
    specs = [
        ("P4_LOW_TAYLOR", 0.03, 0.75, None, None),
        ("P4_HIGH_TAYLOR", 0.40, 1.50, None, None),
    ]
    # The first paired Taylor-only batch is retained as a measured near-zero
    # first-passage direction.  For the two classes requiring a mechanical
    # mechanism, release the existing V4.2 emission/Peierls endpoint factorial
    # rather than inventing another basis or fitting a toughness curve.
    if class_name in ("DBTT", "WEAKT"):
        specs.extend([
            ("P4_EP025_PP010", 0.03, 0.75, 0.25, 0.10),
            ("P4_EP025_PP040", 0.40, 0.75, 0.25, 0.40),
            ("P4_EP055_PP010", 0.03, 1.50, 0.55, 0.10),
            ("P4_EP055_PP040", 0.40, 1.50, 0.55, 0.40),
        ])
    if class_name == "DBTT":
        specs.extend([
            ("P4_EP100_PP040", 0.40, 1.50, 1.00, 0.40),
            ("P4_EP200_PP040", 0.40, 1.50, 2.00, 0.40),
            ("P4_EP300_PP040", 0.40, 1.50, 3.00, 0.40),
            ("P4_EP400_PP040", 0.40, 1.50, 4.00, 0.40),
            ("P4_EP500_PP040", 0.40, 1.50, 5.00, 0.40),
            ("P4_EP600_PP040", 0.40, 1.50, 6.00, 0.40),
        ])
    variants = []
    for label, taylor_endpoint, shared_endpoint, emission_endpoint, peierls_endpoint in specs:
        child = set_transport_endpoint(row, "taylor", taylor_endpoint)
        # Only move the shared emission/Peierls/Taylor stress scale when its
        # inherited 1200 K raw value is invalid (the DBTT historical guard).
        emission = exp_surface(child, "emit", 1.0e11)
        if raw_coordinates(emission, TMAX)[1] <= 0.0 or emission_endpoint is not None:
            child = set_exp_linear_coordinates(
                child, "emit", sigc_1200_GPa=shared_endpoint,
            )
        if emission_endpoint is not None:
            emission_300, _ = raw_coordinates(exp_surface(child, "emit", 1.0e11), T0)
            emit_gT = (float(emission_endpoint) - emission_300) / (TMAX - T0)
            child = set_exp_linear_coordinates(
                child, "emit", gT=emit_gT, sigc_1200_GPa=shared_endpoint,
            )
        if peierls_endpoint is not None:
            child = set_transport_endpoint(child, "peierls", peierls_endpoint)
        variants.append((label, child, {
            "taylor_G0_1200_eV": taylor_endpoint,
            "shared_sigc_1200_GPa_if_repaired_or_released": shared_endpoint,
            "emission_G0_1200_eV": emission_endpoint,
            "peierls_G0_1200_eV": peierls_endpoint,
        }))
    return variants


def parity_300(parent: dict, child: dict) -> dict:
    probes = np.linspace(0.0, 30.0e9, 129)
    result = {}
    for name, getter in (
        ("opening", lambda r: exp_surface(r, "cleave", 1.0e12)),
        ("emission", lambda r: exp_surface(r, "emit", 1.0e11)),
        ("peierls", lambda r: transport_surface(r, "peierls")),
        ("taylor", lambda r: transport_surface(r, "taylor")),
    ):
        delta = np.max(np.abs(getter(parent).values_eV(probes, T0) - getter(child).values_eV(probes, T0)))
        result[name] = float(delta)
    result["maximum_abs_barrier_delta_eV"] = max(result.values())
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--joint-registry", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv); args.out_root.mkdir(parents=True, exist_ok=True)
    bases = list(csv.DictReader(args.joint_registry.open()))
    scan_rows, selected = [], []
    for record in bases:
        class_name = record["candidate_id"].split("_")[1]
        parent = json.loads(record["complete_parameter_vector"])
        scan = scan_p3(parent, record["candidate_id"], class_name)
        scan_rows.extend({k: v for k, v in x.items() if k != "row"} for x in scan)
        p3 = choose_p3(scan, class_name)
        for p4_label, child, p4_parameters in p4_variants(p3["row"], class_name):
            cid = (
                f"R10T_{class_name}_P40_G{int(round(1e6*p3['cleave_gT_eV_per_K'])):04d}"
                f"_S{str(p3['cleave_sigc_1200_GPa']).replace('.', 'P')}GPA_{p4_label}_V1"
            )
            child["candidate_id"] = cid
            audit = domain_audit(child); parity = parity_300(parent, child)
            if not audit["all_processes_valid"]:
                raise RuntimeError(f"selected prospective row is outside raw domain: {cid}")
            if parity["maximum_abs_barrier_delta_eV"] > 2.0e-14:
                raise RuntimeError(f"300 K parity failed for {cid}: {parity}")
            changed = {key: {"parent": parent[key], "offspring": child[key]}
                       for key in child if key in parent and child[key] != parent[key]}
            selected.append({
                "candidate_id": cid, "parent_candidate_id": record["candidate_id"],
                "family": record["family"], "target": "P40",
                "cleavage_hits": child["physics__cleavage_hits"],
                "cleavage_tau_s": child["physics__cleavage_correlation_time_s"],
                "complete_row_sha256": canonical_hash(child),
                "changed_fields_json": json.dumps(changed, sort_keys=True),
                "complete_parameter_vector": json.dumps(child, sort_keys=True),
                "selection_metrics_json": json.dumps({k: v for k, v in p3.items() if k != "row"}, sort_keys=True),
                "p4_parameters_json": json.dumps(p4_parameters, sort_keys=True),
                "raw_domain_audit_json": json.dumps(audit, sort_keys=True),
                "reference_300K_parity_json": json.dumps(parity, sort_keys=True),
                "realization": "existing linear EXP_floor P3 plus existing reference-preserving P4 transport coordinates",
            })
    with (args.out_root / "thermal_accessibility_scan.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scan_rows[0])); writer.writeheader(); writer.writerows(scan_rows)
    with (args.out_root / "thermal_state_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(selected[0])); writer.writeheader(); writer.writerows(selected)
    decision = {
        "schema": "r10.thermal-accessibility-iteration/1",
        "source": {"path": str(args.joint_registry.resolve()),
                   "sha256": hashlib.sha256(args.joint_registry.read_bytes()).hexdigest()},
        "analytical_rows": len(scan_rows), "selected_full_state_rows": len(selected),
        "temperatures_K": TEMPERATURES,
        "selection": [{k: v for k, v in row.items() if k != "complete_parameter_vector"} for row in selected],
        "preserved_controls": [
            "R10J_DBTT_P40_LSHIFT_036MEV_V1 complete-state 300 K fracture/fatigue control",
            "R10_DBTT_P40_THERMAL_ANCHOR_V1 separate low-scale control",
        ],
        "interpretation": [
            "floor action ceiling is a certified opening-channel exclusion only",
            "A0 topology is an acquisition descriptor, not a material-class promotion",
            "paired P4 rows require complete-state causal comparison",
            "all selected rows preserve every sampled 300 K process barrier to floating-point tolerance",
        ],
    }
    (args.out_root / "thermal_accessibility_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    accessibility = {
        "schema": "r10.opening-accessibility-audit/1",
        "contract": {"exposure_s": 16000.0, "pending_mark": 1.0,
                     "multiplicity": 1.0, "channel": "opening"},
        "unit_mark_floor_threshold_eV": {
            str(int(temperature)): maximum_floor_for_mark(
                temperature_K=temperature, attempt_frequency_s=1.0e12,
                hits=3.0, tau_s=1.0e-6, exposure_s=16000.0,
                pending_mark=1.0,
            ) for temperature in (750.0, 1200.0)
        },
        "rows": [],
        "interpretation": "certified necessary opening-channel bound only; not a coupled-model rejection",
    }
    audit_rows = []
    for record in bases:
        if record["candidate_id"] == "R10J_DBTT_P40_LSHIFT_036MEV_V1":
            audit_rows.append((record["candidate_id"], json.loads(record["complete_parameter_vector"]), "preserved_control"))
    audit_rows.extend((row["candidate_id"], json.loads(row["complete_parameter_vector"]), "prospective_selected") for row in selected)
    for candidate_id, row, role in audit_rows:
        surface = exp_surface(row, "cleave", 1.0e12)
        for temperature in (750.0, 1200.0):
            bound = opening_floor_bound(
                surface, temperature_K=temperature,
                hits=float(row["physics__cleavage_hits"]),
                tau_s=float(row["physics__cleavage_correlation_time_s"]),
                exposure_s=16000.0, pending_mark=1.0,
            )
            accessibility["rows"].append({
                "candidate_id": candidate_id, "role": role,
                **bound.__dict__,
            })
    (args.out_root / "accessibility_bound_audit.json").write_text(
        json.dumps(accessibility, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps({"analytical_rows": len(scan_rows), "selected": [x["candidate_id"] for x in selected]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
