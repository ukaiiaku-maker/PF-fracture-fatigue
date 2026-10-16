#!/usr/bin/env python3
"""Apply an established coherent opening stress scale to the reconciled row."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.design_r10_thermal_accessibility_iteration import canonical_hash, domain_audit
from scripts.design_r10_weakt_active_acquisition import surface_audit


PARENT = "R10K_WEAKT_COUPLED_EMIT_SCALE_S3P0_V1"


def tag(value: float) -> str:
    return format(value, ".6g").replace(".", "P")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--lineage-audit", type=Path, required=True)
    p.add_argument("--opening-stress-scale", type=float, action="append", required=True)
    p.add_argument("--out-root", type=Path, required=True)
    a = p.parse_args(argv)
    lineage = json.loads(a.lineage_audit.read_text())
    if lineage["decision"] != "RECONCILED_BEFORE_FURTHER_THERMAL_GENERATION":
        raise RuntimeError("constitutive lineage was not reconciled")
    with a.registry.open(newline="") as stream:
        matches = [x for x in csv.DictReader(stream) if x["candidate_id"] == PARENT]
    if len(matches) != 1:
        raise KeyError(PARENT)
    source = matches[0]
    parent = json.loads(source["complete_parameter_vector"])
    rows = []
    decisions = []
    for scale in a.opening_stress_scale:
        if not 0.0 < scale:
            raise ValueError("opening scale must be positive")
        child = dict(parent)
        child["cleave_sigc0_GPa"] = scale * float(parent["cleave_sigc0_GPa"])
        child["cleave_sT_GPa_per_K"] = scale * float(parent["cleave_sT_GPa_per_K"])
        cid = f"R10K_WEAKT_COUPLED_S3P0_OPEN_SCALE_{tag(scale)}_V1"
        child["candidate_id"] = cid
        child["role"] = "R10 constitutively coupled Weak-T opening-scale continuation"
        child["mechanism_summary"] = (
            "coherent emission and opening sigc(T) scales with unchanged affine "
            "free-energy slopes, nonzero transport entropies, and state evolution"
        )
        metadata = {"candidate_id", "role", "mechanism_summary"}
        changed_physical = {
            k for k in child if child[k] != parent.get(k) and k not in metadata
        }
        if changed_physical != {"cleave_sigc0_GPa", "cleave_sT_GPa_per_K"}:
            raise RuntimeError(f"unexpected physical changes: {sorted(changed_physical)}")
        ratio0 = float(child["cleave_sigc0_GPa"]) / float(parent["cleave_sigc0_GPa"])
        ratioT = float(child["cleave_sT_GPa_per_K"]) / float(parent["cleave_sT_GPa_per_K"])
        if abs(ratio0 - ratioT) > 1.0e-13:
            raise RuntimeError("opening sigc0/sT coupling was broken")
        preserved = (
            "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_exp_a",
            "cleave_exp_n", "cleave_floor_frac", "emit_G00_eV",
            "emit_gT_eV_per_K", "peierls_H0_eV",
            "peierls_activation_entropy_kB", "taylor_H0_eV",
            "taylor_activation_entropy_kB",
        )
        if any(child[k] != parent[k] for k in preserved):
            raise RuntimeError("an energy, entropy, shape, or floor field changed")
        endpoint = domain_audit(child)
        audit = surface_audit(child)
        if not endpoint["all_processes_valid"] or not audit["all_raw_domains_valid"]:
            raise RuntimeError(f"raw domain failure for {cid}")
        changed = {
            k: {"parent": parent.get(k), "offspring": v}
            for k, v in child.items() if parent.get(k) != v
        }
        row = {
            "candidate_id": cid,
            "parent_candidate_id": PARENT,
            "family": source["family"],
            "target": source["target"],
            "cleavage_hits": source["cleavage_hits"],
            "cleavage_tau_s": source["cleavage_tau_s"],
            "complete_row_sha256": canonical_hash(child),
            "changed_fields_json": json.dumps(changed, sort_keys=True),
            "complete_parameter_vector": json.dumps(child, sort_keys=True),
            "selection_metrics_json": json.dumps({
                "opening_stress_scale": scale,
                "parent_refined_hot_roots_MPa_sqrt_m": {"1050": 9.365396066606667, "1200": 8.580168626827168},
                "selection_role": "coupled stress-scale response map; no temperature endpoint target",
            }, sort_keys=True),
            "p4_parameters_json": json.dumps({
                "emission_coordinate": "coherent_emission_characteristic_stress_scale",
                "opening_coordinate": "coherent_opening_characteristic_stress_scale",
                "opening_scale": scale,
            }, sort_keys=True),
            "raw_domain_audit_json": json.dumps(audit, sort_keys=True),
            "endpoint_domain_audit_json": json.dumps(endpoint, sort_keys=True),
            "constitutive_lineage_audit_sha256": canonical_hash(lineage),
            "realization": "original Family-A EXP_floor with coupled stress-scale coordinates",
        }
        rows.append(row)
        decisions.append({
            "candidate_id": cid,
            "complete_row_sha256": row["complete_row_sha256"],
            "opening_stress_scale": scale,
        })
    a.out_root.mkdir(parents=True, exist_ok=True)
    with (a.out_root / "coupled_opening_scale_candidates.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    decision = {
        "schema": "r10.weakt-coupled-opening-scale/1",
        "parent_candidate_id": PARENT,
        "candidates": decisions,
        "independent_endpoint_fit": False,
        "affine_energy_slopes_changed": False,
        "transport_activation_entropy_changed": False,
        "next_gate": "refined 900/1050/1200 K response map",
    }
    (a.out_root / "coupled_opening_scale_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
