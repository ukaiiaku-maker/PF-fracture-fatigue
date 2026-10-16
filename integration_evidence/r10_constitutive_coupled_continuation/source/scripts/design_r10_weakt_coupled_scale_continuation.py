#!/usr/bin/env python3
"""Create the post-reconciliation Weak-T candidate without endpoint fitting."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.design_r10_thermal_accessibility_iteration import (
    canonical_hash, domain_audit, exp_surface,
)
from scripts.design_r10_weakt_active_acquisition import surface_audit


PARENT = "R10T_WEAKT_P40_G1962_S4P5GPA_P4_EP055_PP010_V1"

def candidate_id(target_GPa: float) -> str:
    text = format(float(target_GPa), ".6g")
    if "." not in text:
        text += ".0"
    tag = text.replace(".", "P")
    return f"R10K_WEAKT_COUPLED_EMIT_SCALE_S{tag}_V1"


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--lineage-audit", type=Path, required=True)
    p.add_argument("--out-root", type=Path, required=True)
    p.add_argument("--target-sigc-300-GPa", type=float, action="append")
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
    Tref = float(parent["Tref_K"])
    parent_sigc300 = (
        float(parent["emit_sigc0_GPa"])
        + float(parent["emit_sT_GPa_per_K"]) * (300.0 - Tref)
    )
    targets = a.target_sigc_300_GPa or [3.0]
    rows = []
    decisions = []
    for target in targets:
        scale = float(target) / parent_sigc300
        child = dict(parent)
        child["emit_sigc0_GPa"] = scale * float(parent["emit_sigc0_GPa"])
        child["emit_sT_GPa_per_K"] = scale * float(parent["emit_sT_GPa_per_K"])
        cid = candidate_id(target)
        child["candidate_id"] = cid
        child["role"] = "R10 constitutively coupled Weak-T continuation"
        child["mechanism_summary"] = (
            "single coherent emission sigc(T) scale with original Arrhenius/EXP_floor "
            "energy and entropy laws; Peierls/Taylor inherit the same stress scale"
        )

        allowed_metadata = {"candidate_id", "role", "mechanism_summary"}
        physical_changes = {
            k for k in child if parent.get(k) != child[k] and k not in allowed_metadata
        }
        if physical_changes != {"emit_sigc0_GPa", "emit_sT_GPa_per_K"}:
            raise RuntimeError(f"unexpected physical changes: {sorted(physical_changes)}")
        ratio0 = float(child["emit_sigc0_GPa"]) / float(parent["emit_sigc0_GPa"])
        ratioT = float(child["emit_sT_GPa_per_K"]) / float(parent["emit_sT_GPa_per_K"])
        if abs(ratio0 - ratioT) > 1.0e-13:
            raise RuntimeError("emission sigc0/sT coupling was broken")
        for key in ("emit_G00_eV", "emit_gT_eV_per_K",
                    "peierls_H0_eV", "peierls_activation_entropy_kB",
                    "taylor_H0_eV", "taylor_activation_entropy_kB"):
            if child[key] != parent[key]:
                raise RuntimeError(f"energy/entropy field changed: {key}")

        endpoint = domain_audit(child)
        audit = surface_audit(child)
        if not endpoint["all_processes_valid"] or not audit["all_raw_domains_valid"]:
            raise RuntimeError("coupled candidate left the raw constitutive domain")
        emission = exp_surface(child, "emit", 1.0e11)
        sigc300 = (emission.sigc0_Pa + emission.sT_Pa_per_K * (300.0 - emission.Tref_K)) / 1.0e9
        sigc1200 = (emission.sigc0_Pa + emission.sT_Pa_per_K * (1200.0 - emission.Tref_K)) / 1.0e9
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
            "parent_emit_sigc_300_GPa": parent_sigc300,
            "offspring_emit_sigc_300_GPa": sigc300,
            "offspring_emit_sigc_1200_GPa": sigc1200,
            "coherent_stress_scale": scale,
            "historical_reason_for_300K_scale": (
                "preserved S3.0 native-fatigue result; value reused as a physical scale target, "
                "not as an independently fitted temperature endpoint"
            ),
        }, sort_keys=True),
        "p4_parameters_json": json.dumps({
            "coordinate": "coherent_emission_characteristic_stress_scale",
            "scale": scale,
            "emit_sigc_300_GPa": sigc300,
            "emit_sigc_1200_GPa": sigc1200,
        }, sort_keys=True),
        "raw_domain_audit_json": json.dumps(audit, sort_keys=True),
        "endpoint_domain_audit_json": json.dumps(endpoint, sort_keys=True),
        "constitutive_lineage_audit_sha256": canonical_hash(lineage),
        "realization": "original Family-A EXP_floor with one coupled stress-scale coordinate",
        }
        rows.append(row)
        decisions.append({
            "candidate_id": cid,
            "complete_row_sha256": row["complete_row_sha256"],
            "coherent_stress_scale": scale,
            "emit_sigc_300_GPa": sigc300,
            "emit_sigc_1200_GPa": sigc1200,
        })
    a.out_root.mkdir(parents=True, exist_ok=True)
    out = a.out_root / "coupled_scale_candidates.csv"
    with out.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    decision = {
        "schema": "r10.weakt-coupled-scale-continuation/1",
        "parent_candidate_id": PARENT,
        "candidates": decisions,
        "independent_endpoint_fit": False,
        "activation_entropy_changed": False,
        "next_gate": "refined seven-temperature current-backend fracture panel",
    }
    (a.out_root / "coupled_scale_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
