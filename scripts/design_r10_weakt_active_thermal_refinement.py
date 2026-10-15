#!/usr/bin/env python3
"""P3 hot-fracture refinement of the selected active Weak-T cold row."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.design_r10_thermal_accessibility_iteration import (
    canonical_hash, domain_audit, parity_300, set_exp_linear_coordinates,
)


PARENT = "R10A_WEAKT_ACTIVE_EMIT_S3P0GPA_V1"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args(argv)
    args.out_root.mkdir(parents=True, exist_ok=True)
    records = list(csv.DictReader(args.registry.open()))
    source = next(x for x in records if x["candidate_id"] == PARENT)
    parent = json.loads(source["complete_parameter_vector"])

    output = []
    specs = [(gT, 4.5) for gT in (
        0.00150, 0.00125, 0.00100, 0.001900, 0.001925, 0.001950,
    )]
    specs.extend((0.001900, endpoint) for endpoint in (6.0, 8.0, 10.0))
    # The executed high-sigc side raised the 1050 K root faster than the
    # 1200 K root.  Continue on the informative low-sigc side at the gT value
    # that already brackets the desired hot scale.
    specs.extend((0.001950, endpoint) for endpoint in (3.0, 3.5, 4.0))
    for gT, sigc_1200_GPa in specs:
        child = set_exp_linear_coordinates(
            parent, "cleave", gT=gT, sigc_1200_GPa=sigc_1200_GPa,
        )
        tag = str(int(round(gT * 1.0e6))).zfill(4)
        stress_tag = str(sigc_1200_GPa).replace(".", "P")
        suffix = "" if sigc_1200_GPa == 4.5 else f"_CS{stress_tag}GPA"
        cid = f"R10A_WEAKT_ACTIVE_EMIT_S3P0GPA_CG{tag}{suffix}_V2"
        child["candidate_id"] = cid
        child["role"] = "R10 active Weak-T cold emission plus hot opening refinement"
        child["mechanism_summary"] = (
            "P4 cold emission-knee leverage with P3 opening continuation; "
            "complete 300 K process contract retained from active S3.0 row"
        )
        audit = domain_audit(child)
        parity = parity_300(parent, child)
        if not audit["all_processes_valid"]:
            raise RuntimeError(f"raw process domain failed: {cid}")
        if parity["maximum_abs_barrier_delta_eV"] > 2.0e-14:
            raise RuntimeError(f"300 K contract parity failed: {cid}: {parity}")
        changed = {
            key: {"parent": parent.get(key), "offspring": value}
            for key, value in child.items() if parent.get(key) != value
        }
        output.append({
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
                "failed_parent_refined_roots_MPa_sqrt_m": {
                    "1050": 9.365318329514345,
                    "1200": 6.832184141330956,
                },
                "cleave_gT_eV_per_K": gT,
                "cleave_sigc_1200_GPa": sigc_1200_GPa,
                "selection_role": "sequential P3 recovery bracket after dK refinement failure",
            }, sort_keys=True),
            "p4_parameters_json": source["p4_parameters_json"],
            "raw_domain_audit_json": json.dumps(audit, sort_keys=True),
            "reference_300K_parity_json": json.dumps(parity, sort_keys=True),
            "realization": "existing Family-A EXP_floor P3 thermal continuation on active P4 row",
        })

    path = args.out_root / "thermal_refinement_candidates.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(output)
    decision = {
        "schema": "r10.active-weakt-thermal-refinement/1",
        "parent_candidate_id": PARENT,
        "parent_complete_row_sha256": source["complete_row_sha256"],
        "trigger": "dK=0.01 hot-root refinement invalidated the coarse S3.0 Weak-T span",
        "cold_contract_preserved": True,
        "candidate_ids": [x["candidate_id"] for x in output],
        "selection": "execute refined 1050/1200 K first-event roots; expand only if bracket fails",
    }
    (args.out_root / "thermal_refinement_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(decision, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
