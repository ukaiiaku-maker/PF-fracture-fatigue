#!/usr/bin/env python3
"""Aggregate v10.2.17 signed stochastic Stage 3 cases."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outroot", required=True, type=Path)
    args = parser.parse_args()
    root = args.outroot.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for path in sorted(root.glob("*/T*_th*/stage3_case_status.json")):
        payload = json.loads(path.read_text())
        selection_path = path.parent / "v10_2_17_parameter_selection.json"
        stack_path = path.parent / "v10_2_17_final_signed_stochastic_stack.json"
        selection = json.loads(selection_path.read_text()) if selection_path.is_file() else {}
        stack = json.loads(stack_path.read_text()) if stack_path.is_file() else {}
        summary = payload.get("summary", {})
        rows.append({
            "option_key": selection.get("option_key", path.parent.parent.name),
            "candidate_id": selection.get("candidate_id", ""),
            "temperature_K": payload.get("temperature_K"),
            "hazard_seed": stack.get("cleavage_hazard_seed"),
            "status": payload.get("status"),
            "complete": payload.get("complete"),
            "Kc_first_MPa_sqrt_m": payload.get("Kc_first_MPa_sqrt_m"),
            "projected_extension_um": payload.get("projected_extension_um"),
            "n_advances": summary.get("n_advances"),
            "n_fronts": summary.get("n_fronts"),
            "mpz_length_um": selection.get("mpz_length_um"),
            "mpz_n_bins": selection.get("mpz_n_bins"),
            "cleavage_hazard_mode": stack.get("cleavage_hazard_mode"),
            "event_length_mode": stack.get("event_length_mode"),
            "signed_engine": stack.get("final_engine"),
            "constitutive_K_shield_cap_applied": stack.get(
                "constitutive_K_shield_cap_applied"
            ),
            "case_root": str(path.parent),
        })
    columns = [
        "option_key", "candidate_id", "temperature_K", "hazard_seed", "status",
        "complete", "Kc_first_MPa_sqrt_m", "projected_extension_um",
        "n_advances", "n_fronts", "mpz_length_um", "mpz_n_bins",
        "cleavage_hazard_mode", "event_length_mode", "signed_engine",
        "constitutive_K_shield_cap_applied", "case_root",
    ]
    (root / "stage3_campaign_summary.json").write_text(
        json.dumps(rows, indent=2, sort_keys=True) + "\n"
    )
    with (root / "stage3_campaign_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print(json.dumps({"cases": len(rows), "status_counts": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
