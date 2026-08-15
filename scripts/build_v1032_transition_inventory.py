#!/usr/bin/env python3
"""Build the immutable pre-refinement HCF/LCF inventory.

This is deliberately a merge-only tool.  It reads authoritative terminal
artifacts from the completed campaigns and never writes into their roots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_v1032_hybrid_hcf_lcf as prior


MATERIALS = {
    "DBTT": ("v913_zeroD_sobol_0202500", "v913_paper_dbtt01_0202500_persistent_sites", 21.02530765128298),
    "Peak-T": ("v913_zeroD_sobol_0242980", "v913_paper_peak01_0242980_persistent_sites", 21.289546465050222),
    "weak-T": ("v913_zeroD_sobol_0129902", "v913_paper_weakT01_0129902_persistent_sites", 12.702935563752424),
    "ceramic-like": ("v913_zeroD_sobol_0077080", "v913_paper_ceramic01_0077080_persistent_sites", 12.259477791864454),
}
ID_TO_FAMILY = {identifier: family for family, values in MATERIALS.items() for identifier in values[:2]}
SEEDS = {"DBTT": 1720, "Peak-T": 1720, "weak-T": 2001726, "ceramic-like": 3001729}
FIELDS = [
    "family", "candidate_id", "parameter_option", "original_temperature_class",
    "dimensionality", "integration_mode", "deltaK_MPa_sqrt_m", "normalized_f",
    "da_dN_m_per_cycle", "cycles_to_target", "extension_um", "event_count",
    "median_event_interval_cycles", "minimum_event_interval_cycles",
    "mean_event_interval_cycles", "subcycle_fraction", "fraction_below_10_cycles",
    "fraction_below_0p1_cycle", "censor_status", "plot_kind", "seed",
    "source_run_root", "result_path", "run_contract_path", "repository_head",
    "registry_sha256", "candidate_fingerprint_sha256",
]


def number(value, default=math.nan):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def registry_fingerprints(registry: Path) -> dict[str, str]:
    rows = list(csv.DictReader(registry.open()))
    return {
        row["candidate_id"]: digest_bytes(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        for row in rows
    } | {
        row["option_key"]: digest_bytes(json.dumps(row, sort_keys=True, separators=(",", ":")).encode())
        for row in rows
    }


def contract_for(path: Path) -> tuple[dict, Path | None]:
    base = path if path.is_dir() else path.parent
    for name in ("run_contract.json", "hybrid_launch_contract.json", "high_cycle_run_manifest.json"):
        candidate = base / name
        if candidate.exists():
            try:
                return json.loads(candidate.read_text()), candidate
            except Exception:
                return {}, candidate
    return {}, None


def normalized(row: dict, fingerprints: dict[str, str]) -> dict:
    cls = str(row.get("class", ""))
    family = "Peak-T" if cls == "Peak" else cls
    candidate = str(row.get("candidate_id", ""))
    if candidate in ID_TO_FAMILY:
        family = ID_TO_FAMILY[candidate]
    result = Path(str(row.get("result_path", "")))
    contract, contract_path = contract_for(result)
    option = str(contract.get("parameter_option", ""))
    if not option and family in MATERIALS:
        option = MATERIALS[family][1]
    candidate_id = MATERIALS[family][0] if family in MATERIALS else candidate
    seed = contract.get("seed", contract.get("hazard_seed", SEEDS.get(family, 1720)))
    status = str(row.get("status", ""))
    return {
        "family": family,
        "candidate_id": candidate_id,
        "parameter_option": option,
        "original_temperature_class": family if family in MATERIALS else "mechanism-control",
        "dimensionality": row.get("dimensionality"),
        "integration_mode": row.get("integration_mode"),
        "deltaK_MPa_sqrt_m": number(row.get("deltaK_MPa_sqrt_m")),
        "normalized_f": number(row.get("normalized_f")),
        "da_dN_m_per_cycle": number(row.get("da_dN_m_per_cycle")),
        "cycles_to_target": number(row.get("cycles_to_target")),
        "extension_um": number(row.get("extension_um")),
        "event_count": number(row.get("event_count")),
        "median_event_interval_cycles": number(row.get("median_interval_cycles")),
        "minimum_event_interval_cycles": number(row.get("minimum_interval_cycles")),
        "mean_event_interval_cycles": number(row.get("mean_interval_cycles")),
        "subcycle_fraction": number(row.get("subcycle_fraction")),
        "fraction_below_10_cycles": number(row.get("fraction_below_10_cycles")),
        "fraction_below_0p1_cycle": number(row.get("fraction_below_0p1_cycle")),
        "censor_status": status,
        "plot_kind": row.get("plot_kind"),
        "seed": seed,
        "source_run_root": row.get("source_campaign"),
        "result_path": str(result),
        "run_contract_path": str(contract_path) if contract_path else "",
        "repository_head": contract.get("repository_head", contract.get("git_head", "")),
        "registry_sha256": contract.get("registry_sha256", ""),
        "candidate_fingerprint_sha256": fingerprints.get(candidate, fingerprints.get(option, "")),
    }


def material_accelerated_2d(repo: Path) -> list[dict]:
    path = repo / "runs/1_Final_result_v10_2_30_four_class_1e3_rate_complete_20260810/ladder_analysis/four_class_driving_force_ladder.csv"
    data = pd.read_csv(path)
    mapping = {"peak": "Peak-T", "dbtt": "DBTT", "weakt": "weak-T", "ceramic": "ceramic-like"}
    rows = []
    for _, source in data.iterrows():
        family = mapping[str(source["class"]).lower()]
        candidate, option, _ = MATERIALS[family]
        status = str(source["status"])
        rate = number(source.get("developed_da_dN_m_per_cycle"))
        plot_kind = "censor" if "censor" in status else "partial" if status == "incomplete" else "resolved"
        if plot_kind != "resolved":
            rate = math.nan
        rows.append({
            "class": family, "candidate_id": candidate,
            "deltaK_MPa_sqrt_m": number(source["deltaK_MPa_sqrt_m"]),
            "normalized_f": number(source["f"]), "dimensionality": "2D",
            "integration_mode": "accelerated", "da_dN_m_per_cycle": rate,
            "cycles_to_target": number(source["cycles_to_target"]),
            "extension_um": number(source["projected_extension_um"]),
            "event_count": number(source["event_count"]),
            "subcycle_fraction": number(source["fraction_subcycle_intervals"]),
            "minimum_interval_cycles": number(source["min_event_spacing_cycles"]),
            "median_interval_cycles": number(source["median_event_spacing_cycles"]),
            "mean_interval_cycles": number(source["mean_event_spacing_cycles"]),
            "fraction_below_10_cycles": math.nan, "fraction_below_0p1_cycle": math.nan,
            "status": status, "plot_kind": plot_kind,
            "source_campaign": path.parts[-3], "result_path": str(source["output_root"]),
            "parameter_option": option,
        })
    return rows


def assemble(repo: Path, one_d_root: Path, two_d_root: Path) -> list[dict]:
    prior.CANON_IDS.update({
        MATERIALS["weak-T"][0]: "weak-T", MATERIALS["weak-T"][1]: "weak-T",
        MATERIALS["ceramic-like"][0]: "ceramic-like", MATERIALS["ceramic-like"][1]: "ceramic-like",
    })
    prior.MECHANISMS.update({
        "weak-T": "canonical weak-temperature dependence",
        "ceramic-like": "canonical ceramic-like response",
    })
    rows = (
        prior.accelerated_1d_abcd(repo)
        + prior.accelerated_1d_canonical(repo)
        + prior.explicit_1d(repo / "runs/v914_endurance_knee_ABCD_hybrid_HCF_LCF_v1")
        + prior.accelerated_2d_abcd(repo)
        + material_accelerated_2d(repo)
        + prior.explicit_2d(repo / "runs/v10_2_32_endurance_knee_ABCD_hybrid_HCF_LCF_v1")
    )
    if one_d_root.exists():
        rows += prior.explicit_1d(one_d_root)
    if two_d_root.exists():
        rows += prior.explicit_2d(two_d_root)
    return prior.deduplicate(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--one-d-root", type=Path, default=Path("runs/v914_HCF_LCF_transition_refinement_v2"))
    parser.add_argument("--two-d-root", type=Path, default=Path("runs/v10_2_32_HCF_LCF_transition_refinement_v2"))
    parser.add_argument("--out", type=Path, default=Path("runs/v10_2_32_HCF_LCF_transition_refinement_v2/analysis/existing_hybrid_inventory.csv"))
    args = parser.parse_args(); repo = args.repo.resolve()
    registry = repo / "arrhenius_fracture/data/materials/v10_2_27_paper_four_class_registry.csv"
    fingerprints = registry_fingerprints(registry)
    rows = [normalized(row, fingerprints) for row in assemble(repo, args.one_d_root, args.two_d_root)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=FIELDS).sort_values(
        ["family", "dimensionality", "integration_mode", "deltaK_MPa_sqrt_m"]
    ).to_csv(args.out, index=False)
    audit = {
        "schema": "v10.2.32_transition_existing_inventory_v1",
        "repository": str(repo),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "worktree_status": subprocess.check_output(["git", "status", "--short", "--branch"], cwd=repo, text=True).strip(),
        "canonical_registry": str(registry),
        "canonical_registry_sha256": digest_bytes(registry.read_bytes()),
        "row_count": len(rows),
        "source_campaigns": sorted({str(row["source_run_root"]) for row in rows}),
    }
    args.out.with_name("existing_hybrid_inventory_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
