#!/usr/bin/env python3
"""Authoritative artifact gate for the v10.2.32 transition refinement."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


STEMS = [
    "material_families_four_path_da_dN_vs_deltaK",
    "material_families_hybrid_1D_2D_da_dN_vs_deltaK",
    "material_families_hybrid_normalized_f",
    "material_families_cycles_to_100um_vs_deltaK",
    "material_families_cycles_to_100um_vs_f",
    "material_families_spatial_enhancement_vs_deltaK",
    "material_families_spatial_enhancement_vs_f",
    "material_families_spatial_enhancement_vs_events_per_cycle",
    "material_families_event_density_vs_deltaK",
    "abcd_refined_four_path_da_dN_vs_deltaK",
    "abcd_spatial_enhancement_vs_deltaK",
    "abcd_spatial_enhancement_vs_f",
    "abcd_spatial_enhancement_vs_events_per_cycle",
    "HCF_LCF_switch_parity_map",
]
MATERIALS = {"DBTT", "Peak-T", "weak-T", "ceramic-like"}
IDS = {
    "DBTT": "v913_zeroD_sobol_0202500",
    "Peak-T": "v913_zeroD_sobol_0242980",
    "weak-T": "v913_zeroD_sobol_0129902",
    "ceramic-like": "v913_zeroD_sobol_0077080",
}


def require(condition: bool, message: str, errors: list[str]):
    if not condition:
        errors.append(message)


def verify(repo: Path, out: Path, require_clean: bool = True) -> dict:
    errors: list[str] = []
    main = out / "full_material_hybrid_rates.csv"
    require(main.exists(), "missing full_material_hybrid_rates.csv", errors)
    data = pd.read_csv(main) if main.exists() else pd.DataFrame()
    if not data.empty:
        require(set(data.family) == MATERIALS, "main table does not contain exactly four material families", errors)
        require(set(data.regime_classification).issubset({
            "VHCF_ACCELERATED", "HCF_ACCELERATED", "HCF_LCF_OVERLAP", "LCF_EXPLICIT",
            "SPATIAL_LCF", "NEAR_MONOTONIC_EXPLICIT", "CYCLE_CENSOR", "PARTIAL_UNRESOLVED",
        }), "unexpected regime label", errors)
        for family, candidate in IDS.items():
            require(set(data.loc[data.family.eq(family), "candidate_id"]) == {candidate},
                    f"wrong canonical candidate ID for {family}", errors)
        unresolved = data[data.plot_kind.ne("resolved")]
        require(unresolved.da_dN_m_per_cycle.isna().all(), "unresolved/censored row has a finite rate", errors)
        explicit = data[data.integration_mode.eq("explicit")]
        require(explicit.groupby("family").size().ge(5).all(), "material explicit sampling is incomplete", errors)
        require(set(["accelerated_explicit_ratio", "spatial_enhancement_ratio", "events_per_cycle"]).issubset(data.columns),
                "main diagnostics columns missing", errors)
    for name in ("HCF_LCF_switch_parity.csv", "spatial_enhancement_map.csv",
                 "transition_regime_summary.csv", "explicit_event_density_diagnostics.csv",
                 "D_spatial_bifurcation_state.csv"):
        path = out / name
        require(path.exists() and path.stat().st_size > 200, f"missing/empty {name}", errors)
    report = out / "MATERIAL_HCF_LCF_REFINEMENT_REPORT.md"
    require(report.exists() and report.stat().st_size > 3000, "scientific report missing or too short", errors)
    if report.exists():
        text = report.read_text()
        require(all(f"{i}. **" in text for i in range(1, 11)), "report does not answer all ten questions", errors)
        require("auto` is intentionally not implemented" in text, "auto-mode decision absent", errors)
    for stem in STEMS:
        for suffix in ("png", "pdf", "svg"):
            path = out / f"{stem}.{suffix}"
            require(path.exists() and path.stat().st_size > 1000, f"missing/empty {path.name}", errors)
        plot_data = out / f"{stem}_plot_data.csv"
        require(plot_data.exists() and plot_data.stat().st_size > 100, f"missing/empty {plot_data.name}", errors)
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if require_clean:
        require(not status.strip(), "worktree is not clean", errors)
    result = {"schema": "v10.2.32_transition_refinement_verifier_v1",
              "passed": not errors, "errors": errors, "artifact_root": str(out),
              "figure_stems": len(STEMS), "material_rows": len(data),
              "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
              "worktree_clean": not status.strip()}
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, default=Path("runs/v10_2_32_HCF_LCF_transition_refinement_v2/analysis"))
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    result = verify(args.repo.resolve(), args.out.resolve(), not args.allow_dirty)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
