#!/usr/bin/env python3
"""Executable acceptance gate for the v9.14 barrier/morphology analysis."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
REQUIRED_CSV = [
    "barrier_fatigue_master.csv", "fatigue_curve_points.csv", "fatigue_morphology_descriptors.csv",
    "barrier_geometry_descriptors.csv", "barrier_fatigue_correlation_master.csv",
    "barrier_fatigue_correlations.csv", "barrier_fatigue_partial_correlations.csv",
    "barrier_shape_pca_scores.csv", "fatigue_shape_pca_scores.csv", "fatigue_morphology_clusters.csv",
]
REQUIRED_FIGURES = [
    "correlation_heatmap_regime_slopes", "correlation_heatmap_transition_locations",
    "correlation_heatmap_transition_widths", "correlation_heatmap_kinetic_competition",
    "barrier_crossover_vs_knee_location", "barrier_crossover_vs_lcf_location",
    "vhcf_slope_vs_barrier_sensitivity", "hcf_slope_vs_effective_barrier_sensitivity",
    "lcf_slope_vs_highK_barrier_sensitivity", "knee_slope_change_vs_crossover_sharpness",
    "lcf_slope_recovery_vs_rate_ratio", "barrier_geometry_fatigue_phase_map",
    "barrier_shape_pca", "fatigue_shape_pca", "barrier_mode_vs_fatigue_mode",
    "fatigue_morphology_cluster_map",
]


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def check(condition: bool, message: str, failures: list[str]) -> None:
    if not bool(condition): failures.append(message)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=REPO / "runs/v914_barrier_fatigue_morphology_analysis_v1")
    ap.add_argument("--require-clean", action="store_true")
    args = ap.parse_args(); out = args.out.resolve(); failures: list[str] = []

    for name in REQUIRED_CSV:
        check((out / name).exists() and (out / name).stat().st_size > 100, f"missing/empty {name}", failures)
    check((out / "BARRIER_GEOMETRY_FATIGUE_MORPHOLOGY_REPORT.md").exists(), "missing scientific report", failures)
    check((out / "analysis_audit.json").exists(), "missing analysis audit", failures)
    for stem in REQUIRED_FIGURES:
        check((out / f"{stem}.png").exists() and (out / f"{stem}.png").stat().st_size > 5000, f"missing/invalid {stem}.png", failures)
        check((out / f"{stem}_plot_data.csv").exists(), f"missing plot-data sidecar for {stem}", failures)

    if failures:
        print(json.dumps({"status": "FAIL", "failures": failures}, indent=2)); raise SystemExit(1)

    master = pd.read_csv(out / "barrier_fatigue_master.csv")
    points = pd.read_csv(out / "fatigue_curve_points.csv")
    morph = pd.read_csv(out / "fatigue_morphology_descriptors.csv")
    barriers = pd.read_csv(out / "barrier_geometry_descriptors.csv")
    corr = pd.read_csv(out / "barrier_fatigue_correlations.csv")
    probes = pd.read_csv(out / "mechanism_probe_points.csv")
    audit = json.loads((out / "analysis_audit.json").read_text())

    check(len(master) >= 1280, "master does not cover broad global+local populations", failures)
    check(master.parameter_fingerprint.nunique() == len(master), "master is not one row per physical fingerprint", failures)
    check(master.accelerated_HCF_data_exists.sum() >= 130, "fatigue database coverage regressed", failures)
    check(len(points) >= 1400, "provenance-resolved fatigue point inventory is incomplete", failures)
    check(points.physical_candidate_id.nunique() >= 135, "fatigue candidate coverage regressed", failures)
    check(points.loc[points.censor_marker_class.ne("FINITE_RATE"), "da_dN_m_per_cycle"].isna().all(), "censor/partial rows contain artificial finite rates", failures)
    invalid = points.authoritative_reason.eq("ACCELERATED_HIGH_K_UNVALIDATED_FOR_LCF")
    check(invalid.any() and (~points.loc[invalid, "authoritative_use"]).all(), "invalid high-K accelerated points entered authoritative curve", failures)
    check((points.integration_mode.eq("explicit") & points.authoritative_use).any(), "explicit-cycle records missing", failures)
    check(morph.S_K_HCF.notna().sum() >= 125, "HCF slopes insufficiently resolved", failures)
    check(morph.S_K_LCF.notna().sum() >= 5, "explicit LCF slopes insufficiently resolved", failures)
    check(morph.fatigue_morphology.eq("REENTRY").any(), "arrest/re-entry morphology was lost", failures)
    check(morph.W_knee_f.dropna().ge(0).all(), "invalid knee width", failures)
    check(len(barriers) == len(master), "barrier descriptor/master row mismatch", failures)
    check(barriers.kinetic_crossover_topology.isin(["NONE", "SINGLE", "MULTIPLE"]).all(), "invalid crossover topology", failures)
    check(probes.candidate_id.nunique() == 475, "475-row state-probe population not recovered", failures)
    check("REDUCED_VALID" in set(corr.subset), "reduced-valid analysis subset absent", failures)
    bad_responses = corr.response.astype(str).str.contains(r"_(?:se|ci_low|ci_high|r2|n|span|quality)$", regex=True)
    check(not bad_responses.any(), "fit uncertainty columns were misused as correlation responses", failures)
    check(corr.loc[corr.test_status.eq("EXPLORATORY_N3_N4"), ["pearson_p", "spearman_p"]].isna().all().all(), "small-n exploratory correlations carry inferential p-values", failures)
    check(audit.get("analysis_only_no_simulations_launched") is True and audit.get("physics_changed") is False, "audit does not preserve analysis-only/physics-neutral contract", failures)
    check(audit.get("repository_head") == git("rev-parse", "HEAD"), "artifact HEAD does not match repository HEAD", failures)
    check(audit.get("required_outputs_present") and audit.get("required_figures_present") and audit.get("plot_data_sidecars_present"), "audit acceptance flags not all true", failures)
    report = (out / "BARRIER_GEOMETRY_FATIGUE_MORPHOLOGY_REPORT.md").read_text()
    for phrase in ["VHCF slope predictor", "HCF slope predictor", "LCF slope predictor", "Most efficient prospective tests", "Explicit hypothesis tests", "Central conclusion"]:
        check(phrase in report, f"report omits required answer: {phrase}", failures)
    diff_check = subprocess.run(["git", "diff", "--check"], cwd=REPO, text=True, capture_output=True)
    check(diff_check.returncode == 0, f"git diff --check failed: {diff_check.stdout}{diff_check.stderr}", failures)
    if args.require_clean:
        check(git("status", "--short") == "", "worktree is not clean", failures)

    result = {
        "status": "PASS" if not failures else "FAIL", "failures": failures,
        "branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
        "physical_candidates": len(master), "fatigue_candidates": points.physical_candidate_id.nunique(),
        "fatigue_points": len(points), "probe_candidates": probes.candidate_id.nunique(),
        "resolved_HCF_slopes": int(morph.S_K_HCF.notna().sum()),
        "resolved_LCF_slopes": int(morph.S_K_LCF.notna().sum()),
        "required_figures": len(REQUIRED_FIGURES),
    }
    (out / "verification_result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures: raise SystemExit(1)


if __name__ == "__main__":
    main()
