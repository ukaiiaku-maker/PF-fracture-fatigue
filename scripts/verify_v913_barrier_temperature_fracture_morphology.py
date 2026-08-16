#!/usr/bin/env python3
"""Fail-closed verifier for the v9.13 barrier/temperature analysis artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import pandas as pd
from PIL import Image


REPO=Path(__file__).resolve().parents[1]
DEFAULT_OUT=REPO/"runs/v913_barrier_temperature_fracture_morphology_v1"
SIM_SHA="559425321b9a8739f32788322d8a1c2af8abad73"
BRANCH="codex/v9.13-barrier-temperature-fracture-morphology"
TABLES=["fracture_temperature_master.csv","fracture_response_descriptors.csv","fracture_barrier_geometry_descriptors.csv",
 "fracture_entropy_descriptors.csv","fracture_barrier_interaction_descriptors.csv","fracture_state_at_first_passage.csv",
 "fracture_barrier_temperature_correlations.csv","fracture_barrier_temperature_partial_correlations.csv","fracture_response_pca_scores.csv",
 "fracture_barrier_surface_pca_scores.csv","fracture_temperature_morphology_clusters.csv","fracture_1D_2D_validation_subset.csv"]
FIGURES=["temperature_response_population.png","temperature_response_canonical_four.png","entropy_vs_fracture_slopes.png",
 "differential_entropy_vs_DBTT.png","kinetic_crossover_vs_DBTT_temperature.png","crossover_sharpness_vs_DBTT_width.png",
 "entropy_vs_peak_amplitude.png","transport_entropy_vs_temperature_response.png","barrier_temperature_correlation_heatmap.png",
 "barrier_temperature_partial_correlation_heatmap.png","barrier_temperature_fracture_phase_map.png","barrier_surface_pca.png",
 "fracture_response_pca.png","barrier_modes_vs_fracture_modes.png","canonical_peak_t_mechanistic_overlay.png",
 "canonical_dbtt_mechanistic_overlay.png","canonical_weak_t_mechanistic_overlay.png","canonical_ceramic_like_mechanistic_overlay.png"]


def digest(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""): h.update(block)
    return h.hexdigest()


def fail(message: str):
    raise SystemExit(f"VERIFY_FAIL: {message}")


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=DEFAULT_OUT); ap.add_argument("--require-clean",action="store_true")
    args=ap.parse_args(); out=args.out.resolve()
    required=TABLES+FIGURES+["BARRIER_TEMPERATURE_FRACTURE_MORPHOLOGY_REPORT.md","analysis_audit.json","dataset_inventory.csv",
        "physics_hypothesis_tests.csv","barrier_surface_common_grid.csv","fracture_response_curve_points.csv",
        "barrier_temperature_model_comparison.csv","barrier_descriptor_strong_collinearity.csv"]
    missing=[x for x in required if not (out/x).is_file()]
    if missing: fail(f"missing artifacts: {missing}")
    audit=json.loads((out/"analysis_audit.json").read_text())
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip()
    branch=subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip()
    if branch!=BRANCH: fail(f"branch {branch} != {BRANCH}")
    if audit.get("analysis_git_sha")!=head: fail("analysis audit was not generated from current HEAD")
    if audit.get("simulation_git_sha")!=SIM_SHA or audit.get("physics_changed") is not False or audit.get("new_simulations_launched") is not False:
        fail("simulation provenance/analysis-only contract mismatch")
    if audit.get("candidate_count")!=400 or audit.get("temperature_case_count")!=3920 or audit.get("complete_temperature_case_count")!=3918 or audit.get("censored_temperature_case_count")!=2:
        fail("population counts mismatch")
    inventory=pd.read_csv(out/"dataset_inventory.csv")
    for row in inventory.itertuples():
        if not bool(row.exists): fail(f"missing indexed source {row.path}")
        path=Path(row.path)
        if digest(path)!=row.sha256: fail(f"indexed source hash changed: {path}")
    tables={name:pd.read_csv(out/name) for name in TABLES}
    if len(tables["fracture_temperature_master.csv"])!=3920: fail("temperature-master row count")
    if tables["fracture_temperature_master.csv"].status.value_counts().to_dict()!={"complete":3918,"right_censored_maximum_substeps":2}:
        fail("temperature-master censor semantics")
    temperature=tables["fracture_temperature_master.csv"]
    if temperature.nominal_displacement_rate_m_per_s.isna().any() or temperature.loading_map_sha256.nunique()!=1:
        fail("loading protocol/rate provenance incomplete")
    if tables["fracture_response_descriptors.csv"].candidate_id.nunique()!=400: fail("response population incomplete")
    canonical={"v913_zeroD_sobol_0242980","v913_zeroD_sobol_0202500","v913_zeroD_sobol_0129902","v913_zeroD_sobol_0077080"}
    if set(tables["fracture_1D_2D_validation_subset.csv"].candidate_id)!=canonical: fail("canonical validation rows")
    for name,frame in tables.items():
        if frame.empty: fail(f"empty table {name}")
        if "candidate_id" in frame:
            for col in ["simulation_git_sha","github_repository"]:
                if col not in frame or frame[col].isna().any(): fail(f"missing provenance {name}:{col}")
            if set(frame.simulation_git_sha.astype(str))!={SIM_SHA}: fail(f"mixed simulation provenance {name}")
        else:
            if "simulation_git_sha" not in frame or set(frame.simulation_git_sha.astype(str))!={SIM_SHA}: fail(f"correlation provenance {name}")
    state=tables["fracture_state_at_first_passage.csv"]
    if set(state.state_reconstruction_class)!={"PARTIAL_SAVED_FIRST_PASSAGE_PROXY"}: fail("saved-state scope mislabeled")
    if not state.missing_state_fields.str.contains("K_shield").all(): fail("missing state not disclosed")
    hyp=pd.read_csv(out/"physics_hypothesis_tests.csv")
    if set(hyp.hypothesis)!={f"H{i}" for i in range(1,10)}: fail("H1-H9 incomplete")
    if set(hyp.classification)-{"SUPPORTED","WEAK_SUPPORT","REJECTED","INSUFFICIENT_EVIDENCE"}: fail("invalid hypothesis class")
    models=pd.read_csv(out/"barrier_temperature_model_comparison.csv")
    if len(models)!=15 or models.cv_r2.isna().any() or set(models.canonical_holdout_n)!={4}:
        fail("ridge interaction/transport comparison incomplete")
    report=(out/"BARRIER_TEMPERATURE_FRACTURE_MORPHOLOGY_REPORT.md").read_text()
    for i in range(1,17):
        if f"{i}. **" not in report: fail(f"report question {i} absent")
    for fig in FIGURES:
        path=out/fig; stem=path.stem
        side=out/f"{stem}_plot_data.csv"
        if not side.is_file() or pd.read_csv(side).empty: fail(f"plot sidecar absent/empty: {fig}")
        with Image.open(path) as image:
            if image.width<800 or image.height<500: fail(f"undersized figure: {fig}")
            image.verify()
    if args.require_clean:
        status=subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True)
        if status.strip(): fail("worktree is not clean")
    print(json.dumps({"status":"PASS","branch":branch,"head":head,"tables":len(TABLES),"figures":len(FIGURES),
                      "candidates":400,"temperature_cases":3920,"complete":3918,"censored":2},indent=2))
    return 0


if __name__=="__main__": raise SystemExit(main())
