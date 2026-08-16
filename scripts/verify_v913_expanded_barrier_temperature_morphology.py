#!/usr/bin/env python3
"""Fail-closed completion verifier for expanded v9.13 morphology analysis."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import pandas as pd
from PIL import Image

REPO=Path(__file__).resolve().parents[1]; DEFAULT=REPO/"runs/v913_barrier_temperature_fracture_morphology_v2"
BRANCH="codex/v9.13-barrier-temperature-fracture-morphology"; SIM="559425321b9a8739f32788322d8a1c2af8abad73"
TABLES=["expanded_barrier_temperature_descriptors.csv","expanded_univariate_correlations.csv","expanded_mutual_information.csv",
 "expanded_gam_performance.csv","expanded_interaction_models.csv","expanded_classification_models.csv","expanded_descriptor_collinearity.csv","expanded_feature_family_summary.csv"]
FIGURES=["expanded_descriptor_response_heatmap.png","expanded_mutual_information_rankings.png","relative_barrier_geometry_phase_map.png",
 "timescale_temperature_phase_map.png","barrier_position_width_phase_map.png","entropy_barrier_phase_map.png","DBTT_descriptor_maps.png",
 "PeakT_descriptor_maps.png","weakT_descriptor_maps.png","ceramic_descriptor_maps.png","descriptor_class_distributions.png",
 "expanded_descriptor_pca.png","canonical_four_on_discovered_phase_map.png"]

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def fail(x): raise SystemExit(f"VERIFY_FAIL: {x}")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=DEFAULT); ap.add_argument("--require-clean",action="store_true"); a=ap.parse_args(); out=a.out.resolve()
    extra=["expanded_descriptor_dictionary.md","BARRIER_TEMPERATURE_FRACTURE_MORPHOLOGY_REPORT.md","expanded_analysis_audit.json",
      "expanded_temperature_resolved_descriptors.csv","expanded_initiation_developed_response_descriptors.csv","expanded_hypothesis_tests.csv",
      "expanded_descriptor_pca_scores.csv","canonical_pair_comparison.csv","v1_artifact_manifest.csv"]
    missing=[x for x in TABLES+FIGURES+extra if not (out/x).is_file()]
    if missing: fail(f"missing {missing}")
    audit=json.loads((out/"expanded_analysis_audit.json").read_text()); head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(); branch=subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip()
    if branch!=BRANCH or audit.get("analysis_git_sha")!=head: fail("branch/HEAD audit mismatch")
    if audit.get("simulation_git_sha")!=SIM or audit.get("physics_changed") is not False or audit.get("new_simulations_launched") is not False: fail("analysis-only provenance")
    if audit.get("candidate_count")!=400 or audit.get("complete_temperature_case_count")!=3918 or audit.get("excluded_censored_case_count")!=2: fail("population/censor counts")
    if audit.get("expanded_feature_count",0)<150: fail("descriptor expansion too small")
    master=pd.read_csv(out/"expanded_barrier_temperature_descriptors.csv",low_memory=False); temp=pd.read_csv(out/"expanded_temperature_resolved_descriptors.csv",low_memory=False)
    if len(master)!=400 or master.candidate_id.nunique()!=400 or len(temp)!=3918: fail("descriptor row counts")
    for frame,name in [(master,"candidate"),(temp,"temperature")]:
        if "simulation_git_sha" not in frame or set(frame.simulation_git_sha.astype(str))!={SIM}: fail(f"{name} simulation provenance")
    required_cols=["delta_g_ec_at_K75","log10_rate_ratio_at_K75","delta_K50_MPa_sqrt_m","delta_K90_MPa_sqrt_m","width80_ratio_emit_over_cleave",
      "relative_first_derivative_at_closest","relative_curvature_at_closest","integrated_abs_barrier_separation_eV","integrated_abs_log_rate_separation",
      "kinetic_crossing_count","competition_temperature_width_log1_K","kinetic_crossover_sharpness_T_per_K","differential_entropy_kB",
      "entropy_separation_importance_at_K75","delta_full_dGdT_at_K75_eV_per_K","delta_mixed_derivative_at_K75_eV_per_MPa_sqrt_m_K",
      "log10_tau_c_over_tau_p_at_K75","dlog10_tau_c_over_tau_p_dT","state_delta_Gc_over_kBT"]
    if set(required_cols)-set(master): fail("priority descriptors missing")
    if master[required_cols].isna().all().any(): fail("empty priority descriptor")
    if set(master.kinetic_topology)-{"NO_CROSSING_CLEAVAGE_EASIER","NO_CROSSING_EMISSION_EASIER","SINGLE_CROSSING","MULTIPLE_CROSSING","NEAR_TANGENT","BROAD_NEAR_DEGENERACY"}: fail("bad topology")
    if not temp.state_missing_fields.str.contains("K_shield").all() or temp.state_K_shield_over_K_applied.notna().any(): fail("missing state fabricated")
    if not temp.frozen_path_proxy_not_evolved.all(): fail("path proxy mislabeled")
    dictionary=(out/"expanded_descriptor_dictionary.md").read_text()
    for col in master.columns:
        if col not in {"candidate_id","parameter_fingerprint","source_registry","simulation_git_sha","simulation_sha_provenance","github_repository","historical_branch","canonical_family","canonical_option_key","historical_response_class","is_canonical_holdout","morphology_class"} and f"`{col}`" not in dictionary: fail(f"dictionary missing {col}")
    mi=pd.read_csv(out/"expanded_mutual_information.csv")
    if not {"bias_corrected_MI_nats","bootstrap_std_nats","permutation_p","permutation_q_fdr"}.issubset(mi): fail("MI uncertainty absent")
    gam=pd.read_csv(out/"expanded_gam_performance.csv"); inter=pd.read_csv(out/"expanded_interaction_models.csv")
    if gam.response.nunique()!=8 or not {"RIDGE_PHYSICAL_INTERACTIONS","GRADIENT_BOOSTED_DECISION_STUMPS","ABLATION_PLUS_SERIAL_TRANSPORT","ABLATION_PLUS_SAVED_STATE"}.issubset(set(inter.model)): fail("nonlinear/ablation models incomplete")
    hyp=pd.read_csv(out/"expanded_hypothesis_tests.csv")
    if set(hyp.hypothesis)!={f"H{i}" for i in range(1,13)} or set(hyp.classification)-{"SUPPORTED","WEAK_SUPPORT","REJECTED","INSUFFICIENT_DATA"}: fail("hypothesis gate")
    responses=pd.read_csv(out/"expanded_initiation_developed_response_descriptors.csv")
    expected={"K_first_MPa_sqrt_m","K_10um_MPa_sqrt_m","K_25um_MPa_sqrt_m","K_50um_MPa_sqrt_m","K_checkpoint_MPa_sqrt_m","developed_R_curve_slope_MPa_sqrt_m_per_um"}
    if not expected.issubset(set(responses.response_observable)): fail("initiation/developed separation")
    manifest=pd.read_csv(out/"v1_artifact_manifest.csv")
    for r in manifest.itertuples():
        if not Path(r.path).is_file() or sha(Path(r.path))!=r.sha256: fail(f"v1 preservation hash {r.artifact}")
    report=(out/"BARRIER_TEMPERATURE_FRACTURE_MORPHOLOGY_REPORT.md").read_text()
    for i in range(1,17):
        if f"{i}. **" not in report: fail(f"report answer {i}")
    allfig=audit.get("figure_stems",[])
    if len(allfig)<33: fail("figure suite incomplete")
    for stem in allfig:
        path=out/f"{stem}.png"; side=out/f"{stem}_plot_data.csv"
        if not path.is_file() or not side.is_file() or pd.read_csv(side).empty: fail(f"figure/sidecar {stem}")
        with Image.open(path) as im:
            if im.width<800 or im.height<500: fail(f"undersized {stem}")
            im.verify()
    if a.require_clean and subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True).strip(): fail("worktree not clean")
    print(json.dumps({"status":"PASS","branch":branch,"head":head,"candidates":400,"temperature_rows":3918,"features":audit["expanded_feature_count"],"figures":len(allfig)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
