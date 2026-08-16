#!/usr/bin/env python3
"""Fail-closed verifier for the focused v9.13 amendment artifacts."""
from __future__ import annotations
import argparse,hashlib,json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

REPO=Path(__file__).resolve().parents[1]; DEFAULT=REPO/"runs/v913_barrier_temperature_fracture_morphology_v3_focused"
BRANCH="codex/v9.13-barrier-temperature-fracture-morphology"; SIM="559425321b9a8739f32788322d8a1c2af8abad73"
REQUIRED=["response_independent_barrier_shape_descriptors.csv","activation_window_descriptors.csv","whole_surface_kinetic_competition.csv",
 "plastic_bottleneck_descriptors.csv","state_proxy_temperature_descriptors.csv","feature_family_incremental_models.csv","feature_family_standalone_models.csv",
 "nonlinear_descriptor_models.csv","response_pca_prediction_models.csv","predictor_leakage_audit.csv","response_conditioned_mechanistic_diagnostics.csv",
 "intrinsic_predictor_response_correlations.csv","intrinsic_predictor_mutual_information.csv","focused_model_master.csv","focused_barrier_pca_scores.csv",
 "focused_barrier_pca_metadata.json","preserved_response_pca_and_clusters.csv","response_transition_grid_resolution.csv","v1_artifact_manifest.csv",
 "activation_window_response_tests.csv","FOCUSED_BARRIER_MORPHOLOGY_REPORT.md","focused_analysis_audit.json"]
FIGURES=["activation_window_center_overlap_map","whole_surface_kinetic_competition_map","thermal_barrier_motion_map","plastic_bottleneck_transition_map",
 "feature_family_incremental_R2","response_PC1_barrier_geometry_map","response_PC2_barrier_geometry_map","canonical_four_on_activation_window_map",
 "canonical_four_on_kinetic_competition_map","nonlinear_vs_spearman_comparison"]

def fail(msg): raise SystemExit(f"VERIFY_FAIL: {msg}")
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--out",type=Path,default=DEFAULT); ap.add_argument("--require-clean",action="store_true"); a=ap.parse_args(); out=a.out.resolve()
    missing=[x for x in REQUIRED if not (out/x).is_file()]
    if missing: fail(f"missing required files {missing}")
    audit=json.loads((out/"focused_analysis_audit.json").read_text()); head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=REPO,text=True).strip(); branch=subprocess.check_output(["git","branch","--show-current"],cwd=REPO,text=True).strip()
    if branch!=BRANCH or audit.get("branch")!=branch or audit.get("analysis_git_sha")!=head: fail("branch/HEAD provenance mismatch")
    if audit.get("simulation_git_sha")!=SIM or audit.get("physics_changed") is not False or audit.get("new_simulations_launched") is not False: fail("analysis-only provenance")
    if (audit.get("candidate_count"),audit.get("discovery_candidate_count"),audit.get("canonical_holdout_count"))!=(400,396,4): fail("population/holdout counts")
    intrinsic=pd.read_csv(out/REQUIRED[0],low_memory=False); windows=pd.read_csv(out/REQUIRED[1],low_memory=False); kinetic=pd.read_csv(out/REQUIRED[2],low_memory=False); bottle=pd.read_csv(out/REQUIRED[3],low_memory=False); state=pd.read_csv(out/REQUIRED[4],low_memory=False)
    if len(intrinsic)!=400 or len(windows)!=400 or len(kinetic)!=4000 or len(bottle)!=4000 or len(state)!=400: fail("descriptor row counts")
    if not intrinsic.prediction_eligible.all() or not kinetic.prediction_eligible.all() or state.prediction_eligible.any(): fail("A/C eligibility")
    if not ((windows.activation_window_overlap_Oce>=0)&(windows.activation_window_overlap_Oce<=1)).all() or (windows[["activation_window_wasserstein","activation_window_JS_nats"]]<0).any().any(): fail("activation-window distances")
    if not np.allclose(intrinsic.normalized_activation_window_overlap_delta_T,0): fail("normalized overlap should be T invariant")
    if not ((kinetic.physical_activation_window_overlap_Oce>=0)&(kinetic.physical_activation_window_overlap_Oce<=1)).all(): fail("physical activation-window overlap")
    parts=["dlogR_dT_cleave_gT","dlogR_dT_emit_gT","dlogR_dT_cleave_sT","dlogR_dT_emit_sT","dlogR_dT_shape_weighted_explicit_inverse_T","dlogR_dT_production_prefactor_multihit_correction"]
    if not np.allclose(kinetic.dlog10_rate_ratio_dT_at_z1,kinetic[parts].sum(axis=1),rtol=1e-10,atol=1e-10): fail("thermal derivative decomposition")
    allowed={"EMISSION_LIMITED","PEIERLS_LIMITED","TAYLOR_LIMITED","MIXED_PLASTIC_CONTROL"}
    if set(bottle.plastic_control)-allowed: fail("plastic control labels")
    if set(state.state_reconstruction_class)!={"PARTIAL_SAVED_FIRST_PASSAGE_PROXY"} or not state.missing_state_fields.str.contains("full_active_state_vector").all(): fail("partial state disclosure")
    leak=pd.read_csv(out/"predictor_leakage_audit.csv"); classes={"INTRINSIC_PREDICTOR","STATE_MEDIATOR","RESPONSE_VARIABLE","RESPONSE_DERIVED_DIAGNOSTIC","PROVENANCE"}
    if set(leak.classification)!=classes or (leak.headline_prediction_eligible!=(leak.classification=="INTRINSIC_PREDICTOR")).any(): fail("leakage classification")
    for col in ["authoritative_response_MPa_sqrt_m","first_event_K_MPa_sqrt_m"]:
        q=leak[(leak.dataset.eq("v1_master"))&leak.column.eq(col)];
        if len(q)!=1 or q.classification.iloc[0]!="RESPONSE_VARIABLE": fail(f"legacy response leakage {col}")
    corr=pd.read_csv(out/"intrinsic_predictor_response_correlations.csv");
    if set(corr.predictor_role)!={"INTRINSIC_PREDICTOR"} or set(corr.feature)-set(audit.get("headline_intrinsic_features",[])): fail("headline predictor roles")
    inc=pd.read_csv(out/"feature_family_incremental_models.csv"); expected=["BARRIER_SCALE","BARRIER_SHAPE","RELATIVE_GEOMETRY_PLUS_THERMAL_MOTION","KINETIC_COMPETITION","PLASTIC_BOTTLENECK","STATE_PROXY"]
    if inc.response.nunique()!=8 or len(inc)!=48 or any(g.sort_values("model_stage").added_family.tolist()!=expected for _,g in inc.groupby("response")): fail("six-stage hierarchy")
    if not inc.canonical_holdouts_excluded.all() or inc.n.max()>396 or inc.fold_definition.nunique()!=1: fail("incremental holdout/folds")
    standalone=pd.read_csv(out/"feature_family_standalone_models.csv");
    if standalone.response.nunique()!=8 or standalone.feature_family.nunique()!=7 or len(standalone)!=56: fail("standalone families")
    nonlinear=pd.read_csv(out/"nonlinear_descriptor_models.csv"); required_models={"UNIVARIATE_LINEAR_QUADRATIC_GAM","SHALLOW_REGRESSION_TREE_DEPTH3","LOW_ORDER_PHYSICAL_INTERACTION_RIDGE"}
    if set(nonlinear.model)!=required_models or not nonlinear.canonical_holdouts_excluded.all(): fail("nonlinear models")
    interactions=nonlinear[nonlinear.model.eq("LOW_ORDER_PHYSICAL_INTERACTION_RIDGE")]
    if interactions.interaction_terms.fillna("").str.len().eq(0).any(): fail("empty physical interactions")
    activation_tests=pd.read_csv(out/"activation_window_response_tests.csv")
    if set(["DBTT_magnitude_MPa_sqrt_m","DBTT_width_K","peak_prominence_MPa_sqrt_m","weakT_max_deviation_from_mean_MPa_sqrt_m"])-set(activation_tests.response): fail("activation-window response coverage")
    pca=pd.read_csv(out/"response_pca_prediction_models.csv");
    if set(pca.model)!={"LINEAR_OLS","LINEAR_RIDGE","QUADRATIC_RIDGE","ADDITIVE_SPLINE_GAM"} or set(pca.response_PC)!={"fracture_response_PC1","fracture_response_PC2"} or not pca.canonical_holdouts_excluded.all(): fail("response PCA mapping")
    resolution=pd.read_csv(out/"response_transition_grid_resolution.csv");
    if len(resolution)!=400 or not resolution.interpretation.str.contains("no sub-grid precision").all(): fail("grid resolution disclosure")
    for stem in FIGURES:
        png=out/f"{stem}.png"; csv=out/f"{stem}_plot_data.csv"
        if not png.is_file() or not csv.is_file() or pd.read_csv(csv).empty: fail(f"figure/sidecar {stem}")
        with Image.open(png) as im:
            if im.width<800 or im.height<500: fail(f"undersized figure {stem}")
            im.verify()
    manifest=pd.read_csv(out/"v1_artifact_manifest.csv")
    for r in manifest.itertuples():
        if not Path(r.path).is_file() or sha(r.path)!=r.sha256: fail(f"v1 preservation {r.artifact}")
    report=(out/"FOCUSED_BARRIER_MORPHOLOGY_REPORT.md").read_text()
    for phrase in ["RESPONSE_CONDITIONED_DIAGNOSTIC","PARTIAL_SAVED_FIRST_PASSAGE_PROXY","No fracture simulation was launched","Revised smallest causality test","no sub-grid precision"]:
        if phrase not in report: fail(f"report disclosure {phrase}")
    if a.require_clean and subprocess.check_output(["git","status","--porcelain"],cwd=REPO,text=True).strip(): fail("worktree not clean")
    print(json.dumps({"status":"PASS","branch":branch,"head":head,"candidates":400,"intrinsic_features":audit["headline_intrinsic_feature_count"],"figures":len(FIGURES)},indent=2)); return 0

if __name__=="__main__": raise SystemExit(main())
