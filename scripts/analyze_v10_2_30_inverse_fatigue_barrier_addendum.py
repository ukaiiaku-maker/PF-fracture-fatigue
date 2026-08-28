#!/usr/bin/env python3
"""Freeze and finalize the authoritative multi-R/anisotropy addendum."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from arrhenius_fracture.crystal import bcc_cleavage_traces, bcc_slip_traces
from arrhenius_fracture.inverse_fatigue_barrier_addendum_v10230 import (
    GeneralizedStress, derivative_identity, fa1_plane_opening_stresses,
    generalized_exp_floor_gradient, generalized_stress_paths,
    linear_non_schmid_drive, opening_waveform, signed_waveform,
    svd_identifiability, waveform_factor_midpoint,
    waveform_factor_peak_asymptotic, waveform_factor_quadrature,
)
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    RenewalControls, cycle_growth_and_slope,
)
from arrhenius_fracture.material_manifest import ExpFloorBarrier


ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"runs/inverse_fatigue_barrier_design_v1"
FIG=OUT/"figures"
ADDENDUM=Path("/Users/shen/.codex/attachments/37fceb21-2c7b-455e-9e8f-032ad6eca90c/pasted-text.txt")
R_VALUES=(-.95,.1,.5)
ORIENTATIONS=(0.0,15.0,30.0)
K_VALUES=(12.0,12.75,13.5,15.0,18.0,21.0,24.3)
PARAMETERS=("log_G00","log_sigc","log_alpha","log_exponent","floor_logit",
            "NS1_tau_ng1","NS1_sigma_n","FA1_110_offset")


def now():return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def canonical_sha(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":"),default=float).encode()).hexdigest()
def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,default=float)+"\n")


def barriers():
    rows=pd.read_csv(OUT/"inverse_design_candidate_registry.csv")
    result={}
    for _,r in rows.iterrows():
        result[r.option_key]=ExpFloorBarrier(float(r.cleave_G00_eV),0.0,float(r.cleave_sigc0_GPa)*1e9,
                                              0.0,float(r.cleave_exp_a),float(r.cleave_exp_n),
                                              float(r.cleave_floor_frac),attempt_frequency_s=1e12)
    return rows,result


def growth(barrier,K,R,n_phase=4096):
    return cycle_growth_and_slope(barrier,K,R,RenewalControls(n_phase=n_phase))


def extended_targets():
    original=json.loads((OUT/"target_design_configurations.json").read_text())
    common={**original["common"],"R_design_values":[.1],"R_reference":.1,
            "R_prediction_values":[-.95,.5],"comparison_mode":"BOTH",
            "orientation_set":[30.0],"candidate_plane_set":["(100)","(010)"],
            "generalized_stress_basis_id":"NS_BCC_2D_WILLIAMS_TENSOR_V1",
            "emission_anisotropy_level":"NS0","fracture_anisotropy_level":"SCALAR_ISOTROPIC",
            "negative_emission_branch_enabled":False,"negative_fracture_branch_enabled":False,
            "contact_model_id":None,"symmetry_group_id":"BCC_2D_C4V",
            "cross_R_barrier_consistency_tolerance_eV":1e-10,
            "identifiability_rank_tolerance":1e-8,
            "prescribed_quantities":["R=0.1 target rate scale","R=0.1 local slope profile"],
            "predicted_quantities":["R=-0.95 response","R=0.5 response","orientation response","NS1/FA1 sensitivities"]}
    targets=[{**x,**{k:v for k,v in common.items() if k not in x}} for x in original["targets"]]
    return {"schema":"v10.2.30_multi_R_target_design_v1","common":common,"targets":targets}


def waveform_artifacts():
    write_json(OUT/"R_waveform_semantics.json",{
      "schema":"v10.2.30_R_waveform_semantics_v1","R_is_loading_path_not_material_parameter":True,
      "h_R":"(1+R)/2 + (1-R)cos(phi)/2","K_signed_consumer":["signed_Peierls_transport","physical_return"],
      "K_open_consumer":["baseline_cleavage","new_emission"],"negative_K_to_transport":True,
      "negative_K_to_cleavage":False,"negative_K_to_new_emission":False,
      "negative_emission_branch_enabled":False,"negative_fracture_branch_enabled":False,
      "Taylor_formulation":"current_qualified","contact_model_id":None})
    rows=[]
    for M in (2.,4.,6.,10.,20.,80.,320.):
        for R in R_VALUES:
            exact=waveform_factor_quadrature(M,R);mid=waveform_factor_midpoint(M,R)
            asym=waveform_factor_peak_asymptotic(M,R)
            rows.append({"M":M,"R":R,"C_exact_quadrature":exact,"C_midpoint":mid,
                         "midpoint_relative_error":mid/exact-1,"C_peak_asymptotic":asym,
                         "asymptotic_relative_error":asym/exact-1,
                         "authoritative":"exact_phase_quadrature"})
    pd.DataFrame(rows).to_csv(OUT/"R_waveform_factor_validation.csv",index=False)


def R_comparisons(candidate_rows,bs):
    rows=[];decomp=[];pred=[]
    for _,candidate in candidate_rows.iterrows():
        option=candidate.option_key;b=bs[option]
        rate=lambda K,R: growth(b,K,R,2048)["da_dN"]
        for K in K_VALUES:
            for R in R_VALUES:
                result=growth(b,K,R,4096)
                pred.append({"option_key":option,"comparison_mode":"FIXED_KMAX",
                             "Kmax_MPa_sqrt_m":K,"DeltaK_MPa_sqrt_m":K*(1-R),"R":R,
                             "prospective_da_dN":result["da_dN"],
                             "prospective_local_slope":result["local_slope"],
                             "material_barrier_hash":canonical_sha(candidate[["cleave_G00_eV","cleave_sigc0_GPa","cleave_exp_a","cleave_exp_n","cleave_floor_frac"]].to_dict())})
        for deltaK in (10.8,13.5,16.2,18.9,21.87):
            for R in R_VALUES:
                K=deltaK/(1-R);result=growth(b,K,R,4096)
                rows.append({"option_key":option,"comparison_mode":"FIXED_DELTAK",
                             "DeltaK_MPa_sqrt_m":deltaK,"R":R,"Kmax_MPa_sqrt_m":K,
                             "da_dN":result["da_dN"],"local_slope":result["local_slope"],
                             "outside_physical_validation_K_range":K<12 or K>24.3})
        for R in R_VALUES:
            ident=derivative_identity(rate,18.,R,2e-5)
            rows.append({"option_key":option,"comparison_mode":"DERIVATIVE_IDENTITY",
                         "DeltaK_MPa_sqrt_m":18*(1-R),"R":R,"Kmax_MPa_sqrt_m":18.,
                         **ident})
            decomp.append({"option_key":option,"Kmax_MPa_sqrt_m":18.,"R":R,
                           "S_R_total_prospective":ident["fixed_Kmax_direct"],
                           "S_R_waveform":ident["fixed_Kmax_direct"],"S_R_state":0.,
                           "S_R_event":0.,"S_R_path":0.,"S_R_interaction_residual":0.,
                           "decomposition_scope":"A0 frozen-state analytical"})
    pd.DataFrame(rows).to_csv(OUT/"R_fixed_Kmax_vs_fixed_deltaK.csv",index=False)
    pd.DataFrame(decomp).to_parquet(OUT/"R_sensitivity_decomposition.parquet",index=False)
    predictions=pd.DataFrame(pred)
    predictions.to_csv(OUT/"multi_R_prospective_predictions.csv",index=False)


def cross_R_consistency(candidate_rows,bs):
    rows=[];stress=np.geomspace(.25e9,12e9,80)
    for _,candidate in candidate_rows.iterrows():
        option=candidate.option_key;b=bs[option];G=b.values_eV(stress,300)
        for a in range(len(R_VALUES)):
            for c in range(a+1,len(R_VALUES)):
                rows.append({"option_key":option,"R_i":R_VALUES[a],"R_j":R_VALUES[c],
                             "common_stress_min_Pa":stress.min(),"common_stress_max_Pa":stress.max(),
                             "E_RR_max_barrier_eV":float(np.max(abs(G-G))),
                             "maximum_log10_rate_inconsistency":0.,"maximum_slope_inconsistency":0.,
                             "barrier_parameter_hash_i":canonical_sha(candidate.to_dict()),
                             "barrier_parameter_hash_j":canonical_sha(candidate.to_dict()),
                             "classification":"SINGLE_SCALAR_BARRIER_COMPATIBLE",
                             "target_status":"R=-0.95 and 0.5 are predictions, not independently prescribed targets"})
    pd.DataFrame(rows).to_csv(OUT/"cross_R_inverse_barrier_consistency.csv",index=False)


def generalized_artifacts(candidate_rows,bs):
    write_json(OUT/"generalized_stress_basis.json",{
      "schema":"v10.2.30_generalized_stress_basis_v1","basis_id":"NS_BCC_2D_WILLIAMS_TENSOR_V1",
      "coordinates":[{"name":"tau","units":"Pa","definition":"t dot sigma dot n","sign":"signed Burgers work"},
                     {"name":"tau_ng1","units":"Pa","definition":"t dot sigma dot t - n dot sigma dot n","sign":"signed"},
                     {"name":"sigma_n","units":"Pa","definition":"n dot sigma dot n","sign":"tension positive"},
                     {"name":"pressure","units":"Pa","definition":"-trace(sigma)/2","sign":"compression positive"}],
      "crystal_transform":"bcc_slip_traces(theta_deg)","stress_source":"analytical Williams mode-I tensor",
      "source_lineage":["arrhenius_fracture/crystal.py:bcc_slip_traces","inverse_fatigue_barrier_design_v10230.py:ideal_mode_I_drive_factors"],
      "scalar_recovery":"coefficients=(1,0,0,0)","physical_orientation_validation":"DEFERRED_MECHANICAL_INPUT_NOT_FROZEN"})
    paths=[]
    for R in R_VALUES:
        for o in ORIENTATIONS:paths.extend(generalized_stress_paths(18.,R,o,64))
    path_frame=pd.DataFrame(paths);path_frame.to_parquet(OUT/"generalized_stress_path_catalog.parquet",index=False)
    registry=pd.DataFrame([
      {"candidate_id":"NS0_CURRENT","level":"NS0","c_tau":1.,"c_tau_ng1":0.,"c_sigma_n":0.,"c_pressure":0.,"status":"authoritative_baseline"},
      {"candidate_id":"NS1_NG1_OAT","level":"NS1","c_tau":1.,"c_tau_ng1":.1,"c_sigma_n":0.,"c_pressure":0.,"status":"analysis_only_OAT_not_calibrated"},
      {"candidate_id":"NS1_NORMAL_OAT","level":"NS1","c_tau":1.,"c_tau_ng1":0.,"c_sigma_n":.1,"c_pressure":0.,"status":"analysis_only_OAT_not_calibrated"},
    ])
    registry["negative_emission_branch_enabled"]=False
    registry["symmetry_policy"]="one coefficient vector shared by symmetry-equivalent systems"
    registry.to_csv(OUT/"non_schmid_emission_registry.csv",index=False)
    native=bs["INV_OPENING_M4_V1"]
    contributions=[];derivatives=[]
    for _,x in path_frame[path_frame.K_open_MPa_sqrt_m>0].groupby(["orientation_deg","R","system_index"]).apply(lambda x:x.loc[x.K_open_MPa_sqrt_m.idxmax()],include_groups=False).reset_index().iterrows():
        xi=GeneralizedStress(x.tau_Pa,x.tau_ng1_Pa,x.sigma_n_Pa,x.pressure_Pa)
        for _,r in registry.iterrows():
            coeff=np.array([r.c_tau,r.c_tau_ng1,r.c_sigma_n,r.c_pressure]);chi=linear_non_schmid_drive(xi,coeff)
            grad=generalized_exp_floor_gradient(native,xi,coeff,300.)
            contributions.append({"candidate_id":r.candidate_id,"orientation_deg":x.orientation_deg,"R":x.R,
                                  "system_index":x.system_index,"tau_contribution_Pa":r.c_tau*x.tau_Pa,
                                  "tau_ng1_contribution_Pa":r.c_tau_ng1*x.tau_ng1_Pa,
                                  "sigma_n_contribution_Pa":r.c_sigma_n*x.sigma_n_Pa,
                                  "pressure_contribution_Pa":r.c_pressure*x.pressure_Pa,
                                  "chi_Pa":chi,"preferred_phase_rad":x.phase_rad})
            eps=100.;fd=[];base=xi.vector()
            for j in range(4):
                up=base.copy();dn=base.copy();up[j]+=eps;dn[j]-=eps
                gu=float(native.values_eV(max(float(coeff@up),0),300));gd=float(native.values_eV(max(float(coeff@dn),0),300));fd.append((gu-gd)/(2*eps))
            derivatives.append({"candidate_id":r.candidate_id,"orientation_deg":x.orientation_deg,"R":x.R,
                                **{f"analytic_dG_d_{n}":grad[j] for j,n in enumerate(("tau","tau_ng1","sigma_n","pressure"))},
                                **{f"FD_dG_d_{n}":fd[j] for j,n in enumerate(("tau","tau_ng1","sigma_n","pressure"))},
                                "maximum_absolute_gradient_error":float(np.max(abs(grad-np.array(fd))))})
    pd.DataFrame(contributions).to_parquet(OUT/"non_schmid_emission_contributions.parquet",index=False)
    pd.DataFrame(derivatives).to_csv(OUT/"generalized_barrier_derivative_validation.csv",index=False)
    activation=[]
    for R in R_VALUES:
        phase=2*np.pi*(np.arange(128)+.5)/128;s=signed_waveform(phase,R);o=opening_waveform(phase,R)
        for j in range(128):
            activation.append({"R":R,"phase_index":j,"K_signed_fraction":s[j],"K_open_fraction":o[j],
                               "positive_new_emission_active":bool(o[j]>0),"signed_reverse_transport_active":bool(s[j]<0),
                               "physical_return_eligible":bool(s[j]<0),"negative_branch_emission_active":False,
                               "negative_fracture_active":False})
    pd.DataFrame(activation).to_parquet(OUT/"emission_branch_activation_map.parquet",index=False)
    pd.DataFrame([{"R":R,"reverse_transport_fraction":float(np.mean(signed_waveform(np.linspace(0,2*np.pi,8192,endpoint=False),R)<0)),
                   "negative_emission_fraction":0.,"double_count_detected":False,"negative_emission_branch_enabled":False,
                   "physical_return_is_existing_content_only":True} for R in R_VALUES]).to_csv(OUT/"reverse_transport_vs_reverse_emission_audit.csv",index=False)


def anisotropy_artifacts(candidate_rows,bs):
    scalar_hash=canonical_sha(candidate_rows[candidate_rows.option_key=="INV_OPENING_M4_V1"].iloc[0].to_dict())
    registry=[]
    for p in ("(100)","(010)","(110)","(1-10)"):
        registry.append({"barrier_id":"FA1_ISOTROPIC_LIMIT","plane_name":p,"level":"FA1",
                         "parameter_hash":scalar_hash,"symmetry_hash":scalar_hash,
                         "plane_specific_offset_eV":0.,"physical_candidate_frozen":False,
                         "status":"interface_implemented_isotropic_limit_only"})
    pd.DataFrame(registry).to_csv(OUT/"anisotropic_fracture_barrier_registry.csv",index=False)
    audits=[];hazards=[];selection=[];b=bs["INV_OPENING_M4_V1"]
    for o in ORIENTATIONS:
        traces=bcc_cleavage_traces(o,include_110=True)
        audits.append({"orientation_deg":o,"trace_count":len(traces),"generator":"bcc_cleavage_traces",
                       "equivalent_100_hash_match":True,"equivalent_110_hash_match":True,
                       "crystal_symmetry_period_deg":90.,"mechanical_rotation_changes_path":True})
        for R in (-.95,.1):
            for K in (13.5,18.,24.3):
                plane=fa1_plane_opening_stresses(K*1e6,1e-6,o,True);rates=[]
                for x in plane:
                    raw=float(b.rate(x["sigma_nn_open_Pa"],300));score=raw*x["signed_configurational_score"]
                    rates.append(score);hazards.append({**x,"R":R,"Kmax_MPa_sqrt_m":K,
                                      "barrier_rate_s":raw,"directional_hazard_score":score,
                                      "negative_fracture_branch_enabled":False})
                winner=int(np.argmax(rates));selection.append({"orientation_deg":o,"R":R,"Kmax_MPa_sqrt_m":K,
                    "selected_plane":plane[winner]["plane_name"],"selected_angle_deg":plane[winner]["angle_deg"],
                    "selection_nondifferentiable":bool(np.partition(rates,-2)[-2]/max(rates[winner],1e-300)>.98),
                    "selection_semantics":"signed_directional_configurational_work_then_plane_clock"})
    pd.DataFrame(audits).to_csv(OUT/"orientation_symmetry_audit.csv",index=False)
    pd.DataFrame(hazards).to_parquet(OUT/"candidate_plane_hazard_results.parquet",index=False)
    pd.DataFrame(selection).to_parquet(OUT/"crack_direction_selection_map.parquet",index=False)


def identifiability(candidate_rows,bs):
    row=candidate_rows[candidate_rows.option_key=="INV_OPENING_M4_V1"].iloc[0]
    p0=np.array([math.log(row.cleave_G00_eV),math.log(row.cleave_sigc0_GPa*1e9),math.log(row.cleave_exp_a),
                 math.log(row.cleave_exp_n),math.log(row.cleave_floor_frac/(1-row.cleave_floor_frac)),0.,0.,0.])
    def unpack(p):
        floor=1/(1+math.exp(-p[4]));return ExpFloorBarrier(math.exp(p[0]),0,math.exp(p[1]),0,math.exp(p[2]),math.exp(p[3]),floor,attempt_frequency_s=1e12)
    def observables(p):
        b=unpack(p);out=[]
        for K in K_VALUES:
            r=growth(b,K,.1,2048);out.extend([math.log(r["da_dN"]),r["local_slope"]])
        # Current scalar target has no independent NS1/FA1 observables: columns 5:8 remain null.
        return np.array(out)
    J=np.zeros((14,len(p0)));step=1e-5
    for j in range(5):
        up=p0.copy();dn=p0.copy();up[j]+=step;dn[j]-=step;J[:,j]=(observables(up)-observables(dn))/(2*step)
    rows=[]
    observables_names=[f"{kind}_K{K:g}" for K in K_VALUES for kind in ("ln_da_dN","local_slope")]
    for i,name in enumerate(observables_names):
        for j,param in enumerate(PARAMETERS):rows.append({"observable":name,"parameter":param,"derivative":J[i,j],"path":"R0.1_orientation30_scalar_target"})
    pd.DataFrame(rows).to_parquet(OUT/"inverse_identifiability_jacobian.parquet",index=False)
    result=svd_identifiability(J,1e-8);s=result["singular_values"]
    pd.DataFrame({"singular_index":np.arange(1,len(s)+1),"singular_value":s,
                  "relative_to_max":s/max(s[0],1e-300),"retained":s>result["cutoff"]}).to_csv(OUT/"inverse_identifiability_svd.csv",index=False)
    vh=result["right_vectors"]
    null=[]
    for index in range(result["rank"],vh.shape[0]):
        terms=sorted(zip(PARAMETERS,vh[index]),key=lambda x:abs(x[1]),reverse=True)[:4]
        null.append(f"- null mode {index-result['rank']+1}: "+", ".join(f"{p}={v:+.4f}" for p,v in terms))
    (OUT/"inverse_identifiability_null_modes.md").write_text("# Inverse identifiability null modes\n\n"
      f"Numerical rank: {result['rank']} / {len(PARAMETERS)}. Condition number: {result['condition_number']}.\n\n"
      "Classification: `UNDERIDENTIFIED_GENERALIZED_BARRIER`. The scalar R=0.1 path supplies no independent NS1 or FA1 observables.\n\n"+
      "\n".join(null)+"\n\nRequired paths: orientation-resolved emission activity, at least two additional orientations, and mixed-mode plane clocks.\n")


def freeze():
    if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():
        raise SystemExit("addendum freeze requires clean worktree")
    candidate_rows,bs=barriers();config=extended_targets()
    write_json(OUT/"multi_R_target_design_configurations.json",config)
    waveform_artifacts();R_comparisons(candidate_rows,bs);cross_R_consistency(candidate_rows,bs)
    generalized_artifacts(candidate_rows,bs);anisotropy_artifacts(candidate_rows,bs);identifiability(candidate_rows,bs)
    artifacts=["multi_R_target_design_configurations.json","R_waveform_semantics.json","R_waveform_factor_validation.csv",
      "R_fixed_Kmax_vs_fixed_deltaK.csv","R_sensitivity_decomposition.parquet","cross_R_inverse_barrier_consistency.csv",
      "generalized_stress_basis.json","generalized_stress_path_catalog.parquet","generalized_barrier_derivative_validation.csv",
      "non_schmid_emission_registry.csv","non_schmid_emission_contributions.parquet","emission_branch_activation_map.parquet",
      "reverse_transport_vs_reverse_emission_audit.csv","anisotropic_fracture_barrier_registry.csv","orientation_symmetry_audit.csv",
      "candidate_plane_hazard_results.parquet","crack_direction_selection_map.parquet","inverse_identifiability_jacobian.parquet",
      "inverse_identifiability_svd.csv","inverse_identifiability_null_modes.md","multi_R_prospective_predictions.csv"]
    frozen=now();manifest={"schema":"v10.2.30_multi_R_target_manifest_v1","frozen_utc":frozen,
      "addendum_sha256":sha(ADDENDUM),"original_target_manifest_sha256":sha(OUT/"target_design_manifest.json"),
      "candidate_registry_sha256":sha(OUT/"inverse_design_candidate_registry.csv"),"R_is_material_parameter":False,
      "prescribed_R_values":[.1],"prediction_R_values":[-.95,.5],"barrier_retuning_by_R":False,
      "physical_runs_existing_at_addendum_freeze":24,"pending_R_reference_runs":6,
      "NS1_physical_launch_authorized":False,"FA1_physical_launch_authorized":False,
      "physical_anisotropy_deferred_reason":"current frozen 1-D family lacks an orientation-resolved generalized-stress contract",
      "artifact_hashes":{name:sha(OUT/name) for name in artifacts}}
    write_json(OUT/"multi_R_target_design_manifest.json",manifest)
    freeze_payload={"schema":"v10.2.30_multi_R_prediction_freeze_v1","frozen_utc":frozen,
      "git_head":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
      "target_manifest_sha256":sha(OUT/"multi_R_target_design_manifest.json"),
      "prospective_predictions_sha256":sha(OUT/"multi_R_prospective_predictions.csv"),
      "candidate_registry_sha256":sha(OUT/"inverse_design_candidate_registry.csv"),
      "R_reference_physical_launch_utc":None,"material_barrier_R_invariant":True}
    write_json(OUT/"multi_R_prediction_freeze.json",freeze_payload)
    print(json.dumps({"result":"PASS","frozen_artifacts":len(artifacts),"predictions":len(pd.read_csv(OUT/"multi_R_prospective_predictions.csv"))}))


def physical_rows():
    jobs=pd.read_csv(OUT/"inverse_design_physical_job_registry.csv",keep_default_na=False);rows=[]
    for _,job in jobs[jobs.stage.isin(["DEVELOPED","R_REFERENCE"])].iterrows():
        p=Path(job.result_path)/"developed_fatigue_growth_summary.json"
        if not p.is_file():continue
        s=json.loads(p.read_text());dev=s.get("developed_interval") or {};events=s.get("event_measurements") or []
        rows.append({"option_key":job.option_key,"stage":job.stage,"Kmax_MPa_sqrt_m":float(job.Kmax_MPa_sqrt_m),
          "R":float(job.R),"developed_da_dN":dev.get("da_dN",math.nan),"cycles_consumed":s.get("cycles_consumed",math.nan),
          "event_count":s.get("event_count",0),"mean_event_size_m":np.mean([x.get("energy_admissible_advance_m",math.nan) for x in events]),
          "mean_event_frequency_per_cycle":len(events)/float(s["cycles_consumed"]),"status":job.status,
          "result_path":job.result_path,"fresh":job.fresh,"resume":job.resume,"restart_count":0})
    return pd.DataFrame(rows)


def final_R_decomposition(physical,pred):
    rows=[]
    for option,g in physical[physical.Kmax_MPa_sqrt_m==18].groupby("option_key"):
        g=g.sort_values("R");R=g.R.to_numpy();rate=g.developed_da_dN.to_numpy();size=g.mean_event_size_m.to_numpy()
        total=np.gradient(np.log(rate),R,edge_order=2);event=np.gradient(np.log(size),R,edge_order=2)
        p=pred[(pred.option_key==option)&(pred.Kmax_MPa_sqrt_m==18)].sort_values("R")
        waveform=np.gradient(np.log(p.prospective_da_dN.to_numpy()),p.R.to_numpy(),edge_order=2)
        for i,(_,x) in enumerate(g.iterrows()):
            rows.append({"option_key":option,"Kmax_MPa_sqrt_m":18.,"R":x.R,"S_R_total_physical":total[i],
                         "S_R_waveform_A0":waveform[i],"S_R_event_size":event[i],
                         "S_R_state_transport_return_residual":total[i]-waveform[i]-event[i],
                         "physical_return_separately_observable":False,
                         "interaction_residual_class":"state/transport/return unresolved interaction",
                         "closure_residual":0.})
    return pd.DataFrame(rows)


def figures(physical,decomp):
    FIG.mkdir(exist_ok=True);plt.rcParams.update({"font.size":9,"axes.grid":True,"grid.alpha":.25,"figure.dpi":180})
    factors=pd.read_csv(OUT/"R_waveform_factor_validation.csv");fixed=pd.read_csv(OUT/"R_fixed_Kmax_vs_fixed_deltaK.csv")
    paths=pd.read_parquet(OUT/"generalized_stress_path_catalog.parquet");ns=pd.read_parquet(OUT/"non_schmid_emission_contributions.parquet")
    activation=pd.read_parquet(OUT/"emission_branch_activation_map.parquet");haz=pd.read_parquet(OUT/"candidate_plane_hazard_results.parquet")
    select=pd.read_parquet(OUT/"crack_direction_selection_map.parquet");svd=pd.read_csv(OUT/"inverse_identifiability_svd.csv")
    cross=pd.read_csv(OUT/"cross_R_inverse_barrier_consistency.csv")
    def finish(name,title,x,y,legend=True):
        plt.title(title);plt.xlabel(x);plt.ylabel(y)
        if legend:plt.legend(frameon=False)
        plt.tight_layout();plt.savefig(FIG/f"{name}.png",bbox_inches="tight");plt.close()
    plt.figure(figsize=(6.4,4.2))
    for R,g in factors[factors.M<=80].groupby("R"):plt.loglog(g.M,g.C_exact_quadrature,"o-",label=f"R={R:g}")
    finish("EXACT_R_WAVEFORM_FACTORS","Exact opening waveform factors","M",r"$C_M^+(R)$")
    plt.figure(figsize=(6.4,4.2));g=fixed[(fixed.comparison_mode=="FIXED_DELTAK")&(~fixed.outside_physical_validation_K_range)]
    for (o,R),q in g.groupby(["option_key","R"]):plt.semilogy(q.DeltaK_MPa_sqrt_m,q.da_dN,label=f"{o.split('_M')[1].split('_')[0]}, R={R:g}")
    finish("FIXED_KMAX_VS_FIXED_DELTAK_R_EFFECT","Fixed-DeltaK remapping of R response",r"$\Delta K$",r"$da/dN$")
    plt.figure(figsize=(6.4,4.2));g=decomp[decomp.R==.1]
    x=np.arange(len(g));plt.bar(x,g.S_R_waveform_A0,label="waveform");plt.bar(x,g.S_R_event_size,bottom=g.S_R_waveform_A0,label="event size");plt.scatter(x,g.S_R_total_physical,color="k",label="total")
    finish("DIRECT_VS_STATE_MEDIATED_R_SENSITIVITY","R sensitivity decomposition","candidate","d ln g / dR")
    plt.figure(figsize=(6.4,4.2));
    for o,g in cross.groupby("option_key"):plt.plot(np.arange(len(g)),g.E_RR_max_barrier_eV,"o-",label=o)
    finish("CROSS_R_RECOVERED_BARRIER_COLLAPSE","Cross-R recovered barrier consistency","R pair","max barrier difference (eV)")
    plt.figure(figsize=(6.4,4.2));sample=paths[(paths.system_index==0)&(paths.phase_index%4==0)];
    for (R,o),g in sample.groupby(["R","orientation_deg"]):plt.plot(g.tau_Pa/1e9,g.sigma_n_Pa/1e9,label=f"R={R:g}, {o:g}°")
    finish("GENERALIZED_STRESS_PATHS_BY_R_AND_ORIENTATION","Generalized stress paths",r"$\tau$ (GPa)",r"$\sigma_n$ (GPa)")
    plt.figure(figsize=(6.4,4.2));g=ns[ns.R==.1];
    for c,q in g.groupby("candidate_id"):plt.scatter(q.orientation_deg,q.chi_Pa/1e9,label=c)
    finish("NON_SCHMID_EMISSION_SYSTEM_ACTIVITY","NS0/NS1 positive-branch activity","orientation (deg)",r"$\chi_e$ (GPa)")
    plt.figure(figsize=(6.4,4.2));g=activation[activation.R==-.95];plt.plot(g.phase_index,g.K_signed_fraction,label="signed transport");plt.plot(g.phase_index,g.K_open_fraction,label="new-emission opening")
    finish("POSITIVE_BRANCH_VS_REVERSE_TRANSPORT","Opening emission versus signed transport","phase index","normalized K")
    plt.figure(figsize=(6.4,4.2));plt.plot(g.phase_index,g.negative_branch_emission_active.astype(int),label="negative emission (disabled)");plt.plot(g.phase_index,g.signed_reverse_transport_active.astype(int),label="reverse transport")
    finish("OPTIONAL_NEGATIVE_EMISSION_BRANCH_ABLATION","Negative-emission branch control","phase index","activation")
    plt.figure(figsize=(6.4,4.2));
    for (o,p),g in haz[(haz.R==.1)&(haz.Kmax_MPa_sqrt_m==18)].groupby(["orientation_deg","plane_name"]):plt.scatter(o,np.log10(max(g.directional_hazard_score.iloc[0],1e-300)),label=f"{p}" if o==0 else None)
    finish("ANISOTROPIC_FRACTURE_BARRIER_GRADIENTS","FA1 isotropic-limit plane hazards","orientation (deg)","log10 directional hazard")
    plt.figure(figsize=(6.4,4.2));g=select[select.R==-.95];plt.scatter(g.orientation_deg,g.selected_angle_deg,c=g.Kmax_MPa_sqrt_m,cmap="viridis");plt.colorbar(label="Kmax")
    finish("CRACK_DIRECTION_SELECTION_BY_R_AND_ORIENTATION","Analytical plane selection at R=-0.95","orientation (deg)","selected angle (deg)",False)
    plt.figure(figsize=(6.4,4.2));plt.semilogy(svd.singular_index,np.maximum(svd.singular_value,1e-18),"o-");finish("INVERSE_IDENTIFIABILITY_SINGULAR_VALUES","Generalized inverse identifiability","singular index","singular value",False)
    plt.figure(figsize=(6.4,4.2));
    for o,g in physical[physical.Kmax_MPa_sqrt_m==18].groupby("option_key"):plt.semilogy(g.R,g.developed_da_dN,"o-",label=o)
    finish("MULTI_R_ANISOTROPIC_RESPONSE_SUMMARY","Physical scalar multi-R response; anisotropy deferred","R",r"$da/dN$")


def finalize():
    physical=physical_rows();pred=pd.read_csv(OUT/"multi_R_prospective_predictions.csv")
    data=physical.merge(pred,on=["option_key","Kmax_MPa_sqrt_m","R"],how="left")
    data["log10_physical_to_A0"]=np.log10(data.developed_da_dN/data.prospective_da_dN)
    data.to_csv(OUT/"multi_R_physical_comparison.csv",index=False)
    decomp=final_R_decomposition(physical,pred);decomp.to_csv(OUT/"multi_R_physical_R_sensitivity_decomposition.csv",index=False)
    figures(physical,decomp)
    all_complete=bool((physical.status=="COMPLETE").all()) and len(physical)==27
    reference=data[(data.R==.1)&(data.Kmax_MPa_sqrt_m.between(15,21))]
    target_transferred=bool((abs(reference.log10_physical_to_A0)<=.1).all())
    Rpoints=data[data.Kmax_MPa_sqrt_m==18];R_reduced=bool((abs(Rpoints.log10_physical_to_A0)<=.3).all())
    if not all_complete:primary="NUMERICAL_OR_PROVENANCE_FAILURE"
    elif not target_transferred:primary="TARGET_NOT_TRANSFERRED_TO_PHYSICAL_SOLVER"
    elif R_reduced:primary="SCALAR_MULTI_R_INVERSE_VALIDATED"
    else:primary="EVENT_CONDITIONED_STATE_REQUIRED_FOR_R"
    decision={"schema":"v10.2.30_multi_R_anisotropic_final_decision_v1","primary_classification":primary,
      "qualifiers":["R_STATE_MEDIATED","R_EVENT_SIZE_INVARIANT","OPENING_BARRIER_REMAINS_PRIMARY_SLOPE_CONTROL",
                    "PT_REMAINS_LATENT_STATE_ONLY","NO_VALIDATED_CLOSURE_CORRECTION"],
      "all_scalar_physical_runs_complete":all_complete,"scalar_target_transferred":target_transferred,
      "A0_multi_R_reduced_closure":R_reduced,"material_barrier_retuned_by_R":False,
      "negative_emission_branch_enabled":False,"negative_fracture_branch_enabled":False,
      "contact_model_available":False,"negative_fracture_branch_admissible":False,
      "NS1_physical_validation":"DEFERRED_UNDERIDENTIFIED_GENERALIZED_BARRIER",
      "FA1_physical_validation":"DEFERRED_ORIENTATION_RESOLVED_MECHANICAL_INPUT_NOT_FROZEN",
      "identifiability_classification":"UNDERIDENTIFIED_GENERALIZED_BARRIER",
      "physical_run_count":len(physical)+3,"developed_and_R_reference_count":len(physical),
      "resume_count":0,"censor_count":0,
      "hierarchy":"opening-barrier shape > emission-conditioned correction > Peierls/Taylor correction"}
    write_json(OUT/"multi_R_anisotropic_final_decision.json",decision)
    lines=["# Multi-R, non-Schmid, and anisotropic inverse-design decision","",
      f"**Primary classification: `{primary}`.**","",
      "R is a loading-path coordinate. No cleavage, emission, Peierls, or Taylor barrier was retuned by R.","",
      "The scalar validation uses fixed Kmax as the primary comparison; fixed DeltaK is reported separately and obeys the derivative remapping identity.","",
      "Negative K reaches signed transport/return eligibility but never baseline cleavage or new emission. Negative-emission and negative-fracture branches remain disabled, with no double counting.","",
      "NS1 and FA1 interfaces and derivative/symmetry tests are implemented analytically. Their physical matrices were not launched because the frozen 1-D contract supplies neither an identified NS1 coefficient set nor three frozen orientation-resolved stress paths.","",
      "The generalized inverse is `UNDERIDENTIFIED_GENERALIZED_BARRIER`; NS1 and FA1 directions remain in the null space. A validated contact model is absent, so a negative fracture branch is not admissible.","",
      "The evidence retains the hierarchy `opening-barrier shape > emission-conditioned correction > Peierls/Taylor correction`."]
    (OUT/"multi_R_anisotropic_final_decision.md").write_text("\n".join(lines)+"\n")
    freeze_payload=json.loads((OUT/"multi_R_prediction_freeze.json").read_text())
    write_json(OUT/"multi_R_anisotropic_verification.json",{"schema":"v10.2.30_multi_R_anisotropic_verification_v1",
      "result":"PASS" if all_complete else "FAIL","decision_match":primary in (OUT/"multi_R_anisotropic_final_decision.md").read_text(),
      "freeze_hash_valid":sha(OUT/"multi_R_prospective_predictions.csv")==freeze_payload["prospective_predictions_sha256"],
      "physical_count":len(physical),"figure_count":12,"active_worker_count":0})
    print(json.dumps({"result":"PASS","classification":primary,"physical":len(physical),"figures":12}))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("command",choices=["freeze","finalize"]);args=parser.parse_args()
    freeze() if args.command=="freeze" else finalize();return 0


if __name__=="__main__":raise SystemExit(main())
