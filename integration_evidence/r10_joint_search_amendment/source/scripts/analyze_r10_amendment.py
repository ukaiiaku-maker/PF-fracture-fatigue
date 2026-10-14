#!/usr/bin/env python3
"""Build the additive R10 amendment audit and selection ledger."""
from __future__ import annotations
import argparse,csv,hashlib,json,math
from pathlib import Path
import sys
import numpy as np
from scipy.special import gammainc
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arrhenius_fracture.retained_material_r10 import load_offspring_binding,load_strict_original_binding,load_v2_binding

DBTT="v913_paper_dbtt01_0202500_persistent_sites"

def surface_record(surface,T,stress,q=None,tau=None):
    G=np.asarray(surface.values_eV(stress,T),float);rate=np.asarray(surface.rate(stress,T),float)
    d=np.gradient(G,stress)*1e9
    out={"stress_GPa":(stress/1e9).tolist(),"barrier_eV":G.tolist(),"dG_dsigma_eV_per_GPa":d.tolist(),
         "raw_rate_s-1":rate.tolist(),"attempt_frequency_s-1":surface.attempt_frequency_s,
         "G0_eV":float(surface.G00_eV+surface.gT_eV_per_K*(T-surface.Tref_K)),
         "sigc_GPa":float((surface.sigc0_Pa+surface.sT_Pa_per_K*(T-surface.Tref_K))/1e9),
         "floor_fraction":surface.floor_fraction,"floor_min_eV":surface.floor_min_eV,"floor_max_fraction":surface.floor_max_fraction}
    G0=max(out["G0_eV"],1e-12);raw=max(surface.floor_min_eV,surface.floor_fraction*G0)
    out["floor_eV"]=min(surface.floor_max_fraction*G0,raw)
    out["floor_contact"]="minimum" if surface.floor_min_eV>=surface.floor_fraction*G0 else ("maximum_fraction" if raw>=surface.floor_max_fraction*G0 else "fractional")
    if q is not None:
        out["q"]=q;out["tau_s"]=tau;out["effective_rate_s-1"]=(gammainc(max(q,1.),np.minimum(rate*tau,1e12))/tau if q>1+1e-12 else rate).tolist()
    return out

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--original-registry",type=Path,required=True)
    p.add_argument("--projection-registry",type=Path,required=True);p.add_argument("--joint-registry",type=Path,required=True)
    p.add_argument("--v2-rows",type=Path,required=True);p.add_argument("--v2-manifest",type=Path,required=True);a=p.parse_args(argv)
    out=a.root/"amendment";out.mkdir(parents=True,exist_ok=True)
    bindings={"original_DBTT":load_strict_original_binding(a.original_registry,DBTT),
      "R10_fatigue_projection":load_offspring_binding(a.projection_registry,"R10_DBTT_P40_THERMAL_ANCHOR_V1"),
      "R10J_joint_descendant":load_offspring_binding(a.joint_registry,"R10J_DBTT_P40_LSHIFT_036MEV_V1")}
    stress=np.asarray([0.,.5,1.,2.,5.,10.,20.,30.])*1e9; audit={"schema":"r10.amendment-process-surface-audit/1","rows":{}}
    for name,b in bindings.items():
        q=b.reduced_config.cleavage_hits;tau=b.reduced_config.cleavage_tau_s;audit["rows"][name]={"candidate_id":b.optimized_row["candidate_id"],"q":q,"tau_s":tau,"temperatures":{}}
        processes={"opening":b.manifest.cleavage,"emission":b.manifest.emission,
                   "peierls":b.manifest.peierls.as_surface(b.manifest.emission),"taylor":b.manifest.taylor.as_surface(b.manifest.emission)}
        for T in (300.,750.,1200.):
            audit["rows"][name]["temperatures"][str(int(T))]={k:surface_record(v,T,stress,q,tau) if k=="opening" else surface_record(v,T,stress) for k,v in processes.items()}
    parent=audit["rows"]["original_DBTT"]["temperatures"]["1200"]["opening"]
    child=audit["rows"]["R10_fatigue_projection"]["temperatures"]["1200"]["opening"]
    audit["thermal_anchor_finding"]={"preserved_exactly":["scalar zero-stress opening G0 at 1200 K","scalar opening sigc at 1200 K"],
      "not_preserved":["complete stress-dependent opening surface","opening derivative","floor","fracture trajectory"],
      "G0_difference_eV":child["G0_eV"]-parent["G0_eV"],"sigc_difference_GPa":child["sigc_GPa"]-parent["sigc_GPa"],
      "maximum_sampled_barrier_difference_eV":float(np.max(np.abs(np.asarray(child["barrier_eV"])-parent["barrier_eV"]))) }
    (out/"process_surface_audit.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")

    states=[]
    for campaign in ("r10_dbtt_p40_projection_coarse","r10j_dbtt_p40_lshift_036mev_screen",
                     "r10j_peak_p40_lshift_036mev_screen","r10j_weakt_p40_lshift_036mev_screen"):
      for path in sorted((a.root/"fracture"/campaign).glob("T_*K.json")):
        d=json.loads(path.read_text());stage=(d["events"][0].get("hazard_bearing_state") if d["events"] else d.get("terminal_hazard_bearing_state",{}))
        states.append({"campaign":campaign,"candidate_id":d["candidate_id"],"temperature_K":d["temperature_K"],"stage":"first_event_post_renewal" if d["events"] else "terminal_censor",
          "K_MPa_sqrt_m":stage.get("K_MPa_sqrt_m"),"status":d["status"],**{k:stage.get(k) for k in ("gross_source_activity","emitted_total","mobile_count","retained_count","signed_return","accumulated_source_linked_slip","radius_over_r0","K_shield_MPa_sqrt_m","opening_stress_Pa","cleavage_rate_s-1","hazard_action_current","remaining_action_to_mark")}})
    with (out/"hazard_stage_state_comparison.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(states[0]));w.writeheader();w.writerows(states)

    peak=json.loads((a.root/"bridges/original_peak_seed4810/Kmax_27p790626/result.json").read_text())
    v2=json.loads((a.root/"bridges/dbtt_v2_p25_seed4810/Kmax_27p790626/checkpoint/trajectory_progress.json").read_text())
    vb=load_v2_binding(a.v2_rows,a.v2_manifest,"P25_TJBSV2_S_002987")
    reconciliation={"schema":"r10.action-convention-reconciliation/1","same_load_label_only":True,"records":[
      {"value":peak["overlap"]["explicit_hazard_action"],"units":"dimensionless action/cycle","material":peak["candidate_id"],"q":3.,"tau_s":1e-6,"convention":"one explicit 64-phase cycle from initial state","state":"initial","overlap_passed":True},
      {"value":v2["overlap"]["mean_hazard_action"],"units":"dimensionless action/cycle","material":v2["identity"]["candidate_id"],"q":vb.reduced_config.cleavage_hits,"tau_s":vb.reduced_config.cleavage_tau_s,"convention":"A0 mean-cycle predictor from initial state","state":"initial","overlap_passed":False},
      {"value":v2["overlap"]["explicit_hazard_action"],"units":"dimensionless action/cycle","material":v2["identity"]["candidate_id"],"q":vb.reduced_config.cleavage_hits,"tau_s":vb.reduced_config.cleavage_tau_s,"convention":"explicit 64-phase cycle after state evolution in overlap copy","state":"evolved","overlap_passed":False}],
      "finding":"22.258 belongs to original Peak. 0.092684 and 0.039237 belong to the DBTT V2 row; their mismatch is the measured failed A0/evolved overlap, not inconsistent units or a live binding error."}
    (out/"action_convention_reconciliation.json").write_text(json.dumps(reconciliation,indent=2,sort_keys=True)+"\n")

    windows=list(csv.DictReader((a.root/"current_bridge_windows.csv").open()));joint=[x for x in windows if x["candidate_id"].startswith("R10J_")]
    classes=[
      {"class":"DBTT","family":"A_ORIGINAL_SINGLE_EXP_FLOOR","candidate_id":"R10J_DBTT_P40_LSHIFT_036MEV_V1","evidence":"fresh analytical + complete-state 1-D","fracture":"Kinit(300)=5.218; 450-1200 K right-censored at 80","fatigue":"seed4810 m=3.999/2.574 decades; seed9117 m=4.004/2.574 decades","status":"JOINT_FATIGUE_AND_LOW_T_SCALE_PASS; THERMAL_CLASS_UNRESOLVED","spatial":"withheld pending class-feasible selection"},
      {"class":"DBTT","family":"A_ORIGINAL_SINGLE_EXP_FLOOR","candidate_id":"R10_DBTT_P40_THERMAL_ANCHOR_V1","evidence":"reused + instrumented endpoint replay","fracture":"Kinit(300)=2.282; 750/1200 K right-censored at 80","fatigue":"sparse m=4.005/2.414 decades","status":"REJECT_LOW_T_SCALE; retained frontier control","spatial":"not selected"},
      {"class":"Peak-T","family":"A_ORIGINAL_SINGLE_EXP_FLOOR","candidate_id":"R10J_PEAK_P40_LSHIFT_036MEV_V1","evidence":"fresh analytical + complete-state 1-D","fracture":"Kinit(300)=5.218; 450-1200 K right-censored at 80","fatigue":"A0 moderate band 4.00 decades; full-state withheld after class miss","status":"LOW_T_SCALE_PASS; PEAK_CLASS_MISS/NEAR_MISS","spatial":"not selected"},
      {"class":"Weak-T compensated/ductile","family":"A_ORIGINAL_SINGLE_EXP_FLOOR","candidate_id":"R10J_WEAKT_P40_LSHIFT_036MEV_V1","evidence":"fresh analytical + complete-state 1-D","fracture":"Kinit(300)=5.237; 450-1200 K right-censored at 80","fatigue":"A0 moderate band 4.00 decades; full-state withheld after subtype miss","status":"LOW_T_SCALE_PASS; COMPENSATED_WEAK-T_SUBTYPE_MISS/NEAR_MISS","spatial":"not selected"},
      {"class":"Weak-T compensated/ductile","family":"B_INHERITED_V2_GUARD_ENTROPY_EXTENSION","candidate_id":"P25_TJBSV2_S_026704","evidence":"reused complete/restart state","fracture":"historical exact V2","fatigue":"one developed point 1.106e-10 at DeltaK=4.145","status":"ACTIVE_UNRESOLVED_SUBTYPE; not flat opening control","spatial":"two sealed 300/1200 K rows reused only"},
      {"class":"Ceramic-like control","family":"A/B controls","candidate_id":"retained ceramic rows","evidence":"reused","fracture":"control evidence retained","fatigue":"no Paris requirement","status":"CONTROL","spatial":"sealed historical scope only"}]
    with (out/"class_family_result_table.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(classes[0]));w.writeheader();w.writerows(classes)

    scan=list(csv.DictReader((a.root/"joint_level_iteration/joint_level_scan.csv").open()))
    near=[x for x in scan if x["class"] in ("DBTT","PEAK","WEAKT") and x["delta_G_eV"] in ("0.034","0.036000000000000004")]
    ledger={"schema":"r10.amendment-selection-ledger/1","selected":joint,"near_misses":near,
      "analytical_to_1d":{"candidate_id":"R10J_DBTT_P40_LSHIFT_036MEV_V1","frozen_Kinit_300":5.22,"complete_state_Kinit_300":5.218191433095953,"difference_MPa_sqrt_m":-0.001808566904047},
      "rate_uncertainty":{"shared_amplitudes_seed_ratio_9117_over_4810":[.927025,.927025,.927025],"interpretation":"about 7.3% seed effect; identical ratio arises from shared mark/reward histories in inactive state"},
      "pending_jobs":["joint thermal-coefficient/local-derivative Pareto continuation for DBTT, Peak-T, and compensated weak-T","resume informative active-state V2 checkpoints within declared resource share","same-row 2-D only after a class-feasible 1-D selection"],
      "resource_censored_jobs":["DBTT V2 P25 high-load checkpoints","original Peak low-side checkpoint"],"running_jobs":[]}
    (out/"selection_ledger.json").write_text(json.dumps(ledger,indent=2,sort_keys=True)+"\n")
    state_path=a.root/"continuation_state.json";state=json.loads(state_path.read_text());state["completed"]=list(dict.fromkeys(state["completed"]+["R10 amendment terminal-state and action-convention audit","joint level-shift analytical iteration","DBTT joint descendant sparse and independent five-amplitude fatigue confirmation","DBTT joint descendant seven-temperature monotonic panel"]))
    state["running_jobs"]=[];state["queued_jobs"]=ledger["pending_jobs"];state["next"]=ledger["pending_jobs"][0];state["amendment_ledger"]=str((out/"selection_ledger.json").resolve());state_path.write_text(json.dumps(state,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"surface_audit":str(out/"process_surface_audit.json"),"state_rows":len(states),"class_rows":len(classes),"joint_windows":joint},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
