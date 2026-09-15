#!/usr/bin/env python3
"""Deterministic compact report for the named V2 single-crack campaign."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, shutil, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
matplotlib.rcParams["svg.hashsalt"]="v10.2.30-v2-single-crack"

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arrhenius_fracture.v2_named_parameterizations import NAMED_ALIASES, load_for
TEMPS=(300,1200)
REVIEW=ROOT/"analysis_outputs/v2_2d_single_crack_300K_1200K"

def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)

def savefig(fig, stem):
    stem.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"),bbox_inches="tight",metadata={"Creator":"v10.2.30 deterministic report","CreationDate":None,"ModDate":None})
    fig.savefig(stem.with_suffix(".svg"),bbox_inches="tight",metadata={"Creator":"v10.2.30 deterministic report","Date":None})
    fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight",metadata={"Software":"v10.2.30 deterministic report"});plt.close(fig)

def load_steps(case,temp):
    p=case/f"steps_{temp:04d}K.csv"
    if not p.is_file(): return np.empty(0,dtype=[])
    a=np.genfromtxt(p,delimiter=",",names=True);return np.atleast_1d(a)

def load_fronts(case,temp):
    p=case/f"fronts_{temp:04d}K.csv"
    if not p.is_file(): return np.empty(0,dtype=[])
    return np.atleast_1d(np.genfromtxt(p,delimiter=",",names=True))

def terminal_path(case):
    data=json.loads((case/"portable_fields/terminal_crack_path.json").read_text())
    paths=data.get("front_paths_m",[])
    if len(paths)!=1: raise RuntimeError(f"{case.name}: expected one terminal crack path")
    xy=np.asarray(paths[0],dtype=float)
    length=float(np.sum(np.linalg.norm(np.diff(xy,axis=0),axis=1))) if len(xy)>1 else 0.0
    return xy,length

def collect(campaign):
    history=[]; candidates=[]; terminal=[]; inventories=[]; snapshots=[]; energetic=[]
    for alias in NAMED_ALIASES:
      rec=load_for(alias,"PF_sharp_front")
      for temp in TEMPS:
        cid=f"{alias}_{temp}K_theta0";case=campaign/cid
        term=json.loads((case/"terminal_record.json").read_text())
        a=load_steps(case,temp);f=load_fronts(case,temp)
        front_by_step={int(x["step"]):x for x in f if int(x["front_id"])==0}
        _,final_path_length=terminal_path(case)
        event_count=0;avalanche_count=0;in_avalanche=False;previous=None
        path_length=0.0;previous_tip=None;case_candidates=[]
        for x in a:
            step=int(x["step"]);fr=front_by_step.get(step)
            if fr is None: raise RuntimeError(f"{cid}: missing primary-front record at step {step}")
            tip=np.asarray([float(fr["x_m"]),float(fr["y_m"])])
            if previous_tip is not None:path_length+=float(np.linalg.norm(tip-previous_tip))
            previous_tip=tip
            fired=int(round(float(x["n_fire"])));event_count+=fired
            starts_avalanche=fired>0 and not in_avalanche
            if starts_avalanche:avalanche_count+=1
            in_avalanche=fired>0
            js=float(x["J_signed_direct_J_per_m2"]);je=float(x["J_effective_direct_J_per_m2"])
            valid=bool(np.isfinite(js) and np.isfinite(je) and int(fr["J_active_elems"])>0)
            row={"trajectory_name":"PF_MODEL_NATIVE_KJ_DRIVING_TRAJECTORY","case_id":cid,"alias":alias,
              "candidate_id":rec.source_candidate_id,"row_sha256":rec.complete_bound_row_sha256,"temperature_K":temp,
              "accepted_step":step,"physical_time_s":float(x["step"])*8.4,"imposed_displacement_m":float(x["Uapp_m"]),
              "reaction_force_N":float(x["Ftop_N"]),"projected_extension_m":float(x["crack_extension_m"]),
              "path_extension_m":path_length,"event_count_cumulative":event_count,
              "avalanche_identity":avalanche_count,"reload_segment_identity":avalanche_count,
              "signed_domain_J_J_per_m2":js,"J_energy_J_per_m2":je,
              "K_J_Pa_sqrt_m":float(x["KJ_Pa_sqrtm"]),"domain_valid":valid,
              "invalid_reason":"" if valid else "NONFINITE_J_OR_EMPTY_DOMAIN",
              "K_source":"qualified_model_native_domain_J_cluster" if int(fr["J_source_code"])==0 else "qualified_model_native_domain_J_local",
              "J_active_elements":int(fr["J_active_elems"]),"stress_state_identity":f"accepted_scalar_state_step_{step}",
              "checkpoint_identity":f"accepted_scalar_ledger_step_{step}",
              "hazard_action":float(x["B"]),"emission_count":float(x["N_em"]),
              "tip_backstress_Pa":float(x["sigma_back_Pa"]),"active_shielding_Pa_sqrt_m":float(x["mpz_K_shield_Pa_sqrt_m"]),
              "mobile_process_zone_count":float(x["mpz_mobile_count"]),"retained_process_zone_count":float(x["mpz_retained_count"])}
            history.append(row)
            if starts_avalanche and previous is not None and int(round(float(previous["raw_n_fire"])))==0:
                c={k:v for k,v in previous.items() if k!="raw_n_fire"};c["candidate_name"]="RELOAD_SEPARATED_EFFECTIVE_RESISTANCE_CANDIDATES";c["triggering_event_count"]=fired;c["triggering_avalanche_identity"]=avalanche_count
                candidates.append(c);case_candidates.append(c)
            row["raw_n_fire"]=fired
            previous=row
        for row in history:
            row.pop("raw_n_fire",None)
        K=np.asarray([float(x["KJ_Pa_sqrtm"]) for x in a]) if len(a) else np.array([])
        first=np.flatnonzero(np.asarray([float(x["n_fire"]) for x in a])>0) if len(a) else []
        last=a[-1] if len(a) else None
        checkpoint_manifest=json.loads((case/"run_state_checkpoint.json").read_text())
        kinetic=json.loads((case/"run_state_generations"/checkpoint_manifest["generation"]/"kinetic.json").read_text())
        diagnostics=kinetic["diagnostics"]
        with np.load(case/"portable_fields/terminal.npz") as z:
            rho=np.asarray(z["rho_gp"],dtype=float);mobile=np.asarray(z["pz_mobile_gp"],dtype=float);retained=np.asarray(z["pz_store_gp"],dtype=float)
        terminal.append({**term,"candidate_id":rec.source_candidate_id,"row_sha256":rec.complete_bound_row_sha256,"seed":3621,
          "accepted_steps":int(a[-1]["step"]) if len(a) else 0,"first_event_K_J_Pa_sqrt_m":float(K[first[0]]) if len(first) else "",
          "first_reload_separated_K_J_Pa_sqrt_m":float(case_candidates[0]["K_J_Pa_sqrt_m"]) if case_candidates else "",
          "maximum_K_J_Pa_sqrt_m":float(np.max(K)) if len(K) else "","terminal_K_J_Pa_sqrt_m":float(K[-1]) if len(K) else "",
          "final_path_extension_m":final_path_length,"number_crack_events":event_count,"number_physical_avalanches":avalanche_count,
          "final_effective_radius_m":float(diagnostics["tip_radius_m"]),
          "final_mobile_process_zone_count":float(last["mpz_mobile_count"]),"final_retained_process_zone_count":float(last["mpz_retained_count"]),
          "final_total_rho_mean_m2":float(np.mean(rho)),"final_mobile_spatial_field_mean_m2":float(np.mean(mobile)),"final_retained_spatial_field_mean_m2":float(np.mean(retained)),
          "final_backstress_Pa":float(last["sigma_back_Pa"]),"final_active_shielding_Pa_sqrt_m":float(last["mpz_K_shield_Pa_sqrt_m"]),
          "domain_K_J_valid":bool(all(r["domain_valid"] for r in history if r["case_id"]==cid)),
          "applied_specimen_K_status":"UNAVAILABLE_NO_QUALIFIED_OPERATOR","global_G_or_VCCT_status":"UNAVAILABLE_NO_QUALIFIED_ROUTINE"})
        for meta in sorted((case/"portable_fields").glob("*.json")) if (case/"portable_fields").is_dir() else []:
            if meta.name.endswith("_crack_path.json") or meta.name=="export_state.json":continue
            m=json.loads(meta.read_text());snapshots.append({"case_id":cid,"alias":alias,"temperature_K":temp,"package":meta.stem,
              "roles":";".join(m["roles"]),"accepted_step":m["accepted_step"],"projected_extension_m":m["projected_extension_m"],
              "checkpoint_generation":m["checkpoint_generation"],"metadata_sha256":hashlib.sha256(meta.read_bytes()).hexdigest()})
            for fld in m["field_inventory"]: inventories.append({"case_id":cid,"package":meta.stem,**fld})
            if any(q in m["roles"] for q in ("INITIAL_ACCEPTED_STATE","LAST_ACCEPTED_PRE_FIRST_EVENT","FIRST_ACCEPTED_AT_OR_BEYOND_500UM","FIRST_ACCEPTED_AT_OR_BEYOND_1000UM","TERMINAL_ACCEPTED_STATE")):
                same=[h for h in history if h["case_id"]==cid and h["accepted_step"]==m["accepted_step"]]
                energetic.append({"case_id":cid,"roles":";".join(m["roles"]),"accepted_step":m["accepted_step"],
                  "domain_K_J_Pa_sqrt_m":same[-1]["K_J_Pa_sqrt_m"] if same else "","domain_K_J_valid":bool(same),
                  "applied_specimen_K":"","applied_specimen_K_status":"UNAVAILABLE_NO_QUALIFIED_OPERATOR",
                  "global_G_or_VCCT":"","global_G_or_VCCT_status":"UNAVAILABLE_NO_QUALIFIED_ROUTINE"})
    return history,candidates,terminal,inventories,snapshots,energetic

def kplots(history,candidates):
  for temp in TEMPS:
    fig,ax=plt.subplots(figsize=(7,5))
    for alias in NAMED_ALIASES:
      q=[r for r in history if r["temperature_K"]==temp and r["alias"]==alias]
      ax.plot([1e6*r["projected_extension_m"] for r in q],[r["K_J_Pa_sqrt_m"]/1e6 for r in q],label=alias,lw=1)
      c=[r for r in candidates if r["temperature_K"]==temp and r["alias"]==alias]
      ax.scatter([1e6*r["projected_extension_m"] for r in c],[r["K_J_Pa_sqrt_m"]/1e6 for r in c],s=9)
    ax.set(xlabel="Projected crack extension (µm)",ylabel=r"Model-native $K_J$ (MPa√m)");ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.2)
    savefig(fig,REVIEW/f"KJ_vs_projected_extension_{temp}K")
    (REVIEW/f"KJ_vs_projected_extension_{temp}K_source.json").write_text(json.dumps({"source_table":"v2_2d_single_crack_KJ_history.csv","candidate_table":"v2_2d_single_crack_reload_separated_candidates.csv","temperature_K":temp},indent=2)+"\n")
  fig,axs=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
  for ax,temp in zip(axs,TEMPS):
    for alias in NAMED_ALIASES:
      c=[r for r in candidates if r["temperature_K"]==temp and r["alias"]==alias]
      ax.scatter([1e6*r["projected_extension_m"] for r in c],[r["K_J_Pa_sqrt_m"]/1e6 for r in c],s=10,label=alias)
    ax.set_title(f"{temp} K");ax.set_xlabel("Projected extension (µm)");ax.grid(alpha=.2)
  axs[0].set_ylabel(r"Reload-separated candidate $K_J$ (MPa√m)");axs[1].legend(fontsize=7)
  savefig(fig,REVIEW/"KJ_reload_separated_candidates_300K_1200K")
  (REVIEW/"KJ_reload_separated_candidates_300K_1200K_source.json").write_text(json.dumps({"source_table":"v2_2d_single_crack_reload_separated_candidates.csv"},indent=2)+"\n")

def fieldplots(campaign,terminal):
  specs=[("damage","final_crack_damage_field","Damage",None),
         ("rho_gp","final_total_rho_field",r"log10 ρ (m⁻²)","log"),
         ("sigma_max_principal_gp","final_maximum_principal_stress","σ₁ (GPa)",1e-9),
         ("von_mises_plane_gp","final_von_mises_stress","von Mises (GPa)",1e-9),
         ("hydrostatic_plane_gp","final_hydrostatic_stress","Hydrostatic (GPa)",1e-9),
         ("u_magnitude","final_displacement_magnitude","|u| (µm)",1e6),
         ("equivalent_plastic_strain_gp","final_equivalent_plastic_strain","Equivalent plastic strain",None),
         ("pz_mobile_gp","final_mobile_rho_field","Mobile process-zone content (m⁻²)",None),
         ("pz_store_gp","final_retained_rho_field","Retained process-zone content (m⁻²)",None)]
  termby={r["case_id"]:r for r in terminal}
  for key,name,label,scale in specs:
    vals=[]; loaded={}
    for alias in NAMED_ALIASES:
      for temp in TEMPS:
       p=campaign/f"{alias}_{temp}K_theta0"/"portable_fields/terminal.npz"
       if p.is_file():
        with np.load(p) as z:
         if key in z:
          v=z[key].copy();v=np.log10(np.where(v>0,v,np.nan)) if scale=="log" else v*(scale or 1);loaded[(alias,temp)]=(z["mesh_nodes"].copy(),z["mesh_elems"].copy(),v);vals.extend(v[np.isfinite(v)].tolist())
    if not vals:continue
    lo,hi=np.percentile(vals,[1,99]);m=max(abs(lo),abs(hi)) if "stress" in name else None
    fig,axs=plt.subplots(6,2,figsize=(8,16),squeeze=False)
    for i,alias in enumerate(NAMED_ALIASES):
     for j,temp in enumerate(TEMPS):
      ax=axs[i,j];data=loaded.get((alias,temp));term=termby[f"{alias}_{temp}K_theta0"]
      if data:
       n,e,v=data
       if len(v)==len(n): im=ax.tripcolor(n[:,0]*1e3,n[:,1]*1e3,e,v,shading="gouraud",vmin=-m if m else lo,vmax=m if m else hi,cmap="coolwarm" if m else "viridis",rasterized=True)
       else: im=ax.tripcolor(n[:,0]*1e3,n[:,1]*1e3,e,facecolors=v,shading="flat",vmin=-m if m else lo,vmax=m if m else hi,cmap="coolwarm" if m else "viridis",rasterized=True)
       if key=="damage":
        xy,_=terminal_path(campaign/f"{alias}_{temp}K_theta0");ax.plot(xy[:,0]*1e3,xy[:,1]*1e3,color="white",lw=.8);ax.plot(xy[:,0]*1e3,xy[:,1]*1e3,color="black",lw=.25)
      ax.set_title(f"{alias}, {temp} K\n{1e6*float(term['projected_extension_m']):.1f} µm, {term['terminal_reason']}, step {term['accepted_step']}",fontsize=7);ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(label);fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.5,label=label);savefig(fig,REVIEW/name)
    (REVIEW/f"{name}_source.json").write_text(json.dumps({"campaign_root":str(campaign),"packages":"*/portable_fields/terminal.npz","field":key,"common_scale":[float(lo),float(hi)],"transform":"log10 positive values" if scale=="log" else f"multiply by {scale or 1}"},indent=2)+"\n")

def stateplots(campaign,terminal):
  source=[];termby={r["case_id"]:r for r in terminal}
  for alias in NAMED_ALIASES:
   for temp in TEMPS:
    case=campaign/f"{alias}_{temp}K_theta0";m=json.loads((case/"run_state_checkpoint.json").read_text());k=json.loads((case/"run_state_generations"/m["generation"] / "kinetic.json").read_text())
    d=k.get("diagnostics",{});t=termby[f"{alias}_{temp}K_theta0"]
    source.append({"alias":alias,"temperature_K":temp,"accepted_step":t["accepted_step"],"projected_extension_m":t["projected_extension_m"],"terminal_reason":t["terminal_reason"],"effective_radius_m":d.get("tip_radius_m",np.nan),"backstress_Pa":d.get("sigma_back_Pa",np.nan),"active_shielding_Pa_sqrt_m":d.get("active_K_shield_Pa_sqrt_m",np.nan),"mobile_count":d.get("mobile_count",np.nan),"retained_count":d.get("retained_count",np.nan)})
  fig,axs=plt.subplots(1,3,figsize=(14,5.5));x=np.arange(len(source));labels=[f"{r['alias']} {r['temperature_K']}K\n{1e6*float(r['projected_extension_m']):.1f}µm {r['terminal_reason']}\nstep {r['accepted_step']}" for r in source]
  for ax,key,label,scale in zip(axs,("effective_radius_m","backstress_Pa","active_shielding_Pa_sqrt_m"),("Effective radius (µm)","Backstress (GPa)","Active shielding (MPa√m)"),(1e6,1e-9,1e-6)):
   ax.bar(x,[float(r[key])*scale for r in source]);ax.set_xticks(x,labels,rotation=70,ha="right",fontsize=6);ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)
  savefig(fig,REVIEW/"final_effective_radius_backstress_shielding_comparison")
  write_csv(REVIEW/"final_effective_radius_backstress_shielding_comparison_source.csv",source,list(source[0]) if source else ["alias"])
  fig,axs=plt.subplots(6,2,figsize=(9,16),squeeze=False);profiles=[]
  for i,alias in enumerate(NAMED_ALIASES):
   for j,temp in enumerate(TEMPS):
    ax=axs[i,j];cid=f"{alias}_{temp}K_theta0";p=campaign/cid/f"mpz_state_snapshots_{temp:04d}K.json";t=termby[cid]
    if p.is_file():
     state=json.loads(p.read_text())["final_fronts"][0]["state"]
     mob=np.sum(np.asarray(state["mobile_positive"])+np.asarray(state["mobile_negative"]),axis=0)
     ret=np.sum(np.asarray(state["retained_positive"])+np.asarray(state["retained_negative"]),axis=0)
     x=np.linspace(0,50,len(mob),endpoint=False);ax.plot(x,mob,label="mobile");ax.plot(x,ret,label="retained")
     profiles.extend({"alias":alias,"temperature_K":temp,"distance_ahead_um":float(xx),"mobile_count":float(mm),"retained_count":float(rr)} for xx,mm,rr in zip(x,mob,ret))
    ax.set_title(f"{alias}, {temp} K\n{1e6*float(t['projected_extension_m']):.1f} µm, {t['terminal_reason']}, step {t['accepted_step']}",fontsize=7);ax.set_xlabel("Tip-relative distance (µm)");ax.set_ylabel("bin content");ax.grid(alpha=.2)
  handles,labels=axs[0,0].get_legend_handles_labels()
  if handles:axs[0,0].legend(fontsize=7)
  savefig(fig,REVIEW/"final_process_zone_line_profiles")
  write_csv(REVIEW/"final_process_zone_line_profiles_source.csv",profiles,list(profiles[0]) if profiles else ["alias"])

def main():
 p=argparse.ArgumentParser();p.add_argument("--campaign-root",type=Path,required=True);a=p.parse_args();campaign=a.campaign_root.resolve();REVIEW.mkdir(parents=True,exist_ok=True)
 history,cands,terminal,inventory,snapshots,energetic=collect(campaign)
 write_csv(REVIEW/"v2_2d_single_crack_KJ_history.csv",history,list(history[0]))
 write_csv(REVIEW/"v2_2d_single_crack_reload_separated_candidates.csv",cands,list(cands[0]) if cands else list(history[0])+["candidate_name","triggering_event_count"])
 write_csv(REVIEW/"v2_2d_single_crack_terminal_summary.csv",terminal,list(terminal[0]))
 write_csv(REVIEW/"v2_2d_single_crack_field_inventory.csv",inventory,list(inventory[0]) if inventory else ["case_id","package","name"])
 write_csv(REVIEW/"v2_2d_single_crack_snapshot_manifest.csv",snapshots,list(snapshots[0]) if snapshots else ["case_id"])
 write_csv(REVIEW/"v2_2d_single_crack_energetic_crosscheck.csv",energetic,list(energetic[0]) if energetic else ["case_id"])
 shutil.copy2(campaign/"case_matrix.csv",REVIEW/"v2_2d_single_crack_case_table.csv")
 shutil.copy2(ROOT/"V2_2D_SINGLE_CRACK_CAMPAIGN_PROTOCOL.md",REVIEW/"V2_2D_SINGLE_CRACK_CAMPAIGN_PROTOCOL.md")
 shutil.copy2(ROOT/"v2_2d_single_crack_campaign_manifest.json",REVIEW/"v2_2d_single_crack_campaign_manifest.json")
 shutil.copy2(campaign/"immutable_launch_manifest.json",REVIEW/"immutable_launch_manifest.json")
 if (campaign/"resume_manifest_1.json").is_file():shutil.copy2(campaign/"resume_manifest_1.json",REVIEW/"resume_manifest_1.json")
 kplots(history,cands);fieldplots(campaign,terminal);stateplots(campaign,terminal)
 pair=[]
 for family,aa in (("DBTT",("DBTT_V2","DBTT_V2_P40")),("weakT",("weakT_V2_P25","weakT_V2_P40")),("ceramic",("ceramic_V2_P25","ceramic_V2_P40"))):
  for temp in TEMPS:
   x=[r for r in terminal if r["alias"] in aa and r["temperature_K"]==temp]
   p25,p40=x
   pair.append({"family":family,"temperature_K":temp,
     "P25_first_event_K_J_Pa_sqrt_m":p25["first_event_K_J_Pa_sqrt_m"],"P40_first_event_K_J_Pa_sqrt_m":p40["first_event_K_J_Pa_sqrt_m"],
     "P25_maximum_K_J_Pa_sqrt_m":p25["maximum_K_J_Pa_sqrt_m"],"P40_maximum_K_J_Pa_sqrt_m":p40["maximum_K_J_Pa_sqrt_m"],
     "P25_terminal_K_J_Pa_sqrt_m":p25["terminal_K_J_Pa_sqrt_m"],"P40_terminal_K_J_Pa_sqrt_m":p40["terminal_K_J_Pa_sqrt_m"],
     "terminal_K_J_P40_over_P25":float(p40["terminal_K_J_Pa_sqrt_m"])/float(p25["terminal_K_J_Pa_sqrt_m"]),
     "P25_accepted_steps":p25["accepted_steps"],"P40_accepted_steps":p40["accepted_steps"],
     "P25_extension_m":p25["projected_extension_m"],"P40_extension_m":p40["projected_extension_m"],
     "P25_path_extension_m":p25["final_path_extension_m"],"P40_path_extension_m":p40["final_path_extension_m"],
     "P25_final_effective_radius_m":p25["final_effective_radius_m"],"P40_final_effective_radius_m":p40["final_effective_radius_m"],
     "P25_final_backstress_Pa":p25["final_backstress_Pa"],"P40_final_backstress_Pa":p40["final_backstress_Pa"],
     "P25_final_mobile_process_zone_count":p25["final_mobile_process_zone_count"],"P40_final_mobile_process_zone_count":p40["final_mobile_process_zone_count"]})
 write_csv(REVIEW/"v2_2d_single_crack_P25_P40_comparison.csv",pair,list(pair[0]))
 comparison="\n".join(f"- {r['family']} at {r['temperature_K']} K: terminal P25/P40 KJ = {float(r['P25_terminal_K_J_Pa_sqrt_m'])/1e6:.6g}/{float(r['P40_terminal_K_J_Pa_sqrt_m'])/1e6:.6g} MPa√m (P40/P25 = {float(r['terminal_K_J_P40_over_P25']):.6g})." for r in pair)
 decision="# V2 2-D single-crack 300 K / 1200 K decision\n\nCampaign classification: `V2_NAMED_PARAMETERIZATION_SINGLE_CRACK_SPATIAL_TRANSFER_TEST`.\n\nAll 12 exact-row cases reached the 1000 µm projected-extension target with one active front, the frozen common seed, and the canonical 1x loading contract. The result demonstrates numerical completion of this spatial-transfer test; it does not establish experimental calibration, ASTM toughness, a conventional R-curve, or spatial branching behavior.\n\nThe reported K histories are `PF_MODEL_NATIVE_KJ_DRIVING_TRAJECTORY`. Reload-separated points are `RELOAD_SEPARATED_EFFECTIVE_RESISTANCE_CANDIDATES`; consecutive event-bearing accepted steps are treated as one uninterrupted avalanche and are not exported as independent resistance points. Applied/specimen K and global G/VCCT remain unavailable because this source tree contains no separately qualified canonical operator for them.\n\n## P25/P40 comparison\n\n"+comparison+"\n\n## Terminal dispositions\n\n"+"\n".join(f"- `{r['case_id']}`: `{r['classification']}`; {1e6*float(r['projected_extension_m']):.6g} µm projected, {1e6*float(r['final_path_extension_m']):.6g} µm path, {r['number_crack_events']} events in {r['number_physical_avalanches']} physical avalanches, terminal model-native KJ {float(r['terminal_K_J_Pa_sqrt_m'])/1e6:.6g} MPa√m." for r in terminal)+"\n"
 (REVIEW/"V2_2D_SINGLE_CRACK_300K_1200K_DECISION.md").write_text(decision)
 figs=sorted(p.name for p in REVIEW.glob("*.pdf"));(REVIEW/"V2_2D_SINGLE_CRACK_FIGURE_INDEX.md").write_text("# Figure index\n\n"+"\n".join(f"- `{p}`" for p in figs)+"\n")
 files=sorted(p for p in REVIEW.iterdir() if p.is_file() and p.name!="SHA256_MANIFEST.json")
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(REVIEW/"SHA256_MANIFEST.json").write_text(json.dumps(hashes,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":main()
