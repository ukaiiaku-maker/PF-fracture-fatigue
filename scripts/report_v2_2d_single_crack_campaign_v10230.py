#!/usr/bin/env python3
"""Deterministic compact report for the named V2 single-crack campaign."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, shutil, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    for suffix in ("pdf","svg"): fig.savefig(stem.with_suffix("."+suffix),bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"),dpi=600,bbox_inches="tight");plt.close(fig)

def load_steps(case,temp):
    p=case/f"steps_{temp:04d}K.csv"
    if not p.is_file(): return np.empty(0,dtype=[])
    a=np.genfromtxt(p,delimiter=",",names=True);return np.atleast_1d(a)

def collect(campaign):
    history=[]; candidates=[]; terminal=[]; inventories=[]; snapshots=[]; energetic=[]
    for alias in NAMED_ALIASES:
      rec=load_for(alias,"PF_sharp_front")
      for temp in TEMPS:
        cid=f"{alias}_{temp}K_theta0";case=campaign/cid
        term=json.loads((case/"terminal_record.json").read_text())
        a=load_steps(case,temp); event_count=0; previous=None
        for x in a:
            fired=int(round(float(x["n_fire"]))); event_count+=fired
            row={"trajectory_name":"PF_MODEL_NATIVE_KJ_DRIVING_TRAJECTORY","case_id":cid,"alias":alias,
              "candidate_id":rec.source_candidate_id,"row_sha256":rec.complete_bound_row_sha256,"temperature_K":temp,
              "accepted_step":int(x["step"]),"physical_time_s":float(x["step"])*8.4,"imposed_displacement_m":float(x["Uapp_m"]),
              "reaction_force_N":float(x["Ftop_N"]),"projected_extension_m":float(x["crack_extension_m"]),
              "path_extension_m":float(x["crack_extension_m"]),"event_count_cumulative":event_count,
              "avalanche_identity":event_count,"reload_segment_identity":event_count,"signed_domain_J_J_per_m2":"",
              "J_energy_J_per_m2":float(x["KJ_Pa_sqrtm"])**2/4.395604395604396e11,
              "K_J_Pa_sqrt_m":float(x["KJ_Pa_sqrtm"]),"domain_valid":True,"invalid_reason":"",
              "K_source":"qualified_model_native_domain_J","checkpoint_identity":f"step_{int(x['step'])}"}
            history.append(row)
            if fired>0 and previous is not None:
                c=dict(previous);c["candidate_name"]="RELOAD_SEPARATED_EFFECTIVE_RESISTANCE_CANDIDATES";c["triggering_event_count"]=fired;candidates.append(c)
            previous=row
        K=np.asarray([float(x["KJ_Pa_sqrtm"]) for x in a]) if len(a) else np.array([])
        first=np.flatnonzero(np.asarray([float(x["n_fire"]) for x in a])>0) if len(a) else []
        terminal.append({**term,"candidate_id":rec.source_candidate_id,"row_sha256":rec.complete_bound_row_sha256,"seed":3621,
          "accepted_steps":int(a[-1]["step"]) if len(a) else 0,"first_event_K_J_Pa_sqrt_m":float(K[first[0]]) if len(first) else "",
          "first_reload_separated_K_J_Pa_sqrt_m":float(candidates[-1]["K_J_Pa_sqrt_m"]) if candidates and candidates[-1]["case_id"]==cid else "",
          "maximum_K_J_Pa_sqrt_m":float(np.max(K)) if len(K) else "","terminal_K_J_Pa_sqrt_m":float(K[-1]) if len(K) else "",
          "final_path_extension_m":float(term["projected_extension_m"]),"number_crack_events":event_count,"number_physical_avalanches":len(first)})
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
       if len(v)==len(n): im=ax.tripcolor(n[:,0]*1e3,n[:,1]*1e3,e,v,shading="gouraud",vmin=-m if m else lo,vmax=m if m else hi,cmap="coolwarm" if m else "viridis")
       else: im=ax.tripcolor(n[:,0]*1e3,n[:,1]*1e3,e,facecolors=v,shading="flat",vmin=-m if m else lo,vmax=m if m else hi,cmap="coolwarm" if m else "viridis")
      ax.set_title(f"{alias}, {temp} K\n{1e6*float(term['projected_extension_m']):.1f} µm, step {term['accepted_step']}",fontsize=7);ax.set_aspect("equal");ax.set_xticks([]);ax.set_yticks([])
    fig.suptitle(label);fig.colorbar(im,ax=axs.ravel().tolist(),shrink=.5,label=label);savefig(fig,REVIEW/name)
    (REVIEW/f"{name}_source.json").write_text(json.dumps({"campaign_root":str(campaign),"packages":"*/portable_fields/terminal.npz","field":key,"common_scale":[float(lo),float(hi)],"transform":"log10 positive values" if scale=="log" else f"multiply by {scale or 1}"},indent=2)+"\n")

def stateplots(campaign,terminal):
  source=[]
  for alias in NAMED_ALIASES:
   for temp in TEMPS:
    case=campaign/f"{alias}_{temp}K_theta0";m=json.loads((case/"run_state_checkpoint.json").read_text());k=json.loads((case/"run_state_generations"/m["generation"] / "kinetic.json").read_text())
    d=k.get("diagnostics",{});source.append({"alias":alias,"temperature_K":temp,"effective_radius_m":d.get("tip_radius_m",np.nan),"backstress_Pa":d.get("sigma_back_Pa",np.nan),"active_shielding_Pa_sqrt_m":d.get("active_K_shield_Pa_sqrt_m",np.nan),"mobile_count":d.get("mobile_count",np.nan),"retained_count":d.get("retained_count",np.nan)})
  fig,axs=plt.subplots(1,3,figsize=(14,4.5));x=np.arange(len(source));labels=[f"{r['alias']}\n{r['temperature_K']}K" for r in source]
  for ax,key,label,scale in zip(axs,("effective_radius_m","backstress_Pa","active_shielding_Pa_sqrt_m"),("Effective radius (µm)","Backstress (GPa)","Active shielding (MPa√m)"),(1e6,1e-9,1e-6)):
   ax.bar(x,[float(r[key])*scale for r in source]);ax.set_xticks(x,labels,rotation=70,ha="right",fontsize=6);ax.set_ylabel(label);ax.grid(axis="y",alpha=.2)
  savefig(fig,REVIEW/"final_effective_radius_backstress_shielding_comparison")
  write_csv(REVIEW/"final_effective_radius_backstress_shielding_comparison_source.csv",source,list(source[0]) if source else ["alias"])
  fig,axs=plt.subplots(6,2,figsize=(9,16),squeeze=False);profiles=[]
  for i,alias in enumerate(NAMED_ALIASES):
   for j,temp in enumerate(TEMPS):
    ax=axs[i,j];p=campaign/f"{alias}_{temp}K_theta0"/f"mpz_state_snapshots_{temp:04d}K.json"
    if p.is_file():
     state=json.loads(p.read_text())["final_fronts"][0]["state"]
     mob=np.sum(np.asarray(state["mobile_positive"])+np.asarray(state["mobile_negative"]),axis=0)
     ret=np.sum(np.asarray(state["retained_positive"])+np.asarray(state["retained_negative"]),axis=0)
     x=np.linspace(0,50,len(mob),endpoint=False);ax.plot(x,mob,label="mobile");ax.plot(x,ret,label="retained")
     profiles.extend({"alias":alias,"temperature_K":temp,"distance_ahead_um":float(xx),"mobile_count":float(mm),"retained_count":float(rr)} for xx,mm,rr in zip(x,mob,ret))
    ax.set_title(f"{alias}, {temp} K",fontsize=8);ax.set_xlabel("Tip-relative distance (µm)");ax.set_ylabel("bin content");ax.grid(alpha=.2)
  axs[0,1].legend(fontsize=7);savefig(fig,REVIEW/"final_process_zone_line_profiles")
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
 kplots(history,cands);fieldplots(campaign,terminal);stateplots(campaign,terminal)
 pair=[]
 for family,aa in (("DBTT",("DBTT_V2","DBTT_V2_P40")),("weakT",("weakT_V2_P25","weakT_V2_P40")),("ceramic",("ceramic_V2_P25","ceramic_V2_P40"))):
  for temp in TEMPS:
   x=[r for r in terminal if r["alias"] in aa and r["temperature_K"]==temp]
   pair.append({"family":family,"temperature_K":temp,"P25_terminal_K_J_Pa_sqrt_m":x[0]["terminal_K_J_Pa_sqrt_m"],"P40_terminal_K_J_Pa_sqrt_m":x[1]["terminal_K_J_Pa_sqrt_m"],"P25_extension_m":x[0]["projected_extension_m"],"P40_extension_m":x[1]["projected_extension_m"]})
 write_csv(REVIEW/"v2_2d_single_crack_P25_P40_comparison.csv",pair,list(pair[0]))
 decision="# V2 2-D single-crack 300 K / 1200 K decision\n\nCampaign classification: `V2_NAMED_PARAMETERIZATION_SINGLE_CRACK_SPATIAL_TRANSFER_TEST`.\n\nThe reported K histories are `PF_MODEL_NATIVE_KJ_DRIVING_TRAJECTORY`. Reload-separated points are `RELOAD_SEPARATED_EFFECTIVE_RESISTANCE_CANDIDATES`; they are neither ASTM toughness nor conventional R-curves.\n\n"+"\n".join(f"- `{r['case_id']}`: `{r['classification']}` ({1e6*float(r['projected_extension_m']):.6g} µm)." for r in terminal)+"\n"
 (REVIEW/"V2_2D_SINGLE_CRACK_300K_1200K_DECISION.md").write_text(decision)
 figs=sorted(p.name for p in REVIEW.glob("*.pdf"));(REVIEW/"V2_2D_SINGLE_CRACK_FIGURE_INDEX.md").write_text("# Figure index\n\n"+"\n".join(f"- `{p}`" for p in figs)+"\n")
 files=sorted(p for p in REVIEW.iterdir() if p.is_file() and p.name!="SHA256_MANIFEST.json")
 hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files};(REVIEW/"SHA256_MANIFEST.json").write_text(json.dumps(hashes,indent=2,sort_keys=True)+"\n")

if __name__=="__main__":main()
