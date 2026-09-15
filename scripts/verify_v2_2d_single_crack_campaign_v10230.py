#!/usr/bin/env python3
"""Strict terminal verifier for the 12-case V2 spatial-transfer campaign."""
from __future__ import annotations
import csv,hashlib,json,subprocess,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.run_state_checkpoint_v10230 import load_combined_checkpoint,validate_cross_layer
from arrhenius_fracture.v2_named_parameterizations import NAMED_ALIASES,load_for
TEMPS=(300,1200)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(path):
 with path.open(newline="") as f:return list(csv.DictReader(f))
def main():
 if len(sys.argv)!=2:raise SystemExit("usage: verify_v2_2d_single_crack_campaign_v10230.py CAMPAIGN_ROOT")
 campaign=Path(sys.argv[1]).resolve();review=ROOT/"analysis_outputs/v2_2d_single_crack_300K_1200K";fail=[]
 if not campaign.is_dir():fail.append("campaign root missing")
 for alias in NAMED_ALIASES:
  record=load_for(alias,"PF_sharp_front")
  for temp in TEMPS:
   cid=f"{alias}_{temp}K_theta0";case=campaign/cid
   try:
    term=json.loads((case/"terminal_record.json").read_text());bind=json.loads((case/"v2_exact_parameter_binding.json").read_text());worker=json.loads((case/"worker_status.json").read_text())
    if bind["candidate_id"]!=record.source_candidate_id or bind["complete_bound_row_sha256"]!=record.complete_bound_row_sha256:fail.append(f"{cid}: row binding")
    if term["classification"] not in ("V2_SINGLE_CRACK_2D_PF_REACHED_1000UM",f"V2_SINGLE_CRACK_2D_PF_STOPPED_FAIL_CLOSED_{term['terminal_reason']}"):fail.append(f"{cid}: classification")
    if worker.get("state")!="TERMINAL":fail.append(f"{cid}: worker is not terminal")
    outer,kinetic,arrays=load_combined_checkpoint(case);validate_cross_layer(outer,kinetic)
    if int(outer["geometry"]["committed_event_count"])<0 or len(outer["geometry"]["front_paths"])!=1:fail.append(f"{cid}: multifront/invalid events")
    meta=json.loads((case/"portable_fields/terminal.json").read_text())
    with np.load(case/"portable_fields/terminal.npz") as z:
     for name in ("mesh_nodes","mesh_elems","damage","displacement","rho_gp","ep_gp","sigma_gp"):
      if name not in z or name not in arrays or not np.array_equal(z[name],arrays[name]):fail.append(f"{cid}: terminal {name} differs from checkpoint")
    for name,digest in meta["files"].items():
     if sha(case/"portable_fields"/name)!=digest:fail.append(f"{cid}: portable hash {name}")
    reached=float(term["projected_extension_m"])>=1000e-6-1e-12
    if bool(term["target_reached"])!=reached:fail.append(f"{cid}: inaccurate target label")
   except Exception as exc:fail.append(f"{cid}: {type(exc).__name__}: {exc}")
 required=("V2_2D_SINGLE_CRACK_CAMPAIGN_PROTOCOL.md","v2_2d_single_crack_campaign_manifest.json","immutable_launch_manifest.json","v2_2d_single_crack_case_table.csv","v2_2d_single_crack_terminal_summary.csv","v2_2d_single_crack_KJ_history.csv","v2_2d_single_crack_reload_separated_candidates.csv","v2_2d_single_crack_field_inventory.csv","v2_2d_single_crack_snapshot_manifest.csv","v2_2d_single_crack_energetic_crosscheck.csv","V2_2D_SINGLE_CRACK_300K_1200K_DECISION.md","V2_2D_SINGLE_CRACK_FIGURE_INDEX.md","COMPACT_REVIEW_ARCHIVE.json","SHA256_MANIFEST.json")
 for name in required:
  if not (review/name).is_file():fail.append(f"review output missing {name}")
 if (review/"SHA256_MANIFEST.json").is_file():
  for name,digest in json.loads((review/"SHA256_MANIFEST.json").read_text()).items():
   if not (review/name).is_file() or sha(review/name)!=digest:fail.append(f"review hash {name}")
 figure_stems=("KJ_vs_projected_extension_300K","KJ_vs_projected_extension_1200K","KJ_reload_separated_candidates_300K_1200K","final_crack_damage_field","final_total_rho_field","final_mobile_rho_field","final_retained_rho_field","final_maximum_principal_stress","final_von_mises_stress","final_hydrostatic_stress","final_displacement_magnitude","final_equivalent_plastic_strain","final_process_zone_line_profiles","final_effective_radius_backstress_shielding_comparison")
 for stem in figure_stems:
  for ext in ("pdf","svg","png"):
   if not (review/f"{stem}.{ext}").is_file():fail.append(f"figure missing {stem}.{ext}")
  if not (review/f"{stem}_source.json").is_file() and not (review/f"{stem}_source.csv").is_file():fail.append(f"figure source missing {stem}")
 try:
  terminal_rows=rows(review/"v2_2d_single_crack_terminal_summary.csv")
  history_rows=rows(review/"v2_2d_single_crack_KJ_history.csv")
  candidate_rows=rows(review/"v2_2d_single_crack_reload_separated_candidates.csv")
  if len(terminal_rows)!=12:fail.append(f"terminal table row count {len(terminal_rows)}")
  required_terminal={"candidate_id","row_sha256","seed","target_reached","terminal_reason","accepted_steps","first_event_K_J_Pa_sqrt_m","first_reload_separated_K_J_Pa_sqrt_m","maximum_K_J_Pa_sqrt_m","terminal_K_J_Pa_sqrt_m","projected_extension_m","final_path_extension_m","number_crack_events","number_physical_avalanches","final_effective_radius_m","final_mobile_process_zone_count","final_retained_process_zone_count","final_backstress_Pa","final_active_shielding_Pa_sqrt_m","domain_K_J_valid","applied_specimen_K_status","global_G_or_VCCT_status"}
  if terminal_rows and not required_terminal.issubset(terminal_rows[0]):fail.append("terminal table columns")
  grouped={f"{a}_{t}K_theta0":[] for a in NAMED_ALIASES for t in TEMPS}
  for row in history_rows:
   grouped.get(row["case_id"],[]).append(row)
   for key in ("signed_domain_J_J_per_m2","J_energy_J_per_m2","K_J_Pa_sqrt_m","path_extension_m","hazard_action","emission_count"):
    if not np.isfinite(float(row[key])):fail.append(f"{row['case_id']}: nonfinite history {key}")
   if row["trajectory_name"]!="PF_MODEL_NATIVE_KJ_DRIVING_TRAJECTORY":fail.append(f"{row['case_id']}: trajectory name")
  for cid,reported in grouped.items():
   case=campaign/cid;temp=int(cid.rsplit("_",2)[1][:-1]);raw=np.atleast_1d(np.genfromtxt(case/f"steps_{temp:04d}K.csv",delimiter=",",names=True))
   if len(reported)!=len(raw):fail.append(f"{cid}: history row count")
   for rr,xx in zip(reported,raw):
    if int(rr["accepted_step"])!=int(xx["step"]):fail.append(f"{cid}: history step")
    for rk,xk in (("signed_domain_J_J_per_m2","J_signed_direct_J_per_m2"),("J_energy_J_per_m2","J_effective_direct_J_per_m2"),("K_J_Pa_sqrt_m","KJ_Pa_sqrtm")):
     if not np.isclose(float(rr[rk]),float(xx[xk]),rtol=1e-12,atol=1e-14):fail.append(f"{cid}: history differs from raw {rk}")
  for row in candidate_rows:
   if row["candidate_name"]!="RELOAD_SEPARATED_EFFECTIVE_RESISTANCE_CANDIDATES":fail.append(f"{row['case_id']}: candidate name")
 except Exception as exc:fail.append(f"review table validation: {type(exc).__name__}: {exc}")
 try:
  record=json.loads((review/"COMPACT_REVIEW_ARCHIVE.json").read_text());archive=Path(record["archive_path"])
  if not archive.is_file() or sha(archive)!=record["archive_sha256"]:fail.append("compact archive hash")
  else:
   with zipfile.ZipFile(archive) as zf:
    internal=json.loads(zf.read("ARCHIVE_CONTENT_MANIFEST.json"))
    for name,digest in internal["files"].items():
     if hashlib.sha256(zf.read(name)).hexdigest()!=digest:fail.append(f"compact archive member {name}")
 except Exception as exc:fail.append(f"compact archive validation: {type(exc).__name__}: {exc}")
 if subprocess.run([sys.executable,"-m","compileall","-q","arrhenius_fracture","scripts"],cwd=ROOT).returncode:fail.append("compileall")
 if subprocess.run(["git","diff","--check"],cwd=ROOT).returncode:fail.append("git diff --check")
 if fail:raise SystemExit("STRICT VERIFIER FAILED\n"+"\n".join(f"- {x}" for x in fail))
 print("STRICT_V2_2D_SINGLE_CRACK_CAMPAIGN_VERIFIER_PASS")
if __name__=="__main__":main()
