#!/usr/bin/env python3
"""Strict terminal verifier for the 12-case V2 spatial-transfer campaign."""
from __future__ import annotations
import hashlib,json,subprocess,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.run_state_checkpoint_v10230 import load_combined_checkpoint,validate_cross_layer
from arrhenius_fracture.v2_named_parameterizations import NAMED_ALIASES,load_for
TEMPS=(300,1200)

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 if len(sys.argv)!=2:raise SystemExit("usage: verify_v2_2d_single_crack_campaign_v10230.py CAMPAIGN_ROOT")
 campaign=Path(sys.argv[1]).resolve();review=ROOT/"analysis_outputs/v2_2d_single_crack_300K_1200K";fail=[]
 if not campaign.is_dir():fail.append("campaign root missing")
 for alias in NAMED_ALIASES:
  record=load_for(alias,"PF_sharp_front")
  for temp in TEMPS:
   cid=f"{alias}_{temp}K_theta0";case=campaign/cid
   try:
    term=json.loads((case/"terminal_record.json").read_text());bind=json.loads((case/"v2_exact_parameter_binding.json").read_text())
    if bind["candidate_id"]!=record.source_candidate_id or bind["complete_bound_row_sha256"]!=record.complete_bound_row_sha256:fail.append(f"{cid}: row binding")
    if term["classification"] not in ("V2_SINGLE_CRACK_2D_PF_REACHED_1000UM",f"V2_SINGLE_CRACK_2D_PF_STOPPED_FAIL_CLOSED_{term['terminal_reason']}"):fail.append(f"{cid}: classification")
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
 required=("v2_2d_single_crack_case_table.csv","v2_2d_single_crack_terminal_summary.csv","v2_2d_single_crack_KJ_history.csv","v2_2d_single_crack_reload_separated_candidates.csv","v2_2d_single_crack_field_inventory.csv","v2_2d_single_crack_snapshot_manifest.csv","v2_2d_single_crack_energetic_crosscheck.csv","V2_2D_SINGLE_CRACK_300K_1200K_DECISION.md","V2_2D_SINGLE_CRACK_FIGURE_INDEX.md","SHA256_MANIFEST.json")
 for name in required:
  if not (review/name).is_file():fail.append(f"review output missing {name}")
 if (review/"SHA256_MANIFEST.json").is_file():
  for name,digest in json.loads((review/"SHA256_MANIFEST.json").read_text()).items():
   if not (review/name).is_file() or sha(review/name)!=digest:fail.append(f"review hash {name}")
 for stem in ("KJ_vs_projected_extension_300K","KJ_vs_projected_extension_1200K","KJ_reload_separated_candidates_300K_1200K"):
  for ext in ("pdf","svg","png"):
   if not (review/f"{stem}.{ext}").is_file():fail.append(f"figure missing {stem}.{ext}")
 if subprocess.run([sys.executable,"-m","compileall","-q","arrhenius_fracture","scripts"],cwd=ROOT).returncode:fail.append("compileall")
 if subprocess.run(["git","diff","--check"],cwd=ROOT).returncode:fail.append("git diff --check")
 if fail:raise SystemExit("STRICT VERIFIER FAILED\n"+"\n".join(f"- {x}" for x in fail))
 print("STRICT_V2_2D_SINGLE_CRACK_CAMPAIGN_VERIFIER_PASS")
if __name__=="__main__":main()
