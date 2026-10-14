#!/usr/bin/env python3
"""Create a compact, source-complete R10 continuation evidence package."""
from __future__ import annotations
import argparse,hashlib,json,shutil
from pathlib import Path

def cp(src,dst):dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-root",type=Path,required=True);p.add_argument("--out",type=Path,required=True)
 p.add_argument("--original-registry",type=Path,required=True);p.add_argument("--parent-registry",type=Path,required=True)
 p.add_argument("--v2-rows",type=Path,required=True);p.add_argument("--v2-manifest",type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
 files=["retained_material_comparison.csv","current_bridge_case_table.csv","current_bridge_local_slopes.csv",
  "current_bridge_windows.csv","information_batch_decision.json","continuation_state.json",
  "reused_sealed_2d_terminal_summary.csv","projection_frontier/strict_projection_frontier.csv",
  "projection_frontier/strict_projection_frontier_decision.json","fracture/current_fracture_case_table.csv",
  "fracture/current_fracture_decision.json","figures/current_developed_fatigue.png","figures/current_dbtt_fracture_bridge.png"]
 for name in files:cp(a.run_root/name,a.out/name)
 for src,name in [(a.original_registry,"original_four_registry.csv"),(a.parent_registry,"fatigue_parent_registry.csv"),
                  (a.v2_rows,"v2_candidate_rows.csv"),(a.v2_manifest,"v2_candidate_manifest.json")]:cp(src,a.out/"inputs"/name)
 for campaign in ("p40_transfer_current_seed4810","r10_dbtt_p40_projection_seed4810"):
  for result in (a.run_root/"bridges"/campaign).glob("Kmax_*/result.json"):
   cp(result,a.out/"trajectories"/campaign/result.parent.name/"result.json")
 for campaign in ("original_dbtt_current_coarse","r10_dbtt_p40_projection_coarse"):
  for result in (a.run_root/"fracture"/campaign).glob("T_*K.json"):
   cp(result,a.out/"fracture_cases"/campaign/result.name)
 commands="""# Reproduction commands\n\nUse `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`.\n\n- `python scripts/design_r10_strict_projection_frontier.py --help`\n- `python scripts/run_r10_retained_bridge.py --help`\n- `python scripts/run_r10_current_fracture_panel.py --help`\n- `python scripts/analyze_r10_information_batch.py --root runs/r10_joint_search`\n- `python scripts/analyze_r10_fracture_bridge.py --root runs/r10_joint_search/fracture`\n"""
 (a.out/"COMMANDS.md").write_text(commands)
 readme="""# R10 joint fracture–fatigue continuation\n\nThis package preserves the original strict EXP-floor family and inherited V2 identities. The restored current backend produced a strict-family DBTT-parent projection with developed fatigue slopes 4.081 and 3.962 over 2.414 decades, but the row is **not promoted**: its 300 K monotonic initiation fell to 2.282 MPa√m (the inherited 5–30 gate fails) and its projection is bound-active. The original DBTT current-backend control initiated at 23.271, 22.632, and 32.299 MPa√m at 300/750/1200 K.\n\nThe retained P40 row transferred with global slope 3.705 but only 0.340 decades. Exact inherited V2 2-D results are reused only at their sealed 300/1200 K monotonic scope; no new 2-D run was justified for the rejected projection. Ceramic remains a control. Peak and weak-T remain unresolved, and no joint success is claimed.\n"""
 (a.out/"README.md").write_text(readme)
 manifest={}
 for path in sorted(x for x in a.out.rglob("*") if x.is_file() and x.name!="MANIFEST.json"):
  manifest[str(path.relative_to(a.out))]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
 (a.out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"files":len(manifest),"out":str(a.out.resolve())},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
