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
  "fracture/current_fracture_decision.json","figures/current_developed_fatigue.png","figures/current_dbtt_fracture_bridge.png",
  "joint_level_iteration/joint_level_scan.csv","joint_level_iteration/joint_level_candidates.csv",
  "joint_level_iteration/joint_level_decision.json","amendment/process_surface_audit.json",
  "amendment/hazard_stage_state_comparison.csv","amendment/action_convention_reconciliation.json",
  "amendment/class_family_result_table.csv","amendment/selection_ledger.json"]
 for name in files:cp(a.run_root/name,a.out/name)
 for src,name in [(a.original_registry,"original_four_registry.csv"),(a.parent_registry,"fatigue_parent_registry.csv"),
                  (a.v2_rows,"v2_candidate_rows.csv"),(a.v2_manifest,"v2_candidate_manifest.json")]:cp(src,a.out/"inputs"/name)
 for campaign in ("p40_transfer_current_seed4810","r10_dbtt_p40_projection_seed4810",
                  "r10j_dbtt_p40_lshift_036mev_seed4810","r10j_dbtt_p40_lshift_036mev_seed9117"):
  for result in (a.run_root/"bridges"/campaign).glob("Kmax_*/result.json"):
   cp(result,a.out/"trajectories"/campaign/result.parent.name/"result.json")
 for campaign in ("original_dbtt_current_coarse","r10_dbtt_p40_projection_coarse",
                  "r10j_dbtt_p40_lshift_036mev_screen","r10j_peak_p40_lshift_036mev_screen",
                  "r10j_weakt_p40_lshift_036mev_screen"):
  for result in (a.run_root/"fracture"/campaign).glob("T_*K.json"):
   cp(result,a.out/"fracture_cases"/campaign/result.name)
 root=Path(__file__).resolve().parents[1]
 for name in ("design_r10_joint_level_iteration.py","run_r10_current_fracture_panel.py",
              "run_r10_retained_bridge.py","analyze_r10_information_batch.py",
              "analyze_r10_fracture_bridge.py","analyze_r10_amendment.py","package_r10_joint_search.py"):
  cp(root/"scripts"/name,a.out/"source"/"scripts"/name)
 cp(root/"arrhenius_fracture"/"retained_material_r10.py",a.out/"source"/"arrhenius_fracture"/"retained_material_r10.py")
 commands="""# Reproduction commands\n\nUse `/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python`.\n\n- `python scripts/design_r10_joint_level_iteration.py --help`\n- `python scripts/run_r10_retained_bridge.py --help`\n- `python scripts/run_r10_current_fracture_panel.py --help`\n- `python scripts/analyze_r10_information_batch.py --root runs/r10_joint_search`\n- `python scripts/analyze_r10_fracture_bridge.py --root runs/r10_joint_search/fracture`\n- `python scripts/analyze_r10_amendment.py --help`\n"""
 (a.out/"COMMANDS.md").write_text(commands)
 readme="""# R10 joint fracture–fatigue continuation amendment\n\nThis package preserves the original strict EXP-floor and inherited V2 family identities. The new strict DBTT descendant changes only `cleave_G00_eV` (+0.036 eV) and `cleave_floor_frac` (0.461420 to 0.487264), giving complete-state Kinit=5.218 MPa√m at 300 K. Its developed fatigue response is independently confirmed: seed 4810 gives m=3.999 over 2.574 decades (three amplitudes), and seed 9117 gives m=4.004 with all four adjacent secants 3.868–4.095 over five amplitudes.\n\nThis is not a DBTT-class promotion. The descendant is right-censored at K=80 from 450 through 1200 K. Equivalent Peak and compensated weak-T descendants pass the low-temperature scale but are likewise censored from 450 through 1200 K, so they are retained as class near misses and were not sent to fatigue or 2-D. The original projected DBTT row remains a positive fatigue frontier/control but fails the inherited low-temperature scale at Kinit=2.282. All censors include residual action and synchronized state ledgers; none are called re-entrant failures.\n"""
 (a.out/"README.md").write_text(readme)
 manifest={}
 for path in sorted(x for x in a.out.rglob("*") if x.is_file() and x.name!="MANIFEST.json"):
  manifest[str(path.relative_to(a.out))]={"sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"bytes":path.stat().st_size}
 (a.out/"MANIFEST.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"files":len(manifest),"out":str(a.out.resolve())},indent=2));return 0
if __name__=="__main__":raise SystemExit(main())
