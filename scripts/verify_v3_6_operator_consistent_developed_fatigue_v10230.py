#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,required=True); a=p.parse_args(); r=a.root
 required=['FINAL_REPORT.md','audit_reproduction.json','v3_5_spatial_deformation_reconstruction.csv','old_new_observation_operator_audit.csv','fixed_support_mesh_refinement.csv','developed_events.csv','developed_rate_windows.csv','developed_slope_windows.csv','event_seed_uncertainty.csv','high_load_phase_refinement.csv','high_load_refined_event.csv','exact_bin_shielding.csv','exact_bin_shielding_decision.json','monotonic_anchors.csv','source_provenance.csv','mechanical_contract.json','v3_5_shielding_same_operator_refinement.csv']
 assert all((r/x).is_file() for x in required)
 audit=json.loads((r/'audit_reproduction.json').read_text()); assert audit['all_portable_members_match_exported_bytes']; expected=[7.4523658e-13,.0071600894,.0118813839]
 assert np.allclose([x['reconstructed_fixed_patch_max_engineering_eigenstrain'] for x in audit['reproduced_values']],expected,rtol=2e-8)
 events=pd.read_csv(r/'developed_events.csv'); assert len(events)==18; assert set(events.groupby('Kmax_MPa_sqrt_m').size())=={6}; assert np.max(abs(events.coordinate_motion_plastic_increment))<1e-20
 assert events.groupby('Kmax_MPa_sqrt_m').cycles.max().max()<1e6; assert abs(events.sort_values(['Kmax_MPa_sqrt_m','event_index']).iloc[0].cycles-3644.3807885)<1e-5
 win=pd.read_csv(r/'developed_rate_windows.csv'); assert set(win.window)=={'early_nonoverlap','late_nonoverlap','overall'}; slopes=pd.read_csv(r/'developed_slope_windows.csv'); assert slopes.loglog_slope.between(4.04,4.08).all(); assert not slopes.hard_equality_m4_imposed.any()
 high=pd.read_csv(r/'high_load_phase_refinement.csv').sort_values('phase_steps'); assert high.iloc[0].status=='operator_consistent_kinematic_domain_exit'; assert (high.iloc[1:].status=='target_events_completed').all(); assert abs(high.iloc[-1].final_cycles-high.iloc[-2].final_cycles)<2e-4
 shield=json.loads((r/'exact_bin_shielding_decision.json').read_text()); assert abs(shield['terminal_exact_total_MPa_sqrt_m']-79.8600209369)<1e-8; assert not shield['physical_contact_solved']
 anchors=pd.read_csv(r/'monotonic_anchors.csv')
 for (_,f),g in anchors.groupby(['temperature_K','frequency_Hz']): assert np.all(np.diff(g.sort_values('Kmax_MPa_sqrt_m').virgin_action_per_cycle)>0)
 for (f,K),g in anchors.groupby(['frequency_Hz','Kmax_MPa_sqrt_m']): assert np.all(np.diff(g.sort_values('temperature_K').virgin_action_per_cycle)>0)
 checkpoints=list((r/'developed_checkpoints').glob('*.json')); assert len(checkpoints)==18
 for q in checkpoints:
  d=json.loads(q.read_text()); assert d['schema']=='v10.2.30_v3_6_material_frame_checkpoint_v1'; assert 'rng_state' in d and 'pending_threshold' in d and 'active_ledger' in d
 provenance=pd.read_csv(r/'source_provenance.csv'); assert all(sha(Path(x.path))==x.sha256 for _,x in provenance.iterrows())
 manifest={str(x.relative_to(r)):sha(x) for x in sorted(r.rglob('*')) if x.is_file() and x.name!='SHA256_MANIFEST.json'}
 (r/'SHA256_MANIFEST.json').write_text(json.dumps({'schema':'v10.2.30_v3_6_sha256_manifest_v1','files':manifest},indent=2,sort_keys=True)+'\n')
 print(json.dumps({'status':'PASS','files':len(manifest),'events':len(events),'checkpoints':len(checkpoints),'overall_slope':float(slopes.query("window=='overall'").loglog_slope.iloc[0])},sort_keys=True)); return 0
if __name__=='__main__': raise SystemExit(main())
