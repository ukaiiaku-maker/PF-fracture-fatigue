#!/usr/bin/env python3
"""Bound-triggered hardening expansion after the initial bank found no DBTT/Peak proxy."""
from __future__ import annotations
import argparse,csv,json,hashlib
from pathlib import Path
import sys
from scipy.stats import qmc
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import FAMILY
from scripts.run_r10_cohesive_arrhenius_search import generate,evaluate,write_csv
from scripts.run_r10_global_mechanism_search import HITS,TAU_S,canonical_hash,read_sources

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--original-registry',type=Path,required=True);p.add_argument('--parent-registry',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--log2-bank',type=int,default=13);a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
 sources=read_sources(a.original_registry,a.parent_registry);points=qmc.Sobol(d=5,scramble=True,seed=20162026).random_base2(a.log2_bank);items=[]
 for i,u in enumerate(points):
  item=generate(i,u,sources);item['candidate_id']=f'R10D_HARD_{i:05d}';c=item['cohesive']
  fraction=-0.25-3.75*float(u[4]);c['S_Gamma_J_m2_K']=fraction*c['Gamma_ref_J_m2']/900.;c['thermal_subfamily']='TARGETED_COHESIVE_HARDENING_EXPANSION';items.append(item)
 metrics=[evaluate(x) for x in items];write_csv(a.out/'analytical_bank.csv.gz',metrics,True)
 priority={'RISING_DBTT_PROXY':0,'PEAK_PROXY':1,'WEAK_T_PROXY':2,'OTHER_OR_CENSORED_PROXY':3}
 valid=[m for m in metrics if m.get('domain_pass')];valid.sort(key=lambda m:(priority.get(m['thermal_acquisition_stratum'],9),abs(m['A0_slope_300K_median']-4),m['candidate_id']))
 selected=valid[:24];by={x['candidate_id']:x for x in items};registry=[]
 for m in selected:
  item=by[m['candidate_id']];c=item['cohesive'];parent=item['parent'];ch=canonical_hash(c);ph=canonical_hash(parent)
  registry.append({'candidate_id':item['candidate_id'],'family':FAMILY,'complete_row_sha256':canonical_hash({'family':FAMILY,'candidate_id':item['candidate_id'],'parent_sha256':ph,'cohesive_sha256':ch}),
   'parent_candidate_id':item['parent_candidate_id'],'parent_complete_row_sha256':ph,'cohesive_parameter_sha256':ch,'cleavage_hits':HITS,'cleavage_tau_s':TAU_S,
   'mechanism_acquisition_stratum':m['mechanism_acquisition_stratum'],'thermal_acquisition_stratum':m['thermal_acquisition_stratum'],
   'cohesive_parameter_vector':json.dumps(c,sort_keys=True,separators=(',',':')),'parent_complete_parameter_vector':json.dumps(parent,sort_keys=True,separators=(',',':'))})
 write_csv(a.out/'native_batch_registry.csv',registry);write_csv(a.out/'native_batch_descriptors.csv',selected)
 counts={}
 for m in metrics:counts[m.get('thermal_acquisition_stratum','DOMAIN_REJECT')]=counts.get(m.get('thermal_acquisition_stratum','DOMAIN_REJECT'),0)+1
 decision={'schema':'r10.cohesive-hardening-bound-expansion/1','reason':'INITIAL_32768_BANK_CONTAINED_NO_DBTT_OR_PEAK_PROXY; THERMAL_HARDENING_BOUND_WAS_ACTIVE',
  'family':FAMILY,'allocated':len(items),'analytically_evaluated':len(metrics),'domain_valid':len(valid),'native_queued_not_executed':len(registry),'thermal_counts':counts,
  'expanded_prior':{'Gamma_1200_over_Gamma_300':[1.25,5.0],'temperature_endpoints_independently_fitted':False,'single_S_Gamma_coordinate':True},'files':{}}
 for name in ('analytical_bank.csv.gz','native_batch_registry.csv','native_batch_descriptors.csv'):
  path=a.out/name;decision['files'][name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
 (a.out/'decision.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n');print(json.dumps(decision,indent=2,sort_keys=True));return 0
if __name__=='__main__':raise SystemExit(main())
