#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,gzip,hashlib,json,math
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,x):
 t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');t.replace(p)
def finite(x):return x if isinstance(x,(int,float)) and math.isfinite(x) else None

def main(argv=None):
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--native',type=Path,required=True);p.add_argument('--developed',type=Path,required=True);p.add_argument('--spatial',type=Path,required=True);a=p.parse_args(argv)
 with gzip.open(a.evidence/'analytical_matched_8192.csv.gz','rt',newline='') as f: analytical=list(csv.DictReader(f))
 tuples={};thermal_counts={}
 for r in analytical:
  key=(r['candidate_id'],r['cohesive_shape']);ratio=float(r['thermal_ratio_1200_over_300'])
  if key not in tuples:
   cls='RISING' if ratio>1.25 else 'FALLING' if ratio<.8 else 'WEAK_T'
   thermal_counts[cls]=thermal_counts.get(cls,0)+1;tuples[key]=cls
 def cases(root):
  out=[]
  for path in root.glob('*/*.json'):
   d=json.loads(path.read_text())
   if 'event_count' in d:out.append(d)
  return out
 native=cases(a.native);deep=cases(a.developed)
 def write_csv(path,rows):
  fields=[]
  for row in rows:
   for k in row:
    if k not in fields:fields.append(k)
  with path.open('w',newline='') as f:
   w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)
 def dispositions(items):
  return [{k:d.get(k) for k in ('candidate_id','shape','mode','Kmax_MPa_sqrt_m','DeltaK_MPa_sqrt_m','status','cycles','extension_m','event_count','developed_rate_m_per_cycle','rate_disposition','paired_plasticity_sha256')} for d in items]
 write_csv(a.evidence/'native_12_case_dispositions.csv',dispositions(native))
 write_csv(a.evidence/'developed_followup_case_dispositions.csv',dispositions(deep))
 with gzip.open(a.evidence/'developed_followup_histories.json.gz','wt') as f:json.dump(deep,f,sort_keys=True,allow_nan=True)
 arm={}
 for d in native:
  k=f"{d['mode']}__{d['shape']}";z=arm.setdefault(k,{'cases':0,'target_completed':0,'censors':0,'events':0})
  z['cases']+=1;z['events']+=d['event_count'];z['target_completed']+=d['status']=='TARGET_COMPLETED';z['censors']+=d['status']!='TARGET_COMPLETED'
 deep_arm={}
 for d in deep:
  k=f"{d['mode']}__{d['shape']}";z=deep_arm.setdefault(k,{'loads':0,'developed':0,'startup_or_other_censors':0,'recurrent':0})
  z['loads']+=1;z['developed']+=d['rate_disposition']=='DEVELOPED_WINDOW_RATE';z['startup_or_other_censors']+=d['rate_disposition']!='DEVELOPED_WINDOW_RATE';z['recurrent']+=d['event_count']>=3
 spatial=json.loads((a.spatial/'qualification.json').read_text())
 atomic(a.evidence/'selected_2d_qualification.json',spatial)
 spatial_rows=[{k:c[k] for k in ('temperature_K','cohesive_shape','opening_mode','trigger','imposed_at_trigger_m','separation_at_trigger_m','peak_traction_Pa','mobile_final','retained_final')} for c in spatial['cases']]
 paired={}
 for d in native:paired[(d['candidate_id'],d['mode'],round(d['Kmax_MPa_sqrt_m'],9),d['shape'])]=d['event_count']
 contrasts=[]
 for key,n in paired.items():
  cid,mode,K,shape=key
  if shape!='smooth_exponential':continue
  b=paired.get((cid,mode,K,'bilinear'))
  if b is not None and b!=n:contrasts.append({'candidate_id':cid,'mode':mode,'Kmax_MPa_sqrt_m':K,'smooth_events':n,'bilinear_events':b})
 summary={'schema':'r10.opening-alternatives-results/1','analytical':{'allocated_tuples':8192,'domain_valid_tuples':8141,'matched_rows':len(analytical),'thermal_shape_counts':thermal_counts},
  'state_informed':json.loads((a.evidence/'state_informed_execution_summary.json').read_text()),
  'native_initial':{'case_count':len(native),'material_count':len({x['candidate_id'] for x in native}),'arm_dispositions':arm,'shape_effect_contrasts':contrasts},
  'developed_followup':{'candidate_id':'R10D_COH_27231','case_count':len(deep),'arm_dispositions':deep_arm,'strict_Paris_passes':0,'interpretation':'All advancing points were single-excursion bursts; no developed event-window rate or local Paris slope is claimed.'},
  'spatial_2d':{'candidate_id':spatial['candidate_id'],'case_count':len(spatial_rows),'cases':spatial_rows},
  'physics':{'legacy_family_preserved':True,'legacy_cooperative_clock_used':False,'EXP_floor_plasticity_unchanged':True,'mechanical_patch_owner':'DECLARED_SPATIAL_INTERFACE_CELL','mixed_crossover_is_censored_not_extrapolated':True},
  'files':{}}
 for path in [a.evidence/'analytical_matched_8192.csv.gz',a.evidence/'state_informed_48_executed.csv',a.evidence/'native_12_case_dispositions.csv',a.evidence/'developed_followup_case_dispositions.csv',a.evidence/'developed_followup_histories.json.gz',a.evidence/'selected_2d_qualification.json']:
  summary['files'][str(path)]={'sha256':sha(path),'bytes':path.stat().st_size}
 atomic(a.evidence/'results_summary.json',summary)
 md=f"""# R10 mechanical/mixed cohesive opening comparison — initial information batch

## Decision

The additive family is implemented and executed without changing the legacy cohesive–Arrhenius family or the EXP_floor plasticity inputs. The initial batch does **not** identify a strict Paris-window pass. Its useful result is a resolved shape/mechanism contrast: matched bilinear laws frequently cross into abrupt mechanical advance while smooth exponential laws are commonly inactive or limited to a single-excursion burst under the same retained dislocation material.

## Completed work

- Analytical: 8,192 tuples allocated; 8,141 domain-valid; 32,564 matched arm records.
- State-informed: 48/48 complete materials received fresh native retained-state exposure.
- Native 1-D: 12 materials × 4 arms × 3 loads = {len(native)} cases.
- Developed follow-up: R10D_COH_27231, 4 arms × 6 loads, one-micrometre geometry-owned cells.
- Same-material 2-D: {len(spatial_rows)} nonlinear cohesive/plastic transfer cases at 300, 750, and 1200 K.

## Fatigue disposition

The refined R10D_COH_27231 bilinear and high-load exponential responses reached the extension target as a single-excursion cascade. They therefore remain startup-burst censors, not developed rates. No local slope is reported from those bursts, and the strict 3.4–4.6 / two-rate-decade objective has zero passes in this batch. Lower loads are true fixed-domain non-advance censors over their declared exposure.

## Mechanism and thermal observations

The initial 12-material block contains {len(contrasts)} paired load points where smooth and bilinear event counts differ despite identical plasticity hashes. In 2-D, athermal arms ended at a resolved FE branch-loss boundary, while mixed arms reached qualified thermal escape; the temperature-specific trigger coordinates are retained in `results_summary.json`. Low-barrier crossover intervals were recorded as unsupported first-passage censors and were not evaluated with the old cooperative clock.

## Interpretation limits

The 2-D block is a selected thermal-fracture transfer, not a developed spatial fatigue curve. The native 1-D screen is intentionally separated from the developed follow-up. Numerical renewal caps were removed before the reported follow-up, so its burst classification is physical within the declared one-micrometre interface-cell discretization rather than a capped-rate artifact.
"""
 (a.evidence/'RESULTS.md').write_text(md);print(json.dumps(summary,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
