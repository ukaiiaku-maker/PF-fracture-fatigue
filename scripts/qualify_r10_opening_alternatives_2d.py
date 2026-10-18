#!/usr/bin/env python3
"""Selected exact-material 2-D cohesive/plastic transfer for R10 alternatives."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import (BILINEAR,COMPACT_INDEPENDENT,SMOOTH_EXPONENTIAL,
    COHESIVE_ATHERMAL_MECHANICAL,COHESIVE_MIXED_MECHANICAL_THERMAL,
    BilinearCohesiveLaw,mechanical_completion,COHESIVE_KINETIC_CZM)
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative,evaluate_opening
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding,build_cohesive_current_engine
from scripts.qualify_r10_cohesive_2d import specimen,equilibrium

def atomic(path,p):
    t=path.with_suffix('.tmp');t.write_text(json.dumps(p,indent=2,sort_keys=True,default=str)+'\n');t.replace(path)

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--registry',type=Path,required=True);p.add_argument('--candidate-id',required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--selected-only',action='store_true',
        help='execute only the registry-owned shape/mode for exact-material transfer')
    a=p.parse_args(argv);a.out.mkdir(parents=True,exist_ok=True)
    with a.registry.open(newline='') as f:record=next(r for r in csv.DictReader(f) if r['candidate_id']==a.candidate_id)
    c=json.loads(record['cohesive_parameter_vector'])
    binding=load_cohesive_binding(a.registry,a.candidate_id,execution_identity=COHESIVE_KINETIC_CZM)
    thermal=binding.manifest.cleavage.thermal;rows=[]
    shapes=((record['selected_shape'],) if a.selected_only else
        (SMOOTH_EXPONENTIAL,BILINEAR,COMPACT_INDEPENDENT))
    modes=((record['selected_mode'],) if a.selected_only else
        ('cohesive_athermal','cohesive_mixed'))
    for T in (300.,750.,1200.):
      for shape in shapes:
       for mode in modes:
        mesh,network,elem=specimen(thermal);elem.temperature_K=T;elem.cohesive_shape=shape
        network.execution_identity=(COHESIVE_ATHERMAL_MECHANICAL if mode=='cohesive_athermal' else COHESIVE_MIXED_MECHANICAL_THERMAL)
        engine=build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
        alt=OpeningAlternative(mode,shape,'quasistatic_mechanical' if mode=='cohesive_athermal' else 'qualified_high_barrier_hazard',
            None if mode=='cohesive_athermal' else float(c['activation_area_m2']),
            None if mode=='cohesive_athermal' else float(c['attempt_frequency_s']))
        cutoff=1.25*float(mechanical_completion(shape,0.,thermal,T)['cutoff_m'])
        u=None;action=0.;threshold=math.log(2.);trigger=None;trace=[];kenv=None
        for i,imposed in enumerate(np.linspace(0.,cutoff,81)):
            try:u,d=equilibrium(mesh,network,float(imposed),u)
            except RuntimeError:
                trigger='FE_BRANCH_LOSS';break
            sep=max(float(d['normal_jump_m']),0.);traction=max(float(d['normal_traction_Pa']),0.)
            engine._plastic_half_step(0.5e-6,T,traction);engine._plastic_half_step(0.5e-6,T,traction)
            if imposed>sep and traction>0:kenv=traction/(imposed-sep)
            comp=mechanical_completion(shape,sep,thermal,T)
            kinetic='NOT_APPLICABLE_ATHERMAL'
            if mode=='cohesive_mixed' and kenv:
                opening=evaluate_opening(alt,float(imposed),kenv,thermal,T,{'branch':'BONDED','delta_m':sep})
                kinetic=opening['kinetic_validity']
                if opening.get('elementary_rate_s-1') is not None:action+=float(opening['elementary_rate_s-1'])*1e-6
            trace.append({'step':i,'imposed_jump_m':imposed,'separation_m':sep,'traction_Pa':traction,
                'mobile_count':float(engine.mpz.mobile_count),'retained_count':float(engine.mpz.retained_count),
                'kinetic_validity':kinetic,'action':action})
            if comp['complete'] or (mode=='cohesive_mixed' and action>=threshold):
                trigger='MECHANICAL_COMPLETION' if comp['complete'] else 'THERMAL_ESCAPE';elem.accept_mechanical_separation(sep,1);break
        if trigger is None:trigger='RAMP_LIMIT_CENSOR'
        rows.append({'temperature_K':T,'cohesive_shape':shape,'opening_mode':mode,'trigger':trigger,
            'complete_material_identity':bool(shape==record['selected_shape'] and
                mode==record['selected_mode']),
            'imposed_at_trigger_m':trace[-1]['imposed_jump_m'] if trace else None,
            'separation_at_trigger_m':trace[-1]['separation_m'] if trace else None,
            'peak_traction_Pa':max((x['traction_Pa'] for x in trace),default=math.nan),
            'mobile_final':float(engine.mpz.mobile_count),'retained_final':float(engine.mpz.retained_count),
            'actual_2d_interface_assembly':True,'EXP_floor_state_evolved':True,'trace':trace})
    summary={'schema':'r10.opening-alternatives-2d-transfer/1','candidate_id':a.candidate_id,
        'source_complete_material_sha256':record['complete_row_sha256'],
        'same_complete_material_sha256':record['complete_row_sha256'] if a.selected_only else None,
        'paired_plasticity_sha256':record['parent_complete_row_sha256'],
        'cases':rows,'legacy_cooperative_clock_used':False,
        'scope':('EXACT_SELECTED_MATERIAL_2D_THERMAL_FRACTURE_TRANSFER' if a.selected_only else
                 'MATCHED_PLASTICITY_CROSS_OPENING_2D_COMPARISON')}
    atomic(a.out/'qualification.json',summary);print(json.dumps({'candidate_id':a.candidate_id,'cases':len(rows),'triggers':[r['trigger'] for r in rows]},indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
