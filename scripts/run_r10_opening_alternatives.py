#!/usr/bin/env python3
"""Matched analytical and retained-state campaign for R10 opening alternatives."""
from __future__ import annotations

import argparse,csv,gzip,hashlib,json,math
from pathlib import Path
import sys
import numpy as np
from scipy.special import lambertw
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from arrhenius_fracture.cohesive_arrhenius import (
    BILINEAR,SMOOTH_EXPONENTIAL,COHESIVE_KINETIC_CZM,
    CohesiveThermalLaw,OPENING_ALTERNATIVES_FAMILY,
)
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.fatigue_v1 import FatigueWaveform
from arrhenius_fracture.opening_alternatives_r10 import OpeningAlternative,evaluate_opening
from arrhenius_fracture.retained_material_cohesive_r10 import load_cohesive_binding,build_cohesive_current_engine
from scripts.run_r10_cohesive_arrhenius_search import SEED,generate,gamma_T
from scripts.run_r10_global_mechanism_search import read_sources,canonical_hash

SCHEMA="r10.mechanical-mixed-opening-comparison/1"
TEMPERATURES=(300.,450.,600.,750.,900.,1050.,1200.)
EPRIME=400e9/(1-.22**2)
SHAPES=(SMOOTH_EXPONENTIAL,BILINEAR)
MODES=("cohesive_athermal","cohesive_mixed")

def atomic(path,payload):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,default=str)+'\n');tmp.replace(path)

def write_csv(path,rows,compressed=False):
    fields=[]
    for row in rows:
        for k in row:
            if k not in fields:fields.append(k)
    opener=gzip.open if compressed else open
    with opener(path,'wt',newline='') as f:
        w=csv.DictWriter(f,fields,lineterminator='\n');w.writeheader();w.writerows(rows)

def thermal_from(c):
    return CohesiveThermalLaw(float(c['Gamma_ref_J_m2']),float(c['t_hat_ref_Pa']),
                              float(c['S_Gamma_J_m2_K']),float(c.get('T_ref_K',300.)),
                              float(c.get('compact_b',2.0)),float(c.get('compact_c',0.0)),
                              str(c.get('compact_variant','beta_polynomial')),
                              float(c.get('compact_rise_fraction',.1)),
                              float(c.get('compact_fall_fraction',.85)),
                              float(c.get('compact_plateau_drop',.05)))

def mechanics(thermal,T=300.):
    gamma=thermal.gamma(T);peak=thermal.t_hat(T);zone=EPRIME*gamma/peak**2
    return zone,EPRIME/(2*zone)

def critical_K(thermal,T,shape,radius=1e-6):
    zone,kenv=mechanics(thermal,T);peak=thermal.t_hat(T)
    if shape==SMOOTH_EXPONENTIAL:
        kappa=kenv*thermal.delta0_m**2/thermal.gamma(T)
        if kappa>=math.exp(-2):
            return math.nan
        x=1-float(lambertw(-math.e*kappa,-1).real)
        imposed=x*x/(x-1)*thermal.delta0_m
    else:
        dp=thermal.gamma(T)/(math.e**2*peak)
        imposed=dp+peak/kenv
    free=kenv*imposed
    return free*math.sqrt(2*math.pi*radius)/1e6

def analytical(args):
    args.out.mkdir(parents=True,exist_ok=True)
    sources=read_sources(args.original_registry,args.parent_registry)
    points=qmc.Sobol(d=5,scramble=True,seed=SEED).random_base2(13)
    rows=[];items=[]
    for i,u in enumerate(points):
        item=generate(i,u,sources);c=item['cohesive'];thermal=thermal_from(c)
        if min(gamma_T(c,T) for T in TEMPERATURES)<=.2:continue
        items.append(item)
        for shape in SHAPES:
            roots=[critical_K(thermal,T,shape) for T in TEMPERATURES]
            for mode in MODES:
                rows.append({'candidate_id':item['candidate_id'],'parent_candidate_id':item['parent_candidate_id'],
                    'family':OPENING_ALTERNATIVES_FAMILY,'opening_mode':mode,'cohesive_shape':shape,
                    'dynamics':'quasistatic_mechanical' if mode=='cohesive_athermal' else 'qualified_high_barrier_hazard',
                    'Gamma_ref_J_m2':c['Gamma_ref_J_m2'],'t_hat_ref_Pa':c['t_hat_ref_Pa'],
                    'S_Gamma_J_m2_K':c['S_Gamma_J_m2_K'],'activation_area_m2':c['activation_area_m2'],
                    'delta0_m':c['delta0_m'],'cohesive_zone_scale_m':mechanics(thermal)[0],
                    **{f'Kcritical_{int(T)}K':r for T,r in zip(TEMPERATURES,roots)},
                    'thermal_ratio_1200_over_300':roots[-1]/roots[0],
                    'matched_work_peak_stiffness':True,'legacy_cooperative_clock_used':False})
    write_csv(args.out/'analytical_matched_8192.csv.gz',rows,True)
    # Reuse the completed 48-row material allocation, changing only additive family identity.
    with args.source_registry.open(newline='') as f: source=list(csv.DictReader(f))
    registry=[]
    for r in source:
        row=dict(r);row['legacy_family']=row['family'];row['additive_family']=OPENING_ALTERNATIVES_FAMILY
        row['paired_plasticity_sha256']=row['parent_complete_row_sha256']
        row['opening_alternative_arms']='A_EXP,M_EXP,A_BILIN,M_BILIN'
        row['legacy_complete_row_sha256']=row['complete_row_sha256']
        row['opening_alternative_complete_row_sha256']=canonical_hash({'legacy':row['legacy_complete_row_sha256'],
            'family':OPENING_ALTERNATIVES_FAMILY,'arms':row['opening_alternative_arms']})
        registry.append(row)
    write_csv(args.out/'state_informed_48_registry.csv',registry)
    # Mechanism/thermal round-robin, twelve exact complete materials.
    chosen=[]
    for mech in ('OPENING_DIRECTED','PLASTICITY_DIRECTED'):
        pool=[r for r in registry if r.get('mechanism_acquisition_stratum')==mech]
        pool.sort(key=lambda r:float(json.loads(r['cohesive_parameter_vector'])['t_hat_ref_Pa']))
        for thermal_class in ('WEAK_T_PROXY','OTHER_OR_CENSORED_PROXY'):
            hits=[r for r in pool if r.get('thermal_acquisition_stratum')==thermal_class]
            chosen.extend(hits[:3])
    for r in registry:
        if len(chosen)>=12:break
        if r not in chosen:chosen.append(r)
    write_csv(args.out/'native_12_registry.csv',chosen[:12])
    payload={'schema':SCHEMA,'stage':'ANALYTICAL_COMPLETE','allocated_tuples':8192,
        'domain_valid_tuples':len(items),'matched_arm_rows':len(rows),
        'state_informed_allocated':len(registry),'native_selected':len(chosen[:12]),
        'family':OPENING_ALTERNATIVES_FAMILY,'legacy_family_preserved':True,
        'legacy_clock_used_by_new_arms':False,
        'source_registry_sha256':hashlib.sha256(args.source_registry.read_bytes()).hexdigest()}
    atomic(args.out/'analytical_summary.json',payload);print(json.dumps(payload,indent=2));return 0

def reuse_state(args):
    args.out.mkdir(parents=True,exist_ok=True)
    with args.registry.open(newline='') as f:allocated=list(csv.DictReader(f))
    prior=[]
    for path in args.prior:
        with path.open(newline='') as f:prior.extend(csv.DictReader(f))
    by_id={r['candidate_id']:r for r in prior};rows=[]
    for r in allocated:
        hit=by_id.get(r['candidate_id']) or by_id.get(r['candidate_id']+'_J')
        rows.append({'candidate_id':r['candidate_id'],'paired_plasticity_sha256':r['parent_complete_row_sha256'],
            'prior_native_state_found':hit is not None,'prior_cycles_per_load':None if hit is None else hit.get('cycles_per_load'),
            'prior_mechanism_disposition':None if hit is None else hit.get('mechanism_screen_disposition'),
            'reuse_scope':'PRE_OPENING_EVENT_EXP_FLOOR_STATE_ACQUISITION_ONLY',
            'opening_alternative_response_reused':False})
    write_csv(args.out/'state_informed_48_reuse.csv',rows)
    payload={'schema':SCHEMA,'stage':'STATE_INFORMED_REUSE_COMPLETE','allocated':len(rows),
        'native_prior_found':sum(str(r['prior_native_state_found'])=='True' for r in rows),
        'scope':'Plastic-state acquisition is unchanged and reusable before any opening event; all new arm responses execute independently.',
        'prior_files':[str(p) for p in args.prior]}
    atomic(args.out/'state_informed_summary.json',payload);print(json.dumps(payload,indent=2));return 0

def screen_state(args):
    args.out.mkdir(parents=True,exist_ok=True)
    global binding_rows
    with args.registry.open(newline='') as f:binding_rows=list(csv.DictReader(f))
    rows=[]
    for i,r in enumerate(binding_rows):
        binding=load_cohesive_binding(args.registry,r['candidate_id'],execution_identity=COHESIVE_KINETIC_CZM)
        engine=build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
        c=json.loads(r['cohesive_parameter_vector']);thermal=thermal_from(c)
        K=critical_K(thermal,300.,SMOOTH_EXPONENTIAL)*1.05
        wave=FatigueWaveform(Kmax=K*1e6,R=.1,frequency_Hz=1000.)
        phi=2*math.pi*(np.arange(args.phases)+.5)/args.phases;Ks=wave.K_phase(phi);dt=wave.period_s/args.phases
        initial=(float(engine.mpz.mobile_count),float(engine.mpz.retained_count),float(engine.K_shield()))
        for _ in range(args.cycles):
            for Kv in Ks:
                sig=max(float(engine.sigma_tip(max(float(Kv),0.))),0.)
                engine._plastic_half_step(dt,300.,sig);engine.t=float(getattr(engine,'t',0.))+dt
        final=(float(engine.mpz.mobile_count),float(engine.mpz.retained_count),float(engine.K_shield()))
        rows.append({'candidate_id':r['candidate_id'],'paired_plasticity_sha256':r['parent_complete_row_sha256'],
            'Kmax_MPa_sqrt_m':K,'cycles':args.cycles,'phases':args.phases,
            'mobile_initial':initial[0],'mobile_final':final[0],'retained_initial':initial[1],
            'retained_final':final[1],'K_shield_initial':initial[2],'K_shield_final':final[2],
            'state_exposure_executed':True})
        print(json.dumps({'state_screen':i+1,'total':len(binding_rows),'candidate':r['candidate_id']}),flush=True)
    write_csv(args.out/'state_informed_48_executed.csv',rows)
    atomic(args.out/'state_informed_execution_summary.json',{'schema':SCHEMA,'stage':'STATE_INFORMED_EXECUTION_COMPLETE',
        'materials_executed':len(rows),'cycles':args.cycles,'phases':args.phases,
        'opening_arms_not_collapsed_into_state_screen':True});return 0

def arm(shape,mode,c):
    return OpeningAlternative(mode,shape,
        'quasistatic_mechanical' if mode=='cohesive_athermal' else 'qualified_high_barrier_hazard',
        None if mode=='cohesive_athermal' else float(c['activation_area_m2']),
        None if mode=='cohesive_athermal' else float(c.get('attempt_frequency_s',1e12)))

def commit_patch(engine,length):
    wake=engine.mpz.advance(length);engine.a_adv=float(getattr(engine,'a_adv',0.))+length
    engine.n_adv=int(getattr(engine,'n_adv',0))+1
    engine.checkpoint_advance_total_m=float(getattr(engine,'checkpoint_advance_total_m',0.))+length
    return wake

def run_case(binding,shape,mode,Kmax,cycles,phases,patch_m,seed):
    engine=build_cohesive_current_engine(binding,kernel_family_path=R4_KERNEL_FAMILY)
    c=json.loads(next(r['cohesive_parameter_vector'] for r in binding_rows
                      if r['candidate_id']==binding.provenance['candidate_id']))
    thermal=thermal_from(c);alternative=arm(shape,mode,c);_,kenv=mechanics(thermal)
    wave=FatigueWaveform(Kmax=Kmax*1e6,R=.1,frequency_Hz=1000.)
    phi=2*math.pi*(np.arange(phases)+.5)/phases;Ks=np.asarray(wave.K_phase(phi),float);dt=wave.period_s/phases
    opening_phases=max(64,phases);ophi=2*math.pi*(np.arange(opening_phases)+.5)/opening_phases
    opening_Ks=np.asarray(wave.K_phase(ophi),float);opening_dt=wave.period_s/opening_phases
    rng=np.random.default_rng(seed);threshold=float(rng.exponential());action=0.;state={'branch':'BONDED','delta_m':0.}
    events=[];history=[]
    for cycle in range(cycles):
        valid_counts={};max_sep=0.;max_sig=0.
        for phase,Ksigned in enumerate(Ks):
            K=max(float(Ksigned),0.);half=.5*dt
            if hasattr(engine.mpz,'_reversible_transport_K_signed_Pa_sqrt_m'):
                engine.mpz._reversible_transport_K_signed_Pa_sqrt_m=float(Ksigned)
                engine.mpz._reversible_tip_radius_m=float(engine.r_eff())
            sig=max(float(engine.sigma_tip(K)),0.);engine._plastic_half_step(half,300.,sig)
            sig=max(float(engine.sigma_tip(K)),0.);engine._plastic_half_step(half,300.,sig)
            engine.t=float(getattr(engine,'t',0.))+dt
        trigger=None;trigger_phase=None
        for phase,Ksigned in enumerate(opening_Ks):
            sig=max(float(engine.sigma_tip(max(float(Ksigned),0.))),0.)
            opening=evaluate_opening(alternative,sig/kenv,kenv,thermal,300.,state)
            state=opening['connected_state'];max_sig=max(max_sig,sig)
            if opening.get('connected'):max_sep=max(max_sep,float(opening['connected']['delta_m']))
            validity=opening['kinetic_validity'];valid_counts[validity]=valid_counts.get(validity,0)+1
            mechanical=opening['state'] in {'BONDED_BRANCH_LOST','SEPARATION_COMPLETED'}
            if mode=='cohesive_mixed' and opening.get('elementary_rate_s-1') is not None:
                action+=float(opening['elementary_rate_s-1'])*opening_dt
            thermal_fire=mode=='cohesive_mixed' and action>=threshold
            if mechanical or thermal_fire:
                trigger='MECHANICAL' if mechanical else 'THERMAL_ESCAPE';trigger_phase=phase;break
        renewals=0
        while (trigger is not None and renewals<256 and
               float(engine.checkpoint_advance_total_m)<60e-6):
            wake=commit_patch(engine,patch_m);events.append({'event_index':len(events)+1,
                'cycle':cycle+trigger_phase/opening_phases,'phase':trigger_phase,'trigger':trigger,
                'extension_m':float(engine.checkpoint_advance_total_m),'patch_m':patch_m,
                'mobile_count':float(engine.mpz.mobile_count),'retained_count':float(engine.mpz.retained_count),
                'K_shield_Pa_sqrt_m':float(engine.K_shield()),'wake':wake})
            renewals+=1;state={'branch':'BONDED','delta_m':0.};action=0.;threshold=float(rng.exponential())
            sig=max(float(engine.sigma_tip(max(float(opening_Ks[trigger_phase]),0.))),0.)
            opening=evaluate_opening(alternative,sig/kenv,kenv,thermal,300.,state)
            state=opening['connected_state'];mechanical=opening['state'] in {'BONDED_BRANCH_LOST','SEPARATION_COMPLETED'}
            rate=opening.get('elementary_rate_s-1')
            thermal_fire=(mode=='cohesive_mixed' and rate is not None and
                          float(rate)*opening_dt>=threshold)
            trigger=('MECHANICAL' if mechanical else 'THERMAL_ESCAPE' if thermal_fire else None)
        history.append({'cycle':cycle+1,'extension_m':float(engine.checkpoint_advance_total_m),
            'events':len(events),'mobile_count':float(engine.mpz.mobile_count),
            'retained_count':float(engine.mpz.retained_count),'K_shield_Pa_sqrt_m':float(engine.K_shield()),
            'max_opening_m':max_sep,'max_tip_stress_Pa':max_sig,'kinetic_validity_counts':valid_counts})
        if float(engine.checkpoint_advance_total_m)>=60e-6:break
    ext=float(engine.checkpoint_advance_total_m)
    developed=[e for e in events if 20e-6<=e['extension_m']<=60e-6]
    if len(developed)>=2:
        dc=developed[-1]['cycle']-developed[0]['cycle']
        if dc>0:
            rate=(developed[-1]['extension_m']-developed[0]['extension_m'])/dc
            disposition='DEVELOPED_WINDOW_RATE'
        else:
            rate=math.nan;disposition='SINGLE_EXCURSION_BURST_CENSOR'
    elif ext<20e-6:rate=ext/max(len(history),1);disposition='STARTUP_ONLY_CENSOR'
    else:rate=math.nan;disposition='INSUFFICIENT_RECURRENT_EVENTS_CENSOR'
    return {'status':'TARGET_COMPLETED' if ext>=60e-6 else 'CYCLE_LIMIT_CENSOR',
        'candidate_id':binding.provenance['candidate_id'],'shape':shape,'mode':mode,'Kmax_MPa_sqrt_m':Kmax,
        'DeltaK_MPa_sqrt_m':.9*Kmax,'cycles':len(history),'extension_m':ext,'event_count':len(events),
        'developed_rate_m_per_cycle':rate,'rate_disposition':disposition,'events':events,'history':history,
        'patch_geometry':{'interface_cell_length_m':patch_m,'owner':'DECLARED_SPATIAL_INTERFACE_CELL','stochastic_reward':False},
        'legacy_cooperative_clock_used':False,'paired_plasticity_sha256':binding.provenance['parent_complete_row_sha256']}

binding_rows=[]
def execute(args):
    global binding_rows
    args.out.mkdir(parents=True,exist_ok=True)
    with args.registry.open(newline='') as f:binding_rows=list(csv.DictReader(f))
    wanted=set(args.candidate_ids.split(',')) if args.candidate_ids else None
    rows=[r for r in binding_rows if wanted is None or r['candidate_id'] in wanted][:args.materials];summaries=[]
    # Loader validates legacy hashes; its cohesive law is only a material source.
    for mi,row in enumerate(rows):
        binding=load_cohesive_binding(args.registry,row['candidate_id'],execution_identity=COHESIVE_KINETIC_CZM)
        c=json.loads(row['cohesive_parameter_vector']);thermal=thermal_from(c)
        crit=[critical_K(thermal,300.,s) for s in SHAPES];base=float(np.nanmean(crit))
        factors=[float(x) for x in args.load_factors.split(',')]
        loads=[x*base for x in factors]
        for shape in SHAPES:
            for mode in MODES:
                arm_rows=[]
                for li,K in enumerate(loads):
                    result=run_case(binding,shape,mode,K,args.cycles,args.phases,args.patch_um*1e-6,args.seed+1000*mi+li)
                    case=args.out/row['candidate_id']/f"{mode}__{shape}__K{K:.6f}.json";case.parent.mkdir(parents=True,exist_ok=True);atomic(case,result)
                    arm_rows.append(result);print(json.dumps({'material':mi+1,'candidate':row['candidate_id'],'shape':shape,'mode':mode,'K':K,'status':result['status'],'events':result['event_count']}),flush=True)
                developed=[x for x in arm_rows if x['rate_disposition']=='DEVELOPED_WINDOW_RATE' and x['developed_rate_m_per_cycle']>0]
                slopes=[]
                for lo,hi in zip(developed,developed[1:]):
                    slopes.append(math.log(hi['developed_rate_m_per_cycle']/lo['developed_rate_m_per_cycle'])/math.log(hi['DeltaK_MPa_sqrt_m']/lo['DeltaK_MPa_sqrt_m']))
                summaries.append({'candidate_id':row['candidate_id'],'shape':shape,'mode':mode,
                    'paired_plasticity_sha256':row['parent_complete_row_sha256'],'loads_executed':len(arm_rows),
                    'developed_loads':len(developed),'censored_loads':len(arm_rows)-len(developed),
                    'developed_rate_decades':math.log10(max(x['developed_rate_m_per_cycle'] for x in developed)/min(x['developed_rate_m_per_cycle'] for x in developed)) if len(developed)>=2 else math.nan,
                    'local_slopes_json':json.dumps(slopes),'recurrent_growth':any(x['event_count']>=3 for x in arm_rows)})
    write_csv(args.out/'campaign_dispositions.csv',summaries)
    atomic(args.out/'campaign_summary.json',{'schema':SCHEMA,'stage':'NATIVE_COMPLETE','materials':len(rows),
        'arms_per_material':4,'loads_per_arm':len(factors),'cycles_limit':args.cycles,'phase_count':args.phases,
        'developed_arms':sum(r['developed_loads']>=2 for r in summaries),
        'censors_reported_separately':True,'legacy_cooperative_clock_used':False,
        'results':summaries});return 0

def main(argv=None):
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('analytical');a.add_argument('--original-registry',type=Path,required=True);a.add_argument('--parent-registry',type=Path,required=True);a.add_argument('--source-registry',type=Path,required=True);a.add_argument('--out',type=Path,required=True);a.set_defaults(func=analytical)
    s=sub.add_parser('reuse-state');s.add_argument('--registry',type=Path,required=True);s.add_argument('--prior',type=Path,nargs='+',required=True);s.add_argument('--out',type=Path,required=True);s.set_defaults(func=reuse_state)
    q=sub.add_parser('screen-state');q.add_argument('--registry',type=Path,required=True);q.add_argument('--out',type=Path,required=True);q.add_argument('--cycles',type=int,default=2);q.add_argument('--phases',type=int,default=4);q.set_defaults(func=screen_state)
    e=sub.add_parser('execute');e.add_argument('--registry',type=Path,required=True);e.add_argument('--out',type=Path,required=True);e.add_argument('--materials',type=int,default=12);e.add_argument('--candidate-ids');e.add_argument('--load-factors',default='.78,.88,.96,1.04,1.14');e.add_argument('--cycles',type=int,default=120);e.add_argument('--phases',type=int,default=8);e.add_argument('--patch-um',type=float,default=1.0);e.add_argument('--seed',type=int,default=20261017);e.set_defaults(func=execute)
    args=p.parse_args(argv);return args.func(args)
if __name__=='__main__':raise SystemExit(main())
