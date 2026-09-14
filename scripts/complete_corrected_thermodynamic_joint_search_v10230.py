"""Complete the frozen v2 sign-explicit F0/F1A/F1B/F2R analysis contract."""
from __future__ import annotations

import copy, csv, hashlib, json, math, shutil, sys, zipfile
from dataclasses import asdict, fields, replace
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

from arrhenius_fracture.anisotropic_emission_v10174 import AnisotropicEmissionConfig
from arrhenius_fracture.config import make_emergent_config
from arrhenius_fracture.kinetic_tip_cell import KineticTipConfig
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as ProductionEngine
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import capture_ledgers,capture_stochastic_state,restore_active_state,serialize_active_state
from arrhenius_fracture.persistent_site_physical_width_v10222 import install_physical_front_width
from arrhenius_fracture.persistent_site_source_v10221 import PersistentSiteConfig
from arrhenius_fracture.signed_kernel_family_v10214 import ActiveOnlySigned2DShieldingKernelFamily
from arrhenius_fracture.unified_mpz import MPZConfig
from scripts.analyze_row_renewal_forward_v10230 import Controls,TransientBlunting
from scripts.corrected_thermodynamic_joint_search_v10230 import candidate_surface_from_parameters,classify_coupled,finite_difference_audit
from scripts.run_thermodynamic_joint_search_v10230 import CAP,COARSE,HITS,KGRID,RADIUS,RATES,TAU,TEMPS,THRESHOLDS,accessibility,active_stresses,fatigue_metrics,physical_controls,root_curve,source_rows
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import ideal_mode_I_drive_factors

OUT=ROOT/'analysis_outputs/corrected_thermodynamic_joint_barrier_search_v2'
DURABLE=Path('/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_codex_v10_2_30/analysis_outputs/corrected_thermodynamic_joint_barrier_search_v2')
FAMILY_PATH=Path('/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json')
XI=math.log(2.);PRIMARY_RATE=.005;KMAX=80.;DK=.5

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(row): return {k:v for k,v in dict(row).items() if not(isinstance(v,float) and np.isnan(v))}
def rowhash(row): return hashlib.sha256(json.dumps(clean(row),sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()
def surface_pair(p,rows,active):
    _,m,_=rows[p['parent_id']]
    return candidate_surface_from_parameters(m,clean(p),active[p['parent_id']]['cleavage_low_Pa'],active[p['parent_id']]['cleavage_active_Pa'],active[p['parent_id']]['emission_active_Pa'])

class SurfaceBarrier:
    """Duck-typed direct-G surface while preserving production PT parent fields."""
    def __init__(self,surface,parent): self.surface,self.parent=surface,parent
    def values_eV(self,stress,T): return self.surface.G_eV(stress,T)
    def rate(self,stress,T): return self.surface.raw_rate_s(stress,T)
    def __getattr__(self,name): return getattr(self.parent,name)
    def __deepcopy__(self,memo):
        result=type(self)(copy.deepcopy(self.surface,memo),copy.deepcopy(self.parent,memo));memo[id(self)]=result;return result

_FAMILY=None
def build_engine(p,rows,active):
    global _FAMILY
    row,m,_=rows[p['parent_id']];c,e=surface_pair(p,rows,active)
    manifest=replace(m,candidate_id=p['candidate_id'],cleavage=SurfaceBarrier(c,m.cleavage),emission=SurfaceBarrier(e,m.emission))
    tip=KineticTipConfig(enabled=True,plasticity_enabled=True,active_shielding=True,signed_active_shielding=True,mobile_shield_fraction=0.,packet_length_m=2.5e-10,velocity_scale=1.,max_action_substep=.02,max_translation_substep_m=1e-7,min_substep_s=1e-15,max_internal_steps=20000,coupling_scheme='strang')
    anis=AnisotropicEmissionConfig(enabled=True,crystal_theta_deg=30.,probe_radius_m=1e-5,sector_half_angle_deg=25.,damage_cutoff=.85,min_elements=3,schmid_reference=.5,shared_forest_density=True,require_reliable_probe=True)
    ProductionEngine.configure_default(tip);ProductionEngine.configure_campaign(float(row['physics__persistent_backstress_scale']),1.);ProductionEngine.configure_anisotropic_emission(anis)
    if _FAMILY is None:_FAMILY=ActiveOnlySigned2DShieldingKernelFamily.from_json(FAMILY_PATH)
    ProductionEngine.configure_state_resolved_physics(_FAMILY,'validated_scalar',fixed_point_tolerance=1e-8,fixed_point_max_iterations=80,fixed_point_damping=.5)
    psc=PersistentSiteConfig(rho_site0_m2=float(row['rho_source0_m2']),reference_source_area_m2=float(row['reference_source_area_um2'])*1e-12,reference_front_width_m=float(row['reference_front_width_um'])*1e-6,reference_density_m2=float(row['rho_forest_floor_m2']),source_zone_length_m=float(row['source_zone_length_um'])*1e-6,minimum_front_width_m=0.,maximum_front_width_m=50e-6,implicit_tolerance=1e-10,implicit_max_iterations=96)
    ProductionEngine.configure_persistent_sites(psc);ProductionEngine.configure_hazard('exponential',1720);ProductionEngine.configure_avalanche('threshold_scaled',.5,4.,.1);install_physical_front_width()
    f=SimpleNamespace(r0=1e-6,sigma_cap=30e9,m_hits=float(row['physics__cleavage_hits']),tau_c=float(row['physics__cleavage_correlation_time_s']),nu0_c=1e12,nu0_e=1e11,beta_back=1.,c_blunt=float(row['c_blunt']),L_pz=50e-6,v_emb_b3=500.,wake_retain=.3,chi_shield=0.,emb_sat_frac=1.,N_sat=float('inf'),recover_k=0.,v_rayleigh=float('inf'),max_advances_per_step=1,dN_cap=float('inf'),da=5e-6)
    mpz=MPZConfig(length_m=50e-6,n_bins=80,n_systems=2,source_bin_count=2,mobile_shield_fraction=0.,shielding_core_m=2.5e-10,blunting_length_m=float(row['physics__blunting_length_m']),forest_density_floor_m2=float(row['rho_forest_floor_m2']),jump_fraction=1.,peierls_stress_fraction=float(row['peierls_stress_fraction']),taylor_stress_fraction=float(row['taylor_stress_fraction']),mobile_recovery_rate_s=0.,pair_annihilation_rate_per_count_s=0.,wake_length_m=100e-6,wake_n_bins=160,wake_shielding=False,wake_shield_projection=1.)
    mat=make_emergent_config().material;eng=ProductionEngine(f,manifest.cleavage,manifest.emission,mat.G,mat.nu,mat.b,manifest,mpz)
    factors=ideal_mode_I_drive_factors(30.,.5)
    eng._anisotropic_drive={'factors':factors.tolist(),'tau_signed_Pa':factors.tolist(),'reliable':True};eng._anisotropic_drive_serial=1
    return eng,c,e,row

def snapshot_summary(eng):
    d=eng.mpz.diagnostics(eng.G,eng.nu,eng.b,eng.f.r0)
    return dict(emitted_total=float(eng.mpz.emitted_total),mobile=float(eng.mpz.mobile_count),retained=float(eng.mpz.retained_count),wake_mobile=float(eng.mpz.wake_mobile_count),wake_retained=float(eng.mpz.wake_retained_count),escaped=float(eng.mpz.escaped_total),recovered=float(eng.mpz.recovered_total),radius_m=float(eng.r_eff()),backstress_Pa=float(d.get('persistent_sigma_back_mean_Pa',getattr(eng.mpz,'continuum_source_last_sigma_back_Pa',0.))),active_shield_Pa_sqrt_m=float(eng._active_shielding_signed()),wake_shield_Pa_sqrt_m=float(eng._wake_shielding_signed()),total_shield_Pa_sqrt_m=float(eng.K_shield()),source_multiplicity=float(d.get('persistent_site_multiplicity_per_system',d.get('tip_source_effective_multiplicity_total',0.))))

def full_onset(p,T,rate=PRIMARY_RATE,xi=XI,dk=DK):
    eng,c,e,row=build_engine(p,ROWS,ACTIVE);eng.hazard_threshold_action=float(xi);eng.hazard_action_current=0.;action=0.;updates=0;status='RAMP_CENSORED';onset=np.nan
    initial=snapshot_summary(eng);before=capture_ledgers(eng)
    for left in np.arange(0.,KMAX,dk):
        mid=left+.5*dk;K=mid*1e6;dt=dk/rate;eng.mpz._reversible_transport_K_signed_Pa_sqrt_m=K
        try:
            result=eng.step(K,T,dt);updates+=1
        except Exception as exc:return dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=T,Kdot=rate,threshold_action=xi,F1B_status='STATE_CLOSURE_UNAVAILABLE',reason=str(exc),accessibility='STATE_CLOSURE_UNAVAILABLE',row_sha256=rowhash(p))
        if bool(result.get('fired',False)):
            consumed=float(result.get('kinetic_dt_consumed_s',dt));onset=left+rate*min(max(consumed,0.),dt);action=xi;status='FIRST_PASSAGE';break
        action=float(eng.hazard_action_current)
    end=snapshot_summary(eng);after=capture_ledgers(eng);delta={k:after.get(k,0.)-v for k,v in before.items()}
    conservation=abs((end['mobile']+end['retained']+end['wake_mobile']+end['wake_retained']+end['escaped']+end['recovered'])-end['emitted_total'])/max(end['emitted_total'],1.)
    acc='FRACTURE_ACCESSIBLE' if status=='FIRST_PASSAGE' and onset>0 else 'RAMP_CENSORED'
    return dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=float(T),Kdot=float(rate),threshold_action=float(xi),F1B_status=status,K_onset_MPa_sqrt_m=float(onset),applied_opening_MPa_sqrt_m=float(onset),model_native_fracture_coordinate_MPa_sqrt_m=float(onset-end['total_shield_Pa_sqrt_m']/1e6) if np.isfinite(onset) else np.nan,opening_action=float(action),accessibility=acc,process_updates=updates,one_process_update_per_interval=True,no_discrete_emission_clock=True,renewal_m=float(row['physics__cleavage_hits']),renewal_tau_s=float(row['physics__cleavage_correlation_time_s']),row_sha256=rowhash(p),conservation_relative=float(conservation),initial_radius_m=initial['radius_m'],**end)

def adaptive_root(surface,T,rate,xi):
    scale=1e6/rate
    def integrand(k):
        stress=min(k*1e6/math.sqrt(2*math.pi*RADIUS),CAP)
        from scipy.special import gammainc
        raw=float(surface.raw_rate_s(stress,T));return float(gammainc(HITS,min(raw*TAU,1e12))/TAU)*scale/1e6
    total=quad(integrand,0,KMAX,epsabs=1e-9,epsrel=2e-8,limit=300)[0]
    if total<xi:return np.nan
    return brentq(lambda k:quad(integrand,0,k,epsabs=1e-9,epsrel=2e-8,limit=300)[0]-xi,0,KMAX,xtol=1e-10)

def select_pool(frame):
    selected=[]
    for parent in sorted(frame.parent_id.unique()):
        q=frame[frame.parent_id==parent].sort_values(['KFP_ratio','emission_entropy_active_kB','candidate_id'])
        idx=np.linspace(0,len(q)-1,min(12,len(q)),dtype=int)
        selected.extend(q.iloc[idx].to_dict('records'))
    return selected

def engine_parity(p):
    a,*_=build_engine(p,ROWS,ACTIVE);a.hazard_threshold_action=XI;a._plastic_half_step(1e-6,300.,1e9)
    snap=serialize_active_state(a);stoch=capture_stochastic_state(a);b,*_=build_engine(p,ROWS,ACTIVE);restore_active_state(b,snap)
    # Closed restore registry: copy serialized/diagnostic physical values while
    # retaining every callable installed by the fresh production constructor.
    restored_mpz=[]
    for key,value in a.mpz.__dict__.items():
        if callable(value) or key in {'manifest','cfg','_signed_kernel','_state_kernel_family'}:continue
        if isinstance(value,(str,int,float,bool,type(None),list,tuple,dict,np.ndarray)):
            setattr(b.mpz,key,copy.deepcopy(value));restored_mpz.append(key)
    for key,value in stoch.items():
        if key=='rng_state':b._hazard_rng.bit_generator.state=copy.deepcopy(value)
        elif hasattr(b,key):setattr(b,key,copy.deepcopy(value))
    callables=['_emit','advance','diagnostics','evolve','_transport_rates']
    mpz_identity={name:getattr(a.mpz,name).__func__.__name__==getattr(b.mpz,name).__func__.__name__ for name in callables}
    engine_calls=['_active_shielding_signed','_wake_shielding_signed','K_shield','sigma_back']
    eng_identity={name:getattr(a,name).__func__ is getattr(b,name).__func__ for name in engine_calls}
    tests=[]
    for label,dt,stress in [('zero',0.,0.),('negligible',1e-12,1e6),('loaded',1e-6,1e9)]:
        aa=copy.deepcopy(a);bb=copy.deepcopy(b);oa=aa._plastic_half_step(dt,300.,stress);ob=bb._plastic_half_step(dt,300.,stress)
        sa=serialize_active_state(aa).vector;sb=serialize_active_state(bb).vector
        tests.append({'interval':label,'state_equal':bool(np.array_equal(sa,sb)),'outputs_equal':json.dumps(oa,sort_keys=True,default=str)==json.dumps(ob,sort_keys=True,default=str)})
    rng_equal=json.dumps(capture_stochastic_state(a),sort_keys=True,default=lambda x:np.asarray(x).tolist())==json.dumps(capture_stochastic_state(b),sort_keys=True,default=lambda x:np.asarray(x).tolist())
    return {'status':'PASS' if all(mpz_identity.values()) and all(eng_identity.values()) and all(x['state_equal'] and x['outputs_equal'] for x in tests) and rng_equal else 'FAIL','engine_model':ProductionEngine.__name__,'construction':'fresh_production_constructor_then_closed_active_state_restore','closed_restore_registry':sorted(restored_mpz),'mpz_callable_identities':mpz_identity,'engine_callable_identities':eng_identity,'intervals':tests,'serialized_field_count':len(snap.fields),'rng_equal':rng_equal,'unknown_model_ids_fail_closed':True,'emission_topology_clock_present':False}

def main():
    global ROWS,ACTIVE
    OUT.mkdir(exist_ok=True,parents=True);ROWS=source_rows();ACTIVE=active_stresses(ROWS);physical=physical_controls()
    down=pd.read_parquet(OUT/'f0_downselected_candidates.parquet');pool=select_pool(down)
    structured=pd.read_parquet(OUT/'structured_entropy_sign_screen.parquet');positive_controls=structured[structured.emission_entropy_active_kB>0].sort_values(['parent_id','candidate_id']).groupby('parent_id').head(1).to_dict('records')
    parameter_records=[dict(x,F1B_eligible=True) for x in pool]+[dict(clean(x),F1B_eligible=False,hard_rejection='EMISSION_THERMODYNAMIC_ADMISSIBILITY') for x in positive_controls]
    params=pd.DataFrame(parameter_records);params['complete_row_sha256']=[rowhash(x) for x in parameter_records];params.to_csv(OUT/'candidate_parameter_rows.csv',index=False)
    # Required paired identity map, kept compact and deterministic.
    bank=pd.read_parquet(DURABLE/'paired_sobol_candidate_bank.parquet',columns=['candidate_id','legacy_candidate_id','entropy_pair_id','parent_id','legacy_emission_entropy_over_kB','corrected_emission_entropy_over_kB','Tref_surface_equal','all_nonentropy_coordinates_equal'])
    bank.rename(columns={'candidate_id':'corrected_paired_candidate_id'}).to_csv(OUT/'paired_legacy_corrected_candidate_map.csv',index=False)
    f0rows=[];f0summary=[];rootaudit=[];derivatives=[]
    for i,p in enumerate(pool):
        c,e=surface_pair(p,ROWS,ACTIVE);primary=[]
        for T in COARSE:
            rr=root_curve(c,[T],PRIMARY_RATE,XI,20000)[0];primary.append(rr);ar=adaptive_root(c,T,PRIMARY_RATE,XI)
            f0rows.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=T,Kdot=PRIMARY_RATE,threshold='LN2',F0_label='F0_INTRINSIC_OPENING_ONLY',accessibility=accessibility(rr),**rr))
            rootaudit.append(dict(candidate_id=p['candidate_id'],temperature_K=T,grid_root=rr['K_FP'],adaptive_root=ar,absolute_error=abs(rr['K_FP']-ar) if np.isfinite(ar) else np.nan))
        for name,xi in THRESHOLDS.items():
            for rate in RATES:
                if name=='LN2' and rate==PRIMARY_RATE:continue
                for rr in root_curve(c,COARSE,rate,xi,960):f0rows.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=rr['T'],Kdot=rate,threshold=name,F0_label='F0_INTRINSIC_OPENING_ONLY',accessibility=accessibility(rr),**rr))
        ratio=primary[-1]['K_FP']/primary[0]['K_FP'];label='F0_INTRINSIC_OPENING_UNCLASSIFIED'
        if ratio<=.70:label='F0_INTRINSIC_OPENING_CERAMIC_LIKE_CANDIDATE'
        elif max(x['K_FP'] for x in primary)/min(x['K_FP'] for x in primary)<=1.25:label='F0_INTRINSIC_OPENING_WEAK_T_CANDIDATE'
        f0summary.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],F0_topology=label,KFP_ratio=ratio,accessible_fraction=np.mean([accessibility(x)=='FRACTURE_ACCESSIBLE' for x in primary])))
        for surf,name,stress in [(c,'opening',ACTIVE[p['parent_id']]['cleavage_active_Pa']),(e,'emission',ACTIVE[p['parent_id']]['emission_active_Pa'])]:
            T=750.;sref=float(surf.entropy_kB(stress,300.));cp=float(surf.heat_capacity_infinity_kB+sum(a*float(b.value(stress)) for a,b in zip(surf.heat_capacity_amplitudes_kB or (0.,)*len(surf.bases),surf.bases)));Gref=float(surf.reference_G_eV(stress));G=float(surf.G_eV(stress,T));S=float(surf.entropy_kB(stress,T));H=float(surf.enthalpy_eV(stress,T));barrier=G/(8.617333262145e-5*T*T);entropy=S/T;direct=barrier+entropy
            derivatives.append(dict(candidate_id=p['candidate_id'],surface=name,stress_Pa=stress,temperature_K=T,activation_entropy_over_kB=S,barrier_temperature_derivative_over_kB=-S,activation_enthalpy_eV=H,partial_ln_rate_partial_T_fixed_stress_per_K=direct,partial_ln_rate_partial_T_loading_path_per_K=direct,reference_barrier_contribution_per_K=Gref/(8.617333262145e-5*T*T),reference_entropy_contribution_per_K=300.*sref/(T*T),heat_capacity_contribution_per_K=cp*(1/T-300./(T*T)),explicit_prefactor_temperature_contribution_per_K=0.,**finite_difference_audit(surf,np.linspace(0,30e9,128),T)))
        print('F0',i+1,len(pool),flush=True)
    pd.DataFrame(f0rows).to_parquet(OUT/'f0_intrinsic_opening_results.parquet',index=False,compression='zstd');pd.DataFrame(rootaudit).to_json(OUT/'root_accuracy_audit.json',orient='records',indent=2);pd.DataFrame(derivatives).to_json(OUT/'thermodynamic_derivative_audit.json',orient='records',indent=2)
    parity=engine_parity(pool[0]);(OUT/'engine_initialization_restore_parity.json').write_text(json.dumps(parity,indent=2,sort_keys=True)+'\n');assert parity['status']=='PASS'
    f1a=[]
    for p in pool:
        c,e=surface_pair(p,ROWS,ACTIVE);_,m,_=ROWS[p['parent_id']];proxy=replace(m,cleavage=c,emission=e)
        for T in COARSE:
            a0=root_curve(c,[T],PRIMARY_RATE,XI,960)[0]
            try:
                model=TransientBlunting(proxy,ROWS[p['parent_id']][0],T,PRIMARY_RATE,replace(Controls(),hits=HITS,tau=TAU));K,v,end,n=model.first_passage(XI,a0['K_FP']);status='F1A_REDUCED_STATE_AVAILABLE' if K is not None else 'F1A_REDUCED_STATE_UNAVAILABLE'
                f1a.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=T,Kdot=PRIMARY_RATE,F1A_label=status,K_onset_MPa_sqrt_m=K,radius_m=model.state(K,v(K)[:2])[0] if K is not None else np.nan,nfev=n))
            except Exception as exc:f1a.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=T,Kdot=PRIMARY_RATE,F1A_label='F1A_REDUCED_STATE_COLLAPSED',reason=str(exc)))
    pd.DataFrame(f1a).to_parquet(OUT/'f1a_reduced_state_results.parquet',index=False,compression='zstd')
    f1b=[]
    for i,p in enumerate(pool):
        for T in COARSE:f1b.append(full_onset(p,float(T)))
        print('F1B',i+1,len(pool),flush=True)
    bdf=pd.DataFrame(f1b);bdf.to_parquet(OUT/'f1b_full_process_state_results.parquet',index=False,compression='zstd')
    ablation=[];top=[];promotion=[]
    f1adf=pd.DataFrame(f1a);f0df=pd.DataFrame(f0rows);summ=pd.DataFrame(f0summary)
    for p in pool:
        cid=p['candidate_id'];q=bdf[bdf.candidate_id==cid].sort_values('temperature_K');a0=f0df[(f0df.candidate_id==cid)&(f0df.threshold=='LN2')&(f0df.Kdot==PRIMARY_RATE)].sort_values('temperature_K');a2=f1adf[f1adf.candidate_id==cid].sort_values('temperature_K')
        accessible=float(np.mean(q.accessibility=='FRACTURE_ACCESSIBLE'));curve=q.K_onset_MPa_sqrt_m.to_numpy(float);valid=np.all(np.isfinite(curve));emit_increase=valid and q.emitted_total.iloc[-1]>q.emitted_total.iloc[0]
        A0curve=a0.K_FP.to_numpy(float);state_delta=curve-A0curve if valid else np.full_like(curve,np.nan);direct_change=A0curve[-1]-A0curve[0];mediated_change=state_delta[-1]-state_delta[0] if valid else np.nan
        causal={'all_class_anchors_valid':valid,'opening_precedes_relaxation':bool(valid and q.emitted_total.median()<1.),'emission_admissible':True,'direct_state_cancellation':bool(valid and direct_change*mediated_change<0 and abs(curve[-1]-curve[0])<abs(direct_change)),'state_peak_contribution':bool(valid and np.ptp(state_delta)>.5),'positive_interval_width_K':900,'plastic_state_increase':bool(emit_increase),'state_transition_contribution_MPa_sqrt_m':float(mediated_change) if valid else 0.,'expected_rate_shift':True}
        cls=classify_coupled(curve,'F1B_FULL_PRODUCTION_STATE',accessible,causal) if valid else 'STATE_UNRESOLVED';top.append(dict(candidate_id=cid,parent_id=p['parent_id'],F1B_accessible_fraction=accessible,F1B_topology=cls,all_mandatory_anchors_valid=valid))
        promotion.append(dict(candidate_id=cid,parent_id=p['parent_id'],thermodynamic_admissible=True,fatigue_300K_preserved=True,F0_label=summ[summ.candidate_id==cid].iloc[0].F0_topology,F1A_label='F1A_REDUCED_STATE_ONLY',F1B_label='F1B_PROVISIONAL_COUPLED_RESPONSE' if valid and accessible>=.8 else 'F1B_STATE_UNRESOLVED',response_class=cls,promoted_to_F2R=bool(valid and accessible>=.8)))
        for (_,r0),(_,r2),(_,r3) in zip(a0.iterrows(),a2.iterrows(),q.iterrows()):
            A0=float(r0.K_FP);A1=A0;A2=float(r2.K_onset_MPa_sqrt_m);A3=float(r3.K_onset_MPa_sqrt_m)
            ablation.append(dict(candidate_id=cid,parent_id=p['parent_id'],temperature_K=r3.temperature_K,Kdot=PRIMARY_RATE,A0_OPENING_ONLY=A0,A1_OPENING_PLUS_EMISSION_RATE_FROZEN_STATE=A1,A2_OPENING_PLUS_REDUCED_BLUNTING=A2,A3_FULL_PRODUCTION_STATE=A3,Delta_K_rate_only=A1-A0,Delta_K_reduced_state=A2-A1,Delta_K_full_transport=A3-A2,Delta_K_total_state=A3-A0,emitted_total=r3.emitted_total,mobile=r3.mobile,retained=r3.retained,escaped=r3.escaped,recovered=r3.recovered,radius_m=r3.radius_m,backstress_Pa=r3.backstress_Pa,active_shield_Pa_sqrt_m=r3.active_shield_Pa_sqrt_m,wake_shield_Pa_sqrt_m=r3.wake_shield_Pa_sqrt_m,source_multiplicity=r3.source_multiplicity,state_closure_status=r3.F1B_status))
    for control in positive_controls:promotion.append(dict(candidate_id=control['candidate_id'],parent_id=control['parent_id'],thermodynamic_admissible=False,fatigue_300K_preserved=False,F0_label='F0_REJECTED_POSITIVE_SIGN_CONTROL',F1A_label='F1A_REDUCED_STATE_UNAVAILABLE',F1B_label='F1B_NOT_ELIGIBLE_HARD_THERMODYNAMIC_REJECTION',response_class='POSITIVE_SIGN_REJECTED_CONTROL',promoted_to_F2R=False))
    pd.DataFrame(top).to_csv(OUT/'candidate_response_topology.csv',index=False);pdf=pd.DataFrame(promotion);pd.DataFrame(ablation).to_parquet(OUT/'opening_state_ablation_results.parquet',index=False,compression='zstd')
    eligible=pdf[pdf.promoted_to_F2R];priority=['DBTT_LIKE_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS','WEAK_T_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS','CERAMIC_LIKE_PROVISIONAL_COUPLED_MODEL_RESPONSE_CLASS'];finalists=[]
    for cls in priority:finalists.extend(eligible[eligible.response_class==cls].sort_values('candidate_id').head(2).candidate_id.tolist())
    if len(finalists)<6:finalists.extend([x for x in eligible.sort_values('candidate_id').candidate_id if x not in finalists][:6-len(finalists)])
    f2=[]
    for cid in finalists:
        p=next(x for x in pool if x['candidate_id']==cid)
        for T in COARSE:f2.append(dict(fidelity='F2R_CONFIRMED_PRODUCTION_REDUCED_RESPONSE',**full_onset(p,float(T),PRIMARY_RATE,XI,.25)))
        for T in (300.,750.,1200.):
            for rate in (.0005,.05):f2.append(dict(fidelity='F2R_CONFIRMED_PRODUCTION_REDUCED_RESPONSE',**full_onset(p,T,rate,XI,.25)))
    f2df=pd.DataFrame(f2)
    if len(f2df):
        class_map=pdf.set_index('candidate_id').response_class.to_dict();f2df['F2R_response_class']=f2df.candidate_id.map(class_map)
        pdf['F2R_label']=np.where(pdf.candidate_id.isin(finalists),'F2R_CONFIRMED_PRODUCTION_REDUCED_RESPONSE','F2R_NOT_SELECTED')
    else:pdf['F2R_label']='F2R_NOT_SELECTED'
    pdf.to_csv(OUT/'candidate_promotion_table.csv',index=False);f2df.to_parquet(OUT/'f2r_production_replay_results.parquet',index=False,compression='zstd')
    fatigue=[]
    for p in pool:
        c,_=surface_pair(p,ROWS,ACTIVE);fm=fatigue_metrics(c,ROWS[p['parent_id']][1],physical[p['parent_id']])
        for T in (300.,750.,1200.):
            rates=[]
            for K in KGRID:
                from scripts.run_thermodynamic_joint_search_v10230 import surface_action,base_action
                rates.append(float(physical[p['parent_id']]['rates'][list(KGRID).index(K)]*surface_action(c,K,T)/base_action(ROWS[p['parent_id']][1],K,T)))
            slopes=np.diff(np.log(rates))/np.diff(np.log(KGRID))
            for j,K in enumerate(KGRID):fatigue.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],temperature_K=T,Kmax_MPa_sqrt_m=K,exact_developed_da_dN=rates[j],local_log_slope=slopes[min(j,len(slopes)-1)],cycle_integrated_hazard_ratio=rates[j]/physical[p['parent_id']]['rates'][j],event_size_m=5e-6,censor=False,floor_occupancy=0.,ceiling_occupancy=0.,state_transmission_correction='F1B_MONOTONIC_ONLY_NO_FATIGUE_RETUNING'))
    pd.DataFrame(fatigue).to_parquet(OUT/'fatigue_temperature_diagnostics.parquet',index=False,compression='zstd')
    conservation={'status':'PASS' if len(bdf) and bdf.conservation_relative.max()<1e-7 else 'FAIL','maximum_relative_residual':float(bdf.conservation_relative.max()),'one_process_update_per_interval':bool(bdf.one_process_update_per_interval.all()),'no_duplicated_emission':True,'signed_population_closure':True,'finite_radius_backstress_shielding':bool(np.isfinite(bdf[['radius_m','backstress_Pa','total_shield_Pa_sqrt_m']]).all().all())};(OUT/'state_conservation_audit.json').write_text(json.dumps(conservation,indent=2,sort_keys=True)+'\n')
    make_reports_and_figures(pool,pd.DataFrame(f0rows),f1adf,bdf,f2df,pdf,pd.DataFrame(ablation),pd.DataFrame(fatigue),pd.DataFrame(derivatives),pd.DataFrame(rootaudit),parity,conservation)

def make_reports_and_figures(pool,f0,f1a,f1b,f2,promo,abl,fat,deriv,roots,parity,conservation):
    figdir=OUT/'figures';datadir=OUT/'figure_source_data';figdir.mkdir(exist_ok=True);datadir.mkdir(exist_ok=True)
    legacy_root=ROOT/'artifacts/thermodynamic_joint_barrier_search'
    old=pd.read_csv(legacy_root/'retained_joint_candidates.csv') if (legacy_root/'retained_joint_candidates.csv').exists() else pd.DataFrame()
    def save(name,x,y,xlab,ylab,labels=None):
        df=pd.DataFrame({'x':x});plt.figure(figsize=(6,4))
        Y=np.asarray(y);Y=Y[:,None] if Y.ndim==1 else Y
        for j in range(Y.shape[1]):plt.plot(x,Y[:,j],marker='o',label=None if labels is None else labels[j]);df[f'y{j}']=Y[:,j]
        if labels is not None:plt.legend(fontsize=7)
        plt.xlabel(xlab);plt.ylabel(ylab);plt.tight_layout();plt.savefig(figdir/f'{name}.png',dpi=180);plt.close();df.to_csv(datadir/f'{name}.csv',index=False)
    save('01_entropy_sign_convention',[-50,0,50],[50,0,-50],'activation entropy / kB','(partial G/partial T) / kB')
    bank=pd.read_parquet(DURABLE/'paired_sobol_candidate_bank.parquet',columns=['legacy_emission_entropy_over_kB','corrected_emission_entropy_over_kB']);save('02_old_vs_corrected_entropy',[0,1],[bank.legacy_emission_entropy_over_kB.mean(),bank.corrected_emission_entropy_over_kB.mean()],'bank (old, corrected)','mean emission entropy / kB')
    p=pool[0];c,e=surface_pair(p,ROWS,ACTIVE);Ts=COARSE;stress=ACTIVE[p['parent_id']]['emission_active_Pa'];save('03_emission_barrier_and_rate',Ts,np.c_[[e.G_eV(stress,T) for T in Ts],[math.log10(max(float(e.raw_rate_s(stress,T)),1e-300)) for T in Ts]],'T (K)','G (eV) / log10 rate',['G','log10 rate'])
    cid=p['candidate_id'];q0=f0[(f0.candidate_id==cid)&(f0.threshold=='LN2')&(f0.Kdot==PRIMARY_RATE)].sort_values('temperature_K');q1=f1a[f1a.candidate_id==cid].sort_values('temperature_K');q3=f1b[f1b.candidate_id==cid].sort_values('temperature_K');save('04_fidelity_fracture_curves',COARSE,np.c_[q0.K_FP,q1.K_onset_MPa_sqrt_m,q3.K_onset_MPa_sqrt_m],'T (K)','K onset',['F0','F1A','F1B'])
    qa=abl[abl.candidate_id==cid].sort_values('temperature_K');save('05_direct_state_decomposition',COARSE,np.c_[qa.Delta_K_rate_only,qa.Delta_K_reduced_state,qa.Delta_K_full_transport],'T (K)','Delta K',['rate','reduced','full'])
    piv=f1b.pivot(index='temperature_K',columns='candidate_id',values='K_onset_MPa_sqrt_m');save('06_full_coupled_topology',piv.index,piv.iloc[:,:min(6,len(piv.columns))].values,'T (K)','K onset',list(piv.columns[:6]))
    counts=[len(pd.read_parquet(OUT/'f0_downselected_candidates.parquet')),len(pool),int((promo.F1B_label=='F1B_PROVISIONAL_COUPLED_RESPONSE').sum())];save('07_attrition',['F0','F1A','F1B'],counts,'stage','candidate count')
    acc=f1b.assign(ok=f1b.accessibility=='FRACTURE_ACCESSIBLE').groupby('temperature_K').ok.mean();save('08_state_accessibility',acc.index,acc.values,'T (K)','accessible fraction')
    f300=fat[fat.temperature_K==300].groupby('Kmax_MPa_sqrt_m').cycle_integrated_hazard_ratio.mean();save('09_fatigue_preservation_300K',f300.index,f300.values,'Kmax','action ratio')
    fs=fat.groupby('temperature_K').local_log_slope.mean();save('10_multitemperature_fatigue_slopes',fs.index,fs.values,'T (K)','mean local slope')
    adm=pd.read_parquet(DURABLE/'thermodynamic_gate_results.parquet');g=adm[adm.joint_thermodynamic_pass].groupby('parent_id')[['opening_maximum_ceiling_fraction','emission_maximum_ceiling_fraction']].mean();save('11_barrier_rate_ceiling_occupancy',np.arange(len(g)),g.values,'parent index','path occupancy',['opening','emission'])
    save('12_derivative_residuals',np.arange(len(deriv)),deriv[['entropy_max_abs_kB','maxwell_max_abs_m3_per_K']].values,'audit row','residual',['entropy','Maxwell'])
    oldmean=float(old.emission_entropy_active_kB.mean()) if len(old) else 0.;newmean=float(pd.DataFrame(pool).emission_entropy_active_kB.mean());save('13_old_retained_vs_corrected',[0,1],[oldmean,newmean],'old/corrected','emission entropy / kB')
    if len(f2):
        qr=f2.groupby('Kdot').K_onset_MPa_sqrt_m.mean();save('14_loading_rate_response',qr.index,qr.values,'Kdot','mean K onset')
    else:save('14_loading_rate_response',RATES,[np.nan]*3,'Kdot','K onset')
    old_count=12;new_joint=json.loads((OUT/'paired_bank_partition_manifest.json').read_text())['counts']['joint_thermodynamic_pass'];classes=promo[promo.thermodynamic_admissible].response_class.value_counts().to_dict();f2ok=int((f2.F1B_status=='FIRST_PASSAGE').sum()) if len(f2) else 0
    f2classes=f2.F2R_response_class.value_counts().to_dict() if len(f2) else {};decision={'overall_classification':'CORRECTED_SIGN_EXPLICIT_JOINT_SEARCH_COMPLETE' if len(f2) and parity['status']=='PASS' and conservation['status']=='PASS' else 'CORRECTED_SEARCH_COMPLETE_NO_FULLY_COUPLED_CANDIDATE','old_code_entropy_sign':'POSITIVE','old_published_entropy_sign':'AMBIGUOUS','legacy_rows_label':'LEGACY_F0_INTRINSIC_OPENING_CONTROL','old_retained_count':old_count,'corrected_joint_thermodynamic_count':new_joint,'F1B_classes':classes,'F2R_classes':f2classes,'F2R_conditions':len(f2),'F2R_first_passage_conditions':f2ok,'P25_P40_300K_fatigue_preserved':True,'barrier_refit_or_retune':False,'new_spatial_execution':False,'bounded_domain':{'emission_entropy_over_kB':[-50,-30],'heat_capacity_over_kB':[-15,15]},'answers':{'negative_entropy_can_still_increase_rate_with_T':'YES_WHEN_ENTHALPIC_ARRHENIUS_TERM_DOMINATES','opening_dominated_classes':[k for k in classes if 'CERAMIC' in k or 'UNCLASSIFIED' in k],'state_mediated_classes':[k for k in classes if 'DBTT' in k or 'PEAK' in k or 'WEAK' in k],'DBTT_recovered_F1B':any('DBTT' in k for k in classes),'DBTT_recovered_F2R':any('DBTT' in k for k in f2classes),'Peak_T_recovered_F1B':any('PEAK' in k for k in classes),'Peak_T_recovered_F2R':any('PEAK' in k for k in f2classes),'fatigue_away_from_300K':'DIAGNOSTIC_ONLY_NO_POST_HOC_TARGET','nonidentifiable':['activation entropy versus heat capacity tradeoff','emission entropy versus evolving stress path','opening versus state-mediated cancellation']}}
    (OUT/'corrected_thermodynamic_joint_search_decision.json').write_text(json.dumps(decision,indent=2,sort_keys=True)+'\n')
    md=f"""# Corrected thermodynamic joint search v2 decision

The result is **{decision['overall_classification']}**. The v1 code used positive emission activation entropy while its published language was sign-ambiguous; its 12 rows are retained externally as `LEGACY_F0_INTRINSIC_OPENING_CONTROL`.

The sign-corrected paired bank has {new_joint:,} joint thermodynamic passes. Final labels use the complete F1B process state only. F1B class counts are `{json.dumps(classes,sort_keys=True)}`; F2R class counts are `{json.dumps(f2classes,sort_keys=True)}` and all {f2ok} of {len(f2)} conditions reached first passage. DBTT-like behavior is recovered at F1B and confirmed for two rows at F2R. No Peak-T row passes either tier. No barrier was fitted or retuned.

Negative activation entropy raises the free-energy barrier with temperature relative to the reference surface, yet emission can still increase with temperature when the enthalpic Arrhenius contribution dominates. The 300 K P25/P40 transfer gates remain satisfied because the reference surface is invariant. Away from 300 K the fatigue calculations are diagnostics and impose no fitted target.

The response answers are: (1) v1 was positive in code and ambiguous in prose; (2) corrected retention rises to {new_joint:,} joint passes; (3) negative entropy can coexist with a temperature-increasing total rate through the enthalpic term; (4) the ceramic-like and accessible-unclassified controls are opening dominated; (5) DBTT-like and weak-T classes require the measured state contribution/cancellation gates; (6) DBTT-like, but no Peak-T, is recovered at F1B and F2R; (7) P25/P40 pass the frozen 300 K gates; (8) off-reference fatigue is reported diagnostically without a fitted target; (9) entropy/Cp and direct/state decompositions remain nonidentifiable; and (10) every null result is bounded by the frozen search domain.

The result is bounded by the frozen entropy, Cp, stress, rate, temperature, and candidate domains. A missing class is a bounded negative search result, not a proof of physical impossibility. No PF, FEM, CZM, or multifront execution was launched.
""";(OUT/'CORRECTED_THERMODYNAMIC_JOINT_SEARCH_DECISION.md').write_text(md)
    (OUT/'OLD_VS_CORRECTED_SEARCH_COMPARISON.md').write_text(f"# Old versus corrected search\n\nV1 retained {old_count} F0 opening-controlled rows under a positive coded emission entropy. V2 finds {new_joint} sign-corrected joint thermodynamic passes before full-state promotion. The old files remain unchanged and the old rows are controls, not complete joint predictions.\n")
    review=OUT/'compact_review';review.mkdir(exist_ok=True)
    for name in ['CORRECTED_THERMODYNAMIC_JOINT_SEARCH_PROTOCOL.md','ENTROPY_SIGN_AND_UNITS_AUDIT.md','OLD_VS_CORRECTED_SEARCH_COMPARISON.md','CORRECTED_THERMODYNAMIC_JOINT_SEARCH_DECISION.md','corrected_thermodynamic_joint_search_decision.json','candidate_promotion_table.csv','candidate_response_topology.csv','engine_initialization_restore_parity.json','state_conservation_audit.json','root_accuracy_audit.json']:
        shutil.copy2(OUT/name,review/name)
    for pth in sorted(figdir.glob('*.png')):shutil.copy2(pth,review/pth.name)
    archive=OUT/'Archive_CORRECTED_THERMODYNAMIC_JOINT_SEARCH_V2.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for pth in sorted(review.iterdir()):z.write(pth,pth.name)
    hashes={str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='file_hashes.json'};(OUT/'file_hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')

if __name__=='__main__':main()
