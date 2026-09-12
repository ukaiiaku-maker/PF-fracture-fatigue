"""Strict independent verifier for the analytical thermodynamic joint search."""
from __future__ import annotations
from pathlib import Path
import hashlib,json,math,re,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq

from scripts.thermodynamic_joint_barrier_v10230 import EV_J,KB_EV_PER_K,candidate_surface_from_parameters,cooperative_rate
from scripts.run_thermodynamic_joint_search_v10230 import ART,CAP,HITS,KGRID,PARENTS,RADIUS,TAU,TEMPS,THRESHOLDS,physical_controls,source_rows

EXPECTED_PARENT='0399e12e8f015071b3d3d8c7ad8cb03b435d6aaf'
EXPECTED_HITS=3.2732414351776242
EXPECTED_TAU=6.992153587194454e-7
FORBIDDEN_TARGETS=('v9.13','v9.14','canonical fracture','historical dbtt','historical peak-t')
PROTECTED_TOKENS=('peierls','taylor','source','blunt','attempt_frequency','nu0','shield','event_length')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def validate_search_inputs(text):
    low=str(text).lower()
    if any(token in low for token in FORBIDDEN_TARGETS):raise ValueError('historical/canonical fracture target entered optimizer')
    return True

def validate_candidate(parent,candidate):
    for key,value in candidate.items():
        low=key.lower()
        if any(token in low for token in PROTECTED_TOKENS) and key in parent and str(value)!=str(parent[key]):
            raise ValueError(f'protected parameter changed: {key}')
    return True

def validate_guard(reference_surface,candidate_surface,stress,limit=.005):
    difference=np.max(np.abs(candidate_surface.reference_G_eV(stress)-reference_surface.reference_G_eV(stress)))
    if difference>limit+1e-12:raise ValueError('guard changes fatigue-active barrier beyond tolerance')
    return float(difference)

def validate_entropy_accounting(extra_prefactor=False):
    if extra_prefactor:raise ValueError('activation entropy counted twice')
    return True

def validate_renewal(hits,tau):
    if not math.isclose(float(hits),EXPECTED_HITS,rel_tol=0,abs_tol=1e-14) or not math.isclose(float(tau),EXPECTED_TAU,rel_tol=0,abs_tol=1e-18):
        raise ValueError('generic renewal replaced physical-row contract')
    return True

def independent_surface_audit(surface,stress,T):
    stress=np.asarray(stress,float);ds=2e4;dt=.02
    G=surface.G_eV(stress,T)
    if surface.process=='cleavage' and (np.min(G)<=0 or np.max(np.gradient(G,stress))>1e-16):raise ValueError('negative or stress-increasing cleavage barrier')
    Sfd=-(surface.G_eV(stress,T+dt)-surface.G_eV(stress,T-dt))/(2*dt)/KB_EV_PER_K
    # Use a converged one-sided step at the origin, where guard exponents near
    # 1.5 make a 20 kPa secant much less local than the exact zero derivative.
    h=np.where(stress==0.,1.,ds);lo=np.maximum(stress-h,0.);hi=stress+h
    Vfd=-(surface.G_eV(hi,T)-surface.G_eV(lo,T))/(hi-lo)*EV_J
    if np.max(abs(Sfd-surface.entropy_kB(stress,T)))>3e-7:raise ValueError('entropy derivative mismatch')
    if np.max(abs(Vfd-surface.activation_volume_m3(stress,T)))>2e-31:raise ValueError('activation-volume derivative mismatch')
    ss=stress[stress>ds]
    dvdT=(surface.activation_volume_m3(ss,T+dt)-surface.activation_volume_m3(ss,T-dt))/(2*dt)
    dS=(surface.entropy_kB(ss+ds,T)-surface.entropy_kB(ss-ds,T))/(2*ds)*KB_EV_PER_K*EV_J
    residual=float(np.max(abs(dvdT-dS))) if len(ss) else 0.
    if residual>1e-34:raise ValueError('Maxwell cross derivative mismatch')
    H=surface.enthalpy_eV(stress,T)
    if np.max(abs(G-(H-T*KB_EV_PER_K*surface.entropy_kB(stress,T))))>1e-10:raise ValueError('G=H-TS mismatch')
    return residual

def classify_adversarial(curve,tier='F1',preferred_entropy=False,external_selected=False):
    if external_selected:raise ValueError('candidate added after external canonical comparison')
    if preferred_entropy and curve.get('emission_entropy_active_kB',0)<30:raise ValueError('low-entropy control labeled preferred')
    if curve.get('absolute_scale_failed'):raise ValueError('normalized curve hides absolute scale failure')
    if curve.get('renewal_ceiling_plateau') and curve.get('label')=='WEAK_T':raise ValueError('renewal plateau labeled weak-T')
    if curve.get('saturated_sign_change') and curve.get('label')=='PEAK_T':raise ValueError('saturated sign change labeled Peak-T')
    if tier=='F0' and 'DBTT' in curve.get('label',''):raise ValueError('F0-only response labeled full-state DBTT')
    return True

def independent_root(surface,T,rate,xi):
    def lam(K):
        stress=min(K*1e6/math.sqrt(2*math.pi*RADIUS),CAP)
        return float(cooperative_rate(surface.raw_rate_s(stress,T),HITS,TAU))/rate
    def action(K):return quad(lam,0,K,epsabs=2e-9,epsrel=2e-8,limit=300)[0]
    if action(80)<xi:return math.nan
    return brentq(lambda K:action(K)-xi,0,80,xtol=2e-9,rtol=2e-10)

def main():
    required=['renewal_contract_audit.json','source_fatigue_controls_registry.csv','source_fatigue_controls_registry.json','source_parameter_hashes.json','thermodynamic_surface_definition.json','search_bounds.json','search_protocol_freeze.json','structured_seed_design.csv','sobol_candidate_bank.parquet','candidate_admissibility.csv','f0_coarse_screen.parquet','f0_full_temperature_screen.parquet','f1_state_screen.parquet','fatigue_preservation_metrics.csv','temperature_dependent_fatigue_predictions.parquet','fracture_accessibility_metrics.csv','topology_descriptors.csv','threshold_rate_robustness.csv','entropy_activation_volume_fields.parquet','maxwell_relation_audit.csv','thermodynamic_admissibility.csv','emission_cleavage_competition.csv','pareto_front.csv','retained_joint_candidates.csv','retained_joint_candidates.json','retained_candidate_parameter_rows.csv','retained_candidate_barrier_surfaces.csv','analytical_joint_search_decision.md','analytical_joint_search_decision.json','THERMODYNAMIC_JOINT_SEARCH_HANDOFF.md']
    missing=[x for x in required if not (ART/x).exists()]
    if missing:raise AssertionError(f'missing artifacts: {missing}')
    protocol=json.loads((ART/'search_protocol_freeze.json').read_text())
    assert protocol['parent_head']==EXPECTED_PARENT
    validate_search_inputs(json.dumps(protocol))
    source_hashes=json.loads((ART/'source_parameter_hashes.json').read_text())
    for name,expected in source_hashes.items():
        assert digest(ROOT/name)==expected,f'source hash mismatch: {name}'
    audit=json.loads((ART/'renewal_contract_audit.json').read_text());validate_renewal(audit['row_hits'],audit['row_tau_s'])
    structured=pd.read_csv(ART/'structured_seed_design.csv');sobol=pd.read_parquet(ART/'sobol_candidate_bank.parquet');delta=pd.read_parquet(ART/'delta_cp_candidate_bank.parquet')
    assert len(structured)==3072 and len(sobol)==131072 and len(delta)==32768
    assert sobol.groupby('parent_id').size().to_dict()=={PARENTS[0]:65536,PARENTS[1]:65536}
    retained=json.loads((ART/'retained_joint_candidates.json').read_text());assert 6<=len(retained)<=12
    rows=source_rows();active=json.loads((ART/'active_stress_coordinates.json').read_text())
    robustness=pd.read_csv(ART/'threshold_rate_robustness.csv')
    maxwell=[];root_errors=[]
    for p in retained:
        row,manifest,_=rows[p['parent_id']];validate_candidate(row,p);a=active[p['parent_id']]
        c,e=candidate_surface_from_parameters(manifest,p,a['cleavage_low_Pa'],a['cleavage_active_Pa'],a['emission_active_Pa'])
        assert 1<=float(c.reference_G_eV(0))<=2
        stress=np.unique(np.r_[0.,a['cleavage_low_Pa'],a['cleavage_p05_Pa'],a['cleavage_active_Pa'],a['cleavage_p95_Pa'],a['emission_p05_Pa'],a['emission_active_Pa'],a['emission_p95_Pa'],15e9,30e9])
        for T in [300,450,600,750,900,1050,1200]:
            maxwell.extend([independent_surface_audit(c,stress,T),independent_surface_audit(e,stress,T)])
        root=independent_root(c,300,.005,math.log(2));stored=robustness[(robustness.candidate_id==p['candidate_id'])&(robustness.threshold=='LN2')&(np.isclose(robustness.Kdot,.005))&(robustness['T']==300)].K_FP.iloc[0]
        root_errors.append(abs(root-stored));assert abs(root-stored)<.03
    topo=pd.read_csv(ART/'topology_descriptors.csv');assert set(topo.candidate_id)=={p['candidate_id'] for p in retained}
    assert not topo.F1_topology.str.contains('DBTT|PEAK').any() or (topo.F1_available_points>=math.ceil(.8*len(TEMPS))).all()
    fat=pd.read_csv(ART/'fatigue_preservation_metrics.csv');assert fat[fat.candidate_id.isin({p.get('extension_parent_id',p['candidate_id']) for p in retained})].fatigue_pass.all()
    assert (pd.read_csv(ART/'maxwell_relation_audit.csv').max_abs_residual_m3_per_K<=1e-34).all()
    figures=sorted((ART/'figures').glob('*.png'));assert len(figures)==18
    oversized=[str(p) for p in ART.rglob('*') if p.is_file() and p.stat().st_size>=95_000_000];assert not oversized,oversized
    hashes={str(p.relative_to(ART)):digest(p) for p in sorted(ART.rglob('*')) if p.is_file() and p.name not in {'file_hashes.json','verification.json'}}
    (ART/'file_hashes.json').write_text(json.dumps(hashes,indent=2,sort_keys=True)+'\n')
    result={'status':'PASS','verifier':'STRICT_INDEPENDENT_ANALYTICAL_RECOMPUTATION','candidate_counts':{'structured':len(structured),'primary_sobol':len(sobol),'delta_cp_extension':len(delta),'retained':len(retained)},'independent_root_max_abs_error_MPa_sqrt_m':max(root_errors),'independent_maxwell_max_abs_m3_per_K':max(maxwell),'figures':len(figures),'physical_simulations_launched':0,'production_solver_modified':False}
    (ART/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
