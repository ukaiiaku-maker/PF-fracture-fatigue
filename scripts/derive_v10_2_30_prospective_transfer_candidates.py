"""Bounded design against a once-frozen transfer model; analysis, never solver law."""
from pathlib import Path
import csv
import json
import math
import sys
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import build_frozen_manifest, digest
from arrhenius_fracture.inverse_fatigue_barrier_design_v10230 import (
    ExpFloorBounds, RenewalControls, barrier_from_vector, cycle_growth_and_slope,
)
ART=ROOT/'artifacts/prospective_paris_candidates'
GRID=np.array([12,12.75,13.5,15,16.5,18,19.5,21,24.3])
WINDOW=GRID[(GRID>=13.5)&(GRID<=21)]
FIELDS=['cleave_G00_eV','cleave_sigc0_GPa','cleave_exp_a','cleave_exp_n','cleave_floor_frac']
BOUNDS=ExpFloorBounds(floor_fraction=(.001,.95))
STARTS=[[.4527,2.079,.2126,1.577,.8725],[.48,2,.42,.91,.79],[1,2,.4,1,.90],
        [1.8,2.1,.42,2.9,.78],[.6,3,.6,1.2,.70]]


_MANIFEST = build_frozen_manifest("A_NATIVE")[0]


def prediction(vector,K,model,R=.1):
    radius=10**(model['log10_radius_coefficients'][0]+model['log10_radius_coefficients'][1]*math.log10(K/18))
    correction=10**(model['log10_rate_correction_coefficients'][0]+model['log10_rate_correction_coefficients'][1]*math.log10(K/18))
    barrier=barrier_from_vector(vector,_MANIFEST.cleavage)
    controls=RenewalControls(hits=model['cleavage_hits'],tau_s=model['cleavage_tau_s'],
        temperature_K=300,frequency_Hz=1000,event_length_m=5e-6,n_phase=4096,radius_m=radius)
    answer=cycle_growth_and_slope(barrier,K,R,controls)
    return answer['da_dN']*correction


def main():
    transfer_path=ART/'p40_transfer_update.json'
    model=json.loads(transfer_path.read_text())
    assert model['transfer_update_number']==1
    # The transfer coefficients must already exist in the Git HEAD.
    import subprocess
    assert subprocess.check_output(['git','show','HEAD:artifacts/prospective_paris_candidates/p40_transfer_update.json'],cwd=ROOT)==transfer_path.read_bytes()
    target_rows=list(csv.DictReader((ART/'target_rate_profiles.csv').open()))
    results=[]
    for target,m in [('P40',4.0),('P25',2.5),('P55',5.5)]:
        relevant=[r for r in target_rows if r['profile_id']==target+'_STANDARD_WINDOW']
        ks=np.array([float(r['Kmax_MPa_sqrt_m']) for r in relevant]); rates=np.array([float(r['g_target_m_per_cycle']) for r in relevant]);order=np.argsort(ks);ks=ks[order];rates=rates[order]
        desired=np.exp(np.interp(np.log(WINDOW),np.log(ks),np.log(rates)))
        target_slopes=np.diff(np.log(desired))/np.diff(np.log(WINDOW))
        def residual(v):
            pred=np.array([prediction(v,K,model) for K in WINDOW])
            slopes=np.diff(np.log(pred))/np.diff(np.log(WINDOW))
            return np.r_[np.log10(pred/desired)/.05,(slopes-target_slopes)/.25]
        candidates=[]
        for index,start in enumerate(STARTS):
            fit=least_squares(residual,start,bounds=(BOUNDS.lower(),BOUNDS.upper()),max_nfev=300,
                              ftol=1e-10,xtol=1e-10,gtol=1e-10)
            v=fit.x; margin=np.minimum(v-BOUNDS.lower(),BOUNDS.upper()-v)/(BOUNDS.upper()-BOUNDS.lower())
            pred=np.array([prediction(v,K,model) for K in WINDOW]);slopes=np.diff(np.log(pred))/np.diff(np.log(WINDOW))
            err=np.log10(pred/desired);serr=slopes-target_slopes
            eligible=bool(np.min(margin)>=1e-4 and np.max(abs(err))<=.20 and np.sqrt(np.mean(err**2))<=.10 and np.max(abs(serr))<=.75 and np.sqrt(np.mean(serr**2))<=.5)
            candidates.append(dict(start_index=index,vector=v.tolist(),cost=float(np.sum(residual(v)**2)),interior=bool(np.min(margin)>=1e-4),bound_margins=margin.tolist(),eligible=eligible,success=bool(fit.success),nfev=fit.nfev,max_rate_error=float(np.max(abs(err))),max_slope_error=float(np.max(abs(serr))),window_predictions=pred.tolist(),window_target=desired.tolist()))
        candidates.sort(key=lambda c:c['cost'])
        results.append(dict(target=target,target_middle_slope=m,family='single_EXP_floor',fits=candidates,
            selected=next((c for c in candidates if c['eligible']),None),transfer_model_sha256=digest(model)))
    output=ART/'bounded_single_exp_derivation.json'
    output.write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps([dict(target=r['target'],eligible=r['selected'] is not None,best=r['fits'][0]) for r in results],indent=2))


if __name__=='__main__':main()
