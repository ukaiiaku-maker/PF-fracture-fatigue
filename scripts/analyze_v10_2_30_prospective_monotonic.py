"""Analysis-only no-feedback monotonic first-passage screen, no PF or FEM.

Integrates the existing analytical_screen_v1028 effective cleavage rate under
its default 0.005 MPa sqrt(m)/s ramp to unit hazard action. This is the same
no-plastic cleavage first-passage quantity used by that screening model;
no full state-resolved monotonic validation is claimed.
"""
from pathlib import Path
import csv
import json
import sys
import math
from scipy.integrate import quad
from scipy.optimize import brentq
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from arrhenius_fracture.analytical_screen_v1028 import AnalyticalMechanics,AnalyticalControl,_effective_cleavage_rate,_local_state
from arrhenius_fracture.prospective_paris_candidate_engine_v10230 import build_frozen_manifest
from arrhenius_fracture.prospective_paris_transfer_engine_v10230 import build_transfer_manifest


def main():
    freeze=json.loads((ROOT/'artifacts/prospective_paris_candidates/transfer_candidate_freeze_v1.json').read_text())
    options=['A_NATIVE']+[r['candidate_id'] for r in freeze['candidates']]
    mechanics=AnalyticalMechanics(r0_m=1e-6,sigma_cap_Pa=30e9,cleavage_hits=3.,cleavage_tau_s=1e-6,source_bin_count=2)
    control=AnalyticalControl(temperatures_K=(300.,600.,900.,1200.)).validate()
    rows=[]
    for option in options:
        manifest,_=build_frozen_manifest('A_NATIVE') if option=='A_NATIVE' else build_transfer_manifest(option)
        for T in control.temperatures_K:
            rate=lambda K:_effective_cleavage_rate(manifest,_local_state(K,mechanics)[0],T,mechanics)
            action=lambda K:quad(rate,0,K,epsabs=1e-10,epsrel=1e-9,limit=200)[0]/control.Kdot_MPa_sqrt_m_s
            reached=action(control.Kmax_MPa_sqrt_m)>=1
            Kc=brentq(lambda K:action(K)-1,0,control.Kmax_MPa_sqrt_m,xtol=1e-16,rtol=1e-12) if reached else None
            sigma=_local_state(Kc,mechanics)[0] if reached else None
            rows.append(dict(candidate_id=option,temperature_K=T,K_first_MPa_sqrt_m=Kc,
                classification='FIRST_PASSAGE' if reached else 'RAMP_CENSOR',hazard_action=action(Kc) if reached else action(control.Kmax_MPa_sqrt_m),
                Kdot_MPa_sqrt_m_s=control.Kdot_MPa_sqrt_m_s,barrier_floor_eV=manifest.cleavage.floor_fraction*manifest.cleavage.G00_eV,
                characteristic_cleavage_stress_Pa=manifest.cleavage.sigc0_Pa,sigma_first_Pa=sigma,
                stress_cap_active=bool(sigma is not None and sigma>=mechanics.sigma_cap_Pa),
                zero_stress_cleavage_rate_per_s=rate(0),zero_stress_renewal_fraction=rate(0)*mechanics.cleavage_tau_s,
                renewal_ceiling_active=rate(0)*mechanics.cleavage_tau_s>=.99,
                zero_stress_reference_action_fraction=rate(0)*Kc/control.Kdot_MPa_sqrt_m_s if reached else None,
                zero_stress_dominated_first_passage=bool(reached and rate(0)*Kc/control.Kdot_MPa_sqrt_m_s>=.99),
                topology='pending_four_temperature_comparison',
                scope='analysis_only_existing_no_plastic_monotonic_screen',PF_or_FEM_launched=False))
    native={r['temperature_K']:r['K_first_MPa_sqrt_m'] for r in rows if r['candidate_id']=='A_NATIVE'}
    for option in options:
        subset=[r for r in rows if r['candidate_id']==option]
        values=[r['K_first_MPa_sqrt_m'] for r in subset]
        topology=('nonincreasing_with_high_temperature_renewal_plateau' if all(b<=a*(1+1e-9) for a,b in zip(values,values[1:])) and any(r['renewal_ceiling_active'] for r in subset) else 'strictly_decreasing' if all(b<a for a,b in zip(values,values[1:])) else 'nonmonotonic_or_censored') if all(v is not None for v in values) else 'ramp_censored'
        for r in subset:
            r['topology']=topology;r['A_NATIVE_K_first_MPa_sqrt_m']=native[r['temperature_K']]
            r['log10_K_first_ratio_to_A_NATIVE']=math.log10(r['K_first_MPa_sqrt_m']/native[r['temperature_K']]) if r['K_first_MPa_sqrt_m'] and native[r['temperature_K']] else None
    p=ROOT/'runs/prospective_paris_transfer_v1/analysis_work/monotonic_side_effect_check_all_frozen.csv'
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
