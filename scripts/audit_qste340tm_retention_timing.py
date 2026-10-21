#!/usr/bin/env python3
"""Compare native full-cycle retention with a birth-phase survival quadrature."""
from __future__ import annotations
import json,math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.refit_qste340tm_capacity_policy import DATA,OUT,build_records,evaluate,stress_cycle,unpack
from scripts.calibrate_qste340tm_stateful_exp_floor import NOMINAL_FREQUENCY_HZ,TEMPERATURE_K,exp_floor_barrier

def main():
 records=build_records(DATA);fit=json.load(open(OUT/'fit_fixed_1000.json'));x=np.array(fit['parameters_vector']);p=unpack(x)
 state=evaluate(x,records,'fixed_1000');rows=[]
 for R in (.1,.3,.5,.7):
  ids=[i for i,r in enumerate(records) if float(r['R'])==R]
  for label,i in [('transition',ids[len(ids)//2]),('upper',ids[-1])]:
   s,_,b=stress_cycle(float(records[i]['DeltaK_MPa_sqrt_m']),R,float(state['retained'][i]));se=np.maximum(s-b,0.)
   emit=exp_floor_barrier('audit_emit',p['emission_G0_298_eV'],p['plastic_sigc_298_Pa'],1e11,exp_n=p['plastic_EXP_floor_n'])
   pe=exp_floor_barrier('audit_peierls',p['escape_G0_298_eV'],p['plastic_sigc_298_Pa'],1e12,exp_n=p['plastic_EXP_floor_n'])
   ta=exp_floor_barrier('audit_taylor',p['escape_G0_298_eV'],p['plastic_sigc_298_Pa'],1e11,exp_n=p['plastic_EXP_floor_n'])
   le=emit.rate(se,TEMPERATURE_K);lp=pe.rate(se,TEMPERATURE_K);lt=ta.rate(se,TEMPERATURE_K);lx=1/(1/np.maximum(lp,1e-300)+1/np.maximum(lt,1e-300))
   dt=1/(NOMINAL_FREQUENCY_HZ*len(le));mu_e=float(np.sum(le)*dt);mu_x=float(np.sum(lx)*dt)
   native=mu_e*math.exp(-min(mu_x,700.));remaining=np.cumsum((lx*dt)[::-1])[::-1]
   phase_resolved=float(np.sum(le*dt*np.exp(-np.minimum(remaining,700.))))
   rows.append({'R':R,'location':label,'record_index':i,'DeltaK_MPa_sqrt_m':records[i]['DeltaK_MPa_sqrt_m'],'retained_count':float(state['retained'][i]),
                'mu_emit':mu_e,'mu_escape':mu_x,'native_full_cycle_store':native,'birth_phase_store':phase_resolved,
                'birth_phase_over_native':phase_resolved/max(native,1e-300)})
 payload={'schema':'qste340tm-retention-timing-audit/1','native_identity':'mu_emit*exp(-mu_escape), verified in arrhenius_fracture/fatigue_v1.py',
          'alternative_scope':'diagnostic midpoint-phase birth survival; not substituted into fit because it is not the current native process equation','records':rows}
 (OUT/'retention_timing_audit.json').write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
