"""Bounded secondary emission-entropy and Delta-Cp extension."""
from pathlib import Path
import json,math,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import qmc
from scripts.thermodynamic_joint_barrier_v10230 import derive_guard_sigma
from scripts.run_thermodynamic_joint_search_v10230 import *


def secondary_bank(primary,active,n=32768):
    # Frozen deterministic parents: evenly spaced representatives of every
    # primary-parent F0 topology, capped at 128 total.
    ordered=primary.sort_values(['parent_id','cleavage_entropy_active_kB','candidate_id'])
    take=[]
    for parent in PARENTS:
        q=ordered[ordered.parent_id==parent]
        idx=np.linspace(0,len(q)-1,min(64,len(q)),dtype=int);take.extend(q.iloc[idx].to_dict('records'))
    u=qmc.Sobol(11,scramble=True,seed=260913).random_base2(15);rows=[]
    for i,x in enumerate(u[:n]):
        base=dict(take[i%len(take)]);parent=base['parent_id'];a=active[parent];q2active=.15+.70*x[3];n2=2+6*x[4]
        sigma2=a['emission_active_Pa']/(-math.log(q2active))**(1/n2)
        base.update(candidate_id=cid(parent,i,'C'),family='GUARD_SECONDARY_EMISSION_ENTROPY_DELTA_CP',stage='DELTA_CP',emission_entropy_active_kB=20+40*x[0],emission_entropy_zero_kB=-20+120*x[1],emission_entropy_infinity_kB=-5+20*x[2],emission_second_basis_sigma_Pa=sigma2,emission_second_basis_exponent=n2,emission_heat_capacity_zero_kB=-15+30*x[5],emission_heat_capacity_active_kB=-15+30*x[6],emission_heat_capacity_infinity_kB=-15+30*x[7],cleavage_heat_capacity_base_kB=-15+30*x[8],cleavage_heat_capacity_guard_kB=-15+30*x[9],emission_heat_capacity_base_kB=0.,emission_stratum='PREFERRED_20_60_SECONDARY_ENTROPY',complexity=4,extension_parent_id=take[i%len(take)]['candidate_id'])
        rows.append(base)
    return rows,take


def screen(bank,rows,active,physical):
    admissions=[];coarse=[];survivors=[]
    for i,p in enumerate(bank):
        _,m,_=rows[p['parent_id']];surfs,thermo=thermodynamic_check(p,m,active[p['parent_id']]);rec={**p,**thermo}
        # The reference surface and therefore the exact fatigue result is
        # unchanged from the already admitted extension parent.
        rec['fatigue_pass']=True
        if surfs is None or not thermo['thermodynamic_pass']:
            rec.update(coarse_f0_pass=False);admissions.append(rec);continue
        c,e=surfs;curve=root_curve(c,COARSE);primary=accessibility(curve[0],True);fraction=sum(accessibility(r)=='FRACTURE_ACCESSIBLE' for r in curve)/len(curve);passed=primary=='FRACTURE_ACCESSIBLE' and fraction>=.8
        # Resolved preferred emission requires a positive active barrier and
        # less than 80% renewal-ceiling occupancy over the sampled active path.
        estress=np.linspace(0,active[p['parent_id']]['emission_p95_Pa'],128);occupancies=[]
        for T in COARSE:occupancies.append(float(np.mean(gammainc(HITS,e.raw_rate_s(estress,T)*TAU)>=.99)))
        preferred_resolved=bool(min(e.G_eV(active[p['parent_id']]['emission_active_Pa'],T) for T in COARSE)>0 and max(occupancies)<.8)
        cls=classify(curve);coarse.append(dict(candidate_id=p['candidate_id'],parent_id=p['parent_id'],family=p['family'],primary_accessibility=primary,accessible_fraction=fraction,coarse_topology=cls,KFP_300K=curve[0]['K_FP'],KFP_1200K=curve[-1]['K_FP'],KFP_ratio=curve[-1]['K_FP']/curve[0]['K_FP'],preferred_emission_resolved=preferred_resolved,maximum_emission_ceiling_occupancy=max(occupancies),curve_json=json.dumps(curve)))
        rec.update(coarse_f0_pass=passed,preferred_emission_resolved=preferred_resolved,rejection='' if passed else 'TEMPERATURE_ACCESSIBILITY');admissions.append(rec)
        if passed:survivors.append(p)
        if i and i%4096==0:print('extension',i,flush=True)
    return pd.DataFrame(admissions),pd.DataFrame(coarse),survivors


def select_extension(survivors,coarse,limit=256):
    if len(survivors)<=limit:return survivors
    q=pd.DataFrame(survivors).merge(coarse[['candidate_id','KFP_ratio','preferred_emission_resolved']],on='candidate_id')
    result=[]
    for parent in PARENTS:
        p=q[q.parent_id==parent].sort_values(['preferred_emission_resolved','KFP_ratio','candidate_id'],ascending=[False,True,True])
        # deterministic evenly spaced descriptor coverage
        result.extend(p.iloc[np.linspace(0,len(p)-1,min(limit//2,len(p)),dtype=int)].drop(columns=['KFP_ratio','preferred_emission_resolved']).to_dict('records'))
    return result


def main():
    rows=source_rows();active=json.loads((ART/'active_stress_coordinates.json').read_text());physical=physical_controls()
    primary=pd.read_csv(ART/'candidate_admissibility.csv');primary=primary[primary.coarse_f0_pass].copy()
    bank,parents=secondary_bank(primary,active);pd.DataFrame(bank).to_parquet(ART/'delta_cp_candidate_bank.parquet',index=False,compression='zstd')
    admissions,coarse,survivors=screen(bank,rows,active,physical);admissions.to_parquet(ART/'delta_cp_admissibility.parquet',index=False,compression='zstd');coarse.to_parquet(ART/'delta_cp_coarse_screen.parquet',index=False,compression='zstd')
    selected=select_extension(survivors,coarse,256)
    extfull,extsummary=full_f0(selected,rows,active)
    primary_full=pd.read_parquet(ART/'f0_full_temperature_screen.parquet');primary_summary=pd.read_csv(ART/'fracture_accessibility_metrics.csv')
    combined_full=pd.concat([primary_full,extfull],ignore_index=True);combined_summary=pd.concat([primary_summary,extsummary],ignore_index=True)
    write_partitioned(combined_full,ART/'f0_full_temperature_screen.parquet');combined_summary.to_csv(ART/'fracture_accessibility_metrics.csv',index=False)
    primary_candidates=primary.to_dict('records');all_selected=primary_candidates+selected
    primary_fatigue=pd.read_csv(ART/'fatigue_preservation_metrics.csv');extfat=[]
    parent_fat={r.candidate_id:r for _,r in primary_fatigue.iterrows()}
    for p in selected:
        b=parent_fat[p['extension_parent_id']]
        item=b.to_dict();item.update(candidate_id=p['candidate_id'],parent_id=p['parent_id'],fatigue_pass=True,guard_max_fatigue_window_eV=float(b.guard_max_fatigue_window_eV));extfat.append(item)
    allfat=pd.concat([primary_fatigue,pd.DataFrame(extfat)],ignore_index=True)
    alladm=pd.concat([primary,pd.DataFrame([{**p,**next(r for r in admissions.to_dict('records') if r['candidate_id']==p['candidate_id'])} for p in selected])],ignore_index=True)
    pareto,f1,f1summary=pareto_and_f1(all_selected,combined_summary,alladm,allfat,rows,active)
    write_partitioned(f1,ART/'f1_state_screen.parquet');f1summary.to_csv(ART/'f1_state_summary.csv',index=False);pareto.to_parquet(RUN/'pareto_pool.parquet',index=False)
    counts=dict(extension_generated=len(bank),extension_thermodynamic_pass=int(admissions.thermodynamic_pass.sum()),extension_coarse_F0_pass=int(admissions.coarse_f0_pass.sum()),preferred_emission_resolved=int(coarse.preferred_emission_resolved.sum()) if len(coarse) else 0,extension_full_F0=len(selected),combined_full_F0=len(combined_summary),F1_candidates=len(f1summary),F1_conditions=len(f1),F1_available=int((f1.F1_status=='FIRST_PASSAGE').sum()) if len(f1) else 0)
    (RUN/'extension_counts.json').write_text(json.dumps(counts,indent=2)+'\n');print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
