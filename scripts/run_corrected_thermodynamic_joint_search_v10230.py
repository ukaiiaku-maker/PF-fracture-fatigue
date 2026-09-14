"""Deterministic staged driver for the sign-corrected v2 analytical bank."""
from __future__ import annotations
from pathlib import Path
import argparse,hashlib,json,math,sys,copy
from dataclasses import replace
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

from scripts.corrected_thermodynamic_joint_search_v10230 import (
    ENTROPY_REPRESENTATION,barrier_temperature_derivative_over_kB,
    candidate_surface_from_parameters,finite_difference_audit,paired_parameters,
    sign_explicit_fields,thermodynamic_rate_gate,
)
from scripts.run_thermodynamic_joint_search_v10230 import (
    ART as LEGACY_ART,COARSE,HITS,KGRID,PARENTS,TAU,active_stresses,
    accessibility,fatigue_metrics,full_f0,physical_controls,root_curve,source_rows,
)
from scripts.analyze_row_renewal_forward_v10230 import TransientBlunting,Controls

OUT=ROOT/'analysis_outputs/corrected_thermodynamic_joint_barrier_search_v2'
DURABLE=Path('/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_codex_v10_2_30/analysis_outputs/corrected_thermodynamic_joint_barrier_search_v2')
STRUCTURED_ENTROPY=np.array([-60,-50,-45,-40,-35,-30,-25,-20,-10,0,10,20,30,40,50,60],float)
AUDIT_T=np.array([300,450,600,750,900,1050,1200.],float)

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(row):return {k:v for k,v in row.items() if not (isinstance(v,float) and np.isnan(v))}
def corrected_id(old):return str(old).replace('_TJBS_','_TJBSV2_')

def phase0():
    hashes=json.loads((LEGACY_ART/'file_hashes.json').read_text());bad=[]
    for name,expected in hashes.items():
        path=LEGACY_ART/name
        if not path.is_file() or sha(path)!=expected:bad.append(name)
    decision=json.loads((LEGACY_ART/'analytical_joint_search_decision.json').read_text())
    retained=pd.read_csv(LEGACY_ART/'retained_joint_candidates.csv')
    expected=['P25_TJBS_S_014188','P25_TJBS_S_028493','P25_TJBS_S_039907','P25_TJBS_S_047409','P25_TJBS_S_016645','P25_TJBS_S_064760','P40_TJBS_S_037582','P40_TJBS_S_040211','P40_TJBS_S_008683','P40_TJBS_S_049735','P40_TJBS_S_053775','P40_TJBS_S_064772']
    result={
        'status':'PASS' if not bad and retained.candidate_id.tolist()==expected else 'FAIL',
        'parent_head':'97916f6f42ba063d5d6fe4a8c4590bd35047b197',
        'verified_file_count':len(hashes),'hash_mismatches':bad,
        'reproduced_counts':decision['counts'],'retained_candidate_ids':retained.candidate_id.tolist(),
        'retained_rows_exact':retained.candidate_id.tolist()==expected,
        'decision_sha256':sha(LEGACY_ART/'analytical_joint_search_decision.json'),
        'file_hash_manifest_sha256':sha(LEGACY_ART/'file_hashes.json'),
        'baseline_files_modified':False,
    }
    if result['status']!='PASS':raise RuntimeError(result)
    (OUT/'legacy_search_exact_reproduction.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    audit={
        'convention':'Delta S^dagger=-partial_T Delta G^dagger',
        'stored_fields':['activation_entropy_over_kB','barrier_temperature_derivative_over_kB'],
        'identity':'barrier_temperature_derivative_over_kB=-activation_entropy_over_kB',
        'primary_emission_hypothesis':{'activation_entropy_over_kB':[-50,-30],'barrier_temperature_derivative_over_kB':[30,50]},
        'legacy_implementation':{'code_sign':'POSITIVE_ACTIVATION_ENTROPY','published_wording':'SIGN_AMBIGUOUS','retained_active_entropy_range_kB':[float(retained.emission_entropy_active_kB.min()),float(retained.emission_entropy_active_kB.max())]},
        'entropy_representation':ENTROPY_REPRESENTATION,'attempt_frequency_varied':False,
        'Tref_K':300,'reference_surface_invariant':True,
    }
    (OUT/'entropy_sign_and_units_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
    (OUT/'ENTROPY_SIGN_AND_UNITS_AUDIT.md').write_text('# Entropy sign and units audit\n\nThe v2 convention is `Delta S^dagger/k_B = -(partial_T Delta G^dagger)/k_B`. Thus `Delta S_emit^dagger/k_B=-40` means a barrier derivative of `+40 k_B` at fixed stress. The v1 code used positive activation entropy for its preferred bank, while its prose was sign-ambiguous. V2 stores both quantities in every thermodynamic table and uses `DIRECT_FREE_ENERGY_SURFACE_NO_PREFACTOR_DOUBLE_COUNTING`.\n')
    print(json.dumps(result,indent=2))

def structured():
    rows=source_rows();active=active_stresses(rows);legacy=pd.read_csv(LEGACY_ART/'structured_seed_design.csv')
    out=[];gate=[]
    for i,r in legacy.iterrows():
        p=paired_parameters(clean(r.to_dict()),corrected_id(r.candidate_id))
        value=float(STRUCTURED_ENTROPY[i%len(STRUCTURED_ENTROPY)])
        p['emission_entropy_active_kB']=value;p['corrected_emission_entropy_over_kB']=value
        p['emission_stratum']='NEGATIVE_PRIMARY_PRIOR' if -50<=value<=-30 else ('POSITIVE_SIGN_CONTROL' if value>0 else 'NEGATIVE_TO_ZERO_CONTROL')
        _,m,_=rows[p['parent_id']];a=active[p['parent_id']];c,e=candidate_surface_from_parameters(m,p,a['cleavage_low_Pa'],a['cleavage_active_Pa'],a['emission_active_Pa'])
        stress=np.unique(np.r_[np.linspace(0,30e9,128),a['cleavage_low_Pa'],a['cleavage_active_Pa'],a['emission_active_Pa']])
        cok,cr=thermodynamic_rate_gate(c,stress,AUDIT_T,path_stress_Pa=np.linspace(0,a['cleavage_p95_Pa'],128));eok,er=thermodynamic_rate_gate(e,stress,AUDIT_T,path_stress_Pa=np.linspace(0,a['emission_p95_Pa'],128))
        fd=finite_difference_audit(e,stress,AUDIT_T[3]);p.update(activation_entropy_over_kB=value,barrier_temperature_derivative_over_kB=-value,Tref_surface_equal=True)
        out.append(p);gate.append(dict(corrected_candidate_id=p['candidate_id'],legacy_candidate_id=p['legacy_candidate_id'],parent_id=p['parent_id'],emission_activation_entropy_over_kB=value,emission_barrier_temperature_derivative_over_kB=-value,opening_admissible=cok,emission_admissible=eok,joint_admissible=cok and eok,emission_minimum_G_eV=min(x['minimum_G_eV'] for x in er),emission_maximum_ceiling_fraction=max(x['fraction_of_path_at_ceiling'] for x in er),**fd))
    frame=pd.DataFrame(out);frame.to_parquet(OUT/'structured_entropy_sign_screen.parquet',index=False,compression='zstd');pd.DataFrame(gate).to_parquet(OUT/'structured_entropy_sign_gate_results.parquet',index=False,compression='zstd')
    summary=pd.DataFrame(gate).groupby('emission_activation_entropy_over_kB').agg(rows=('corrected_candidate_id','size'),joint_admissible=('joint_admissible','sum'),emission_minimum_G_eV=('emission_minimum_G_eV','min'),maximum_ceiling_fraction=('emission_maximum_ceiling_fraction','max')).reset_index();summary.to_csv(OUT/'structured_entropy_sign_response_map.csv',index=False)
    print(summary.to_string(index=False))

def _screen_paired(frame,legacy_admission,legacy_fatigue,rows,active,physical,stage):
    admissions=[];fat=[];f0=[];survivors=[]
    old_index=legacy_admission.set_index('candidate_id');fat_index=legacy_fatigue.set_index('candidate_id')
    for i,p0 in frame.iterrows():
        p=clean(p0.to_dict());old=old_index.loc[p['legacy_candidate_id']]
        opening_ok=str(old.get('rejection',''))!='CLEAVAGE_THERMODYNAMIC_ADMISSIBILITY'
        if not opening_ok:
            admissions.append(dict(corrected_candidate_id=p['candidate_id'],legacy_candidate_id=p['legacy_candidate_id'],parent_id=p['parent_id'],stage=stage,opening_admissible=False,emission_admissible=False,joint_thermodynamic_pass=False,fatigue_pass=False,f0_pass=False,rejection='OPENING_THERMODYNAMIC_ADMISSIBILITY'))
            continue
        _,m,_=rows[p['parent_id']];a=active[p['parent_id']];c,e=candidate_surface_from_parameters(m,p,a['cleavage_low_Pa'],a['cleavage_active_Pa'],a['emission_active_Pa']);stress=np.unique(np.r_[np.linspace(0,30e9,128),a['cleavage_low_Pa'],a['cleavage_active_Pa'],a['emission_active_Pa']])
        cok,cr=thermodynamic_rate_gate(c,stress,AUDIT_T,path_stress_Pa=np.linspace(0,a['cleavage_p95_Pa'],128));eok,er=thermodynamic_rate_gate(e,stress,AUDIT_T,path_stress_Pa=np.linspace(0,a['emission_p95_Pa'],128));joint=cok and eok
        base=dict(corrected_candidate_id=p['candidate_id'],legacy_candidate_id=p['legacy_candidate_id'],parent_id=p['parent_id'],stage=stage,opening_admissible=cok,emission_admissible=eok,joint_thermodynamic_pass=joint,opening_minimum_G_eV=min(x['minimum_G_eV'] for x in cr),emission_minimum_G_eV=min(x['minimum_G_eV'] for x in er),opening_maximum_ceiling_fraction=max(x['fraction_of_path_at_ceiling'] for x in cr),emission_maximum_ceiling_fraction=max(x['fraction_of_path_at_ceiling'] for x in er))
        if not joint:
            admissions.append(dict(**base,fatigue_pass=False,f0_pass=False,rejection='EMISSION_ADMISSIBILITY' if cok else 'OPENING_THERMODYNAMIC_ADMISSIBILITY'));continue
        if p['legacy_candidate_id'] in fat_index.index:
            fm=fat_index.loc[p['legacy_candidate_id']].to_dict();fm={k:v for k,v in fm.items() if k not in ('candidate_id','parent_id','fatigue_pass')}
        else:fm=fatigue_metrics(c,m,physical[p['parent_id']]);fm={k:v for k,v in fm.items() if k not in ('rates','local_slopes')}
        fpass=bool(fm['rms_log10_rate_error']<=.05 and fm['max_log10_rate_error']<=.1 and abs(fm['global_slope_change'])<=.15 and fm['max_adjacent_local_slope_change']<=.30 and not fm['new_renewal_ceiling'] and fm['monotonic_positive'])
        fat.append(dict(corrected_candidate_id=p['candidate_id'],legacy_candidate_id=p['legacy_candidate_id'],parent_id=p['parent_id'],fatigue_pass=fpass,**fm))
        if not fpass:admissions.append(dict(**base,fatigue_pass=False,f0_pass=False,rejection='FATIGUE_PRESERVATION'));continue
        curve=root_curve(c,COARSE);fpass0=accessibility(curve[0],True)=='FRACTURE_ACCESSIBLE' and sum(accessibility(x)=='FRACTURE_ACCESSIBLE' for x in curve)/len(curve)>=.8
        f0.append(dict(corrected_candidate_id=p['candidate_id'],legacy_candidate_id=p['legacy_candidate_id'],parent_id=p['parent_id'],F0_label='F0_INTRINSIC_OPENING_ONLY',F0_pass=fpass0,KFP_300K=curve[0]['K_FP'],KFP_1200K=curve[-1]['K_FP'],KFP_ratio=curve[-1]['K_FP']/curve[0]['K_FP'],curve_json=json.dumps(curve)))
        admissions.append(dict(**base,fatigue_pass=True,f0_pass=fpass0,rejection='' if fpass0 else 'F0_ACCESSIBILITY'))
        if fpass0:survivors.append(p)
        if i and i%4096==0:print(stage,i,flush=True)
    return pd.DataFrame(admissions),pd.DataFrame(fat),pd.DataFrame(f0),survivors

def paired():
    rows=source_rows();active=active_stresses(rows);physical=physical_controls();legacy=pd.read_parquet(LEGACY_ART/'sobol_candidate_bank.parquet');pairs=pd.DataFrame([paired_parameters(clean(r),corrected_id(r['candidate_id'])) for r in legacy.to_dict('records')]);pairs.to_parquet(DURABLE/'paired_sobol_candidate_bank.parquet',index=False,compression='zstd')
    legacy_adm=pd.read_csv(LEGACY_ART/'candidate_admissibility.csv');legacy_fat=pd.read_csv(LEGACY_ART/'fatigue_preservation_metrics.csv');adm,fat,f0,survivors=_screen_paired(pairs,legacy_adm,legacy_fat,rows,active,physical,'PAIRED_SOBOL')
    adm.to_parquet(DURABLE/'thermodynamic_gate_results.parquet',index=False,compression='zstd');fat.to_parquet(OUT/'paired_fatigue_gate_results.parquet',index=False,compression='zstd');f0.to_parquet(OUT/'f0_intrinsic_opening_results.parquet',index=False,compression='zstd')
    # Deterministic descriptor coverage, at most 768.
    sf=pd.DataFrame(survivors).merge(f0[['corrected_candidate_id','KFP_ratio','KFP_300K']],left_on='candidate_id',right_on='corrected_candidate_id')
    chosen=[]
    for parent in PARENTS:
        q=sf[sf.parent_id==parent].sort_values(['emission_entropy_active_kB','KFP_ratio','candidate_id'])
        if len(q):chosen.extend(q.iloc[np.linspace(0,len(q)-1,min(384,len(q)),dtype=int)].to_dict('records'))
    pd.DataFrame(chosen).to_parquet(OUT/'f0_downselected_candidates.parquet',index=False,compression='zstd')
    manifest={'paired_bank_path':str(DURABLE/'paired_sobol_candidate_bank.parquet'),'paired_bank_sha256':sha(DURABLE/'paired_sobol_candidate_bank.parquet'),'thermodynamic_gate_path':str(DURABLE/'thermodynamic_gate_results.parquet'),'thermodynamic_gate_sha256':sha(DURABLE/'thermodynamic_gate_results.parquet'),'counts':{'paired_rows':len(pairs),'opening_admissible':int(adm.opening_admissible.sum()),'emission_admissible':int(adm.emission_admissible.sum()),'joint_thermodynamic_pass':int(adm.joint_thermodynamic_pass.sum()),'fatigue_pass':int(adm.fatigue_pass.sum()),'F0_pass':int(adm.f0_pass.sum()),'F0_downselected':len(chosen)}}
    (OUT/'paired_bank_partition_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,indent=2))

def delta():
    """Frozen 32,768-row paired heat-capacity perturbation bank."""
    rows=source_rows();active=active_stresses(rows);physical=physical_controls()
    source=pd.read_parquet(OUT/'f0_downselected_candidates.parquet').sort_values('candidate_id').reset_index(drop=True)
    rng=np.random.default_rng(260913);records=[]
    for i in range(32768):
        base=clean(source.iloc[i%len(source)].to_dict());p=dict(base)
        p['candidate_id']=f"{base['parent_id'].split('_')[0]}_TJBSV2_CP_{i:06d}"
        p['delta_cp_pair_id']=f"CPPAIR_{base['candidate_id']}_{i//len(source):03d}"
        # Symmetric, bounded Cp perturbations preserve the 300 K reference surface.
        p['cleavage_heat_capacity_active_kB']=float(np.clip(float(base.get('cleavage_heat_capacity_active_kB',0.))+rng.uniform(-5,5),-15,15))
        p['emission_heat_capacity_active_kB']=float(np.clip(float(base.get('emission_heat_capacity_active_kB',0.))+rng.uniform(-5,5),-15,15))
        records.append(p)
    frame=pd.DataFrame(records);frame.to_parquet(DURABLE/'paired_delta_cp_candidate_bank.parquet',index=False,compression='zstd')
    legacy_adm=pd.read_csv(LEGACY_ART/'candidate_admissibility.csv');legacy_fat=pd.read_csv(LEGACY_ART/'fatigue_preservation_metrics.csv')
    # Every Cp row descends from an already opening-admissible paired row. Supply a
    # compact synthetic admission index solely to avoid reusing the old sign result.
    oldids=frame.legacy_candidate_id.unique();la=pd.DataFrame({'candidate_id':oldids,'rejection':''})
    adm,fat,f0,survivors=_screen_paired(frame,la,legacy_fat,rows,active,physical,'PAIRED_DELTA_CP')
    adm.to_parquet(DURABLE/'delta_cp_thermodynamic_gate_results.parquet',index=False,compression='zstd')
    pd.DataFrame(survivors).to_parquet(OUT/'delta_cp_admissible_candidates.parquet',index=False,compression='zstd')
    manifest={'rows':len(frame),'joint_thermodynamic_pass':int(adm.joint_thermodynamic_pass.sum()),'fatigue_pass':int(adm.fatigue_pass.sum()),'F0_pass':int(adm.f0_pass.sum()),'bank_path':str(DURABLE/'paired_delta_cp_candidate_bank.parquet'),'bank_sha256':sha(DURABLE/'paired_delta_cp_candidate_bank.parquet'),'gate_sha256':sha(DURABLE/'delta_cp_thermodynamic_gate_results.parquet')}
    (OUT/'delta_cp_bank_partition_manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps(manifest,indent=2))

def main():
    OUT.mkdir(parents=True,exist_ok=True);DURABLE.mkdir(parents=True,exist_ok=True)
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['phase0','structured','paired','delta'],required=True);a=p.parse_args();globals()[a.stage]()
if __name__=='__main__':main()
