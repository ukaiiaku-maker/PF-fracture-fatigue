"""Strict independent artifact verifier for corrected joint-search v2."""
from pathlib import Path
import hashlib,json,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

OUT=ROOT/'analysis_outputs/corrected_thermodynamic_joint_barrier_search_v2'
DURABLE=Path('/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_codex_v10_2_30/analysis_outputs/corrected_thermodynamic_joint_barrier_search_v2')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    required=['CORRECTED_THERMODYNAMIC_JOINT_SEARCH_PROTOCOL.md','corrected_joint_search_protocol.json','ENTROPY_SIGN_AND_UNITS_AUDIT.md','entropy_sign_and_units_audit.json','legacy_search_exact_reproduction.json','paired_legacy_corrected_candidate_map.csv','structured_entropy_sign_screen.parquet','f0_intrinsic_opening_results.parquet','f1a_reduced_state_results.parquet','f1b_full_process_state_results.parquet','f2r_production_replay_results.parquet','candidate_promotion_table.csv','candidate_parameter_rows.csv','candidate_response_topology.csv','opening_state_ablation_results.parquet','fatigue_temperature_diagnostics.parquet','engine_initialization_restore_parity.json','state_conservation_audit.json','thermodynamic_derivative_audit.json','root_accuracy_audit.json','determinism_audit.json','OLD_VS_CORRECTED_SEARCH_COMPARISON.md','CORRECTED_THERMODYNAMIC_JOINT_SEARCH_DECISION.md','corrected_thermodynamic_joint_search_decision.json','Archive_CORRECTED_THERMODYNAMIC_JOINT_SEARCH_V2.zip','file_hashes.json']
    missing=[x for x in required if not (OUT/x).is_file()]
    if missing:raise AssertionError(missing)
    protocol=json.loads((OUT/'corrected_joint_search_protocol.json').read_text());assert protocol['parent_head']=='97916f6f42ba063d5d6fe4a8c4590bd35047b197'
    legacy=json.loads((OUT/'legacy_search_exact_reproduction.json').read_text());assert legacy['status']=='PASS' and legacy['retained_rows_exact'] and not legacy['baseline_files_modified']
    manifest=json.loads((OUT/'paired_bank_partition_manifest.json').read_text());assert manifest['counts']=={'F0_downselected':768,'F0_pass':3345,'emission_admissible':4505,'fatigue_pass':4182,'joint_thermodynamic_pass':4182,'opening_admissible':4406,'paired_rows':131072};assert sha(DURABLE/'paired_sobol_candidate_bank.parquet')==manifest['paired_bank_sha256'];assert sha(DURABLE/'thermodynamic_gate_results.parquet')==manifest['thermodynamic_gate_sha256']
    cp=json.loads((OUT/'delta_cp_bank_partition_manifest.json').read_text());assert cp['rows']==32768 and cp['joint_thermodynamic_pass']>0 and cp['F0_pass']>0;assert sha(DURABLE/'paired_delta_cp_candidate_bank.parquet')==cp['bank_sha256']
    f0=pd.read_parquet(OUT/'f0_intrinsic_opening_results.parquet');f1a=pd.read_parquet(OUT/'f1a_reduced_state_results.parquet');f1b=pd.read_parquet(OUT/'f1b_full_process_state_results.parquet');f2=pd.read_parquet(OUT/'f2r_production_replay_results.parquet')
    assert f0.candidate_id.nunique()==24 and f1a.shape[0]==168 and f1a.candidate_id.nunique()==24
    assert f1b.shape[0]==168 and f1b.candidate_id.nunique()==24 and set(f1b.groupby('candidate_id').size())=={7};assert f2.shape[0]==78 and f2.candidate_id.nunique()==6 and set(f2.groupby('candidate_id').size())=={13}
    assert f1b.one_process_update_per_interval.all() and f1b.no_discrete_emission_clock.all();assert np.isclose(f1b.renewal_m,3.2732414351776242).all() and np.isclose(f1b.renewal_tau_s,6.992153587194454e-7,rtol=0,atol=1e-20).all()
    parity=json.loads((OUT/'engine_initialization_restore_parity.json').read_text());assert parity['status']=='PASS' and parity['rng_equal'] and all(parity['mpz_callable_identities'].values()) and all(parity['engine_callable_identities'].values()) and all(x['state_equal'] and x['outputs_equal'] for x in parity['intervals'])
    conservation=json.loads((OUT/'state_conservation_audit.json').read_text());assert conservation['status']=='PASS' and conservation['maximum_relative_residual']<1e-7
    promo=pd.read_csv(OUT/'candidate_promotion_table.csv');assert (promo.response_class=='POSITIVE_SIGN_REJECTED_CONTROL').sum()>=2;assert not promo[promo.F0_label.str.startswith('F0_',na=False)].F0_label.str.contains('PROVISIONAL_COUPLED_MODEL_RESPONSE').any();assert not promo.F1A_label.str.contains('PROVISIONAL_COUPLED_MODEL_RESPONSE').any();assert not promo[~promo.thermodynamic_admissible].promoted_to_F2R.any()
    roots=pd.read_json(OUT/'root_accuracy_audit.json');assert roots.absolute_error.max()<.001
    derivative=pd.read_json(OUT/'thermodynamic_derivative_audit.json');assert derivative.entropy_max_abs_kB.max()<3e-6 and derivative.maxwell_max_abs_m3_per_K.max()<1e-34;assert np.allclose(derivative.activation_entropy_over_kB,-derivative.barrier_temperature_derivative_over_kB)
    fatigue=pd.read_parquet(OUT/'fatigue_temperature_diagnostics.parquet');assert set(fatigue.temperature_K)=={300.,750.,1200.} and not fatigue.censor.any()
    frozen=pd.read_parquet(OUT/'paired_fatigue_gate_results.parquet');ids=set(f1b.candidate_id);frozen=frozen[frozen.corrected_candidate_id.isin(ids)]
    assert len(frozen)==24 and frozen.fatigue_pass.all()
    assert (frozen.rms_log10_rate_error<=.05).all() and (frozen.max_log10_rate_error<=.1).all() and (frozen.max_adjacent_local_slope_change<=.30).all()
    assert len(list((OUT/'figures').glob('*.png')))==14 and len(list((OUT/'figure_source_data').glob('*.csv')))==14
    decision=json.loads((OUT/'corrected_thermodynamic_joint_search_decision.json').read_text());assert decision['overall_classification'] in {'CORRECTED_SIGN_EXPLICIT_JOINT_SEARCH_COMPLETE','CORRECTED_SEARCH_COMPLETE_NO_FULLY_COUPLED_CANDIDATE'} and not decision['new_spatial_execution'] and not decision['barrier_refit_or_retune']
    hashes=json.loads((OUT/'file_hashes.json').read_text());bad=[name for name,value in hashes.items() if not (OUT/name).is_file() or sha(OUT/name)!=value];assert not bad,bad
    import scripts.complete_corrected_thermodynamic_joint_search_v10230 as complete
    complete.ROWS=complete.source_rows();complete.ACTIVE=complete.active_stresses(complete.ROWS)
    candidate=pd.read_parquet(OUT/'f0_downselected_candidates.parquet').sort_values('candidate_id').iloc[0].dropna().to_dict()
    first=complete.full_onset(candidate,300.,complete.PRIMARY_RATE,complete.XI,complete.DK);second=complete.full_onset(candidate,300.,complete.PRIMARY_RATE,complete.XI,complete.DK)
    deterministic=json.dumps(first,sort_keys=True,default=str)==json.dumps(second,sort_keys=True,default=str);assert deterministic
    det={'status':'PASS','candidate_id':candidate['candidate_id'],'common_seed':1720,'condition':[300.,complete.PRIMARY_RATE,complete.XI],'bitwise_record_equal':deterministic,'first_record_sha256':hashlib.sha256(json.dumps(first,sort_keys=True,default=str).encode()).hexdigest(),'second_record_sha256':hashlib.sha256(json.dumps(second,sort_keys=True,default=str).encode()).hexdigest()};(OUT/'determinism_audit.json').write_text(json.dumps(det,indent=2,sort_keys=True)+'\n')
    result={'status':'PASS','verifier':'STRICT_CORRECTED_SIGN_EXPLICIT_F0_F1A_F1B_F2R','counts':{'paired':131072,'delta_cp':32768,'F0_candidates':24,'F1A_conditions':168,'F1B_conditions':168,'F2R_conditions':78,'positive_sign_controls':int((promo.response_class=='POSITIVE_SIGN_REJECTED_CONTROL').sum())},'root_max_abs_error_MPa_sqrt_m':float(roots.absolute_error.max()),'state_conservation_max_relative':conservation['maximum_relative_residual'],'figures':14,'physical_2D_runs_launched':0}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
