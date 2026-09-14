"""Strict verifier for the V2.1 review and bounded production 1-D transfer."""
from pathlib import Path
import hashlib, json, subprocess, sys, zipfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd

OUT=ROOT/'analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer'
PARENT='97916f6f42ba063d5d6fe4a8c4590bd35047b197'
V2='6d56fd936699d244004d5e66c35d3718d8fcc357'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    required=['CORRECTED_JOINT_SEARCH_V2_1_REVIEW.md','corrected_joint_search_v2_1_decision.json','candidate_promotion_table_v2_1.csv','candidate_parameter_rows_full.csv','candidate_ranking_protocol.json','candidate_ranking_table.csv','candidate_selection_decision.md','source_diff_manifest.json','v2_archive_verification.json','production_candidate_manifest.json','production_candidate_rows.csv','production_candidate_source_hashes.json','production_row_loading_audit.json','production_engine_sentinel_audit.json','production_1d_fracture_results.parquet','production_1d_fracture_results.csv','production_1d_fracture_state_at_onset.parquet','production_1d_fracture_transfer_comparison.csv','PRODUCTION_1D_FRACTURE_TRANSFER_DECISION.md','production_300K_fatigue_case_table.csv','production_300K_fatigue_event_ledger.parquet','production_300K_fatigue_local_slopes.csv','production_300K_fatigue_transfer_comparison.csv','PRODUCTION_300K_FATIGUE_PILOT_DECISION.md','optional_tier2_fatigue_plan.csv','corrected_joint_search_v2_1_final_decision.json','CORRECTED_JOINT_SEARCH_V2_1_AND_1D_TRANSFER_DECISION.md','Archive_CORRECTED_JOINT_SEARCH_V2_1_AND_1D_TRANSFER.zip','SHA256_MANIFEST.json']
    missing=[x for x in required if not (OUT/x).is_file()]
    if missing:raise AssertionError(f'missing outputs: {missing}')
    source=json.loads((OUT/'source_diff_manifest.json').read_text())
    assert source['published_production_head']==PARENT and source['corrected_search_record_head']==V2 and source['published_is_ancestor']
    assert source['production_physics_difference_status']=='PASS_NO_PRODUCTION_PHYSICS_DIFFERENCE'
    assert not source['groups']['production_constitutive_physics_files']
    assert json.loads((OUT/'v2_archive_verification.json').read_text())['status']=='PASS'
    promotion=pd.read_csv(OUT/'candidate_promotion_table_v2_1.csv')
    assert 'promoted_to_F2R' not in promotion and {'eligible_for_F2R','executed_in_F2R','F2R_execution_condition_count'}<=set(promotion)
    assert set(promotion.loc[promotion.executed_in_F2R,'F2R_execution_condition_count'])=={13}
    assert (promotion.loc[~promotion.executed_in_F2R,'F2R_execution_condition_count']==0).all()
    review=json.loads((OUT/'corrected_joint_search_v2_1_decision.json').read_text())
    assert review['F2R_conditions_total']==78
    assert set(review['F2R_candidate_counts'].values())=={2} and set(review['F2R_condition_counts'].values())=={26}
    full=pd.read_csv(OUT/'candidate_parameter_rows_full.csv')
    assert len(full)==24 and full.candidate_id.nunique()==24
    for c in ['opening_surface_json','emission_surface_json','parent_complete_registry_row_json','parent_material_manifest_json','physics__cleavage_hits','physics__cleavage_correlation_time_s','opening_attempt_frequency_s','emission_attempt_frequency_s','complete_bound_row_sha256']:
        assert c in full and full[c].notna().all(),c
    assert np.allclose(full.physics__cleavage_hits,3.2732414351776242,rtol=0,atol=1e-15)
    assert np.allclose(full.physics__cleavage_correlation_time_s,6.992153587194454e-7,rtol=0,atol=1e-20)
    ranking=pd.read_csv(OUT/'candidate_ranking_table.csv')
    assert list(ranking.global_rank)==list(range(1,25)) and ranking.candidate_id.nunique()==24
    p40=json.loads((OUT/'p40_f2r_transfer_results/p40_f2r_transfer_decision.json').read_text())
    assert p40['preregistered_P40_DBTT_candidates']==['P40_TJBSV2_S_038503','P40_TJBSV2_S_006687']
    assert p40['P40_DBTT_status']=='P40_DBTT_F2R_TRANSFER_CONFIRMED' and p40['P40_weak_T_status']=='P40_WEAK_T_F2R_TRANSFER_CONFIRMED'
    p40rows=pd.read_parquet(OUT/'p40_f2r_transfer_results/p40_f2r_transfer_results.parquet')
    assert len(p40rows)==39 and set(p40rows.groupby('candidate_id').size())=={13}
    manifest=json.loads((OUT/'production_candidate_manifest.json').read_text())['candidates']
    assert len(manifest)==8 and len({x['candidate_id'] for x in manifest})==8
    assert json.loads((OUT/'production_row_loading_audit.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'production_engine_sentinel_audit.json').read_text())['status']=='PASS'
    fracture=pd.read_parquet(OUT/'production_1d_fracture_results.parquet')
    assert len(fracture)==128 and fracture.candidate_id.nunique()==8 and (fracture.reported_quantity=='MODEL_NATIVE_1D_FIRST_PASSAGE_K_ONSET').all()
    assert (fracture.threshold_action==np.log(2)).all() and (fracture.conservation_relative<1e-7).all()
    fdecision=json.loads((OUT/'production_1d_fracture_transfer_decision.json').read_text())['rows']
    assert len(fdecision)==8 and all(x['transfer_status']=='PRODUCTION_1D_RESPONSE_TOPOLOGY_REPRODUCED' for x in fdecision)
    fatigue=pd.read_csv(OUT/'production_300K_fatigue_case_table.csv')
    assert len(fatigue)==12 and fatigue.candidate_id.nunique()==4
    assert set(fatigue.Kmax_fraction)=={.55,.75,.95} and (fatigue.hazard_seed==1001721).all()
    assert (fatigue.R==.1).all() and (fatigue.frequency_Hz==1000).all() and (fatigue.fixed_event_quantum_m==7e-9).all()
    assert fatigue.no_hidden_fatigue_floor.all() and fatigue.no_bulk_hazard.all()
    assert np.allclose(fatigue.DeltaK_MPa_sqrt_m,.9*fatigue.Kmax_MPa_sqrt_m)
    assert not np.isfinite(fatigue.developed_da_dN).any()
    slopes=pd.read_csv(OUT/'production_300K_fatigue_local_slopes.csv')
    assert len(slopes)==4 and (slopes.classification=='FATIGUE_SLOPE_UNRESOLVED_CENSORED').all()
    tier2=pd.read_csv(OUT/'optional_tier2_fatigue_plan.csv');assert len(tier2)==6 and tier2.status.str.contains('NOT_EXECUTED').all()
    final=json.loads((OUT/'corrected_joint_search_v2_1_final_decision.json').read_text())
    assert final['classification']=='PRODUCTION_1D_COUPLED_RESPONSE_CONFIRMED'
    assert final['fatigue_classification'] in {'PRODUCTION_1D_STATE_CLOSURE_FAILED','PRODUCTION_1D_CENSORED_IN_TESTED_DOMAIN'}
    assert not final['barrier_refit_or_retune'] and not final['tier2_executed'] and not final['two_dimensional_PF_FEM_CZM_executed']
    assert len(list((OUT/'figures').glob('*.png')))==10 and len(list((OUT/'figure_source_data').glob('*.csv')))==10
    hashes=json.loads((OUT/'SHA256_MANIFEST.json').read_text())['files']
    bad=[name for name,value in hashes.items() if not (OUT/name).is_file() or sha(OUT/name)!=value]
    assert not bad,bad
    with zipfile.ZipFile(OUT/'Archive_CORRECTED_JOINT_SEARCH_V2_1_AND_1D_TRANSFER.zip') as archive:
        assert set(hashes)<=set(archive.namelist())
        assert all(hashlib.sha256(archive.read(name)).hexdigest()==value for name,value in hashes.items())
    result={'status':'PASS','verifier':'STRICT_V2_1_REVIEW_AND_PRODUCTION_1D_TRANSFER','review_candidates':24,'p40_F2R_conditions':39,'production_candidates':8,'fracture_conditions':128,'fatigue_trajectories':12,'figures':10,'two_dimensional_runs':0,'tier2_runs':0}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n');print(json.dumps(result,indent=2,sort_keys=True))
if __name__=='__main__':main()
