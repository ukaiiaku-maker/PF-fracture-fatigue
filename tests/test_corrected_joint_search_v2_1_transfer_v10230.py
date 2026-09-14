import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import run_corrected_joint_search_v2_1_transfer_v10230 as run


def test_v2_1_promotion_semantics_and_frozen_ranking_exist():
    out=Path('analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer')
    promotion=pd.read_csv(out/'candidate_promotion_table_v2_1.csv')
    assert 'promoted_to_F2R' not in promotion
    assert (promotion.F2R_execution_condition_count==13*promotion.executed_in_F2R.astype(int)).all()
    ranking=pd.read_csv(out/'candidate_ranking_table.csv')
    assert list(ranking.global_rank)==list(range(1,25))


def test_fixed_quantum_checkpoint_repair_is_pre_event_only(tmp_path):
    path=tmp_path/'high_cycle_live_checkpoint.json'
    path.write_text(json.dumps({'stochastic':{'hazard_event_index':0,'hazard_threshold_history':[],
        'avalanche_base_checkpoint_m':7e-9,'avalanche_event_advance_m':2e-6,
        'avalanche_event_length_factor':.4}}))
    run.repair_pre_event_fixed_quantum_checkpoint(tmp_path)
    repaired=json.loads(path.read_text())['stochastic']
    assert repaired['avalanche_event_advance_m']==7e-9
    assert repaired['avalanche_event_length_factor']==1.0
    assert json.loads((tmp_path/'checkpoint_binding_repair.json').read_text())['active_state_vector_modified'] is False


def test_fracture_transfer_uses_model_native_quantity_and_exact_renewal():
    out=Path('analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer')
    rows=pd.read_parquet(out/'production_1d_fracture_results.parquet')
    assert set(rows.reported_quantity)=={'MODEL_NATIVE_1D_FIRST_PASSAGE_K_ONSET'}
    assert np.allclose(rows.renewal_m,3.2732414351776242,rtol=0,atol=1e-15)
    assert np.allclose(rows.renewal_tau_s,6.992153587194454e-7,rtol=0,atol=1e-20)
