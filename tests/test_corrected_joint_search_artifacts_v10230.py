import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis_outputs/corrected_thermodynamic_joint_barrier_search_v2'

def test_full_fidelity_condition_contract():
    a=pd.read_parquet(OUT/'f1a_reduced_state_results.parquet');b=pd.read_parquet(OUT/'f1b_full_process_state_results.parquet');r=pd.read_parquet(OUT/'f2r_production_replay_results.parquet')
    assert (len(a),len(b),len(r))==(168,168,78)
    assert set(b.groupby('candidate_id').size())=={7}
    assert set(r.groupby('candidate_id').size())=={13}

def test_initialized_restored_engine_and_intervals_match():
    p=json.loads((OUT/'engine_initialization_restore_parity.json').read_text())
    assert p['status']=='PASS' and p['rng_equal']
    assert all(p['mpz_callable_identities'].values())
    assert all(x['state_equal'] and x['outputs_equal'] for x in p['intervals'])

def test_production_rows_use_exact_renewal_contract_and_no_emission_clock():
    b=pd.read_parquet(OUT/'f1b_full_process_state_results.parquet')
    assert np.isclose(b.renewal_m,3.2732414351776242).all()
    assert np.isclose(b.renewal_tau_s,6.992153587194454e-7,rtol=0,atol=1e-20).all()
    assert b.no_discrete_emission_clock.all()

def test_censored_or_missing_states_are_never_zero_filled():
    b=pd.read_parquet(OUT/'f1b_full_process_state_results.parquet')
    failed=b.F1B_status!='FIRST_PASSAGE'
    assert b.loc[failed,'K_onset_MPa_sqrt_m'].isna().all()

def test_positive_sign_controls_are_retained_but_not_promoted():
    p=pd.read_csv(OUT/'candidate_promotion_table.csv');q=p[p.response_class=='POSITIVE_SIGN_REJECTED_CONTROL']
    assert len(q)>=2 and not q.promoted_to_F2R.any() and not q.thermodynamic_admissible.any()

def test_root_and_conservation_audits_pass():
    roots=pd.read_json(OUT/'root_accuracy_audit.json');state=json.loads((OUT/'state_conservation_audit.json').read_text())
    assert roots.absolute_error.max()<1e-3
    assert state['status']=='PASS' and state['maximum_relative_residual']<1e-7
