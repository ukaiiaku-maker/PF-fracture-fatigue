from pathlib import Path
from scripts import v10230_dense_deltaK_supervisor as dense

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"runs/v10_2_30_four_class_qualification_7a5133f_20260804"

def test_dense_matrix_exact_launch_accounting_and_order():
    rows=dense.matrix(); assert len(rows)==32 and len({r['case'] for r in rows})==32
    assert [r['fraction'] for r in rows[::4]]==list(dense.FRACTIONS)
    assert sum(r['mode']=='fresh' for r in rows)==28 and sum(r['mode']=='resumed' for r in rows)==4
    assert not ({.55,.95}&{r['fraction'] for r in rows})
    assert all(r['Kmax_MPa_sqrt_m']==r['deltaK_MPa_sqrt_m']/0.9 for r in rows)

def test_real_0p75_checkpoints_are_exact_monotonic_horizon_resumes():
    for row in [r for r in dense.matrix() if r['fraction']==.75]:
        info=dense.inspect_resume(SOURCE,row)
        assert info['starting_cycles']==1e12 and info['old_cycle_horizon']==1e12 and info['new_cycle_horizon']==1e14
        assert all(info['checks'].values()) and info['stochastic']['rng_state']

def test_after_growth_horizon_is_terminal_censor_not_restart(tmp_path):
    case=tmp_path/"case"; out=case/"output"; out.mkdir(parents=True)
    (out/"exit_code.txt").write_text("0\n")
    import json
    (out/"developed_fatigue_growth_summary.json").write_text(json.dumps({"target_reached":False,"cycles_consumed":1e14,"event_count":3}))
    assert dense.classify(case)=="censored"

def test_run_rebinds_generic_full_precision_matrix_guard(tmp_path,monkeypatch):
    monkeypatch.setattr(dense,"validate_staged",lambda _root:{})
    observed={}
    def fake_run(_args):
        rows=dense.q.matrix(); observed['guarded']=all(dense.q.EXPECTED_MATRIX[(r['label'],r['fraction'])]==r['deltaK_MPa_sqrt_m'] for r in rows); return 0
    monkeypatch.setattr(dense.q,"run",fake_run)
    args=type('Args',(),{'minimum_free_gib':10,'no_progress_seconds':900,'recover_stale_lock':False})()
    assert dense.run(tmp_path,args)==0 and observed['guarded']
