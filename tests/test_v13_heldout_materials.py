import json
import pytest
from scripts import run_v13_heldout_materials as heldout


def test_storage_continuation_never_relaunches_or_accepts_changed_history(monkeypatch,tmp_path):
    monkeypatch.setattr(heldout,'OUT',tmp_path)
    dest=tmp_path/'ensemble';dest.mkdir();monkeypatch.setattr(heldout,'DEST',dest)
    (dest/'done').mkdir();terminal=dest/'done/terminal.json';terminal.write_text('{}')
    record={'completed_cases':[{'case':'done','terminal_sha256':heldout.sha256(terminal)}],
            'unlaunched_cases':['next']}
    (tmp_path/'HELDOUT_QUEUE_PAUSE.json').write_text(json.dumps(record))
    (dest/'queue_status.json').write_text(json.dumps(dict(active={},pending=[],paused=True)))
    (dest/'queue_pause.json').write_text(json.dumps(dict(reason='less than 1 GiB free',pending=['next'])))
    plan={'cases':['done','next']}
    assert heldout.continuation_pending(plan)==['next']
    terminal.write_text('{"changed":true}')
    with pytest.raises(RuntimeError,match='terminal changed'):heldout.continuation_pending(plan)
    terminal.write_text('{}');(dest/'next').mkdir()
    with pytest.raises(RuntimeError,match='relaunch'):heldout.continuation_pending(plan)
    (dest/'next').rmdir()
    (dest/'queue_status.json').write_text(json.dumps(dict(active={'next':123},pending=[],paused=True)))
    with pytest.raises(RuntimeError,match='drained'):heldout.continuation_pending(plan)


def test_only_heldout_classes_and_identical_conditions():
    base=json.loads((heldout.ACCEPTED/'transition_ensemble_plan.json').read_text())
    p=heldout.build_plan(base)
    assert len(set(p['cases']))==32
    assert set(p['groups'])=={'DBTT_300K','DBTT_1000K','ceramic_300K','ceramic_1000K'}
    assert all('Peak' not in c and 'weakT' not in c for c in p['cases'])
    for key in ('condition','theta_deg','rate_multiplier','family','seeds','maximum_intervals',
                'maximum_workers','right_censor_threshold_um','daughter_growth_after_first_branch_um'):
        assert p[key]==base[key]
    assert p['parameterizations']=={'DBTT':'v913_zeroD_sobol_0202500','ceramic':'oneD_v2_focused_ceramic_like_0018'}


def test_no_alternate_condition():
    base=json.loads((heldout.ACCEPTED/'transition_ensemble_plan.json').read_text())
    base['theta_deg']=30
    with pytest.raises(AssertionError):heldout.build_plan(base)


def test_worker_delegates_to_same_frozen_function(monkeypatch,tmp_path):
    c='theta15_rate1x_DBTT_300K_seed3621';calls=[]
    monkeypatch.setattr(heldout,'verify_source',lambda:{'cases':[c]})
    monkeypatch.setattr(heldout,'DEST',tmp_path/'ensemble');monkeypatch.setattr(heldout,'PLAN',tmp_path/'plan.json')
    monkeypatch.setattr(heldout.frozen_worker,'DEST',None);monkeypatch.setattr(heldout.frozen_worker,'ENSEMBLE_PLAN',None)
    monkeypatch.setattr(heldout.frozen_worker,'worker',lambda case:calls.append(case))
    heldout.worker(c)
    assert calls==[c] and heldout.frozen_worker.DEST==tmp_path/'ensemble'
    with pytest.raises(RuntimeError):heldout.worker('theta15_rate1x_Peak_300K_seed3621')
    assert calls==[c]
