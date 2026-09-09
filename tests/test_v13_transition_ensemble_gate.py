import json
from types import SimpleNamespace as NS

import pytest

from scripts import run_v13_transition_ensemble as execution


def test_followup_cannot_launch_without_useful_transition(tmp_path,monkeypatch):
    p=tmp_path/'gate.json';p.write_text(json.dumps(dict(useful_conditions=[],followup_ensemble_authorized_by_gate=False)))
    out=tmp_path/'ensemble.json';monkeypatch.setattr(execution,'ENSEMBLE_PLAN',out)
    with pytest.raises(RuntimeError,match='not passed'):
        execution.prepare(p,'theta15_rate1x')
    assert not out.exists()


def test_eight_common_seeds_for_four_groups_only_after_gate(tmp_path,monkeypatch):
    p=tmp_path/'gate.json';p.write_text(json.dumps(dict(useful_conditions=['theta15_rate1x'],followup_ensemble_authorized_by_gate=True)))
    out=tmp_path/'ensemble.json';monkeypatch.setattr(execution,'ENSEMBLE_PLAN',out)
    execution.prepare(p,'theta15_rate1x')
    plan=json.loads(out.read_text())
    assert len(plan['cases'])==32 and plan['seeds']==list(range(3621,3629))
    for seed in plan['seeds']:
        assert sum(c.endswith(f'_seed{seed}') for c in plan['cases'])==4
    assert plan['right_censor_threshold_um']==75 and plan['daughter_growth_after_first_branch_um']==20
    assert plan['maximum_workers']==2 and plan['no_second_branch']


def test_theta40_followup_forbidden_even_with_supplied_positive_gate(tmp_path,monkeypatch):
    p=tmp_path/'gate.json';p.write_text(json.dumps(dict(useful_conditions=['theta40_rate1x'],followup_ensemble_authorized_by_gate=True)))
    monkeypatch.setattr(execution,'ENSEMBLE_PLAN',tmp_path/'ensemble.json')
    with pytest.raises(RuntimeError,match='theta40'):
        execution.prepare(p,'theta40_rate1x')


def cp(extension,generations):
    return NS(projected_extension_m=extension*1e-6,state=NS(crack_network=NS(
        branches=[NS(generation=g) for g in generations])))


def test_censor_is_neutral_and_only_before_first_branch():
    boundary=execution.UnbranchedBoundary()
    assert not boundary.accept(checkpoint=cp(74,[0]))
    state=cp(77,[0]);assert boundary.accept(checkpoint=state)
    assert state.projected_extension_m==77e-6
    assert not execution.UnbranchedBoundary().accept(checkpoint=cp(80,[0,1,1]))
    # Branch history persists even if only one descendant remains active.
    assert not execution.UnbranchedBoundary().accept(checkpoint=cp(80,[0,1]))
