import math

import numpy as np
import pytest

from arrhenius_fracture.opening_transaction_r10 import (
    MaterialCellState, StochasticOwner, commit_completed_cell,
    consume_constant_rate_successors, locate_constant_rate_event,
    update_cell_branch,
)


def test_event_far_inside_phase_accepts_prefix_not_trial_endpoint():
    owner=StochasticOwner(threshold=0.75)
    proposal=locate_constant_rate_event(owner,rate_s=10.,duration_s=1.)
    assert proposal["event"]
    assert math.isclose(proposal["consumed_time_s"],0.075)
    assert proposal["accepted_action"]==0.75
    assert proposal["discarded_trial_endpoint_action"]==10.
    assert owner.action==0.0


def test_successor_changed_rate_has_distinct_positive_wait():
    marks=iter((0.8,10.))
    owner=StochasticOwner(threshold=0.5)
    result=consume_constant_rate_successors(
        1.0,owner,lambda cell:(10. if cell==0 else 2.),lambda:next(marks))
    assert len(result["receipts"])==2
    assert np.allclose([r["accepted_time_s"] for r in result["receipts"]],
                       [0.05,0.45])
    assert owner.accepted_events==2
    assert math.isclose(owner.action,1.1)


def test_incomplete_branch_loss_cannot_commit_or_transfer_transit():
    old=MaterialCellState(4)
    update_cell_branch(old,"TRANSIT",2e-10,completion=False,physical_time_s=1e-9)
    with pytest.raises(RuntimeError,match="incomplete"):
        commit_completed_cell(old,physical_time_s=1e-9,
                              successor_branch="TRANSIT",
                              successor_separation_m=2e-10)


def test_two_explicit_mechanical_completions_can_share_time():
    first=MaterialCellState(0,branch="SEPARATION_COMPLETED",
                            separation_m=8e-10,completed=True)
    r0,second=commit_completed_cell(first,physical_time_s=2e-6,
        successor_branch="BONDED",successor_separation_m=1e-10)
    update_cell_branch(second,"SEPARATION_COMPLETED",8e-10,
                       completion=True,physical_time_s=2e-6)
    r1,third=commit_completed_cell(second,physical_time_s=2e-6,
        successor_branch="BONDED",successor_separation_m=1e-10)
    assert r0["accepted_event_time_s"]==r1["accepted_event_time_s"]==2e-6
    assert third.cell_id==2 and third.branch=="BONDED"
