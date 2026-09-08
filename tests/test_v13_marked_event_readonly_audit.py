from types import SimpleNamespace

import pytest

from scripts.audit_v13_marked_event_bookkeeping import audit_pair,opportunities
from arrhenius_fracture.tip_directional_observation_v11 import selected_event_owner


@pytest.mark.parametrize('case',['Peak_300K_seed3621','Peak_1000K_seed3621'])
def test_frozen_peak_marked_event_bookkeeping(case):
    result=audit_pair(case)
    assert result['outcome']=='INTENDED_MARKED_EVENT_SEMANTICS'
    assert all(result['checks'].values())


def test_missing_post_primary_times_are_not_fabricated():
    row=opportunities('Peak_1000K_seed3621')[0]
    assert row['T_primary_next_s'] is None
    assert row['T_companion_over_T_primary_next'] is None
    assert len(row['both_candidate_clocks'])==2


def test_existing_multi_tip_owner_rejection_is_a_state_boundary_not_output():
    observations=[SimpleNamespace(candidate_id='i',tip_id='daughter_i'),
                  SimpleNamespace(candidate_id='j',tip_id='daughter_j')]
    with pytest.raises(RuntimeError,match='multiple pre-event tips'):
        selected_event_owner(SimpleNamespace(member_candidate_ids=('i','j')),observations)
