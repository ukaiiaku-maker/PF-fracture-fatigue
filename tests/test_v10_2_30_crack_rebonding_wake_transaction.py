from __future__ import annotations

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls
from arrhenius_fracture.crack_rebonding_v10230 import RebondingWakeState


def _cfg(**overrides):
    fields = dict(
        wake_length_m=5.0e-6,
        wake_weight_length_m=1.0e-6,
        restored_work_of_separation_J_m2=1.0,
        rebond_K_geometry_factor=1.0,
    )
    fields.update(overrides)
    return CrackRebondingControls(**fields).validate()


def test_no_initial_wake_by_default():
    state = RebondingWakeState(_cfg())
    assert state.active == []


def test_accepted_event_creates_exactly_one_patch():
    state = RebondingWakeState(_cfg())
    state.commit_event(accepted_length_m=2.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    assert len(state.active) == 1
    assert state.active[0].length_m == pytest.approx(2.0e-7)
    assert state.active[0].s_j_m == pytest.approx(0.0)
    assert state.total_created_length_m == pytest.approx(2.0e-7)


def test_rejected_event_creates_no_patch():
    state = RebondingWakeState(_cfg())
    snap = state.snapshot()
    # a rejected event simply never calls commit_event; restoring a snapshot
    # taken before any attempt must reproduce the same (empty) state.
    state.restore(snap)
    assert state.active == []
    assert state.total_created_length_m == 0.0


def test_truncated_event_creates_patch_with_accepted_not_proposed_length():
    state = RebondingWakeState(_cfg())
    proposed_length = 5.0e-7
    accepted_length = 3.1e-7  # truncated by the energy gate
    state.commit_event(accepted_length_m=accepted_length, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    assert state.active[0].length_m == pytest.approx(accepted_length)
    assert state.active[0].length_m != pytest.approx(proposed_length)


def test_multiple_events_translate_and_insert_in_order():
    state = RebondingWakeState(_cfg())
    state.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    first_patch_id = state.active[0].patch_id
    state.commit_event(accepted_length_m=2.0e-7, event_index=1, pre_event_states=None, Eprime_Pa=2.0e11)
    assert len(state.active) == 2
    first = next(p for p in state.active if p.patch_id == first_patch_id)
    # first patch translated by the SECOND event's accepted length
    assert first.s_j_m == pytest.approx(2.0e-7)
    second = state.active[-1]
    assert second.s_j_m == pytest.approx(0.0)
    assert second.length_m == pytest.approx(2.0e-7)


def test_fresh_patch_never_inherits_pre_event_bonding():
    cfg = _cfg(fresh_surface_clean_fraction=0.3)
    state = RebondingWakeState(cfg)
    # seed one heavily-bonded existing patch
    state.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    state.active[0].p_B = 0.9
    state.active[0].p_C = 0.1
    state.active[0].p_P = 0.0
    state.commit_event(accepted_length_m=1.0e-7, event_index=1, pre_event_states=None, Eprime_Pa=2.0e11)
    new_patch = state.active[-1]
    assert new_patch.p_B == 0.0
    assert new_patch.p_C == pytest.approx(0.3)
    assert new_patch.p_P == pytest.approx(0.7)


def test_pre_event_states_applied_before_translation():
    state = RebondingWakeState(_cfg())
    state.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    pid = state.active[0].patch_id
    trial_state = np.array([0.2, 0.3, 0.5])
    state.commit_event(
        accepted_length_m=1.0e-7,
        event_index=1,
        pre_event_states={pid: trial_state},
        Eprime_Pa=2.0e11,
    )
    updated = next(p for p in state.active if p.patch_id == pid)
    assert updated.p_P == pytest.approx(0.2)
    assert updated.p_C == pytest.approx(0.3)
    assert updated.p_B == pytest.approx(0.5)
    assert updated.s_j_m == pytest.approx(1.0e-7)


def test_rollback_restores_full_wake_state():
    state = RebondingWakeState(_cfg())
    state.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    snap = state.snapshot()
    state.commit_event(accepted_length_m=2.0e-7, event_index=1, pre_event_states=None, Eprime_Pa=2.0e11)
    assert len(state.active) == 2
    state.restore(snap)
    assert len(state.active) == 1
    assert state.total_created_length_m == pytest.approx(1.0e-7)


def test_retirement_beyond_wake_length_never_touches_geometry_and_balances_totals():
    cfg = _cfg(wake_length_m=1.5e-7, wake_weight_length_m=5.0e-8)
    state = RebondingWakeState(cfg)
    state.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    # translate the first patch past the retirement threshold via a second event
    state.commit_event(accepted_length_m=2.0e-7, event_index=1, pre_event_states=None, Eprime_Pa=2.0e11)
    assert len(state.retired) == 1
    assert state.retired[0].s_j_m == pytest.approx(2.0e-7)
    total_active = state.active_length_m()
    total_retired = state.retired_length_m()
    assert total_active + total_retired == pytest.approx(state.total_created_length_m)


def test_no_event_block_commit_applies_end_states_without_new_patch():
    state = RebondingWakeState(_cfg())
    state.commit_event(accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11)
    pid = state.active[0].patch_id
    n_before = len(state.active)
    end_states = {pid: np.array([0.1, 0.2, 0.7])}
    state.commit_no_event_block(end_states, Eprime_Pa=2.0e11)
    assert len(state.active) == n_before
    assert state.active[0].p_B == pytest.approx(0.7)
    # length/s_j unchanged -- no geometry transaction occurred
    assert state.active[0].s_j_m == pytest.approx(0.0)


def test_checkpoint_round_trip_preserves_state_exactly():
    from arrhenius_fracture.crack_rebonding_v10230 import (
        serialize_rebonding_checkpoint,
        restore_rebonding_checkpoint,
    )

    class FakeEngine:
        pass

    engine = FakeEngine()
    engine._rebonding_state = RebondingWakeState(_cfg())
    engine._rebonding_state.commit_event(
        accepted_length_m=1.0e-7, event_index=0, pre_event_states=None, Eprime_Pa=2.0e11
    )
    payload = serialize_rebonding_checkpoint(engine)
    assert payload is not None

    engine2 = FakeEngine()
    engine2._rebonding_state = RebondingWakeState(_cfg())
    restore_rebonding_checkpoint(engine2, payload)

    a, b = engine._rebonding_state, engine2._rebonding_state
    assert len(a.active) == len(b.active)
    for pa, pb in zip(a.active, b.active):
        assert pa.patch_id == pb.patch_id
        assert pa.length_m == pytest.approx(pb.length_m)
        assert pa.p_B == pytest.approx(pb.p_B)
    assert a.total_created_length_m == pytest.approx(b.total_created_length_m)
    assert a.H_b == pytest.approx(b.H_b)


def test_checkpoint_serialize_returns_none_when_disabled():
    from arrhenius_fracture.crack_rebonding_v10230 import serialize_rebonding_checkpoint

    class FakeEngine:
        pass

    engine = FakeEngine()  # no _rebonding_state attribute at all
    assert serialize_rebonding_checkpoint(engine) is None
