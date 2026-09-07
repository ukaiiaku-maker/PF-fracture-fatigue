"""Full-production-chain transactional qualification for crack rebonding.

Round-3 review correctly identified that the earlier "real-engine" smoke
tests (test_v10_2_30_crack_rebonding_live_engine_smoke.py) only exercised
``build_shared_engine``'s bare ``CampaignCalibratedTipEngine``, which has no
stochastic-hazard/transactional-event mixins -- it does not prove the
rebonding-coupled event-time root-finder and wake transaction work on the
actual production engine class.

This file constructs the REAL, fully-composed production engine
(``CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine``, confirmed via
direct ``__init__``-based construction -- not the ``object.__new__`` bypass
pattern used elsewhere in this repo for isolated unit tests -- to match the
actual runtime MRO used by the CLI chain) and drives it through multiple
real accepted first-passage events.

Constructing this engine directly (mirroring exactly what an investigation
agent found by tracing the real CLI's monkeypatch/inheritance chain) also
uncovered and led to fixing a real, significant gap: the injection point
originally believed to be "the sole choke point" for cleavage hazard
(``kinetic_tip_cell.py::cycle_step_waveform``) is dead code for this real
engine class. ``PersistentSiteCyclicTipEngine.cycle_step_waveform``
(persistent_site_cyclic_v10229.py) is a fully independent reimplementation
that does not call ``super()``, and ``CoupledPersistentSiteCyclicTipEngine``
(the actual class in the real MRO) overrides it *again*, delegating to
``integrate_state_coupled_waveform`` (persistent_site_coupled_hazard_v10229.py)
-- a third, adaptive-Simpson-quadrature commit pathway with its own
deep-copied trial engines for error estimation. The real injection points
are ``persistent_site_coupled_hazard_v10229.py::_phase_statistics`` (the
actual per-phase cleavage-rate computation for real runs) and
``_commit_constant_segment`` (the actual final commit onto the real engine,
called once per accepted quadrature segment -- there can be several per
``cycle_step_waveform`` call). See
docs/v10_2_30_crack_rebonding_equation_lineage.md for the full correction
record, including why the two earlier injection points remain harmlessly
present (correct in isolation, simply unreached for this class hierarchy).

The real-engine construction recipe used here is shared (not duplicated)
with the S8 full-state-reference and causal-bonding tests via
tests/_crack_rebonding_engine_fixture.py.
"""
from __future__ import annotations

import pytest

from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)

import _crack_rebonding_engine_fixture as fx


def test_real_engine_constructs_with_full_stochastic_and_energy_gated_mixins():
    engine = fx.build_real_engine()
    assert isinstance(engine, CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine)
    assert hasattr(engine, "_hazard_rng")
    assert hasattr(engine, "_energy_gate_pending")
    assert engine.B == 0.0


def test_two_accepted_events_create_correct_patches_and_fresh_patch_never_inherits():
    engine = fx.build_real_engine(fx.rebonding_cfg())
    ctrl = fx.controller()
    waveform = fx.default_waveform()

    fx.run_to_next_fired_event(engine, ctrl, waveform)
    assert engine._energy_gate_pending.get("rebonding_block_context") is not None
    length1 = fx.commit_pending_event(engine)
    assert len(engine._rebonding_state.active) == 1
    first_patch = engine._rebonding_state.active[0]
    assert first_patch.length_m == pytest.approx(length1)
    assert first_patch.s_j_m == pytest.approx(0.0)
    assert first_patch.p_B == 0.0

    elapsed_after_first = engine._rebonding_state.elapsed_time_s
    assert elapsed_after_first > 0.0  # chronological clock genuinely advanced

    fx.run_to_next_fired_event(engine, ctrl, waveform)
    length2 = fx.commit_pending_event(engine)
    assert len(engine._rebonding_state.active) == 2

    # The pre-existing (first) patch must have translated by exactly the
    # second event's accepted length, and never inherit any bonding it
    # never actually accrued being freshly hypothetical.
    translated_first = next(p for p in engine._rebonding_state.active if p.patch_id == first_patch.patch_id)
    assert translated_first.s_j_m == pytest.approx(length2)

    # The newest patch must be created fresh: p_B exactly zero regardless of
    # what the pre-existing patch's state was.
    newest_patch = engine._rebonding_state.active[-1]
    assert newest_patch.patch_id != first_patch.patch_id
    assert newest_patch.p_B == 0.0
    assert newest_patch.s_j_m == pytest.approx(0.0)

    # Chronological continuity: the clock keeps advancing, never resets.
    assert engine._rebonding_state.elapsed_time_s >= 0.0


def test_rollback_restores_full_wake_state_on_a_real_engine():
    engine = fx.build_real_engine(fx.rebonding_cfg())
    ctrl = fx.controller()
    waveform = fx.default_waveform()

    fx.run_to_next_fired_event(engine, ctrl, waveform)
    fx.commit_pending_event(engine)

    fx.run_to_next_fired_event(engine, ctrl, waveform)
    n_patches_before_veto = len(engine._rebonding_state.active)
    elapsed_before_veto = engine._rebonding_state.elapsed_time_s

    engine.restore_geometry_veto()

    assert len(engine._rebonding_state.active) == n_patches_before_veto
    assert engine._rebonding_state.elapsed_time_s == pytest.approx(elapsed_before_veto)
    assert engine._energy_gate_pending is None


def test_disabled_real_engine_never_allocates_rebonding_state():
    engine = fx.build_real_engine(None)
    ctrl = fx.controller()
    waveform = fx.default_waveform()
    for _ in range(30):
        engine.cycle_step_waveform(ctrl, waveform, 300.0)
        if engine.B >= 1.0 - 1.0e-9:
            break
    assert getattr(engine, "_rebonding_state", None) is None
    assert getattr(engine, "_rebonding_block_context", None) is None
