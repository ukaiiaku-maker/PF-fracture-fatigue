"""S8D Part 2 (round-3 follow-up), honestly scoped.

Round-3 review's point 5 noted the permanent qualification test calls
``engine.commit_energy_gated_event(...)`` directly, bypassing the real
outer ``EnergyGatedAvalancheBackend.advance()`` -> ``finalize_engine_event()``
chain that a genuine CLI/production run goes through
(``hazard_energy_event_gate_v10230.py:625,637,661,667``).

Investigation (recorded in docs/v10_2_30_crack_rebonding_equation_lineage.md)
confirmed that exercising the FULL chain -- including
``EnergyGatedAvalancheBackend.advance()`` -- requires a real FEM mesh,
boundary condition set, damage field, and displacement field
(``energy_gate_event_length``'s ``kwargs["mesh"]``/``damage``/
``displacement``/``boundary`` and an ``OBSERVER.snapshot`` mechanics
snapshot matching that mesh), which in turn requires either a real,
previously-built production kernel-family JSON (validated by a slow
FEM/mesh-resolving subprocess, `scripts/ensure_v10_2_28_signed_kernel.py`)
or a from-scratch synthetic FEM fixture -- neither is available as a
lightweight fixture in this repo, and constructing either is out of scope
for this pass (it would require hours of unplanned FEM/mesh engineering,
not a "prephysical qualification" activity).

What IS feasible, and implemented here: ``finalize_engine_event()`` itself
(the actual glue between the backend and ``commit_energy_gated_event``,
including the real ``register_engine``/``_engine_from_id`` weakref-registry
lookup that a genuine CLI run uses to find the correct live engine instance
from a bare descriptor) requires no mesh at all -- it is exercised directly
here, closing one more real layer of the gap beyond the direct-method-call
qualification test, while the mesh-dependent
``EnergyGatedAvalancheBackend.advance()`` layer remains a documented,
honest, out-of-scope gap for this pass.
"""
from __future__ import annotations

import pytest

import _crack_rebonding_engine_fixture as fx

from arrhenius_fracture.hazard_energy_event_gate_v10230 import _engine_from_id, finalize_engine_event
from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import (
    restore_checkpoint,
    write_checkpoint,
)


def test_finalize_engine_event_reaches_the_real_engine_via_the_weakref_registry():
    """register_engine(self) is called inside the real engine's __init__
    (persistent_site_cyclic_energy_gated_v10230.py:81) -- confirms the
    construction recipe in tests/_crack_rebonding_engine_fixture.py produces
    a properly registered engine, findable by engine_id exactly as a real
    backend-driven commit would find it, not just directly callable."""
    engine = fx.build_real_engine(fx.rebonding_cfg())
    ctrl = fx.controller()
    waveform = fx.default_waveform()
    fx.run_to_next_fired_event(engine, ctrl, waveform)

    pending = engine._energy_gate_pending
    assert pending.get("rebonding_block_context") is not None
    descriptor = pending["descriptor"]
    committed_length = pending["proposal_m"]

    found = _engine_from_id(int(descriptor["energy_gate_engine_id"]))
    assert found is engine

    gate = {
        "energy_admissible_event_length_m": committed_length,
        "arrest_reason": "test_finalize_glue",
        "hazard_resistance_J_per_m2": 1.0,
        "orientation_gamma_relative": 1.0,
    }

    n_patches_before = len(engine._rebonding_state.active)
    finalize_engine_event(descriptor, committed_length, gate)

    # The real transactional commit ran: a new patch exists, using the
    # accepted length, and the pending transaction is cleared -- same
    # invariants as the direct-call qualification test, now reached through
    # the actual production glue function instead of a direct method call.
    assert len(engine._rebonding_state.active) == n_patches_before + 1
    new_patch = engine._rebonding_state.active[-1]
    assert new_patch.length_m == pytest.approx(committed_length)
    assert new_patch.p_B == 0.0
    assert engine._energy_gate_pending is None


def test_finalize_engine_event_raises_when_the_registered_engine_is_gone():
    """finalize_engine_event must fail closed (not silently no-op) if the
    weakref registry lookup fails -- e.g. a stale/garbage-collected engine
    reference, confirming this is genuinely a lookup-and-dispatch layer with
    its own failure mode, not a pass-through."""
    with pytest.raises(RuntimeError, match="no longer available"):
        finalize_engine_event(
            {"energy_gate_engine_id": -999999, "energy_gate_result_ref": None},
            1.0e-6,
            {"energy_admissible_event_length_m": 1.0e-6},
        )


def test_checkpoint_round_trip_after_a_finalize_engine_event_commit(monkeypatch, tmp_path):
    """The real, file-based checkpoint mechanism
    (persistent_site_high_cycle_checkpoint_v10230.py::write_checkpoint/
    restore_checkpoint, driven by V10230_HIGH_CYCLE_CHECKPOINT_DIR exactly
    as a real run would configure it) must contain the enabled rebonding
    state and restore it correctly after a commit reached through the real
    finalize_engine_event glue, not only after a direct
    commit_energy_gated_event call on a hand-built engine."""
    monkeypatch.setenv("V10230_HIGH_CYCLE_CHECKPOINT_DIR", str(tmp_path))

    engine = fx.build_real_engine(fx.rebonding_cfg())
    ctrl = fx.controller()
    waveform = fx.default_waveform()
    fx.run_to_next_fired_event(engine, ctrl, waveform)

    pending = engine._energy_gate_pending
    descriptor = pending["descriptor"]
    committed_length = pending["proposal_m"]
    gate = {
        "energy_admissible_event_length_m": committed_length,
        "arrest_reason": "test_finalize_glue",
        "hazard_resistance_J_per_m2": 1.0,
        "orientation_gamma_relative": 1.0,
    }
    finalize_engine_event(descriptor, committed_length, gate)

    payload = write_checkpoint(engine, waveform=waveform, temperature_K=300.0, reason="first_passage")
    assert payload is not None
    assert payload.get("crack_rebonding") is not None

    restored = fx.build_real_engine(fx.rebonding_cfg())
    # restore_checkpoint_payload requires the engine's geometry signature to
    # already match the checkpoint's (n_adv/a_adv/advance totals) -- in a
    # real run, the OUTER FEM/mesh state is restored via its own separate
    # mechanism first, and only then is this kinetic checkpoint layered on
    # top. There is no outer mesh in this fixture, so mirror the same
    # geometry-relevant scalars directly, matching what a real restore
    # sequence would have already produced.
    restored.n_adv = engine.n_adv
    restored.a_adv = engine.a_adv
    restored.micro_advance_total_m = engine.micro_advance_total_m
    restored.checkpoint_advance_total_m = engine.checkpoint_advance_total_m
    restored.mpz.advance_total_m = engine.mpz.advance_total_m
    restore_checkpoint(restored, tmp_path)
    assert len(restored._rebonding_state.active) == len(engine._rebonding_state.active)
    assert restored._rebonding_state.elapsed_time_s == pytest.approx(
        engine._rebonding_state.elapsed_time_s
    )


def test_disabled_engine_checkpoint_has_no_crack_rebonding_key(monkeypatch, tmp_path):
    """Structural parity: a checkpoint written for a disabled engine must
    not carry a crack_rebonding key at all (not present-with-null), through
    the same real file-based mechanism used above."""
    monkeypatch.setenv("V10230_HIGH_CYCLE_CHECKPOINT_DIR", str(tmp_path))
    engine = fx.build_real_engine(None)
    payload = write_checkpoint(engine, reason="test_disabled_parity")
    assert payload is not None
    assert "crack_rebonding" not in payload
