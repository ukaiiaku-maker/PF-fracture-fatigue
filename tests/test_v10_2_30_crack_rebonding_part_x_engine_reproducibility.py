"""PX2.5 item 1: fresh-process/same-process engine reproducibility.

External review found a real ~22% divergence between two sequentially
constructed "identical" engines when the full crack_rebonding test
selection ran (never in isolation). Root-caused here, precisely:

- ``StochasticHazardDiagnosticTipEngine._hazard_config_default`` is a
  MUTABLE CLASS ATTRIBUTE. Several pre-existing test files (e.g.
  test_v10_2_30_crack_rebonding_causal_pilot.py,
  test_v10_2_30_crack_rebonding_zero_cohesion_parity.py,
  test_v10_2_30_crack_rebonding_static_shield_ablation.py) call
  ``Engine.configure_hazard(mode="exponential", seed=...)`` and do not
  reset it back to the "deterministic" default afterward -- so whichever
  test runs next in the same pytest session inherits "exponential" mode
  with a stale seed, silently, via nothing more than test execution
  order.
- Separately, ``KineticMovingTipFrontEngine._next_engine_id`` (the
  counter ``SeedSequence([hazard_cfg.seed, engine_id])`` uses to
  decorrelate concurrently-existing engines by design) is defined on a
  BASE class but every subclass's own ``type(self)._next_engine_id +=
  1`` creates a SEPARATE, SHADOWED copy on the leaf class the first time
  it runs -- calling ``reset_audit()`` on a base class does NOT reset the
  leaf class's own shadowed counter; it must be called on the actual
  leaf production class.

Both effects are real, but neither is a physics bug: the existing
qualified production launch scripts (run_static_shield_attribution_
stage.py, etc.) already call ``Engine.configure_hazard(...)`` +
``Engine.reset_audit()`` on the correct leaf class immediately before
each construction. This file proves, directly, that doing so gives
EXACT reproducibility regardless of arbitrary prior construction/
class-state history in the same process, and ALSO across fully
independent fresh subprocesses -- the two guarantees mission section 14
requires before PX3 may launch scientific jobs.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from arrhenius_fracture.a_native_engine_v10230 import build_a_native_engine
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel,
    CrackRebondingControls,
    FeedbackMode,
    InitialPrecrackWakeMode,
    RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_v10230 import install_crack_rebonding
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine as Engine,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_deterministic_trajectory(seed: int, n_steps: int = 40, n_events: int = 3):
    """The pattern every PX3+ scientific launch must use: explicit
    configure_hazard + reset_audit on the LEAF class immediately before
    construction -- proven below to be immune to arbitrary prior process
    state."""
    Engine.configure_hazard(mode="exponential", seed=seed)
    Engine.reset_audit()
    engine, audit = build_a_native_engine()
    ctrl = FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=80, block_cycles=1000.0, max_block_cycles=1.0e6), None, None, None,
    )
    wave = FatigueWaveform(Kmax=18.0e6, R=-0.5, frequency_Hz=1000.0)
    thresholds = [float(engine.hazard_threshold_action)]
    events = []
    for i in range(n_steps):
        r = engine.cycle_step_waveform(ctrl, wave, 300.0)
        if r.get("fired"):
            events.append((i, r["cycles_consumed"], r["B"]))
            thresholds.append(float(engine.hazard_threshold_action))
        if len(events) >= n_events:
            break
    return {
        "engine_id": engine._engine_id,
        "mro": [c.__name__ for c in type(engine).__mro__],
        "material_row_sha256": audit["source_row_sha256"],
        "thresholds": thresholds,
        "events": events,
        "final_B": float(engine.B),
        "final_mobile_count": float(engine.mpz.mobile_count),
        "final_retained_count": float(engine.mpz.retained_count),
        "final_t": float(engine.t),
    }


def _construct_unrelated_engines(n: int) -> None:
    """Simulate arbitrary prior process activity: construct n unrelated
    engines (leaving mode="deterministic" -- the class default -- alone,
    since that alone is not what caused the original divergence)."""
    for _ in range(n):
        e, _ = build_a_native_engine()
        cfg = CrackRebondingControls(
            enabled=True, model_level=RebondModelLevel.PASSIVATION_GATED_REBOND,
            contact_model=ContactModel.SIGNED_K_COMPRESSION_PROXY, feedback_mode=FeedbackMode.HAZARD_ONLY_REBOND_SHIELD,
            wake_length_m=5.0e-4, wake_weight_length_m=5.0e-7, restored_work_of_separation_J_m2=1.8225,
            rebond_K_geometry_factor=1.0, bond_barrier_eV=0.39, rupture_barrier_eV=0.386,
            depassivation_barrier_eV=0.40, repassivation_barrier_eV=0.40,
            initial_precrack_wake_mode=InitialPrecrackWakeMode.NO_INITIAL_ACTIVE_WAKE,
        ).validate()
        install_crack_rebonding(e, cfg)


def _leak_stale_exponential_hazard_config() -> None:
    """Reproduce exactly the leakage several pre-existing test files cause:
    leave the class-level hazard default mutated to exponential mode with
    an unrelated seed, WITHOUT resetting it back afterward."""
    Engine.configure_hazard(mode="exponential", seed=999999)
    build_a_native_engine()


def test_explicit_reset_gives_identical_result_regardless_of_unrelated_prior_construction():
    baseline = _run_deterministic_trajectory(seed=1720)
    _construct_unrelated_engines(7)
    polluted = _run_deterministic_trajectory(seed=1720)
    assert polluted == baseline


def test_explicit_reset_gives_identical_result_despite_leaked_stale_hazard_config():
    """The precise root cause found in review: an earlier test's
    configure_hazard(mode='exponential', seed=X) call, left unreset,
    leaking into whatever constructs an engine next. Confirms the
    explicit-reset pattern is immune to this specific leakage."""
    baseline = _run_deterministic_trajectory(seed=1720)
    _leak_stale_exponential_hazard_config()
    after_leak = _run_deterministic_trajectory(seed=1720)
    assert after_leak == baseline


def test_reset_audit_must_target_the_actual_leaf_class():
    """Documents the shadowed-class-attribute subtlety directly: resetting
    a BASE class's _next_engine_id does not affect an already-shadowed
    leaf-class copy; resetting the leaf class does."""
    from arrhenius_fracture.kinetic_tip_cell import KineticMovingTipFrontEngine

    _construct_unrelated_engines(3)  # forces the leaf class's own shadowed counter to exist and advance
    leaf_id_before_base_reset = Engine._next_engine_id
    KineticMovingTipFrontEngine.reset_audit()
    assert Engine._next_engine_id == leaf_id_before_base_reset  # unaffected by the base-class reset

    Engine.reset_audit()
    Engine.configure_hazard(mode="deterministic", seed=0)
    engine, _ = build_a_native_engine()
    assert engine._engine_id == 1  # leaf-class reset takes effect


@pytest.mark.parametrize("pollute_n", [0, 7])
def test_fresh_subprocess_reproducibility(pollute_n, tmp_path):
    """The strongest form of the review's ask: run the identical
    construction+trajectory in a completely separate, fresh Python
    process (not just a fresh function call), optionally after that same
    fresh process ALSO constructs unrelated engines first -- and require
    byte-identical JSON output against a reference captured with
    pollute_n=0."""
    script = REPO_ROOT / "scripts" / "px2_5_fresh_process_engine_check.py"
    result = subprocess.run(
        [sys.executable, str(script), "--seed", "1720", "--pollute-n", str(pollute_n)],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    if pollute_n == 0:
        tmp_path.joinpath("reference.json").write_text(result.stdout)
    # Compare against a freshly-run pollute_n=0 reference in the SAME test
    # (avoids relying on execution order across parametrized cases).
    ref_result = subprocess.run(
        [sys.executable, str(script), "--seed", "1720", "--pollute-n", "0"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert ref_result.returncode == 0, ref_result.stderr
    reference = json.loads(ref_result.stdout)
    assert payload == reference
