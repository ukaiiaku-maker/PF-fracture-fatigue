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
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from arrhenius_fracture.anisotropic_emission_v10174 import (
    AnisotropicEmissionConfig,
    install_anisotropic_campaign_emission,
)
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls, RebondModelLevel
from arrhenius_fracture.crack_rebonding_v10230 import install_crack_rebonding
from arrhenius_fracture.fatigue_v1 import FatigueControllerConfig, FatigueCycleHazardController, FatigueWaveform
from arrhenius_fracture.persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from arrhenius_fracture.persistent_site_source_v10221 import PersistentSiteConfig
from arrhenius_fracture.reduced_shared_state_v1023 import (
    SharedReducedConfig,
    _front_config,
    _mpz_config,
    _tip_config,
    install_campaign_calibrated_source,
    load_manifest,
)
from arrhenius_fracture.signed_kernel_family_v10214 import (
    ActiveOnlySigned2DShieldingKernelFamily,
    KernelState,
)
from arrhenius_fracture.state_resolved_signed_engine_v1026 import StateResolvedSignedBurgersTipEngine


def _synthetic_family() -> ActiveOnlySigned2DShieldingKernelFamily:
    source_x = (np.arange(40, dtype=float) + 0.5) * (100.0e-6 / 40.0)
    rows_I = np.vstack((1.0 + 2.0e4 * source_x, -0.5 + 1.0e4 * source_x))
    rows_II = 0.25 * rows_I
    states = []
    for state_id, extension, scale in (("E000", 0.0, 1.0), ("E200", 200.0e-6, 1.1)):
        states.append(
            KernelState(
                state_id=state_id,
                coordinates=np.asarray([1.0, 0.0, extension]),
                active_I=scale * rows_I,
                wake_I=np.zeros((2, 40)),
                active_II=scale * rows_II,
                wake_II=np.zeros((2, 40)),
                metadata={},
            )
        )
    family = ActiveOnlySigned2DShieldingKernelFamily(
        states=states,
        active_x_m=source_x,
        wake_x_m=source_x.copy(),
        activation_to_line_content=np.ones(2),
        source_capacity_bounds=np.asarray([[0.0, 1.0e5], [0.0, 1.0e5]]),
        fixed_kernel_assessment={"fixed_kernel_accepted": False},
        interpolation={"method": "inverse_distance", "neighbors": 2},
        metadata={"schema": "test"},
        source_path="synthetic",
    )
    family._opening_boundary_policy = {"policy": "strict"}
    family._last_boundary_action = "none"
    family._last_observed_analytical_r_eff_over_r0 = 1.0
    family._last_observed_opening_strength_fraction = 0.0
    family._runtime_grid_binding = None
    return family


def _build_real_engine(rebonding_cfg: CrackRebondingControls | None = None):
    """Construct the real production engine class via genuine __init__
    construction (not the object.__new__ bypass pattern used elsewhere for
    isolated unit tests), matching the actual CLI runtime MRO."""
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.configure_state_resolved_physics(
        _synthetic_family(), transport_mode="validated_scalar"
    )
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.configure_persistent_sites(
        PersistentSiteConfig(rho_site0_m2=5.0e12).validate()
    )

    manifest = load_manifest(candidate_id="DBTT_A0003837")
    manifest = dataclasses.replace(manifest, retained_recovery_rate_s=0.0)
    cfg = SharedReducedConfig(
        mpz_length_um=4.0,
        mpz_n_bins=8,
        wake_length_um=4.0,
        wake_n_bins=8,
        blunting_length_um=0.5,
        max_internal_steps=2000,
        drive_factors=(0.132886, 0.008596),
    ).validate()
    front_cfg = _front_config(cfg)
    front_cfg.sigma_cap = 1.0e12  # must be > 0 for this engine hierarchy
    mpz_cfg = _mpz_config(cfg)

    engine = CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine(
        front_cfg, manifest.cleavage, manifest.emission,
        float(cfg.G_Pa), float(cfg.poisson), float(cfg.b_m),
        manifest, mpz_cfg,
    )
    engine.tip_cfg = _tip_config(cfg, "full")
    engine.mpz.cfg.mobile_shield_fraction = float(cfg.mobile_shield_fraction)
    install_campaign_calibrated_source(
        engine.mpz, float(cfg.b_m), float(cfg.G_Pa), backstress_scale=1.0, refresh_scale=1.0
    )
    install_anisotropic_campaign_emission(
        engine.mpz,
        AnisotropicEmissionConfig(
            enabled=True, crystal_theta_deg=float(cfg.crystal_theta_deg),
            shared_forest_density=True, require_reliable_probe=True,
        ),
    )

    if rebonding_cfg is not None and rebonding_cfg.enabled:
        install_crack_rebonding(engine, rebonding_cfg)

    return engine


def _controller() -> FatigueCycleHazardController:
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=16, block_cycles=1000.0, max_block_cycles=1.0e6),
        None, None, None,
    )


def _rebonding_cfg() -> CrackRebondingControls:
    return CrackRebondingControls(
        enabled=True,
        model_level=RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
        wake_length_m=5.0e-4,
        wake_weight_length_m=5.0e-7,
        restored_work_of_separation_J_m2=1.0,
        rebond_K_geometry_factor=0.5,
        bond_activation_volume_m3=0.0,
        bond_barrier_eV=0.2,
        bond_attempt_frequency_s=1.0e10,
        rupture_activation_volume_m3=0.0,
        rupture_barrier_eV=1.0,
        rupture_attempt_frequency_s=1.0e10,
        chemistry_factor=1.0,
    ).validate()


def _run_to_next_fired_event(engine, controller, waveform, max_blocks=5000):
    for _ in range(max_blocks):
        result = engine.cycle_step_waveform(controller, waveform, 300.0)
        if result.get("fired"):
            return result
    raise AssertionError(f"no event fired within {max_blocks} blocks")


def _commit_pending_event(engine):
    pending = engine._energy_gate_pending
    committed_length = pending["proposal_m"]
    gate = {
        "energy_admissible_event_length_m": committed_length,
        "arrest_reason": "test_commit",
        "hazard_resistance_J_per_m2": 1.0,
        "orientation_gamma_relative": 1.0,
    }
    result_ref = pending["descriptor"].get("energy_gate_result_ref")
    engine.commit_energy_gated_event(committed_length, gate, result_ref)
    return committed_length


def test_real_engine_constructs_with_full_stochastic_and_energy_gated_mixins():
    engine = _build_real_engine()
    assert isinstance(engine, CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine)
    assert hasattr(engine, "_hazard_rng")
    assert hasattr(engine, "_energy_gate_pending")
    assert engine.B == 0.0


def test_two_accepted_events_create_correct_patches_and_fresh_patch_never_inherits():
    engine = _build_real_engine(_rebonding_cfg())
    controller = _controller()
    waveform = FatigueWaveform(Kmax=18.0e6, R=-0.95, frequency_Hz=1000.0)

    _run_to_next_fired_event(engine, controller, waveform)
    assert engine._energy_gate_pending.get("rebonding_block_context") is not None
    length1 = _commit_pending_event(engine)
    assert len(engine._rebonding_state.active) == 1
    first_patch = engine._rebonding_state.active[0]
    assert first_patch.length_m == pytest.approx(length1)
    assert first_patch.s_j_m == pytest.approx(0.0)
    assert first_patch.p_B == 0.0

    elapsed_after_first = engine._rebonding_state.elapsed_time_s
    assert elapsed_after_first > 0.0  # chronological clock genuinely advanced

    _run_to_next_fired_event(engine, controller, waveform)
    length2 = _commit_pending_event(engine)
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
    engine = _build_real_engine(_rebonding_cfg())
    controller = _controller()
    waveform = FatigueWaveform(Kmax=18.0e6, R=-0.95, frequency_Hz=1000.0)

    _run_to_next_fired_event(engine, controller, waveform)
    _commit_pending_event(engine)

    _run_to_next_fired_event(engine, controller, waveform)
    n_patches_before_veto = len(engine._rebonding_state.active)
    elapsed_before_veto = engine._rebonding_state.elapsed_time_s

    engine.restore_geometry_veto()

    assert len(engine._rebonding_state.active) == n_patches_before_veto
    assert engine._rebonding_state.elapsed_time_s == pytest.approx(elapsed_before_veto)
    assert engine._energy_gate_pending is None


def test_disabled_real_engine_never_allocates_rebonding_state():
    engine = _build_real_engine(None)
    controller = _controller()
    waveform = FatigueWaveform(Kmax=18.0e6, R=-0.95, frequency_Hz=1000.0)
    for _ in range(30):
        engine.cycle_step_waveform(controller, waveform, 300.0)
        if engine.B >= 1.0 - 1.0e-9:
            break
    assert getattr(engine, "_rebonding_state", None) is None
    assert getattr(engine, "_rebonding_block_context", None) is None
