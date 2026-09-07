"""Shared real-production-engine construction helpers for crack-rebonding
qualification tests (S7's full-production-qualification test and the S8
full-state-reference/causal-bonding tests all need the identical
construction recipe -- factored out here so there is exactly one
implementation of "how to build the real, fully-composed production
engine", not several that could silently drift apart).

See tests/test_v10_2_30_crack_rebonding_full_production_qualification.py
for why this specific construction recipe (direct __init__, not the
object.__new__ bypass pattern used elsewhere in this repo for isolated unit
tests) is required to match the actual CLI runtime MRO, and
docs/v10_2_30_crack_rebonding_equation_lineage.md's "Injection-point
correction history" section for the underlying investigation.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from arrhenius_fracture.anisotropic_emission_v10174 import (
    AnisotropicEmissionConfig,
    install_anisotropic_campaign_emission,
)
from arrhenius_fracture.crack_rebonding_kinetics_v10230 import (
    ContactModel,
    CrackRebondingControls,
    RebondModelLevel,
)
from arrhenius_fracture.crack_rebonding_v10230 import install_crack_rebonding
from arrhenius_fracture.fatigue_v1 import (
    FatigueControllerConfig,
    FatigueCycleHazardController,
    FatigueWaveform,
)
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


def synthetic_family() -> ActiveOnlySigned2DShieldingKernelFamily:
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


def build_real_engine(
    rebonding_cfg: CrackRebondingControls | None = None,
    *,
    retained_recovery_rate_s: float = 0.0,
):
    """Construct the real production engine class via genuine __init__
    construction (not the object.__new__ bypass pattern used elsewhere for
    isolated unit tests), matching the actual CLI runtime MRO."""
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.configure_state_resolved_physics(
        synthetic_family(), transport_mode="validated_scalar"
    )
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.configure_persistent_sites(
        PersistentSiteConfig(rho_site0_m2=5.0e12).validate()
    )

    manifest = load_manifest(candidate_id="DBTT_A0003837")
    manifest = dataclasses.replace(
        manifest, retained_recovery_rate_s=retained_recovery_rate_s
    )
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


def controller(n_phase: int = 16) -> FatigueCycleHazardController:
    return FatigueCycleHazardController(
        FatigueControllerConfig(n_phase=n_phase, block_cycles=1000.0, max_block_cycles=1.0e6),
        None, None, None,
    )


def default_waveform() -> FatigueWaveform:
    return FatigueWaveform(Kmax=18.0e6, R=-0.95, frequency_Hz=1000.0)


def rebonding_cfg(
    *,
    wake_length_m: float = 5.0e-4,
    model_level: RebondModelLevel = RebondModelLevel.CLEAN_REVERSIBLE_REBOND,
    contact_model: ContactModel = ContactModel.SIGNED_K_COMPRESSION_PROXY,
    bond_barrier_eV: float = 0.2,
    rupture_barrier_eV: float = 1.0,
    bond_attempt_frequency_s: float = 1.0e10,
    rupture_attempt_frequency_s: float = 1.0e10,
) -> CrackRebondingControls:
    return CrackRebondingControls(
        enabled=True,
        model_level=model_level,
        contact_model=contact_model,
        wake_length_m=wake_length_m,
        wake_weight_length_m=5.0e-7,
        restored_work_of_separation_J_m2=1.0,
        rebond_K_geometry_factor=0.5,
        bond_activation_volume_m3=0.0,
        bond_barrier_eV=bond_barrier_eV,
        bond_attempt_frequency_s=bond_attempt_frequency_s,
        rupture_activation_volume_m3=0.0,
        rupture_barrier_eV=rupture_barrier_eV,
        rupture_attempt_frequency_s=rupture_attempt_frequency_s,
        chemistry_factor=1.0,
    ).validate()


def run_to_next_fired_event(engine, ctrl, waveform, max_blocks: int = 5000):
    for _ in range(max_blocks):
        result = engine.cycle_step_waveform(ctrl, waveform, 300.0)
        if result.get("fired"):
            return result
    raise AssertionError(f"no event fired within {max_blocks} blocks")


def commit_pending_event(engine):
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
