"""Audited v10.2.30 hazard-energy-gated persistent-site fatigue entry."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import crystal as _crystal
from . import fatigue_controller_delegate_v10229 as _delegate
from . import fatigue_v1 as _fatigue_v1
from . import fem as _fem
from . import hazard_energy_event_gate_v10230 as _energy_gate
from . import persistent_site_cyclic_coupled_v10229 as _coupled_commit
from . import persistent_site_forward_robust_v10230 as _forward_hazard
from . import persistent_site_forward_selector_v10230 as _forward_selector
from . import sharp_front_v10_1_7_3 as _avalanche
from . import sharp_front_v10_2_29_fatigue_audited as _v10229
from .hazard_energy_event_gate_v10230 import (
    OBSERVER,
    audit_payload,
    build_energy_gated_avalanche_backend,
    config_from_environment,
    reset_runtime_state,
    set_latest_probe_K,
    wrap_assemble_mechanics,
    wrap_cleave_direction_competition,
    wrap_cleavage_branch_candidates,
    write_last_energy_gate_diagnostics,
)
from .hazard_energy_event_gate_mesh_consistent_v10230 import (
    MODEL_ID as MESH_SEARCH_MODEL_ID,
    energy_gate_event_length_mesh_consistent,
)
from .persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
    MODEL_ID as CORRECTED_ENGINE_MODEL_ID,
)
from .persistent_site_trial_clone_v10230 import (
    MODEL_ID as TRIAL_CLONE_MODEL_ID,
    install_fast_trial_clone,
    restore_fast_trial_clone,
)


MODEL_ID = "v10.2.30_hazard_energy_gated_persistent_site_fatigue"


def _has_option(args: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in args)


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _observed_waveform_factory(original):
    """Capture the incoming FEM probe K before fixed-DeltaK replacement."""

    def observed(*args, **kwargs):
        incoming = kwargs.get("Kmax")
        if incoming is None and args:
            incoming = args[0]
        if incoming is not None:
            set_latest_probe_K(float(incoming))
        return original(*args, **kwargs)

    observed.__name__ = getattr(original, "__name__", "ObservedFatigueWaveform")
    observed.__doc__ = getattr(original, "__doc__", None)
    return observed


def _write_audit(args: list[str]) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = audit_payload()
    payload.update(
        {
            "schema": MODEL_ID,
            "base_fatigue_entry": (
                "arrhenius_fracture.sharp_front_v10_2_29_fatigue_audited"
            ),
            "transactional_engine": (
                "CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine"
            ),
            "transactional_engine_model_id": CORRECTED_ENGINE_MODEL_ID,
            "event_length_search_model_id": MESH_SEARCH_MODEL_ID,
            "vhcf_block_selector_model_id": _forward_selector.MODEL_ID,
            "vhcf_block_search_strategy": (
                "predictor_only_forward_marcher_authoritative"
            ),
            "vhcf_forward_marcher_model_id": _forward_hazard.MODEL_ID,
            "vhcf_full_horizon_first_trial": False,
            "vhcf_raw_population_increment_targets_active": False,
            "vhcf_active_state_endpoint_error_control": True,
            "vhcf_state_profile_endpoint_error_control": True,
            "vhcf_backstress_endpoint_error_control": True,
            "vhcf_segment_size_carried_across_outer_calls": True,
            "vhcf_explicit_transient_segment_cap_active": True,
            "vhcf_recursive_depth_first_commit": False,
            "vhcf_partial_cycle_return_supported": True,
            "vhcf_two_half_step_state_committed": True,
            "vhcf_third_commit_integration": False,
            "trial_clone_model_id": TRIAL_CLONE_MODEL_ID,
            "trial_rng_seedsequence_reduce_path_avoided": True,
            "persistent_site_source": True,
            "anisotropic_direction_competition": True,
            "four_class_registry_preserved": True,
            "parameter_refit": False,
            "cleavage_first_passage_rate_changed": False,
            "continuum_energy_comparison_diagnostic_only": True,
            "continuum_energy_comparison_affects_hazard": False,
            "stochastic_proposal_distribution_changed": False,
            "event_length_commit_changed": True,
            "moving_mpz_and_geometry_commit_atomic": True,
            "waiting_cycle_tip_translation": False,
            "zero_length_hazard_attempts_consumed": True,
            "energy_gate_event_load": "Kmax_geometry_transaction_load",
            "energy_gate_barrier_load": "Kmax_geometry_transaction_load",
            "phase_resolved_hazard_integration_preserved": True,
            "fixed_deltaK_probe_K_captured_before_waveform_replacement": True,
            "mesh_resolved_geometry_commit_required": True,
            "directional_J_subgrid_value_used_for_search_only": True,
            "athermal_fracture_parameter_active": False,
            "Gc0_athermal_active": False,
        }
    )
    (root / "v10_2_30_hazard_energy_gate_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if not _has_option(args, "--fatigue-cycles"):
        return _v10229.main(args)

    cfg = config_from_environment()
    if not cfg.enabled:
        raise SystemExit("v10.2.30 requires V10230_ENERGY_GATE_ENABLED=1")
    reset_runtime_state(cfg)

    original_engine = _v10229.AuditedCoupledPersistentSiteCyclicTipEngine
    original_avalanche_builder = _avalanche.build_avalanche_backend
    original_assemble = _fem.assemble_mechanics
    original_compete = _crystal.cleave_direction_competition
    original_discrete = _crystal.cleavage_branch_candidates
    original_energy_search = _energy_gate.energy_gate_event_length
    original_waveform = _fatigue_v1.FatigueWaveform
    original_attach_prediction_context = _delegate.attach_prediction_context
    original_select_nonlinear_block = _delegate.select_nonlinear_block
    original_coupled_commit = _coupled_commit.integrate_state_coupled_waveform

    OBSERVER.original_assemble = original_assemble
    _fem.assemble_mechanics = wrap_assemble_mechanics(original_assemble)
    _crystal.cleave_direction_competition = wrap_cleave_direction_competition(
        original_compete
    )
    _crystal.cleavage_branch_candidates = wrap_cleavage_branch_candidates(
        original_discrete
    )
    _energy_gate.energy_gate_event_length = energy_gate_event_length_mesh_consistent
    _v10229.AuditedCoupledPersistentSiteCyclicTipEngine = (
        CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine
    )
    _fatigue_v1.FatigueWaveform = _observed_waveform_factory(original_waveform)
    _delegate.attach_prediction_context = _forward_selector.attach_prediction_context
    _delegate.select_nonlinear_block = _forward_selector.select_nonlinear_block
    _coupled_commit.integrate_state_coupled_waveform = (
        _forward_hazard.integrate_state_coupled_waveform
    )
    install_fast_trial_clone()

    def gated_builder(
        local_args,
        geom,
        original_builder,
        default_subsegment_fraction=0.1,
    ):
        return build_energy_gated_avalanche_backend(
            local_args,
            geom,
            original_builder,
            default_subsegment_fraction=default_subsegment_fraction,
            original_avalanche_builder=original_avalanche_builder,
        )

    _avalanche.build_avalanche_backend = gated_builder
    try:
        print(
            "  v10.2.30 hazard-energy-gated fatigue: "
            "trigger=cleavage_first_passage "
            "resistance=gamma_rel*m_hits*DeltaG_eff/b^2 "
            "event=min(stochastic_proposal,energy_arrest) "
            "event_load=Kmax continuum_gate=diagnostic_only "
            "block_control=partition_robust_forward_two_half_step "
            "active_state_error=profile+backstress+source_rate "
            "partial_return=on work_budget=on "
            "trial_rng_clone=state_exact Gc0_athermal=off"
        )
        result = _v10229.main(args)
        out = _option_value(args, "--out")
        if out:
            write_last_energy_gate_diagnostics(out)
            _write_audit(args)
        return result
    finally:
        _avalanche.build_avalanche_backend = original_avalanche_builder
        restore_fast_trial_clone()
        _coupled_commit.integrate_state_coupled_waveform = original_coupled_commit
        _delegate.select_nonlinear_block = original_select_nonlinear_block
        _delegate.attach_prediction_context = original_attach_prediction_context
        _fatigue_v1.FatigueWaveform = original_waveform
        _v10229.AuditedCoupledPersistentSiteCyclicTipEngine = original_engine
        _energy_gate.energy_gate_event_length = original_energy_search
        _crystal.cleavage_branch_candidates = original_discrete
        _crystal.cleave_direction_competition = original_compete
        _fem.assemble_mechanics = original_assemble
        OBSERVER.original_assemble = None


if __name__ == "__main__":
    main()
