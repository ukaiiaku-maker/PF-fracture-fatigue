"""Audited v10.2.30 hazard-energy-gated persistent-site fatigue entry."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import crystal as _crystal
from . import fem as _fem
from . import hazard_energy_event_gate_v10230 as _energy_gate
from . import sharp_front_v10_1_7_3 as _avalanche
from . import sharp_front_v10_2_29_fatigue_audited as _v10229
from .hazard_energy_event_gate_v10230 import (
    OBSERVER,
    audit_payload,
    build_energy_gated_avalanche_backend,
    config_from_environment,
    reset_runtime_state,
    wrap_assemble_mechanics,
    wrap_cleave_direction_competition,
    wrap_cleavage_branch_candidates,
    write_last_energy_gate_diagnostics,
)
from .hazard_energy_event_gate_mesh_consistent_v10230 import (
    MODEL_ID as MESH_SEARCH_MODEL_ID,
    energy_gate_event_length_mesh_consistent,
)
from .persistent_site_cyclic_energy_gated_eventload_v10230 import (
    EventLoadConsistentHazardEnergyGatedPersistentSiteCyclicTipEngine,
    MODEL_ID as EVENT_LOAD_ENGINE_MODEL_ID,
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
                "EventLoadConsistentHazardEnergyGatedPersistentSiteCyclicTipEngine"
            ),
            "transactional_engine_model_id": EVENT_LOAD_ENGINE_MODEL_ID,
            "event_length_search_model_id": MESH_SEARCH_MODEL_ID,
            "persistent_site_source": True,
            "anisotropic_direction_competition": True,
            "four_class_registry_preserved": True,
            "parameter_refit": False,
            "cleavage_first_passage_changed": False,
            "stochastic_proposal_distribution_changed": False,
            "event_length_commit_changed": True,
            "moving_mpz_and_geometry_commit_atomic": True,
            "waiting_cycle_tip_translation": False,
            "energy_gate_event_load": "Kmax_geometry_transaction_load",
            "energy_gate_barrier_load": "Kmax_geometry_transaction_load",
            "phase_resolved_hazard_integration_preserved": True,
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

    OBSERVER.original_assemble = original_assemble
    _fem.assemble_mechanics = wrap_assemble_mechanics(original_assemble)
    _crystal.cleave_direction_competition = wrap_cleave_direction_competition(
        original_compete
    )
    _crystal.cleavage_branch_candidates = wrap_cleavage_branch_candidates(
        original_discrete
    )
    _energy_gate.energy_gate_event_length = (
        energy_gate_event_length_mesh_consistent
    )
    _v10229.AuditedCoupledPersistentSiteCyclicTipEngine = (
        EventLoadConsistentHazardEnergyGatedPersistentSiteCyclicTipEngine
    )

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
            "event_load=Kmax mesh_commit=required Gc0_athermal=off"
        )
        result = _v10229.main(args)
        out = _option_value(args, "--out")
        if out:
            write_last_energy_gate_diagnostics(out)
            _write_audit(args)
        return result
    finally:
        _avalanche.build_avalanche_backend = original_avalanche_builder
        _v10229.AuditedCoupledPersistentSiteCyclicTipEngine = original_engine
        _energy_gate.energy_gate_event_length = original_energy_search
        _crystal.cleavage_branch_candidates = original_discrete
        _crystal.cleave_direction_competition = original_compete
        _fem.assemble_mechanics = original_assemble
        OBSERVER.original_assemble = None


if __name__ == "__main__":
    main()
