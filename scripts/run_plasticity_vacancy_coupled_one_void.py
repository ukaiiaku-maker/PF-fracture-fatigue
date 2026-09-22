#!/usr/bin/env python3
"""Bounded causal plasticity -> vacancy -> one-void production sentinel."""
from __future__ import annotations

from dataclasses import replace
import json
import math
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import (
    material_bundle, require_bound_identity,
)
from arrhenius_fracture.vacancy_coupling_v1 import (
    REFERENCE_VACANCY_ROW, equilibrium_vacancy_fraction, vacancy_conservation,
    vacancy_diffusivity_m2_s,
)
from arrhenius_fracture.voiding_production_v5 import (
    deterministic_trajectory, downstream_front_transaction,
    equilibrate_fixed_load_with_production_fem, fresh_sharp_front_engine,
    ligament_transaction, refine_downstream_source,
)
from arrhenius_fracture.voiding_v5 import VoidPhase


TEMPERATURE_K = 900.0
RETAINED_CRACK_PATH_M = (
    (0.0, 8.572527594031472e-20),
    (0.0005725993004046688, 1.5602085896855586e-20),
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def checkpoint_exact(state, path):
    fingerprint = complete_accepted_state_fingerprint(state)
    write_checkpoint(state, path)
    restored = restore_checkpoint(path)
    require(complete_accepted_state_fingerprint(restored) == fingerprint,
            "checkpoint/restart changed complete accepted ownership")
    require(np.array_equal(restored.vacancy_fraction_gp, state.vacancy_fraction_gp),
            "checkpoint/restart changed vacancy field")
    require(dict(restored.vacancy_ledgers) == dict(state.vacancy_ledgers),
            "checkpoint/restart changed vacancy ledgers")
    return restored


def require_conservation(state, label):
    residual = vacancy_conservation(state)["vacancy_conservation_residual_m2"]
    require(abs(residual) <= 1.0e-24, label + " vacancy conservation failed")


def main():
    bundle = material_bundle("DBTT")
    trace = []
    state, rows = deterministic_trajectory(
        bundle=bundle, stop_before_ligament=True,
        crack_path_m=RETAINED_CRACK_PATH_M,
        boundary_segments=128, radial_layers=48,
        shape_regular_local_level="B", state_trace=trace,
        sentinel_initial_vacancy_fraction_gp=1.0,
        vacancy_coupled=True,
    )
    by_operation = {row["operation"]: row for row in rows}
    for label, accepted in trace:
        require_conservation(accepted, label)

    plastic = by_operation["plastic_vacancy_production"]["plastic_vacancy_audit"]
    vacancy_update = plastic["vacancy_update"]
    require(vacancy_update["vacancy_plastic_production_area_m2"] > 0.0,
            "accepted plastic interval produced no vacancies")
    require(by_operation["plastic_vacancy_production"]["vacancy_fraction_max"]
            > equilibrium_vacancy_fraction(TEMPERATURE_K),
            "explicit sentinel field is not supersaturated")
    print("PLASTIC_VACANCY_PRODUCTION", flush=True)
    print("VACANCY_SUPERSATURATION", flush=True)
    print("MULTI_HIT_EMBRYO", flush=True)
    print("STABLE_SUBGRID_VOID", flush=True)

    growth_rows = (
        by_operation["subgrid_growth"], by_operation["resolved_growth"],
    )
    for row in growth_rows:
        audit = row["growth_audit"]
        require(audit["Rdot_diff_m_s"] > 0.0 and audit["Rdot_plast_m_s"] > 0.0,
                "series growth did not own both required branches")
        require(audit["Rdot_m_s"] > 0.0,
                "series growth has no accepted radius rate")
    print("VACANCY_LIMITED_GROWTH", flush=True)
    print("PLASTIC_ACCOMMODATION", flush=True)
    print("GEOMETRIC_PROMOTION", flush=True)

    state = equilibrate_fixed_load_with_production_fem(
        replace(state, displacement=2.0 * state.displacement)
    )
    connected, ligament = ligament_transaction(state)
    require(ligament.accepted, "ligament connection was rejected")
    require(connected.void_state.cavities[0].phase == VoidPhase.CONNECTED_VOID,
            "ligament did not create CONNECTED_VOID")
    require_conservation(connected, "ligament connection")
    print("VOID_CONNECTED", flush=True)

    source, source_audit = refine_downstream_source(
        connected, max_refinement_levels=1,
        refinement_region="complete_cavity_ring",
        quality_improvement="constrained_v1",
    )
    require(source_audit["status"] == "SOURCE_TENSOR_QUALIFIED",
            "downstream source did not qualify")
    require_conservation(source, "downstream source")
    child, birth, _, _ = downstream_front_transaction(source)
    require(birth is not None and birth.accepted, "downstream child was not accepted")
    require_conservation(child, "downstream child")
    print("DOWNSTREAM_CHILD_CREATED", flush=True)
    terminal, continuation, _, _ = downstream_front_transaction(child, continuation=True)
    require(continuation is not None and continuation.accepted,
            "downstream child continuation was not accepted")
    require_conservation(terminal, "child continuation")
    print("DOWNSTREAM_CHILD_CONTINUED", flush=True)

    stage_states = dict(trace)
    stage_states.update({
        "annihilation": stage_states["plastic_vacancy_production"],
        "embryo": stage_states["multi_hit_2"],
        "ligament_connection": connected,
        "downstream_source": source,
        "child_continuation": terminal,
    })
    required_stages = {
        "plastic_vacancy_production", "annihilation", "multi_hit_1", "embryo",
        "stabilization", "subgrid_growth", "geometric_promotion",
        "resolved_growth", "ligament_connection", "downstream_source",
        "child_continuation",
    }
    require(required_stages <= set(stage_states), "restart stage coverage is incomplete")
    with tempfile.TemporaryDirectory(prefix="plasticity-vacancy-one-void-") as temporary:
        directory = Path(temporary)
        for label in sorted(required_stages):
            checkpoint_exact(stage_states[label], directory / f"{label}.json")

    identity = dict(require_bound_identity(terminal))
    cavity = terminal.void_state.cavities[0]
    r_tip = fresh_sharp_front_engine(terminal.material, bundle).r_eff()
    require(r_tip != cavity.radius_m, "r_tip was aliased to R_void")
    conservation = vacancy_conservation(terminal)
    final_growth = by_operation["resolved_growth"]["growth_audit"]
    report = {
        "temperature_K": TEMPERATURE_K,
        "c_eq": equilibrium_vacancy_fraction(TEMPERATURE_K),
        "D_v_m2_s": vacancy_diffusivity_m2_s(TEMPERATURE_K),
        "maximum_c_v": float(np.max(terminal.vacancy_fraction_gp)),
        "integrated_c_v_area_m2": conservation["vacancy_field_inventory_area_m2"],
        "plastic_production_area_m2": terminal.vacancy_ledgers["vacancy_plastic_production_area_m2"],
        "sink_annihilation_area_m2": terminal.vacancy_ledgers["vacancy_annihilation_area_m2"],
        "cavity_consumption_area_m2": terminal.vacancy_ledgers["vacancy_consumed_by_void_area_m2"],
        "Rdot_diff_m_s": final_growth["Rdot_diff_m_s"],
        "Rdot_plast_m_s": final_growth["Rdot_plast_m_s"],
        "Rdot_m_s": final_growth["Rdot_m_s"],
        "controlling_branch": final_growth["controlling_branch"],
        "peierls_taylor_material_identity": identity["plasticity_fingerprint"],
        "vacancy_row_id": REFERENCE_VACANCY_ROW.row_id,
        "vacancy_parameter_status": REFERENCE_VACANCY_ROW.calibration_status,
        "r_tip_m": r_tip,
        "R_void_m": cavity.radius_m,
        "vacancy_conservation_residual_m2": conservation["vacancy_conservation_residual_m2"],
        "checkpoint_restart_stage_count": len(required_stages),
    }
    require(all(math.isfinite(float(report[key])) for key in (
        "temperature_K", "c_eq", "D_v_m2_s", "maximum_c_v",
        "integrated_c_v_area_m2", "Rdot_diff_m_s", "Rdot_plast_m_s",
        "Rdot_m_s", "r_tip_m", "R_void_m", "vacancy_conservation_residual_m2",
    )), "terminal report contains a nonfinite quantity")
    print(json.dumps(report, sort_keys=True, indent=2))
    print("TEST_ONLY_EXTREME_RESERVOIR_PLUMBING = PASS")
    print("PHYSICAL_PLASTICITY_GENERATED_VOID_NUCLEATION = NOT_DEMONSTRATED")


if __name__ == "__main__":
    main()
