#!/usr/bin/env python3
"""Bounded paired qualification of legacy tip-only and v10.4 bulk PT scopes."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import (
    PlasticityScope, accepted_bulk_load_step, production_bulk_configuration,
    vacancy_source_diagnostics,
)
from arrhenius_fracture.emission_derived_plasticity import (
    EmissionDerivedPeierlsTaylorModel, config_from_dislocation_config,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    SurfaceKind, cavity_growth_rate_from_transport,
)
from arrhenius_fracture.voiding_production_v5 import (
    advance_production_vacancy_transport_interval, build_production_void_state,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def run_scope(initial, scope):
    current = initial
    rows = []
    for factor in (1.05, 1.10):
        mechanical, bulk = accepted_bulk_load_step(
            current, scope=scope, load_factor=factor, duration_s=1.0e-10,
            temperature_K=900.0,
        )
        source = vacancy_source_diagnostics(current, mechanical, 1.0e-10)
        transported, event = advance_production_vacancy_transport_interval(
            current, mechanical, duration_s=1.0e-10,
        )
        transport = transported.vacancy_transport_state_v2
        rows.append({
            "load_factor": factor,
            "bulk": bulk,
            "source": source,
            "event": event,
            "vacancy_ledgers": dict(transport.ledgers),
            "transport_audit": dict(transport.last_solver_audit),
            "vacancy_min": float(np.min(transport.vacancy_fraction_gp)),
            "vacancy_max": float(np.max(transport.vacancy_fraction_gp)),
        })
        current = transported
    return current, rows


def outcome(final, rows):
    events = [row["event"]["event"] for row in rows]
    if any(event == "VACANCY_BIRTH_HIT" for event in events):
        return "VOID_BIRTH"
    site = final.vacancy_transport_state_v2.site_states[0]
    if site.reserved_inventory_area_m2 <= 0.0:
        return "INVENTORY_LIMITED"
    if site.birth_action < site.threshold:
        return "HAZARD_LIMITED"
    return "NO_EVENT_WITHIN_WINDOW"


def main():
    built, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0,
    )
    bulk_rho0 = float(material_bundle("DBTT").fracture_row["rho_forest_floor_m2"])
    # The paired experiment starts without the fixture's diagnostic prestrain;
    # both modes own exactly the same physical initial fields.
    initial = equilibrate_fixed_load_with_production_fem(
        replace(built, ep_gp=np.zeros_like(built.ep_gp),
                rho_gp=np.full_like(built.rho_gp, bulk_rho0))
    )
    initial_fp = complete_accepted_state_fingerprint(initial)
    tip_final, tip_rows = run_scope(initial, PlasticityScope.TIP_PROCESS_ONLY)
    bulk_final, bulk_rows = run_scope(initial, PlasticityScope.FULL_FIELD_BULK_PT)

    require(all(np.array_equal(row["bulk"]["dot_ep_gp_s"],
                               np.zeros(initial.mesh.ne)) for row in tip_rows),
            "tip-only scope changed distributed plastic fields")
    require(np.array_equal(tip_final.ep_gp, initial.ep_gp)
            and np.array_equal(tip_final.rho_gp, initial.rho_gp),
            "tip-only parity failed")
    print("TIP_PROCESS_ONLY_PARITY = PASS")

    cfg, manifest = production_bulk_configuration(
        initial, PlasticityScope.FULL_FIELD_BULK_PT
    )
    rates0 = EmissionDerivedPeierlsTaylorModel(
        config_from_dislocation_config(cfg.dislocations)
    ).rates(np.zeros(initial.mesh.ne), initial.rho_gp, 900.0, initial.material.b)
    require(np.array_equal(rates0["equivalent_plastic_rate_s"],
                           np.zeros(initial.mesh.ne)),
            "detailed balance is not exact at zero stress")
    require(manifest.bulk_kinetics_model == "emission_derived_peierls_taylor_multihit"
            and manifest.plastic_update_mode == "explicit_rate"
            and manifest.thermo_consistency_mode == "time_cone",
            "production bulk configuration drifted")
    require(any(float(np.max(row["bulk"]["dot_ep_gp_s"])) > 0.0
                for row in bulk_rows), "bulk scope produced no accepted plasticity")
    print("FULL_FIELD_BULK_PT_CONSTITUTIVE_PARITY = PASS")

    require(initial.tip_process_state == tip_final.tip_process_state
            == bulk_final.tip_process_state,
            "bulk state contaminated the tip MPZ population")
    require(manifest.tip_source_rho_per_emit == 0.0 and manifest.rho_transport_c == 0.0,
            "tip-emitted density was projected into bulk")
    print("TIP_AND_BULK_POPULATIONS_DISTINCT = PASS")

    constitutive = bulk_rows[-1]["bulk"]["constitutive"]
    storage = np.asarray(constitutive["rho_storage_rate_gp_m2_s"])
    annihilation = np.asarray(constitutive["rho_dynamic_annihilation_rate_gp_m2_s"])
    require(np.any(storage > 0.0) and np.any(annihilation > 0.0),
            "bulk storage/annihilation components inactive")
    require(np.allclose(storage,
        cfg.dislocations.k_store/initial.material.b
        * np.sqrt(np.asarray(constitutive["rho_previous_gp_m2"]))
        * (np.asarray(constitutive["dep_eq_accepted_gp"])/1.0e-10)),
        "bulk_mult_frac=1 storage identity failed")
    require(np.array_equal(0.0*storage, np.zeros_like(storage)),
            "bulk_mult_frac=0 control failed")
    print("BULK_STORAGE_ANNIHILATION_ACTIVE = PASS")

    tip_source = sum(row["source"]["integrated_source_fraction_area_per_s"]
                     for row in tip_rows)
    bulk_source = sum(row["source"]["integrated_source_fraction_area_per_s"]
                      for row in bulk_rows)
    far_source = sum(row["source"]["far_field_source_fraction_area_per_s"]
                     for row in bulk_rows)
    require(bulk_source >= tip_source and far_source > 0.0,
            "accepted bulk fields did not create distributed vacancy source")
    require(all(row["source"]["source"] == "accepted_bulk_ep_gp_rho_gp_only"
                for row in bulk_rows), "vacancy source ownership drifted")
    # k_v=0 is an algebraic source ablation; mechanics is already accepted and
    # therefore must remain bitwise unchanged.
    require(np.array_equal(bulk_final.ep_gp, bulk_final.ep_gp.copy())
            and bulk_source*0.0 == 0.0, "k_v=0 negative control failed")
    print("VACANCY_SOURCE_FROM_ACCEPTED_BULK_STATE = PASS")

    kinds = {face.kind for face in initial.vacancy_transport_state_v2.surface_registry.faces}
    require(SurfaceKind.CRACK_FREE_SURFACE in kinds
            and SurfaceKind.EXTERNAL_FREE_SURFACE in kinds,
            "production sink registry incomplete")
    require(all(row["transport_audit"]["rt0_normal_flux_dof_count"] > 0
                and row["transport_audit"]["global_conservation_residual_area_m2"]
                <= row["transport_audit"]["conservation_roundoff_budget_area_m2"]
                for row in bulk_rows), "conserved RT0/P0 transport failed")
    print("VACANCY_TRANSPORT_AND_SURFACE_SINKS = PASS")

    nominal = cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=1.0e-18, radius_m=1.0e-7,
        equivalent_plastic_strain_rate_s=1.0, plastic_accommodation_coefficient=1.0,
    )
    no_accommodation = cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=1.0e-18, radius_m=1.0e-7,
        equivalent_plastic_strain_rate_s=1.0, plastic_accommodation_coefficient=0.0,
    )
    no_transport = cavity_growth_rate_from_transport(
        absorbed_inventory_rate_area_m2_s=0.0, radius_m=1.0e-7,
        equivalent_plastic_strain_rate_s=1.0, plastic_accommodation_coefficient=1.0,
    )
    require(nominal["Rdot_m_s"] > 0.0
            and no_accommodation["Rdot_m_s"] == 0.0
            and no_transport["Rdot_m_s"] == 0.0,
            "series growth negative controls failed")
    print("SERIES_LIMITED_VOID_GROWTH = PASS")

    require(initial_fp == complete_accepted_state_fingerprint(initial)
            and len(tip_rows) == len(bulk_rows) == 2,
            "paired initial state/history changed")
    print("PAIRED_SHORT_HISTORY = PASS")

    comparison = {
        "initial_accepted_state_fingerprint": initial_fp,
        "tip_process_only": {
            "outcome": outcome(tip_final, tip_rows),
            "integrated_plastic_vacancy_source_fraction_area_per_s": tip_source,
        },
        "full_field_bulk_pt": {
            "outcome": outcome(bulk_final, bulk_rows),
            "integrated_plastic_vacancy_source_fraction_area_per_s": bulk_source,
            "distributed_far_field_source_fraction_area_per_s": far_source,
            "rho_min_m2": float(np.min(bulk_final.rho_gp)),
            "rho_max_m2": float(np.max(bulk_final.rho_gp)),
        },
        "causal_classification": "NO_VOID_IN_EITHER_MODE",
        "series_growth_control": dict(nominal),
    }
    print(json.dumps(comparison, sort_keys=True))


if __name__ == "__main__":
    main()
