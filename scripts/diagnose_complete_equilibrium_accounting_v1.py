#!/usr/bin/env python3
"""Same-state complete-force diagnosis at the 8.47499-s candidate checkpoint."""
from __future__ import annotations

import json
from pathlib import Path
import sys
from collections.abc import Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import PlasticityScope
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.dislocation_interaction_voiding_v1 import accepted_interaction_opportunity
from arrhenius_fracture.fem import assemble_mechanics
from arrhenius_fracture.topology_transaction_v11 import (
    complete_equilibrium_force_accounting,
    equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from scripts.run_active_two_mechanism_natural_void_study import site_source_exposure
from scripts.run_sustained_growth_candidate_v1 import canonical_drive_check


RUN = ROOT/"runs/managed_sustained_growth_v1_20260920/output"
CHECKPOINT = RUN/"restart_000008475ms.json"
PREVIOUS = RUN/"restart_000008450ms.json"
OUT = ROOT/"runs/managed_sustained_growth_v1_20260920/complete_force_diagnosis_v1.json"


def rel(a: float, b: float, floor: float = 1.0e-300) -> float:
    return abs(float(a)-float(b))/max(abs(float(b)), floor)


def plain(value):
    if isinstance(value, Mapping):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> None:
    state = restore_checkpoint(CHECKPOINT)
    previous = restore_checkpoint(PREVIOUS)
    _, residual, sigma, *_ = assemble_mechanics(
        state.mesh, state.displacement, state.ep_gp, state.rho_gp,
        state.damage, state.elasticity_D, state.material,
        cohesive_network=state.cohesive_network)
    accounting = complete_equilibrium_force_accounting(state, residual)
    repaired = equilibrate_fixed_load_with_production_fem(state)
    repaired_accounting = repaired.energy_ledgers[
        "complete_equilibrium_force_accounting"]

    _, K_old, canonical_old, rows_old = canonical_drive_check(state)
    _, K_new, canonical_new, rows_new = canonical_drive_check(repaired)
    duration = 0.025
    old_exposure, old_source = site_source_exposure(previous, state, duration)
    new_exposure, new_source = site_source_exposure(previous, repaired, duration)
    epdot_old = equivalent_plastic_strain_rate_s(
        previous.ep_gp, state.ep_gp, duration)
    epdot_new = equivalent_plastic_strain_rate_s(
        previous.ep_gp, repaired.ep_gp, duration)
    direct_old = [accepted_interaction_opportunity(
        state, site, temperature_K=1200.0,
        accepted_equivalent_plastic_strain_rate_s=epdot_old)
        for site in state.void_state.dislocation_interaction_state.sites]
    direct_new = [accepted_interaction_opportunity(
        repaired, site, temperature_K=1200.0,
        accepted_equivalent_plastic_strain_rate_s=epdot_new)
        for site in repaired.void_state.dislocation_interaction_state.sites]
    eq_old = state.equilibrium_observables
    eq_new = repaired.equilibrium_observables
    payload = {
        "schema": "unified-2d.complete-equilibrium-stop-diagnosis/1",
        "checkpoint": str(CHECKPOINT),
        "physical_time_s": state.junction_process_state["production_time_s"],
        "cause": (
            "TOP_BOTTOM_DIAGNOSTIC_OMITTED_THE_EXPLICITLY_CONSTRAINED_"
            "LEFT_AUXILIARY_PIN_Y_REACTION"),
        "repair_kind": "OBSERVATION_AND_ACCEPTANCE_ACCOUNTING_ONLY",
        "boundary_value_problem_changed": False,
        "physical_clocks_advanced": False,
        "complete_force_accounting": accounting,
        "repaired_complete_force_accounting": repaired_accounting,
        "same_state_re_equilibration": {
            "maximum_displacement_change_m": float(np.max(np.abs(
                repaired.displacement-state.displacement))),
            "plastic_strain_bitwise_equal": bool(np.array_equal(repaired.ep_gp, state.ep_gp)),
            "density_bitwise_equal": bool(np.array_equal(repaired.rho_gp, state.rho_gp)),
            "damage_bitwise_equal": bool(np.array_equal(repaired.damage, state.damage)),
            "stored_energy_old_J_per_m": state.stored_energy_J_per_m,
            "stored_energy_new_J_per_m": repaired.stored_energy_J_per_m,
            "stored_energy_relative_change": rel(
                repaired.stored_energy_J_per_m, state.stored_energy_J_per_m),
            "top_reaction_old_N_per_m": eq_old.top_normal_resultant_N_per_m,
            "top_reaction_new_N_per_m": eq_new.top_normal_resultant_N_per_m,
            "bottom_reaction_old_N_per_m": eq_old.bottom_normal_resultant_N_per_m,
            "bottom_reaction_new_N_per_m": eq_new.bottom_normal_resultant_N_per_m,
            "top_bottom_diagnostic_old": eq_old.top_bottom_balance_residual,
            "top_bottom_diagnostic_new": eq_new.top_bottom_balance_residual,
            "boundary_work_old_J_per_m": eq_old.external_reaction_work_J_per_m,
            "boundary_work_new_J_per_m": eq_new.external_reaction_work_J_per_m,
            "energy_identity_old": eq_old.energy_identity_residual,
            "energy_identity_new": eq_new.energy_identity_residual,
        },
        "canonical_drive_comparison": {
            "K_old_Pa_sqrt_m": K_old, "K_new_Pa_sqrt_m": K_new,
            "K_relative_change": rel(K_new, K_old),
            "G_old_J_per_m2": rows_old[0]["G_kinetic_used_J_per_m2"],
            "G_new_J_per_m2": rows_new[0]["G_kinetic_used_J_per_m2"],
            "G_relative_change": rel(
                rows_new[0]["G_kinetic_used_J_per_m2"],
                rows_old[0]["G_kinetic_used_J_per_m2"]),
            "old": canonical_old, "new": canonical_new,
        },
        "plastic_vacancy_source_comparison": {
            "site_exposure_old_fraction_area": old_exposure,
            "site_exposure_new_fraction_area": new_exposure,
            "integrated_old_fraction_area": (
                old_source["integrated_source_fraction_area_per_s"]*duration),
            "integrated_new_fraction_area": (
                new_source["integrated_source_fraction_area_per_s"]*duration),
        },
        "direct_interaction_opportunity_comparison": {
            "old": direct_old, "new": direct_new,
        },
    }
    OUT.write_text(json.dumps(plain(payload), indent=2, sort_keys=True)+"\n")
    print(OUT)
    print("ALL_SUPPORT_RESULTANT_RELATIVE",
          accounting["all_support_resultant_relative"])
    print("ALL_SUPPORT_MOMENT_RELATIVE",
          accounting["all_support_moment_relative"])
    print("MAXIMUM_DISPLACEMENT_CHANGE_M",
          payload["same_state_re_equilibration"]["maximum_displacement_change_m"])
    print("K_RELATIVE_CHANGE", payload["canonical_drive_comparison"]["K_relative_change"])


if __name__ == "__main__":
    main()
