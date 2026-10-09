#!/usr/bin/env python3
"""Unscaled reference/combined natural-voiding comparison after preflight."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, build_production_void_state, crack_tip_tensor,
    fresh_sharp_front_engine,
)

TEMPERATURE_K = 900.0
HOLD_S = 3.0e5
VARIANTS = ("reference", "combined")


def _preflight_module():
    spec = importlib.util.spec_from_file_location(
        "vacancy_preflight", ROOT/"scripts/preflight_vacancy_energy_variants.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def finite(value):
    if isinstance(value, dict): return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)): return [finite(v) for v in value]
    if isinstance(value, np.ndarray): return finite(value.tolist())
    if isinstance(value, np.generic): return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def field_inventory(transport):
    return float(np.sum(np.asarray(transport.transport_mesh.area_e)
                        *np.asarray(transport.vacancy_fraction_gp))
                 *transport.parameter_row.inventory_ratio)


def run_variant(variant, outdir):
    preflight = _preflight_module()
    initial, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=TEMPERATURE_K,
        vacancy_energy_variant=variant,
        void_site_centers_m=preflight.SITES_M, stochastic=True, seed=3621)
    mechanical, epdot, plastic_audit = preflight.accepted_plastic_endpoint(
        initial, TEMPERATURE_K)
    tensor, tensor_audit = crack_tip_tensor(mechanical, branch_id=ROOT_BRANCH_ID)
    engine = fresh_sharp_front_engine(mechanical.material, _bundle_from_state(mechanical))
    sigma = max(float(np.linalg.eigvalsh(np.asarray(tensor))[-1]), 0.0)
    K = sigma*math.sqrt(2.0*math.pi*engine.r_eff())
    history = {
        "kind": "hold", "temperature_K": TEMPERATURE_K, "maximum_events": 16,
        "stress_tensor_Pa": tensor,
        "K_effective_by_front": {ROOT_BRANCH_ID: K},
    }
    after_load, load_audit = advance_global_physical_interval(
        initial, mechanical, preflight.INTERVAL_S, history)
    final, hold_audit = advance_global_physical_interval(
        after_load, after_load, HOLD_S, history)
    checkpoint = outdir/f"{variant}_terminal.json"
    write_checkpoint(final, checkpoint, compression="gzip")
    restored = restore_checkpoint(checkpoint)
    restart_exact = (complete_accepted_state_fingerprint(restored)
                     == complete_accepted_state_fingerprint(final))
    transport = final.vacancy_transport_state_v2
    sites = list(transport.site_states)
    hit_inventory = min(site.required_embryo_inventory_area_m2/site.required_hit_count
                        for site in sites)
    total_available = (float(transport.ledgers["initial_owned_inventory_area_m2"])
                       +float(transport.ledgers["plastic_production_area_m2"])
                       -float(transport.ledgers["crack_surface_absorption_area_m2"])
                       -float(transport.ledgers["external_surface_absorption_area_m2"])
                       -float(transport.ledgers["dislocation_climb_annihilation_area_m2"])
                       +float(transport.ledgers["dislocation_climb_emission_area_m2"]))
    reserved_total = math.fsum(site.reserved_inventory_area_m2 for site in sites)
    selected = [row["selected_event"] for row in
                load_audit["events"]+hold_audit["events"] if row["selected_event"]]
    if any(event == "DIRECTIONAL_CLEAVAGE" for event in selected):
        outcome = "DIRECT_CRACK_BEFORE_VOID"
    elif any(event == "VACANCY_BIRTH_HIT" for event in selected):
        outcome = "VOID_BIRTH_BEFORE_CRACK"
    elif reserved_total < hit_inventory:
        outcome = "INVENTORY_LIMITED"
    elif all(site.birth_action < site.next_birth_threshold() for site in sites):
        outcome = "HAZARD_LIMITED"
    else:
        outcome = "NO_EVENT_WITHIN_WINDOW"
    centers = np.asarray(transport.transport_mesh.nodes)[
        np.asarray(transport.transport_mesh.elems)].mean(axis=1)
    np.savez_compressed(
        outdir/f"{variant}_fields.npz", centers_xy_m=centers,
        vacancy_fraction=np.asarray(transport.vacancy_fraction_gp),
        rt0_flux_vector_at_element_centroid=np.asarray(
            transport.last_solver_audit["rt0_flux_vector_at_element_centroid"]),
        accepted_epdot_s=np.asarray(epdot), rho_m2=np.asarray(final.rho_gp))
    return finite({
        "variant": variant, "outcome": outcome,
        "temperature_K": TEMPERATURE_K, "hold_s": HOLD_S,
        "physical_rate_scaling": 1.0,
        "initial_cavity_count": 0,
        "predefined_site_count": len(initial.void_state.sites),
        "final_cavity_count": len(final.void_state.cavities),
        "site_states": [site.__dict__ for site in sites],
        "parameter_row": transport.parameter_row.__dict__,
        "parameter_row_fingerprint": transport.parameter_row.fingerprint,
        "initial_fingerprint": complete_accepted_state_fingerprint(initial),
        "mechanical_fingerprint": complete_accepted_state_fingerprint(mechanical),
        "final_fingerprint": complete_accepted_state_fingerprint(final),
        "checkpoint_restart_exact": restart_exact,
        "accepted_maximum_epdot_s": float(np.max(epdot)),
        "plasticity_model": plastic_audit["bulk_kinetics_model"],
        "field_inventory_area_m2": field_inventory(transport),
        "total_available_inventory_upper_bound_area_m2": total_available,
        "reserved_site_inventory_area_m2": reserved_total,
        "minimum_inventory_per_hit_area_m2": hit_inventory,
        "inventory_shortfall_factor": hit_inventory/max(total_available, 1.0e-300),
        "reserved_inventory_shortfall_factor": hit_inventory/max(reserved_total, 1.0e-300),
        "vacancy_fraction_min": float(np.min(transport.vacancy_fraction_gp)),
        "vacancy_fraction_max": float(np.max(transport.vacancy_fraction_gp)),
        "ledgers": dict(transport.ledgers),
        "last_transport_solver_audit": dict(transport.last_solver_audit),
        "selected_events": selected,
        "load_scheduler_segments": load_audit["events"],
        "hold_scheduler_segments": hold_audit["events"],
        "crack_tip_tensor_Pa": tensor,
        "crack_tip_tensor_audit": tensor_audit,
        "mechanism_history": {
            "embryo_count": sum(site.phase.value == "EMBRYO" for site in final.void_state.sites),
            "stable_void_count": sum(site.phase.value == "STABLE_SUBGRID_VOID"
                                     for site in final.void_state.sites),
            "second_void_target_reached": len(final.void_state.cavities) >= 2,
        },
        "fields_file": str((outdir/f"{variant}_fields.npz").relative_to(ROOT)),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT/"artifacts/natural_voiding_energy_comparison")
    args = parser.parse_args(); args.out.mkdir(parents=True, exist_ok=True)
    rows = [run_variant(variant, args.out) for variant in VARIANTS]
    payload = {
        "schema": "unified-2d.natural-voiding-energy-comparison/1",
        "calibration_status": "UNCALIBRATED_COMPARATIVE_STUDY",
        "common_protocol": {"temperature_K": TEMPERATURE_K,
                            "plastic_load_factor": 1.05,
                            "plastic_increment_duration_s": 1.0e-10,
                            "hold_s": HOLD_S, "physical_rate_scaling": 1.0},
        "rows": rows,
    }
    (args.out/"comparison.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("NATURAL_VOIDING_ENERGY_COMPARISON = PASS")
    for row in rows:
        print(row["variant"], row["outcome"], row["final_cavity_count"],
              row["reserved_inventory_shortfall_factor"],
              row["checkpoint_restart_exact"])


if __name__ == "__main__":
    main()
