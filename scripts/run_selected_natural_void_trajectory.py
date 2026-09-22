#!/usr/bin/env python3
"""Selected unscaled combined-row natural-void trajectory with step restarts."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import (
    PlasticityScope, accepted_bulk_load_step, vacancy_source_diagnostics,
)
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import accepted_interaction_opportunity
from arrhenius_fracture.global_physical_scheduler_v1 import advance_global_physical_interval
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from scripts.run_active_two_mechanism_natural_void_study import (
    TEMPERATURE_K, canonical_drive_check, finite, make_initial, site_source_exposure,
)

OUT = ROOT/"artifacts/informative_selected_natural_void_trajectory"
STEP_DURATIONS_S = (1.0e-3, 9.0e-3, 9.0e-2, 9.0e-1)
INITIAL_LOAD_FACTOR = 6.0


def dump(path, value):
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True)+"\n")


def make_plot(history):
    import matplotlib.pyplot as plt
    time = np.asarray([row["time_s"] for row in history])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for site in ("site-1", "site-2"):
        axes[0].plot(time, [row["vacancy_birth_action"][site] for row in history],
                     "o-", label=site+" vacancy")
        axes[0].plot(time, [row["direct_interaction_action"][site] for row in history],
                     "s--", label=site+" direct")
    axes[0].set_yscale("log"); axes[0].set_xscale("log")
    axes[0].set_xlabel("time (s)"); axes[0].set_ylabel("accumulated action")
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.2)
    axes[1].plot(time, [row["transport_ledgers"]["plastic_production_area_m2"]
                        for row in history], "o-", label="plastic production")
    axes[1].plot(time, [row["transport_ledgers"]["dislocation_climb_annihilation_area_m2"]
                        for row in history], "s-", label="climb annihilation")
    axes[1].plot(time, [sum(row["site_reserved_inventory_area_m2"].values())
                        for row in history], "^-", label="site reserved inventory")
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("time (s)"); axes[1].set_ylabel("cumulative area ledger (m²)")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(OUT/"source_capture_sink_action.png", dpi=180)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoints = sorted(OUT.glob("step_*_restart.json"))
    progress_path = OUT/"progress.json"
    if checkpoints:
        state = restore_checkpoint(checkpoints[-1])
        history = json.loads(progress_path.read_text())["history"]
        start_index = len(history)
    else:
        state = make_initial("combined")
        history = []
        start_index = 0
        write_checkpoint(state, OUT/"step_000_restart.json", compression="gzip")
    for index in range(start_index, len(STEP_DURATIONS_S)):
        dt = STEP_DURATIONS_S[index]
        previous = state
        mechanical, plastic = accepted_bulk_load_step(
            previous, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
            load_factor=INITIAL_LOAD_FACTOR if index == 0 else 1.0,
            duration_s=dt, temperature_K=TEMPERATURE_K,
            maximum_stagger_iterations=80,
        )
        exposure, source = site_source_exposure(previous, mechanical, dt)
        tensor, K, canonical, load_rows = canonical_drive_check(mechanical)
        epdot = equivalent_plastic_strain_rate_s(previous.ep_gp, mechanical.ep_gp, dt)
        direct = [accepted_interaction_opportunity(
            mechanical, site, temperature_K=TEMPERATURE_K,
            accepted_equivalent_plastic_strain_rate_s=epdot,
        ) for site in mechanical.void_state.dislocation_interaction_state.sites]
        state, scheduler = advance_global_physical_interval(
            previous, mechanical, dt,
            {"kind": "hold", "temperature_K": TEMPERATURE_K,
             "stress_tensor_Pa": tensor,
             "K_effective_by_front": {ROOT_BRANCH_ID: K},
             "directional_load_rows_by_front": {ROOT_BRANCH_ID: load_rows},
             "maximum_events": 16},
        )
        checkpoint = OUT/f"step_{index+1:03d}_restart.json"
        manifest = write_checkpoint(state, checkpoint, compression="gzip")
        restored = restore_checkpoint(checkpoint)
        transport = state.vacancy_transport_state_v2
        direct_owner = state.void_state.dislocation_interaction_state
        history.append(finite({
            "step": index+1, "dt_s": dt,
            "time_s": math.fsum(STEP_DURATIONS_S[:index+1]),
            "load_factor": INITIAL_LOAD_FACTOR if index == 0 else 1.0,
            "canonical_drive": canonical,
            "K_effective_Pa_sqrt_m": K,
            "maximum_plastic_increment": float(np.max(epdot)*dt),
            "plastic_work_J_per_m": plastic["accepted_endpoint_path_plastic_work_J_per_m"],
            "integrated_plastic_source_fraction_area": source[
                "integrated_source_fraction_area_per_s"]*dt,
            "site_source_exposure_fraction_area": exposure,
            "site_reserved_inventory_area_m2": {
                site.site_id: site.reserved_inventory_area_m2
                for site in transport.site_states},
            "vacancy_birth_action": {
                site.site_id: site.birth_action for site in transport.site_states},
            "direct_interaction_action": {
                site.site_id: site.interaction_action for site in direct_owner.sites},
            "direct_opportunity": direct,
            "transport_ledgers": dict(transport.ledgers),
            "transport_solver_audit": dict(transport.last_solver_audit),
            "site_phases": {site.site_id: site.phase.value for site in state.void_state.sites},
            "cavity_count": len(state.void_state.cavities),
            "selected_events": [event["selected_event"] for event in scheduler["events"]
                                if event["selected_event"]],
            "all_owners_previewed": scheduler["all_owners_previewed"],
            "restart_manifest": checkpoint.name,
            "restart_state_file": manifest["state_file"],
            "direct_restored_equal": (
                complete_accepted_state_fingerprint(state)
                == complete_accepted_state_fingerprint(restored)),
        }))
        dump(progress_path, {"history": history})
        print("SELECTED_NATURAL_STEP_COMPLETE", index+1, flush=True)
    final = {
        "classification": "SELECTED_UNSCALED_COMBINED_NATURAL_VOID_TRAJECTORY",
        "protocol_selected_before_outcome": True,
        "material_family": "DBTT", "vacancy_energy_variant": "combined",
        "temperature_K": TEMPERATURE_K, "initial_load_factor": INITIAL_LOAD_FACTOR,
        "step_durations_s": STEP_DURATIONS_S,
        "history": history,
        "terminal_site_phases": {site.site_id: site.phase.value
                                 for site in state.void_state.sites},
        "terminal_cavity_count": len(state.void_state.cavities),
        "latest_valid_restart": history[-1]["restart_manifest"],
        "first_site_outcome": state.void_state.sites[0].phase.value,
        "second_site_outcome": state.void_state.sites[1].phase.value,
        "outcome": ("VOID_BIRTH_BEFORE_CRACK" if state.void_state.cavities else
                    "HAZARD_LIMITED" if any(
                        site.birth_action > 0.0 for site in state.vacancy_transport_state_v2.site_states)
                    else "NO_EVENT_WITHIN_WINDOW"),
    }
    dump(OUT/"result.json", final)
    transport = state.vacancy_transport_state_v2
    np.savez_compressed(
        OUT/"terminal_fields.npz",
        vacancy_fraction_gp=np.asarray(transport.vacancy_fraction_gp),
        mechanics_ep_gp=np.asarray(state.ep_gp), rho_gp_m2=np.asarray(state.rho_gp),
        transport_nodes_m=np.asarray(transport.transport_mesh.nodes),
        transport_elements=np.asarray(transport.transport_mesh.elems),
    )
    make_plot(history)


if __name__ == "__main__":
    main()
