#!/usr/bin/env python3
"""Resumable full 2-D trajectory for the prospectively selected hypothesis."""
from __future__ import annotations

from dataclasses import replace
import json
import math
import os
from pathlib import Path
import sys
import time
from types import MappingProxyType

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import PlasticityScope, accepted_bulk_load_step
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.dislocation_interaction_voiding_v1 import (
    DislocationInteractionParameterRowV1,
    accepted_interaction_opportunity,
)
from arrhenius_fracture.embryo_growth_v1 import CavityEmissionParameterRowV1
from arrhenius_fracture.global_physical_scheduler_v1 import (
    AcceptedIntervalContinuationV1,
    advance_global_physical_interval,
)
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_coupling_v1 import equivalent_plastic_strain_rate_s
from arrhenius_fracture.voiding_production_v5 import build_production_void_state
from scripts.run_active_two_mechanism_natural_void_study import (
    canonical_drive_check,
    finite,
    site_source_exposure,
)


TEMPERATURE_K = 1200.0
SEED = 3621
SITES_M = ((6.0e-4, 0.0), (7.5e-4, 0.0))
PARTITION = (0.50, 0.10, 0.40)
TOTAL_TIME_S = 20.0
MAX_INTERNAL_DT_S = 0.10
EARLY_STEPS_S = (1e-5, 2e-5, 7e-5, 2e-4, 7e-4, 2e-3, 7e-3, 2e-2, 7e-2)
MAX_EQUIVALENT_PLASTIC_STRAIN = 0.05
CHECKPOINT_INTERVAL_S = 0.5


def atomic_dump(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(finite(value), indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def pickle_safe_metadata(value):
    """Copy immutable audit views without altering accepted physical objects."""
    if isinstance(value, (dict, MappingProxyType)):
        return {key: pickle_safe_metadata(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(pickle_safe_metadata(item) for item in value)
    if isinstance(value, list):
        return [pickle_safe_metadata(item) for item in value]
    return value


def accepted_time(state) -> float:
    return float(state.junction_process_state.get(
        "production_time_s", state.vacancy_transport_state_v2.transport_time_s,
    ))


def phase_summary(state):
    growth = {
        item.site_id: {
            "radius_m": item.radius_m,
            "area_m2": item.area_m2,
            "vacancy_derived_area_m2": item.vacancy_derived_area_m2,
            "dislocation_derived_area_m2": item.dislocation_derived_area_m2,
            "emitted_loop_count_effective": (
                item.cavity_emission_state.cumulative_loop_count_effective
            ),
            "emission_sources": tuple({
                "source_id": source.source_id,
                "regime": source.availability_flow_regime,
                "available_fraction": source.available_fraction,
                "integrated_action": source.integrated_action,
                "realized_loop_count": source.realized_loop_count,
                "continuum_moment_loop_count": source.continuum_moment_loop_count,
            } for source in item.cavity_emission_state.sources),
        }
        for item in state.void_state.embryo_growth_states
    }
    direct = state.void_state.dislocation_interaction_state
    transport = state.vacancy_transport_state_v2
    return {
        "site_phases": {item.site_id: item.phase.value for item in state.void_state.sites},
        "direct_action": {item.site_id: item.interaction_action for item in direct.sites},
        "direct_hits": {item.site_id: item.interaction_hit_count for item in direct.sites},
        "vacancy_birth_action": {item.site_id: item.birth_action for item in transport.site_states},
        "vacancy_reserved_area_m2": {
            item.site_id: item.reserved_inventory_area_m2 for item in transport.site_states
        },
        "growth": growth,
        "cavity_count": len(state.void_state.cavities),
        "crack_length_m": state.crack_network.total_physical_crack_length_m,
    }


def create_initial(scout):
    direct = DislocationInteractionParameterRowV1(**scout["selected_parameter_row"])
    emission = CavityEmissionParameterRowV1(
        **scout["selected_cavity_emission_parameter_row"]
    )
    bundle = material_bundle("DBTT")
    return build_production_void_state(
        bundle=bundle, temperature_K=TEMPERATURE_K,
        vacancy_energy_variant="combined", void_site_centers_m=SITES_M,
        stochastic=True, seed=SEED,
        direct_interaction_birth_enabled=True,
        direct_interaction_parameter_row=direct,
        cavity_surface_loop_emission_enabled=True,
        cavity_emission_parameter_row=emission,
        interaction_population_partition=PARTITION,
        bulk_initial_density_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
        initial_plastic_strain_gp=0.0,
    )[0]


def main():
    out = Path(os.environ.get(
        "NATURAL_VOID_RUN_OUT", ROOT / "runs/natural_void_hypothesis_v1"
    )).resolve()
    out.mkdir(parents=True, exist_ok=True)
    atomic_dump(out / "active_worker.json", {
        "pid": os.getpid(), "source_root": str(ROOT),
        "started_wall_time_s": time.time(),
        "wall_time_chunk_limit_s": float(os.environ.get("MAX_WALL_SECONDS", "1800")),
    })
    scout_path = ROOT / "artifacts/natural_void_hypothesis_scout_v1/scout.json"
    scout = json.loads(scout_path.read_text())
    declared = {
        "schema": "unified-2d.natural-void-hypothesis-run/1",
        "classification": "PROSPECTIVE_UNCALIBRATED_PARAMETER_HYPOTHESIS",
        "source_scout": str(scout_path),
        "source_scout_sha256": scout["selected_parameter_row_fingerprint"],
        "direct_parameter_row": scout["selected_parameter_row"],
        "cavity_emission_parameter_row": scout[
            "selected_cavity_emission_parameter_row"],
        "vacancy_energy_variant": "combined",
        "temperature_K": TEMPERATURE_K,
        "site_centers_m": SITES_M,
        "stochastic_seed": SEED,
        "population_partition": PARTITION,
        "initial_load_factor": 1.5,
        "subsequent_load_factor": 1.0,
        "maximum_physical_time_s": TOTAL_TIME_S,
        "maximum_equivalent_plastic_strain": MAX_EQUIVALENT_PLASTIC_STRAIN,
        "parameters_fixed_during_run": True,
    }
    inputs = out / "declared_inputs.json"
    if inputs.exists():
        existing_declared = json.loads(inputs.read_text())
        candidate_declared = finite(declared)
        # The immutable source directory may change across execution chunks.
        # Its scout path is provenance, not a physical input; the scout digest
        # and every declared parameter remain exact resume gates.
        comparable_existing = dict(existing_declared)
        comparable_candidate = dict(candidate_declared)
        comparable_existing.pop("source_scout", None)
        comparable_candidate.pop("source_scout", None)
        if comparable_existing != comparable_candidate:
            raise RuntimeError("run directory contains a different declared hypothesis")
        declared = existing_declared
    else:
        atomic_dump(inputs, declared)

    progress_path = out / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else None
    if progress and progress.get("latest_restart"):
        state, continuation = restore_checkpoint(
            out / progress["latest_restart"], with_provider_runtime=True)
        if continuation is not None and not isinstance(
                continuation, AcceptedIntervalContinuationV1):
            raise RuntimeError("restart contains an unknown interval continuation owner")
        checkpoint_time = accepted_time(state)
        history = [
            row for row in progress["history"]
            if float(row["end_time_s"]) <= checkpoint_time + 1.0e-12
        ]
        next_checkpoint_time = float(progress["next_checkpoint_time_s"])
        if len(history) != len(progress["history"]):
            progress["resume_history_rows_discarded_after_latest_restart"] = (
                len(progress["history"])-len(history)
            )
    else:
        state = create_initial(scout)
        continuation = None
        history = []
        manifest = write_checkpoint(state, out / "restart_000000ms.json", compression="gzip")
        progress = {
            "history": history, "latest_restart": "restart_000000ms.json",
            "latest_restart_state_file": manifest["state_file"],
            "next_checkpoint_time_s": CHECKPOINT_INTERVAL_S,
        }
        next_checkpoint_time = CHECKPOINT_INTERVAL_S
        atomic_dump(progress_path, progress)

    wall_limit = float(os.environ.get("MAX_WALL_SECONDS", "1800"))
    output_limit = int(os.environ.get("MAX_OUTPUT_BYTES", str(8 * 1024**3)))
    wall_start = time.monotonic()
    stop_reason = None
    while accepted_time(state) < TOTAL_TIME_S - 1e-12:
        now = accepted_time(state)
        if time.monotonic() - wall_start >= wall_limit:
            stop_reason = "WALL_TIME_CHUNK_LIMIT"
            break
        if sum(path.stat().st_size for path in out.rglob("*") if path.is_file()) >= output_limit:
            stop_reason = "OUTPUT_STORAGE_LIMIT"
            break
        index = len(history)
        if continuation is None:
            dt = (EARLY_STEPS_S[index] if index < len(EARLY_STEPS_S)
                  else min(MAX_INTERNAL_DT_S, TOTAL_TIME_S-now))
            macro_start = now
            previous = state
            mechanical, plastic = accepted_bulk_load_step(
                previous, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
                load_factor=1.5 if index == 0 else 1.0,
                duration_s=dt, temperature_K=TEMPERATURE_K,
                maximum_stagger_iterations=80,
            )
            epdot = equivalent_plastic_strain_rate_s(
                previous.ep_gp, mechanical.ep_gp, dt)
            maximum_ep = float(np.max(np.sqrt(
                mechanical.ep_gp[0]**2 + mechanical.ep_gp[1]**2
                + 2.0*mechanical.ep_gp[2]**2
            )))
            if maximum_ep > MAX_EQUIVALENT_PLASTIC_STRAIN:
                stop_reason = "DECLARED_PLASTIC_STRAIN_LIMIT"
                break
            exposure, source = site_source_exposure(previous, mechanical, dt)
            tensor, K, canonical, load_rows = canonical_drive_check(mechanical)
            direct = [accepted_interaction_opportunity(
                mechanical, site, temperature_K=TEMPERATURE_K,
                accepted_equivalent_plastic_strain_rate_s=epdot,
            ) for site in mechanical.void_state.dislocation_interaction_state.sites]
            driver_context = {
                "step": index + 1,
                "maximum_equivalent_plastic_strain": maximum_ep,
                "plastic_work_J_per_m": plastic[
                    "accepted_endpoint_path_plastic_work_J_per_m"],
                "site_source_exposure_fraction_area": exposure,
                "integrated_plastic_source_fraction_area": (
                    source["integrated_source_fraction_area_per_s"] * dt),
                "canonical_drive": canonical,
                "direct_opportunity": direct,
            }
            load_history = {
                "kind": "hold", "temperature_K": TEMPERATURE_K,
                "stress_tensor_Pa": tensor,
                "K_effective_by_front": {ROOT_BRANCH_ID: K},
                "directional_load_rows_by_front": {ROOT_BRANCH_ID: load_rows},
                "maximum_events": int(os.environ.get(
                    "MAX_SCHEDULER_EVENTS_PER_INTERVAL", "4096")),
                "maximum_transactions_per_call": int(os.environ.get(
                    "MAX_SCHEDULER_TRANSACTIONS_PER_CHECKPOINT", "32")),
            }
            selected_before = ()
            transactions_before = 0
            all_owners_before = True
        else:
            dt = continuation.remaining_duration_s
            macro_start = continuation.macro_start_time_s
            mechanical = continuation.mechanical_endpoint_state
            load_history = dict(continuation.load_history)
            driver_context = dict(continuation.driver_context)
            selected_before = continuation.selected_events
            transactions_before = continuation.completed_transaction_count
            all_owners_before = continuation.all_owners_previewed
        call_history = dict(load_history)
        total_transaction_budget = int(load_history["maximum_events"])
        remaining_transaction_budget = (
            total_transaction_budget-transactions_before)
        if remaining_transaction_budget <= 0:
            raise RuntimeError(
                "global event scheduler exhausted the original macro-interval "
                "transaction budget")
        call_history["maximum_events"] = remaining_transaction_budget
        advanced, scheduler = advance_global_physical_interval(
            state, mechanical, dt, call_history,
        )
        selected_chunk = tuple(
            item["selected_event"] for item in scheduler["events"]
            if item["selected_event"] is not None)
        selected = selected_before + selected_chunk
        transactions = transactions_before + len(scheduler["events"])
        all_owners = all_owners_before and scheduler["all_owners_previewed"]
        remaining_macro = max(dt-scheduler["advanced_duration_s"], 0.0)
        macro_tolerance = max(1.0e-12, 1.0e-8 * (
            continuation.requested_duration_s if continuation is not None else dt))
        if remaining_macro > macro_tolerance:
            requested = (continuation.requested_duration_s
                         if continuation is not None else dt)
            continuation = AcceptedIntervalContinuationV1(
                mechanical_endpoint_state=mechanical,
                remaining_duration_s=remaining_macro,
                requested_duration_s=requested,
                macro_start_time_s=macro_start,
                load_history=pickle_safe_metadata(load_history),
                driver_context=pickle_safe_metadata(driver_context),
                selected_events=selected,
                completed_transaction_count=transactions,
                all_owners_previewed=all_owners,
            )
            state = advanced
            checkpoint = out / "inflight_restart.json"
            manifest = write_checkpoint(
                state, checkpoint, compression="gzip",
                provider_runtime=continuation)
            progress.update({
                "latest_restart": checkpoint.name,
                "latest_restart_state_file": manifest["state_file"],
                "accepted_time_s": accepted_time(state),
                "history": history,
                "inflight_interval": {
                    **dict(continuation.audit_payload()),
                    "local_physical_time_s": accepted_time(state),
                    "phase": phase_summary(state),
                    "transport_balance": dict(
                        state.vacancy_transport_state_v2.ledgers),
                    "current_mechanics_stage": "ACCEPTED_GLOBAL_TRANSACTION_BOUNDARY",
                },
                "phase": phase_summary(state),
                "stop_reason": None,
            })
            atomic_dump(progress_path, progress)
            print("NATURAL_VOID_INFLIGHT", accepted_time(state),
                  transactions, remaining_macro, phase_summary(state), flush=True)
            continue
        continuation = None
        row = finite({
            **driver_context, "start_time_s": macro_start,
            "accepted_duration_s": accepted_time(advanced)-macro_start,
            "end_time_s": accepted_time(advanced),
            "mesh_nodes": advanced.mesh.nn, "mesh_elements": advanced.mesh.ne,
            "selected_events": selected,
            "scheduler_transaction_count": transactions,
            "all_owners_previewed": all_owners,
            "state": phase_summary(advanced),
            "accepted_state_fingerprint": complete_accepted_state_fingerprint(advanced),
        })
        history.append(row)
        state = advanced
        progress.pop("inflight_interval", None)
        checkpoint_due = accepted_time(state) >= next_checkpoint_time - 1e-12
        if selected or checkpoint_due or accepted_time(state) >= TOTAL_TIME_S - 1e-12:
            label = int(round(accepted_time(state) * 1000.0))
            checkpoint = out / f"restart_{label:06d}ms.json"
            manifest = write_checkpoint(state, checkpoint, compression="gzip")
            restored = restore_checkpoint(checkpoint)
            row["restart_manifest"] = checkpoint.name
            row["restart_state_file"] = manifest["state_file"]
            row["direct_restored_equal"] = (
                complete_accepted_state_fingerprint(state)
                == complete_accepted_state_fingerprint(restored)
            )
            progress["latest_restart"] = checkpoint.name
            progress["latest_restart_state_file"] = manifest["state_file"]
            while next_checkpoint_time <= accepted_time(state) + 1e-12:
                next_checkpoint_time += CHECKPOINT_INTERVAL_S
        progress.update({
            "history": history,
            "accepted_time_s": accepted_time(state),
            "next_checkpoint_time_s": next_checkpoint_time,
            "last_accepted_event": selected[-1] if selected else None,
            "solver_activity": {
                "mesh_nodes": state.mesh.nn, "mesh_elements": state.mesh.ne,
                "transport_solver": state.vacancy_transport_state_v2.last_solver_audit,
            },
            "phase": phase_summary(state),
            "stop_reason": stop_reason,
        })
        atomic_dump(progress_path, progress)
        print("NATURAL_VOID_PROGRESS", accepted_time(state),
              phase_summary(state)["site_phases"], selected, flush=True)

    # Always publish a valid chunk-end restart when work remains.
    if accepted_time(state) < TOTAL_TIME_S - 1e-12:
        if continuation is None:
            label = int(round(accepted_time(state) * 1000.0))
            checkpoint = out / f"restart_{label:06d}ms.json"
        else:
            checkpoint = out / "inflight_restart.json"
        manifest = write_checkpoint(
            state, checkpoint, compression="gzip",
            provider_runtime=continuation)
        progress.update({
            "latest_restart": checkpoint.name,
            "latest_restart_state_file": manifest["state_file"],
            "accepted_time_s": accepted_time(state),
            "history": history, "stop_reason": stop_reason,
        })
        atomic_dump(progress_path, progress)
    outcome = (
        "COMPLETED_DECLARED_WINDOW" if accepted_time(state) >= TOTAL_TIME_S - 1e-12
        else stop_reason or "CHUNK_STOP"
    )
    atomic_dump(out / "result.json", {
        **declared, "outcome": outcome,
        "accepted_time_s": accepted_time(state),
        "latest_restart": progress["latest_restart"],
        "phase": phase_summary(state), "history": history,
    })
    print("NATURAL_VOID_HYPOTHESIS_CHUNK", outcome, accepted_time(state), flush=True)


if __name__ == "__main__":
    main()
