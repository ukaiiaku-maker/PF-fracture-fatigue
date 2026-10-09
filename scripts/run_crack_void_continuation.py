#!/usr/bin/env python3
"""Restartable post-void propagation pilot using accepted scheduler events.

The unscaled prefix is a finite full-field-plasticity hold.  Repeated topology
is then exercised with explicitly labelled event-targeted directional-rate
scales; those scales are software controls and are never physical predictions.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import (
    PlasticityScope, accepted_bulk_load_step,
)
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.global_physical_scheduler_v1 import (
    advance_global_physical_interval, preview_downstream_first_passage,
)
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint,
)
from arrhenius_fracture.voiding_production_v5 import (
    _bundle_from_state, downstream_source_observation,
    restore_sharp_front_engine,
)

SCHEMA = "unified-2d.crack-void-continuation/1"
DEFAULT_START = ROOT/"artifacts/restricted_production_closeout/bounded_scheduler_positive_child.json"
RETAINED_CONNECTED = ROOT/"artifacts/restricted_production_closeout/bounded_scheduler_ligament_connected.json"
DEFAULT_OUT = ROOT/"artifacts/crack_void_continuation"
BEHAVIOR_FILES = (
    ROOT/"arrhenius_fracture/global_physical_scheduler_v1.py",
    ROOT/"arrhenius_fracture/voiding_production_v5.py",
    ROOT/"arrhenius_fracture/embryo_growth_v1.py",
    ROOT/"arrhenius_fracture/bulk_plasticity_v104.py",
)


def finite(value):
    if is_dataclass(value):
        return finite(asdict(value))
    if isinstance(value, dict):
        return {str(key): finite(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [finite(item) for item in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, np.generic):
        return finite(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return "Infinity" if value > 0.0 else "-Infinity"
    return value


def digest(value) -> str:
    return hashlib.sha256(json.dumps(
        finite(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode()).hexdigest()


def source_hashes():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in BEHAVIOR_FILES}


def front_id(state):
    active = tuple(state.crack_network.active_tip_ids)
    if len(active) != 1:
        return None
    return active[0]


def equivalent_ep(state):
    ep = np.asarray(state.ep_gp)
    exx, eyy, gamma = ep
    ezz = -(exx+eyy)
    return np.sqrt(np.maximum(
        2.0/3.0*(exx**2+eyy**2+ezz**2+0.5*gamma**2), 0.0))


def state_row(state, *, elapsed_s, accepted_extensions, classification,
              load_factor, event=None, wall_s=0.0, control=None):
    cavity = state.void_state.cavities[0]
    active = front_id(state)
    tip = None if active is None else state.crack_network.branch(active).tip
    entry, exit_point = cavity.connection_entry_m, cavity.connection_exit_m
    free_span = (0.0 if entry is None or exit_point is None
                 else math.dist(entry, exit_point))
    growth = next((item for item in state.void_state.embryo_growth_states
                   if item.cavity_id == cavity.cavity_id), None)
    transport = state.vacancy_transport_state_v2
    area = np.asarray(transport.transport_mesh.area_e)
    ratio = float(transport.parameter_row.inventory_ratio)
    field_inventory = float(np.sum(
        area*np.asarray(transport.vacancy_fraction_gp))*ratio)
    obs = state.equilibrium_observables
    front_payload = None
    front_engine = None
    if active is not None:
        owner = state.crack_network.branch(active).local_state.get(
            "front_engine_state_owner")
        front_payload = state.tip_process_state.get("by_branch", {}).get(owner, {})
        if front_payload:
            front_engine = restore_sharp_front_engine(
                state.material, _bundle_from_state(state), front_payload)
    row = {
        "accepted_state_fingerprint": complete_accepted_state_fingerprint(state),
        "classification": classification,
        "elapsed_time_s": float(elapsed_s),
        "canonical_accepted_time_s": float(
            state.junction_process_state.get(
                "canonical_accepted_time_v1").seconds()),
        "transport_time_s": float(transport.transport_time_s),
        "load_factor": float(load_factor),
        "event": event,
        "accepted_downstream_extension_count": int(accepted_extensions),
        "active_front_ids": list(state.crack_network.active_tip_ids),
        "active_front_tip_m": None if tip is None else list(map(float, tip)),
        "newly_fractured_material_length_m": float(
            state.crack_network.total_physical_crack_length_m),
        "connected_void_free_span_m": float(free_span),
        "cavity": {
            "cavity_id": cavity.cavity_id, "phase": cavity.phase.value,
            "radius_m": float(cavity.radius_m), "area_m2": float(cavity.area_m2),
            "inventory_area_m2": float(cavity.inventory_area_m2),
        },
        "growth_mechanisms": (None if growth is None else {
            "vacancy_derived_area_m2": float(growth.vacancy_derived_area_m2),
            "dislocation_derived_area_m2": float(growth.dislocation_derived_area_m2),
            "captured_unaccommodated_vacancy_area_m2": float(
                growth.captured_unaccommodated_vacancy_area_m2),
            "emitted_loop_count_effective": float(
                growth.emitted_loop_count_effective),
        }),
        "vacancy_field_inventory_area_m2": field_inventory,
        "vacancy_ledgers": dict(transport.ledgers),
        "mesh_elements": int(state.mesh.ne),
        "reaction_N_per_m": (None if obs is None else
                              float(obs.mean_reaction_magnitude_N_per_m)),
        "opening_m": (None if obs is None else float(obs.total_opening_m)),
        "compliance_m2_per_N": (None if obs is None else
                                 float(obs.compliance_m2_per_N)),
        "recoverable_energy_J_per_m": (None if obs is None else
                                        float(obs.recoverable_elastic_energy_J_per_m)),
        "total_internal_energy_J_per_m": float(state.stored_energy_J_per_m),
        "canonical_r_tip_m": (None if front_engine is None else
                               float(front_engine.r_eff())),
        "front_process": (None if front_engine is None else {
            "owner": owner,
            "time_s": float(front_engine.t),
            "n_adv": int(front_engine.n_adv),
            "a_adv": float(front_engine.a_adv),
            "B": float(front_engine.B),
            "observation_source": "ACCEPTED_CANONICAL_FRONT_ENGINE",
        }),
        "accepted_bulk_plasticity_scope": state.junction_process_state.get(
            "accepted_bulk_plasticity_v104", {}).get("scope"),
        "source_availability_restriction":
            "RESOLVED_CAVITY_LOOP_SOURCE_UNAVAILABLE_WHERE_NOT_QUALIFIED",
        "wall_time_s": float(wall_s),
        "control": control,
    }
    return finite(row)


def repair_front_history_diagnostics(history, start_state):
    """Upgrade old diagnostic rows without altering any accepted state."""
    if not history or not any(row.get("canonical_r_tip_m") is None for row in history):
        return history
    start = state_row(
        start_state, elapsed_s=history[0]["elapsed_time_s"],
        accepted_extensions=history[0]["accepted_downstream_extension_count"],
        classification=history[0]["classification"],
        load_factor=history[0]["load_factor"], event=history[0].get("event"),
    )
    history[0]["canonical_r_tip_m"] = start["canonical_r_tip_m"]
    history[0]["front_process"] = start["front_process"]
    for index, row in enumerate(history):
        receipt = (row.get("control") or {}).get("receipt") or {}
        advances = receipt.get("legacy_front_process") or []
        if not advances:
            continue
        audit = advances[-1]
        step = audit.get("step_info", {})
        row["canonical_r_tip_m"] = step.get("r_eff")
        row["front_process"] = {
            "owner": (row.get("front_process") or {}).get("owner", "void-front-1"),
            "time_s": row["elapsed_time_s"],
            "n_adv": audit.get("post_n_adv"),
            "a_adv": audit.get("post_a_adv_m"),
            "B": audit.get("post_B"),
            "observation_source": "ACCEPTED_EVENT_FRONT_PROCESS_AUDIT",
        }
        # The immediately preceding no-event row is exactly the pre-event
        # front state stored by the next accepted transaction.
        if index and history[index-1].get("canonical_r_tip_m") is None:
            previous = history[index-1]
            previous["canonical_r_tip_m"] = step.get("r_eff_pre_renewal")
            previous["front_process"] = {
                "owner": (previous.get("front_process") or {}).get(
                    "owner", "void-front-1"),
                "time_s": previous["elapsed_time_s"],
                "n_adv": audit.get("pre_n_adv"),
                "a_adv": audit.get("pre_a_adv_m"),
                "B": audit.get("pre_B"),
                "observation_source": "NEXT_ACCEPTED_EVENT_PRE_STATE_AUDIT",
            }
    return history


def bound_source(state, continuation=True):
    source = dict(downstream_source_observation(state, continuation=continuation))
    source["accepted_state_fingerprint"] = complete_accepted_state_fingerprint(state)
    source["continuation"] = bool(continuation)
    return source


def preview_with_source(state, source, *, duration_s, scale):
    return preview_downstream_first_passage(
        state, continuation=True, temperature_K=900.0,
        maximum_duration_s=float(duration_s),
        functional_test_rate_scale=float(scale),
        test_only_functional_qualification=(float(scale) != 1.0),
        state_bound_source_observation=source,
    )


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(finite(value), indent=2, sort_keys=True)+"\n")


def save_checkpoint_with_receipt(state, path, receipt):
    write_checkpoint(state, path, compression="gzip")
    body = {
        "schema": SCHEMA+".checkpoint-receipt/1",
        "state_fingerprint": complete_accepted_state_fingerprint(state),
        "receipt": finite(receipt),
    }
    body["authentication_sha256"] = digest(body)
    write_json(path.with_suffix(path.suffix+".receipt.json"), body)


def field_snapshot(state, path):
    np.savez_compressed(
        path,
        nodes=np.asarray(state.mesh.nodes), elems=np.asarray(state.mesh.elems),
        equivalent_plastic_strain=equivalent_ep(state),
        dislocation_density_m2=np.asarray(state.rho_gp),
        transport_nodes=np.asarray(state.vacancy_transport_state_v2.transport_mesh.nodes),
        transport_elems=np.asarray(state.vacancy_transport_state_v2.transport_mesh.elems),
        vacancy_fraction=np.asarray(state.vacancy_transport_state_v2.vacancy_fraction_gp),
    )


def make_plots(outdir, states, history):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    # The retained connected checkpoint owns the exact ligament edge that was
    # committed.  Removing only that recorded edge reconstructs the accepted
    # preconnection topology for the geometry panel; no cavity-interior crack
    # span is drawn or charged as fractured material.
    geometry_rows = []
    connected = states.get("connected")
    if connected is not None:
        root = connected.crack_network.branch(connected.crack_network.primary_branch_id)
        geometry_rows.append(("preconnection\n(from committed-edge receipt)", connected, {
            root.branch_id: np.asarray(root.path[:-1]),
        }))
    geometry_rows.extend((label, states[key], {}) for label, key in (
        ("accepted child birth", "child"), ("sustained final", "final"))
                         if key in states)
    fig, axes = plt.subplots(1, len(geometry_rows),
                             figsize=(5*len(geometry_rows), 4), squeeze=False)
    for ax, (label, state, overrides) in zip(axes[0], geometry_rows):
        for branch in state.crack_network.branches:
            path = overrides.get(branch.branch_id, np.asarray(branch.path))
            if len(path) == 0:
                continue
            ax.plot(path[:, 0]*1e3, path[:, 1]*1e3, "k-", lw=1.6)
        cavity = state.void_state.cavities[0]
        ax.add_patch(Circle(
            np.asarray(cavity.center_m)*1e3, cavity.radius_m*1e3,
            facecolor="white", edgecolor="tab:red", lw=2, zorder=3,
        ))
        ax.set_title(label); ax.set_aspect("equal"); ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
    fig.tight_layout(); fig.savefig(outdir/"geometry_progression.png", dpi=180); plt.close(fig)

    x = np.arange(len(history)); times = np.asarray([r["elapsed_time_s"] for r in history])
    length = np.asarray([r["newly_fractured_material_length_m"] for r in history])*1e3
    front = np.asarray([np.nan if r["active_front_tip_m"] is None
                        else r["active_front_tip_m"][0]*1e3 for r in history])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(times, length, "o-", label="fractured material")
    ax.plot(times, front, "s-", label="front x"); ax.set_xlabel("accepted elapsed time (s)")
    ax.set_ylabel("length / position (mm)"); ax.legend(); fig.tight_layout()
    fig.savefig(outdir/"length_and_front_vs_time.png", dpi=180); plt.close(fig)

    opening = np.asarray([np.nan if r["opening_m"] is None else r["opening_m"] for r in history])
    reaction = np.asarray([np.nan if r["reaction_N_per_m"] is None else r["reaction_N_per_m"] for r in history])
    fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(opening*1e6, reaction, "o-")
    ax.set_xlabel("opening (µm)"); ax.set_ylabel("reaction (N/m)"); fig.tight_layout()
    fig.savefig(outdir/"load_displacement.png", dpi=180); plt.close(fig)

    radius = np.asarray([r["cavity"]["radius_m"] for r in history])*1e6
    vacancy = np.asarray([r["growth_mechanisms"]["vacancy_derived_area_m2"]
                          if r["growth_mechanisms"] else 0.0 for r in history])
    loops = np.asarray([r["growth_mechanisms"]["dislocation_derived_area_m2"]
                        if r["growth_mechanisms"] else 0.0 for r in history])
    fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(x, radius, "o-", label="radius (µm)")
    ax2 = ax.twinx(); ax2.plot(x, vacancy, "s--", label="vacancy area", color="tab:green")
    ax2.plot(x, loops, "^--", label="loop area", color="tab:orange")
    ax.set_xlabel("accepted record"); ax.set_ylabel("cavity radius (µm)")
    ax2.set_ylabel("mechanism area (m²)"); fig.tight_layout()
    fig.savefig(outdir/"cavity_mechanisms.png", dpi=180); plt.close(fig)

    selected = states["final"]
    tri = np.asarray(selected.mesh.elems)
    fields = ((equivalent_ep(selected), "equivalent plastic strain", False),
              (np.asarray(selected.rho_gp), "dislocation density", True))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, (values, title, logscale) in zip(axes[:2], fields):
        data = np.log10(np.maximum(values, 1.0)) if logscale else values
        im = ax.tripcolor(selected.mesh.nodes[:, 0]*1e3,
                          selected.mesh.nodes[:, 1]*1e3, tri, data, shading="flat")
        ax.set_title(("log10 " if logscale else "")+title); fig.colorbar(im, ax=ax)
    tm = selected.vacancy_transport_state_v2.transport_mesh
    vacancy_values = np.log10(np.maximum(
        np.asarray(selected.vacancy_transport_state_v2.vacancy_fraction_gp), 1e-30))
    im = axes[2].tripcolor(tm.nodes[:, 0]*1e3, tm.nodes[:, 1]*1e3,
                           np.asarray(tm.elems), vacancy_values, shading="flat")
    axes[2].set_title("log10 vacancy fraction"); fig.colorbar(im, ax=axes[2])
    for ax in axes: ax.set_aspect("equal"); ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)")
    fig.tight_layout(); fig.savefig(outdir/"final_fields.png", dpi=180); plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=Path, default=DEFAULT_START)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--target-extensions", type=int, default=10)
    parser.add_argument("--maximum-extensions", type=int, default=20)
    parser.add_argument("--physical-dt-s", type=float, default=1.0e-10)
    parser.add_argument("--controlled-dt-s", type=float, default=1.0e-10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.target_extensions <= args.maximum_extensions <= 20:
        raise ValueError("extension targets must satisfy 1 <= target <= cap <= 20")
    outdir = args.outdir; outdir.mkdir(parents=True, exist_ok=True)
    history_path = outdir/"history.json"
    latest_path = outdir/"latest.json"
    if args.resume and latest_path.exists() and history_path.exists():
        state = restore_checkpoint(latest_path)
        history = json.loads(history_path.read_text())
        history = repair_front_history_diagnostics(
            history, restore_checkpoint(outdir/"start.json"))
        write_json(history_path, history)
        completed = int(history[-1]["accepted_downstream_extension_count"])
        initial_tip_x = float(history[0]["active_front_tip_m"][0])
        elapsed = float(history[-1]["elapsed_time_s"])
        states = {
            "connected": restore_checkpoint(RETAINED_CONNECTED),
            "child": restore_checkpoint(outdir/"start.json"),
        }
        if (outdir/"mid.json").exists(): states["mid"] = restore_checkpoint(outdir/"mid.json")
    else:
        state = restore_checkpoint(args.start)
        if front_id(state) != "void-front-1":
            raise RuntimeError("continuation pilot requires the canonical active child")
        completed = 0; elapsed = 0.0
        initial_tip_x = float(state.crack_network.branch("void-front-1").tip[0])
        history = [state_row(
            state, elapsed_s=elapsed, accepted_extensions=0,
            classification="RETAINED_RATE_ACCELERATED_CHILD_PREFIX",
            load_factor=1.0, event="START_CHILD_CHECKPOINT")]
        states = {
            "connected": restore_checkpoint(RETAINED_CONNECTED),
            "child": state,
        }
        save_checkpoint_with_receipt(state, outdir/"start.json", history[-1])
        field_snapshot(state, outdir/"fields_start.npz")

        # Declared finite physical pilot: fixed opening, DBTT/900 K, unscaled
        # directional rates, one 1e-10 s full-field PT/rho interval.
        t0 = time.monotonic()
        mechanical, bulk = accepted_bulk_load_step(
            state, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
            load_factor=1.0, duration_s=args.physical_dt_s,
            temperature_K=900.0)
        source = bound_source(mechanical)
        physical_preview = preview_with_source(
            mechanical, source, duration_s=args.physical_dt_s, scale=1.0)
        state, audit = advance_global_physical_interval(
            state, mechanical, args.physical_dt_s, {
                "kind": "hold", "temperature_K": 900.0,
                "maximum_events": 8,
                "state_bound_topology_preview": physical_preview,
            })
        elapsed += float(audit["advanced_duration_s"])
        physical_events = [row["selected_event"] for row in audit["events"]
                           if row["selected_event"] is not None]
        history.append(state_row(
            state, elapsed_s=elapsed, accepted_extensions=completed,
            classification="UNSCALED_PHYSICAL_FULL_FIELD_PILOT",
            load_factor=1.0, event=(physical_events or ["NO_EVENT_WITHIN_WINDOW"])[0],
            wall_s=time.monotonic()-t0,
            control={"temperature_K": 900.0, "duration_s": args.physical_dt_s,
                     "directional_rate_scale": 1.0,
                     "bulk_scope": PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO.value,
                     "bulk_manifest_fingerprint": bulk["manifest_fingerprint"]}))
        save_checkpoint_with_receipt(state, latest_path, history[-1])
        write_json(history_path, history)

    stopping_reason = None
    while completed < args.target_extensions and completed < args.maximum_extensions:
        active = front_id(state)
        if active is None:
            stopping_reason = "NO_ACTIVE_DOWNSTREAM_FRONT"
            break
        terminal = state.junction_process_state.get(
            "downstream_front_terminal_status", {})
        if terminal.get("reason") == "SPECIMEN_BOUNDARY_FIXED_INCREMENT_EXHAUSTED":
            stopping_reason = "SPECIMEN_BOUNDARY_FIXED_INCREMENT_EXHAUSTED"
            break
        tip_before = np.asarray(state.crack_network.branch(active).tip, float)
        if float(np.max(state.mesh.nodes[:, 0])-tip_before[0]) < 0.25e-5:
            stopping_reason = "SPECIMEN_BOUNDARY_REACHED"
            break
        t0 = time.monotonic()
        try:
            mechanical, bulk = accepted_bulk_load_step(
                state, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
                load_factor=1.0, duration_s=args.controlled_dt_s,
                temperature_K=900.0)
            source = bound_source(mechanical)
            baseline = preview_with_source(
                mechanical, source, duration_s=args.controlled_dt_s, scale=1.0)
            rows = baseline.commit_payload["directional_preview"].commit_payload["audit"]
            finite_crossings = [float(row["crossing_time_s"]) for row in rows
                                if math.isfinite(float(row["crossing_time_s"]))
                                and float(row["crossing_time_s"]) > 0.0]
            if not finite_crossings:
                stopping_reason = "NO_KINETICALLY_ACTIVE_CONTINUATION_SOURCE"
                break
            unscaled_wait = min(finite_crossings)
            scale = unscaled_wait/args.controlled_dt_s
            if not math.isfinite(scale) or scale < 1.0:
                scale = 1.0
            preview = preview_with_source(
                mechanical, source, duration_s=args.controlled_dt_s, scale=scale)
            if preview.event_type != "CHILD_CONTINUATION":
                stopping_reason = "CONTROLLED_CONTINUATION_PREVIEW_UNAVAILABLE"
                break
            duration = float(preview.crossing_time_s)
            advanced, audit = advance_global_physical_interval(
                state, mechanical, duration, {
                    "kind": "hold", "temperature_K": 900.0,
                    "maximum_events": 8,
                    "functional_test_directional_rate_scale": scale,
                    "test_only_functional_qualification": True,
                    "state_bound_topology_preview": preview,
                })
        except RuntimeError as exc:
            stopping_reason = "NUMERICAL_OR_GEOMETRIC_STOP:"+str(exc)
            break
        selected = [row for row in audit["events"]
                    if row["selected_event"] == "CHILD_CONTINUATION"]
        if len(selected) != 1:
            stopping_reason = "CONTINUATION_NOT_COMMITTED_EXACTLY_ONCE"
            break
        tip_after = np.asarray(advanced.crack_network.branch(active).tip, float)
        increment = float(np.linalg.norm(tip_after-tip_before))
        if increment <= 0.0:
            stopping_reason = "NONPOSITIVE_CRACK_EXTENSION"
            break
        state = advanced; completed += 1; elapsed += float(audit["advanced_duration_s"])
        receipt = {
            "event_index": completed, "selected_event": "CHILD_CONTINUATION",
            "input_state_fingerprint": selected[0]["input_state_fingerprint"],
            "output_state_fingerprint": selected[0]["output_state_fingerprint"],
            "elapsed_time_s": selected[0]["elapsed_time_s"],
            "selected_owner": selected[0]["selected_owner"],
            "tip_before_m": tip_before.tolist(), "tip_after_m": tip_after.tolist(),
            "positive_extension_m": increment,
            "front_process_load_policy": selected[0]["front_process_load_policy"],
            "legacy_front_process": selected[0]["legacy_front_process"],
        }
        history.append(state_row(
            state, elapsed_s=elapsed, accepted_extensions=completed,
            classification="CONTROLLED_FUNCTIONAL_SUSTAINED_PROPAGATION",
            load_factor=1.0, event="CHILD_CONTINUATION",
            wall_s=time.monotonic()-t0,
            control={"temperature_K": 900.0, "duration_s": duration,
                     "unscaled_wait_s": unscaled_wait,
                     "directional_rate_scale": scale,
                     "bulk_scope": PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO.value,
                     "bulk_manifest_fingerprint": bulk["manifest_fingerprint"],
                     "rate_control_classification":
                         "EVENT_TARGETED_SOFTWARE_CONTROL_NOT_PHYSICAL_PREDICTION",
                     "receipt": receipt}))
        save_checkpoint_with_receipt(state, latest_path, receipt)
        write_json(history_path, history)
        if completed == max(1, args.target_extensions//2):
            states["mid"] = state
            save_checkpoint_with_receipt(state, outdir/"mid.json", receipt)
            field_snapshot(state, outdir/"fields_mid.npz")
        print("ACCEPTED_EXTENSION", completed, "TIP_X_M", tip_after[0],
              "DELTA_M", increment, "MESH_ELEMENTS", state.mesh.ne,
              "WALL_S", history[-1]["wall_time_s"], flush=True)

    if stopping_reason is None:
        stopping_reason = ("TARGET_REACHED" if completed >= args.target_extensions
                           else "MAXIMUM_EXTENSION_CAP_REACHED")
    states["final"] = state
    field_snapshot(state, outdir/"fields_final.npz")
    save_checkpoint_with_receipt(state, latest_path, history[-1])
    make_plots(outdir, states, history)
    final_tip = state.crack_network.branch(front_id(state)).tip if front_id(state) else (math.nan, math.nan)
    summary = {
        "schema": SCHEMA,
        "classification": {
            "physical_pilot": "UNSCALED_PHYSICAL_FULL_FIELD_PT_EVOLVING_RHO",
            "repeated_propagation": "CONTROLLED_FUNCTIONAL_NOT_PHYSICAL_PREDICTION",
            "starting_prefix": "RETAINED_RATE_ACCELERATED_CHILD_CHECKPOINT",
            "prescribed_void": "EXISTING_ACCEPTED_INITIAL_CONDITION_NOT_NEW_VOID_PREDICTION",
            "connection_and_child_prefix": "VALIDLY_RESUMED_RETAINED_ACCEPTED_STATES",
        },
        "configuration": {
            "material": "DBTT", "temperature_K": 900.0,
            "loading": "FIXED_ACCEPTED_OPENING",
            "physical_time_horizon_s": args.physical_dt_s,
            "controlled_interval_s": args.controlled_dt_s,
            "load_factor": 1.0,
            "bulk_plasticity_scope": PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO.value,
            "target_extensions": args.target_extensions,
            "maximum_extensions": args.maximum_extensions,
        },
        "source_sha256": source_hashes(),
        "start_state_fingerprint": history[0]["accepted_state_fingerprint"],
        "final_state_fingerprint": complete_accepted_state_fingerprint(state),
        "accepted_downstream_extensions": completed,
        "post_void_travel_m": float(final_tip[0]-initial_tip_x),
        "initial_void_diameter_m": 2.0*float(states["child"].void_state.cavities[0].radius_m),
        "stopping_reason": stopping_reason,
        "additional_void_sites_activated": sum(
            item.site_id != "site-1" and item.phase.value != "AVAILABLE_SITE"
            for item in state.void_state.sites),
        "serial_resolved_void_capability": "ONE_RESOLVED_CAVITY_ONLY",
        "resolved_cavity_loop_source_restriction":
            "KNOWN_UNAVAILABLE_WHERE_SOURCE_OBSERVER_IS_NOT_QUALIFIED",
        "history_rows": len(history),
        "checkpoints": [name for name in ("start.json", "mid.json", "latest.json")
                        if (outdir/name).exists()],
        "plots": [name for name in (
            "geometry_progression.png", "length_and_front_vs_time.png",
            "load_displacement.png", "cavity_mechanisms.png", "final_fields.png")],
    }
    summary["authentication_sha256"] = digest(summary)
    write_json(outdir/"summary.json", summary)
    print("ACCEPTED_DOWNSTREAM_EXTENSIONS =", completed)
    print("POST_VOID_TRAVEL_M =", summary["post_void_travel_m"])
    print("STOPPING_REASON =", stopping_reason)
    print("CRACK_VOID_CONTINUATION = PASS" if completed >= 2 else
          "CRACK_VOID_CONTINUATION = PARTIAL")
    return 0 if completed >= 2 else 2


if __name__ == "__main__":
    raise SystemExit(main())
