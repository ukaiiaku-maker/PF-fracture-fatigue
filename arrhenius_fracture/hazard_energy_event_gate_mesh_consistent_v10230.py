"""Mesh-consistent event-length search for the v10.2.30 hazard energy gate.

The sharp-wake representation changes topology in finite mesh-resolved packets.
Subincrements that do not yet alter the damage field use the already active
signed directional-J derivative, J=K^2/E', only as a search continuation. Once a
trial changes topology, admissibility is decided by the full fixed-opening,
re-equilibrated elastic-energy drop. A no-damage trial is never committed as a
physical geometry event.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import hazard_energy_event_gate_v10230 as _base


MODEL_ID = "v10.2.30_mesh_consistent_hazard_energy_event_search"


def energy_gate_event_length_mesh_consistent(
    *,
    kwargs: dict[str, Any],
    descriptor: dict[str, Any],
) -> dict[str, Any]:
    cfg = _base.OBSERVER.config
    proposal = max(float(descriptor["event_advance_m"]), 0.0)
    if proposal <= 0.0:
        raise ValueError("stochastic event proposal must be positive")
    snapshot = _base.OBSERVER.snapshot
    if not isinstance(snapshot, dict) or "mesh" not in snapshot:
        raise RuntimeError("no valid mechanics snapshot is available for energy gating")

    mesh = kwargs["mesh"]
    boundary = kwargs["boundary"]
    damage = np.asarray(kwargs["damage"], dtype=float)
    displacement = np.asarray(kwargs["displacement"], dtype=float)
    if snapshot["mesh"] is not mesh:
        raise RuntimeError("energy-gate mechanics snapshot does not match event mesh")
    if snapshot["damage"].shape != damage.shape or not np.array_equal(
        snapshot["damage"], damage
    ):
        raise RuntimeError("energy-gate mechanics snapshot does not match event damage")

    direction = np.asarray(kwargs["direction"], dtype=float).reshape(2)
    direction /= max(float(np.linalg.norm(direction)), 1.0e-300)
    p0 = np.asarray(kwargs["p0"], dtype=float).reshape(2)
    kill_r = float(kwargs["kill_r"])

    gamma, direction_audit = _base.current_direction_gamma(direction)
    barrier_J = max(float(descriptor.get("hazard_barrier_J", 0.0)), 0.0)
    hits = max(float(descriptor.get("hazard_cooperative_hits", 1.0)), 1.0)
    b = max(
        abs(float(descriptor.get("hazard_burgers_vector_m", 0.0))),
        1.0e-300,
    )
    resistance = _base.hazard_resistance_J_per_m2(
        barrier_J=barrier_J,
        cooperative_hits=hits,
        burgers_vector_m=b,
        gamma_relative=gamma,
    )

    event_K = max(float(descriptor.get("event_K_Pa_sqrt_m", 0.0)), 0.0)
    probe_K = _base._probe_K_Pa_sqrt_m(event_K)
    energy_scale = (
        (event_K / probe_K) ** 2
        if event_K > 0.0 and probe_K > 0.0
        else 1.0
    )

    ep_gp = np.asarray(snapshot["ep_gp"], dtype=float)
    rho_gp = np.asarray(snapshot["rho_gp"], dtype=float)
    D = np.asarray(snapshot["D"], dtype=float)
    mat = snapshot["mat"]
    cohesive = snapshot.get("cohesive_network")
    Eprime = max(float(getattr(mat, "Eprime", 0.0)), 1.0e-300)
    directional_J = event_K * event_K / Eprime

    u_pre, _, energy_pre_probe = _base._equilibrate_fixed_opening(
        mesh=mesh,
        boundary=boundary,
        u_initial=displacement,
        ep_gp=ep_gp,
        rho_gp=rho_gp,
        damage=damage,
        D=D,
        mat=mat,
        cohesive_network=cohesive,
    )

    ntrial = max(int(math.ceil(1.0 / cfg.trial_fraction)), 1)
    candidates = [proposal * i / ntrial for i in range(1, ntrial + 1)]
    rows: list[dict[str, Any]] = []
    accepted_length = 0.0
    accepted_u = u_pre
    accepted_topology = False
    first_failed = None
    first_topology_length = None

    def evaluate(length: float) -> tuple[float, np.ndarray, dict[str, Any], bool]:
        p1 = p0 + float(length) * direction
        dtrial = _base._damage_for_segment(mesh, damage, p0, p1, kill_r)
        newly_killed = int(
            np.count_nonzero((dtrial > damage) & (dtrial > 0.0))
        )
        topology_changed = newly_killed > 0
        if topology_changed:
            utrial, _, energy_post_probe = _base._equilibrate_fixed_opening(
                mesh=mesh,
                boundary=boundary,
                u_initial=u_pre,
                ep_gp=ep_gp,
                rho_gp=rho_gp,
                damage=dtrial,
                D=D,
                mat=mat,
                cohesive_network=cohesive,
            )
            released_probe = max(energy_pre_probe - energy_post_probe, 0.0)
            released = released_probe * energy_scale
            release_source = "fixed_opening_re_equilibrated_energy_drop"
        else:
            utrial = u_pre
            energy_post_probe = energy_pre_probe
            released_probe = directional_J * float(length) / max(energy_scale, 1.0e-300)
            released = directional_J * float(length)
            release_source = "directional_J_search_continuation_only"

        dissipated = resistance * float(length)
        tolerance = max(
            cfg.absolute_energy_tolerance_J_per_m,
            cfg.relative_energy_tolerance
            * max(abs(released), abs(dissipated), 1.0e-300),
        )
        residual = released - dissipated
        row = {
            "trial_length_m": float(length),
            "stored_energy_pre_probe_J_per_m": energy_pre_probe,
            "stored_energy_post_probe_J_per_m": energy_post_probe,
            "elastic_release_probe_J_per_m": released_probe,
            "probe_to_event_energy_scale": energy_scale,
            "elastic_release_event_J_per_m": released,
            "hazard_dissipation_J_per_m": dissipated,
            "energy_residual_J_per_m": residual,
            "energy_tolerance_J_per_m": tolerance,
            "admissible": bool(residual + tolerance >= 0.0),
            "newly_killed_nodes": newly_killed,
            "topology_changed": topology_changed,
            "energy_release_source": release_source,
            "directional_J_event_J_per_m2": directional_J,
            "subgrid_search_continuation_only": not topology_changed,
        }
        return residual + tolerance, utrial, row, topology_changed

    for length in candidates:
        residual_with_tol, utrial, row, topology_changed = evaluate(length)
        rows.append(row)
        if topology_changed and first_topology_length is None:
            first_topology_length = float(length)
        if residual_with_tol >= 0.0:
            if topology_changed:
                accepted_length = float(length)
                accepted_u = utrial
                accepted_topology = True
            continue
        if topology_changed:
            first_failed = float(length)
            break

    if accepted_topology and first_failed is not None:
        lo = accepted_length
        hi = first_failed
        ulo = accepted_u
        for _ in range(cfg.bisection_iterations):
            mid = 0.5 * (lo + hi)
            residual_with_tol, umid, row, topology_changed = evaluate(mid)
            row["bisection"] = True
            rows.append(row)
            if residual_with_tol >= 0.0 and topology_changed:
                lo = mid
                ulo = umid
            else:
                hi = mid
        accepted_length = lo
        accepted_u = ulo

    all_proposal = accepted_topology and accepted_length >= proposal * (
        1.0 - 1.0e-12
    )
    committed = min(proposal, accepted_length) if accepted_topology else 0.0
    return {
        "energy_gate_model_id": _base.MODEL_ID,
        "energy_gate_search_model_id": MODEL_ID,
        "stochastic_proposed_event_length_m": proposal,
        "energy_admissible_event_length_m": accepted_length,
        "committed_event_length_m": committed,
        "arrest_reason": (
            "stochastic_proposal_reached"
            if all_proposal
            else (
                "hazard_derived_energy_arrest"
                if committed > 0.0
                else (
                    "no_mesh_resolved_admissible_increment"
                    if first_topology_length is not None
                    else "no_mesh_resolved_topology_change"
                )
            )
        ),
        "hazard_barrier_J": barrier_J,
        "hazard_cooperative_hits": hits,
        "hazard_burgers_vector_m": b,
        "orientation_gamma_relative": gamma,
        "hazard_resistance_J_per_m2": resistance,
        "event_K_Pa_sqrt_m": event_K,
        "probe_K_Pa_sqrt_m": probe_K,
        "probe_to_event_energy_scale": energy_scale,
        "directional_J_event_J_per_m2": directional_J,
        "first_mesh_topology_change_length_m": first_topology_length,
        "mesh_resolved_commit_required": True,
        "subgrid_directional_J_used_only_for_search": True,
        "mechanics_serial": int(_base.OBSERVER.mechanics_serial),
        "latest_probe_K_Pa_sqrt_m": _base.OBSERVER.latest_probe_K_Pa_sqrt_m,
        "direction_serial": int(_base.OBSERVER.direction_serial),
        "direction_audit": direction_audit,
        "trial_rows": rows,
        "equilibrated_displacement": accepted_u,
        "athermal_Gc_used": False,
        "independent_toughness_floor_used": False,
        "paris_law_used": False,
    }


__all__ = ["MODEL_ID", "energy_gate_event_length_mesh_consistent"]
