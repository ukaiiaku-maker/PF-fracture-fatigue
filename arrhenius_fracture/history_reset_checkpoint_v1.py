"""Persist the qualified P3 history reset as a restartable native checkpoint."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .config import ElasticProperties
from .fem import assemble_mechanics, plane_strain_D, solve_dirichlet
from .mesh import BoundaryData, rebuild_tri_mesh
from .run_state_checkpoint_v10230 import (
    load_combined_checkpoint, validate_cross_layer, write_combined_checkpoint,
)


MODEL_ID = "p3-history-reset-reequilibrated-checkpoint/1"
RESET_FIELDS = frozenset({
    "mobile_positive", "mobile_negative", "retained_positive",
    "retained_negative", "accumulated_slip_positive",
    "accumulated_slip_negative", "wake_mobile_positive",
    "wake_mobile_negative", "wake_retained_positive",
    "wake_retained_negative", "wake_slip_positive", "wake_slip_negative",
})


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _reequilibrate(outer: dict, arrays: dict[str, np.ndarray]) -> tuple[np.ndarray, dict]:
    mesh = rebuild_tri_mesh(arrays["mesh_nodes"], arrays["mesh_elems"])
    displacement = np.asarray(arrays["displacement"], float).copy()
    ep_gp = np.asarray(arrays["ep_gp"], float)
    rho_gp = np.asarray(arrays["rho_gp"], float)
    damage = np.asarray(arrays["damage"], float)
    y = mesh.nodes[:, 1]
    tolerance = max(float(np.ptp(mesh.nodes[:, 0])), float(np.ptp(y))) * 1e-12
    top = np.flatnonzero(np.abs(y - y.max()) <= tolerance)
    bottom = np.flatnonzero(np.abs(y - y.min()) <= tolerance)
    boundary = BoundaryData(
        top, bottom, int(bottom[np.argmin(mesh.nodes[bottom, 0])]),
        int(bottom[np.argmax(mesh.nodes[bottom, 0])]), np.asarray([], int),
    )
    opening = float(outer["driver"]["Uapp_accepted"])
    material = ElasticProperties()
    stiffness, residual, *_ = assemble_mechanics(
        mesh, displacement, ep_gp, rho_gp, damage,
        plane_strain_D(material), material,
    )
    before = displacement.copy()
    displacement, reaction = solve_dirichlet(
        stiffness, residual, displacement, boundary, opening / 2.0, -opening / 2.0,
    )
    _, residual_after, *_ = assemble_mechanics(
        mesh, displacement, ep_gp, rho_gp, damage,
        plane_strain_D(material), material,
    )
    fixed = np.zeros(mesh.ndof, bool)
    fixed[2 * top + 1] = True
    fixed[2 * bottom + 1] = True
    fixed[2 * boundary.left_bot] = True
    fixed[2 * boundary.right_bot] = True
    residual_norm = float(np.linalg.norm(residual_after[~fixed], ord=np.inf))
    receipt = {
        "converged": bool(np.isfinite(residual_norm) and residual_norm < 1e-3),
        "residual_norm_N_per_m": residual_norm,
        "displacement_correction_norm_m": float(np.linalg.norm(displacement - before)),
        "reaction_N_per_m": float(reaction),
        "native_mechanics": True,
        "opening_m": opening,
    }
    if not receipt["converged"]:
        raise RuntimeError("native post-reset mechanics re-equilibration did not converge")
    return displacement, receipt


def transform_history_reset_checkpoint(
    source: str | Path, destination: str | Path,
) -> dict[str, Any]:
    """Create a new reset generation without mutating the accepted source."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if source == destination:
        raise ValueError("history-reset destination must differ from its source")
    outer, kinetic, arrays = load_combined_checkpoint(source)
    validate_cross_layer(outer, kinetic)
    vector = np.asarray(arrays.pop("kinetic_active_vector"), float).copy()
    before_sha = _sha256_array(vector)
    removed: dict[str, float] = {}
    observed: set[str] = set()
    for field in kinetic.get("fields", []):
        if field.get("owner") != "mpz" or field.get("name") not in RESET_FIELDS:
            continue
        name = str(field["name"])
        start, stop = int(field["start"]), int(field["stop"])
        removed[name] = float(np.sum(vector[start:stop]))
        vector[start:stop] = 0.0
        observed.add(name)
    missing = RESET_FIELDS - observed
    if missing:
        raise RuntimeError(f"checkpoint lacks required reset fields: {sorted(missing)}")
    if not any(abs(value) > 0.0 for value in removed.values()):
        raise RuntimeError("history reset requires a physically populated source checkpoint")
    ledgers_before = dict(kinetic.get("ledgers", {}))
    displacement, equilibrium = _reequilibrate(outer, arrays)
    arrays["displacement"] = displacement
    receipt = {
        "schema": MODEL_ID,
        "intervention": "history_reset_reequilibrated",
        "qualification_only": True,
        "production_default": False,
        "source_checkpoint": str(source),
        "pre_reset_population_sha256": before_sha,
        "post_reset_population_sha256": _sha256_array(vector),
        "removed_population": removed,
        "removed_population_total": float(sum(removed.values())),
        "counterfactual_removed_work": {
            "W_emit_J_per_m": float(outer.get("driver", {}).get("W_ext_acc", 0.0))
            if not outer.get("history", {}).get("hist", {}).get("W_emit")
            else float(outer["history"]["hist"]["W_emit"][-1]),
        },
        "physical_loss_ledgers_unchanged": ledgers_before == kinetic.get("ledgers", {}),
        "native_reequilibration": equilibrium,
        "tip_bulk_population_transfer_added": False,
    }
    kinetic = dict(kinetic)
    kinetic["reason"] = "p3_history_reset_reequilibrated"
    kinetic["high_cycle_cache"] = {}
    metadata = dict(kinetic.get("metadata", {}))
    metadata["diagnostic_history_reset"] = receipt
    kinetic["metadata"] = metadata
    outer = dict(outer)
    outer["driver"] = dict(outer["driver"])
    outer["driver"]["Ftop_prev"] = equilibrium["reaction_N_per_m"]
    outer["history"] = dict(outer["history"])
    history = dict(outer["history"].get("hist", {}))
    if history.get("Ftop"):
        history["Ftop"] = list(history["Ftop"])
        history["Ftop"][-1] = equilibrium["reaction_N_per_m"]
    outer["history"]["hist"] = history
    interventions = list(outer["history"].get("diagnostic_interventions", []))
    interventions.append(receipt)
    outer["history"]["diagnostic_interventions"] = interventions
    manifest = write_combined_checkpoint(
        destination, outer=outer, arrays=arrays, kinetic=kinetic,
        kinetic_vector=vector,
    )
    # Prove the exact persisted generation is readable and cross-layer valid.
    restored_outer, restored_kinetic, restored_arrays = load_combined_checkpoint(destination)
    validate_cross_layer(restored_outer, restored_kinetic)
    restored_vector = restored_arrays["kinetic_active_vector"]
    if _sha256_array(restored_vector) != receipt["post_reset_population_sha256"]:
        raise RuntimeError("persisted reset vector differs from the qualified reset")
    if not np.array_equal(restored_arrays["displacement"], displacement):
        raise RuntimeError("persisted post-reset equilibrium displacement changed")
    return {**receipt, "destination_checkpoint": str(destination), "manifest": manifest}


__all__ = ["MODEL_ID", "RESET_FIELDS", "transform_history_reset_checkpoint"]
