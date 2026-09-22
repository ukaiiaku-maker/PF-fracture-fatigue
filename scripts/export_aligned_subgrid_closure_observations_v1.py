#!/usr/bin/env python3
"""Export complete consumer-specific observations for the first OneD closure.

No mechanics solve is launched.  Every row comes from a durable accepted
natural-trajectory checkpoint and its matching scheduler observation.  For a
subgrid cavity the production mechanics has no mesh hole; the matched no-void
far-field companion is therefore an exact representation identity, while the
local cavity-source tensor remains an accepted constitutive observation.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.embryo_growth_v1 import source_local_fem_tensors
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint,
)
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    accepted_elastoplastic_stress_3d,
)


RUN = Path(os.environ.get(
    "NATURAL_VOID_RUN_OUT",
    ROOT/"runs/managed_natural_void_v1_20260919/output",
))
OUT = Path(os.environ.get(
    "ALIGNED_SUBGRID_OBSERVATIONS_OUT",
    ROOT/"artifacts/aligned_subgrid_closure_v1/observations.json",
))
CHECKPOINT_SPLIT = {
    "restart_001900ms.json": "TRAIN_NO_VOID",
    "restart_002000ms.json": "HOLDOUT_NO_VOID",
    "restart_002100ms.json": "TRAIN_NO_VOID",
    "restart_002200ms.json": "HOLDOUT_SUBGRID",
    "restart_002600ms.json": "TRAIN_SUBGRID",
}


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode()).hexdigest()


def _plain(value: Any) -> Any:
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if hasattr(value, "value") and isinstance(value.value, str):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _accepted_time(state) -> float:
    return float(state.junction_process_state.get(
        "production_time_s", state.vacancy_transport_state_v2.transport_time_s))


def _triangle_quality(state) -> dict[str, float]:
    xyz = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)]
    sides = np.stack((
        np.linalg.norm(xyz[:, 1]-xyz[:, 0], axis=1),
        np.linalg.norm(xyz[:, 2]-xyz[:, 1], axis=1),
        np.linalg.norm(xyz[:, 0]-xyz[:, 2], axis=1),
    ), axis=1)
    area = np.asarray(state.mesh.area_e, float)
    quality = 4.0*math.sqrt(3.0)*area/np.sum(sides*sides, axis=1)
    return {
        "element_count": int(len(area)),
        "minimum_triangle_quality": float(np.min(quality)),
        "median_triangle_quality": float(np.median(quality)),
        "minimum_element_area_m2": float(np.min(area)),
        "median_element_area_m2": float(np.median(area)),
    }


def _history_by_fingerprint() -> dict[str, Mapping[str, Any]]:
    progress = json.loads((RUN/"progress.json").read_text())
    return {row["accepted_state_fingerprint"]: row for row in progress["history"]}


def _row(checkpoint: str, split: str, history_by_fp) -> dict[str, Any]:
    state = restore_checkpoint(RUN/checkpoint)
    accepted_id = complete_accepted_state_fingerprint(state)
    history = history_by_fp.get(accepted_id)
    if history is None:
        raise RuntimeError(f"{checkpoint} has no matching accepted scheduler observation")
    drive_rows = history["canonical_drive"]["canonical_directional_rows"]
    if len(drive_rows) != 1:
        raise RuntimeError("aligned closure requires exactly one canonical tip row")
    drive = drive_rows[0]
    source_commits = {
        item["source_commit"] for item in drive.get("marginal_trial_rows", [])
        if item.get("source_commit") is not None
    }
    if drive.get("source_commit") is not None:
        source_commits.add(drive["source_commit"])
    if len(source_commits) != 1:
        raise RuntimeError("canonical drive has no unique implementation commit")
    source_commit = next(iter(source_commits))
    stress = accepted_elastoplastic_stress_3d(state)
    centers = np.asarray(state.mesh.nodes)[np.asarray(state.mesh.elems)].mean(axis=1)
    sites = []
    for site in state.void_state.sites:
        eid = int(np.argmin(np.linalg.norm(
            centers-np.asarray(site.center_m, float), axis=1)))
        sites.append({
            "site_id": site.site_id,
            "center_m": list(site.center_m),
            "phase": site.phase.value,
            "element_id": eid,
            "accepted_constitutive_stress_Pa": {
                key: float(value[eid]) for key, value in stress.items()
                if key != "source"
            },
            "equivalent_plastic_components": np.asarray(state.ep_gp)[:, eid].tolist(),
            "forest_density_m2": float(np.asarray(state.rho_gp)[eid]),
        })
    growth_rows = []
    for growth in state.void_state.embryo_growth_states:
        tensors, sigma_zz = source_local_fem_tensors(state, growth)
        growth_rows.append({
            "site_id": growth.site_id,
            "representation": growth.representation,
            "radius_m": growth.radius_m,
            "area_m2": growth.area_m2,
            "vacancy_derived_area_m2": growth.vacancy_derived_area_m2,
            "dislocation_derived_area_m2": growth.dislocation_derived_area_m2,
            "emitted_loop_count_effective": growth.emitted_loop_count_effective,
            "source_tensors_xy_Pa": tensors.tolist(),
            "source_sigma_zz_Pa": sigma_zz.tolist(),
            "source_states": [_plain(source)
                              for source in growth.cavity_emission_state.sources],
            "source_tensor_status": (
                "ACCEPTED_3D_ELASTOPLASTIC_CONSTITUTIVE_SUBGRID"),
        })
    cavities = [_plain(item) for item in state.void_state.cavities]
    explicit_geometry = any(item["phase"] != "STABLE_SUBGRID_VOID"
                            for item in cavities)
    if explicit_geometry:
        raise RuntimeError("first closure exporter is subgrid/preconnection only")
    equilibrium = state.equilibrium_observables.to_dict()
    transport = state.vacancy_transport_state_v2
    result = {
        "schema": "unified-2d.aligned-subgrid-closure-observation/1",
        "checkpoint": checkpoint,
        "split": split,
        "accepted_time_s": _accepted_time(state),
        "accepted_state_fingerprint": accepted_id,
        "checkpoint_state_sha256": json.loads((RUN/checkpoint).read_text())["state_sha256"],
        "implementation_identity": {
            "source_commit": source_commit,
            "mechanical_model": "CURRENT_UNIFIED_2D_PRODUCTION",
            "void_representation": "ANALYTICAL_SUBGRID_NO_MESH_HOLE",
        },
        "material_fingerprint": equilibrium["material_fingerprint"],
        "topology_regime": "PRECONNECTION",
        "topology_fingerprint": drive["topology_fingerprint"],
        "tip": {
            "x_tip_m": state.crack_network.branch(
                state.crack_network.active_tip_ids[0]).tip[0],
            "drive": drive,
            "adjacent_constitutive_tensor_Pa": history[
                "canonical_drive"]["adjacent_element_tensor_Pa"],
        },
        "applied_load": {
            "opening_m": equilibrium["total_opening_m"],
            "top_reaction_N_per_m": equilibrium["top_normal_resultant_N_per_m"],
            "bottom_reaction_N_per_m": equilibrium["bottom_normal_resultant_N_per_m"],
            "reaction_balance": equilibrium["top_bottom_balance_residual"],
            "compliance_m2_per_N": equilibrium["compliance_m2_per_N"],
            "recoverable_energy_J_per_m": equilibrium[
                "recoverable_elastic_energy_J_per_m"],
            "total_internal_energy_J_per_m": equilibrium[
                "total_stored_internal_energy_J_per_m"],
            "energy_identity_residual": equilibrium["energy_identity_residual"],
        },
        "plastic_history": {
            "maximum_equivalent_plastic_strain": history[
                "maximum_equivalent_plastic_strain"],
            "plastic_work_J_per_m": history["plastic_work_J_per_m"],
            "sites": sites,
        },
        "cavities": cavities,
        "growth": growth_rows,
        "direct_opportunity": history["direct_opportunity"],
        "transport": {
            "state_fingerprint": transport.fingerprint,
            "transport_time_s": transport.transport_time_s,
            "vacancy_fraction_min": float(np.min(transport.vacancy_fraction_gp)),
            "vacancy_fraction_max": float(np.max(transport.vacancy_fraction_gp)),
            "site_states": [_plain(item) for item in transport.site_states],
            "ledgers": _plain(transport.ledgers),
            "last_solver_audit": _plain(transport.last_solver_audit),
        },
        "mesh": _triangle_quality(state),
        "matched_no_void_companion": {
            "kind": "FIXED_ACCEPTED_HISTORY_DIAGNOSTIC_COMPANION",
            "mesh_displacement_plastic_history_identical": True,
            "explicit_cavity_geometry_removed": False,
            "reason": "subgrid owner has no mechanical mesh hole",
            "far_field_tip_delta_G_J_per_m2": 0.0,
            "delta_reaction_N_per_m": 0.0,
            "delta_compliance_m2_per_N": 0.0,
            "delta_recoverable_energy_J_per_m": 0.0,
            "status": "EXACT_PRODUCTION_REPRESENTATION_IDENTITY_NOT_A_FEM_DIFFERENCE_FIT",
        },
    }
    result["observation_sha256"] = _sha(result)
    return result


def main() -> None:
    history = _history_by_fingerprint()
    rows = [_row(name, split, history)
            for name, split in CHECKPOINT_SPLIT.items()]
    payload = {
        "schema": "unified-2d.aligned-subgrid-closure-dataset/1",
        "scope": "DBTT_1200K_ALIGNED_PRECONNECTION_SUBGRID",
        "split_declared_by_checkpoint_before_fit": CHECKPOINT_SPLIT,
        "far_field_correction_policy": (
            "EXACT_ZERO_WHILE_PRODUCTION_REPRESENTATION_HAS_NO_MESH_HOLE"),
        "local_source_policy": "ACCEPTED_3D_ELASTOPLASTIC_CONSTITUTIVE_TENSOR",
        "rows": rows,
    }
    payload["dataset_sha256"] = _sha(payload)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUT.with_suffix(OUT.suffix+".tmp")
    temporary.write_text(json.dumps(_plain(payload), indent=2, sort_keys=True,
                                    allow_nan=False)+"\n")
    os.replace(temporary, OUT)
    print(OUT)
    print("OBSERVATIONS", len(rows))
    print("DATASET_SHA256", payload["dataset_sha256"])


if __name__ == "__main__":
    main()
