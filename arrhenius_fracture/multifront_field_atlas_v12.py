"""Sparse, read-only field export for the V12 multifront field atlas.

The observer is deliberately outside the accepted-interval lifecycle.  It
receives already accepted states, copies their arrays into portable files, and
never returns data to the mechanics, topology, process-zone, or hazard code.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any, Iterable, Mapping

import numpy as np

from .current_source_multifront_hooks_v12 import (
    restore_complete_current_source_engine,
)
from .general_multifront_v12 import MultiFrontRuntimeState, canonical_hash
from .network_metrics_v11 import crack_growth_metrics


SCHEMA = "v12.multifront-field-atlas-snapshot/1"
CLAIM_LABEL = "CAPABILITY_DEMONSTRATION_NOT_VALIDATED_BRANCHING_PHYSICS"
MILESTONES_UM = (250.0, 500.0, 750.0, 1000.0)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "unnamed"


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _array_inventory(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    return {
        key: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sorted(arrays.items())
    }


def _write_vtu(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    """Write a compact ASCII VTU usable without a Python environment."""
    nodes = np.asarray(arrays["nodes_m"], dtype=float)
    elems = np.asarray(arrays["elements"], dtype=np.int64)
    point_fields = {
        "ux_m": arrays["ux_m"], "uy_m": arrays["uy_m"],
        "displacement_magnitude_m": arrays["displacement_magnitude_m"],
        "damage": arrays["damage_nodal"],
    }
    cell_fields = {
        key: value for key, value in arrays.items()
        if key in {
            "plastic_strain_xx", "plastic_strain_yy", "plastic_strain_xy",
            "dislocation_density_m-2", "sigma_xx_Pa", "sigma_yy_Pa",
            "sigma_xy_Pa", "strain_xx", "strain_yy", "engineering_shear_strain_xy",
            "maximum_principal_stress_Pa", "minimum_principal_stress_Pa",
            "von_Mises_stress_Pa", "equivalent_plastic_strain",
            "material_region_id", "mesh_region_id", "element_damage",
        }
    }

    def values(a: np.ndarray) -> str:
        flat = np.asarray(a).reshape(-1)
        if np.issubdtype(flat.dtype, np.integer):
            return " ".join(str(int(v)) for v in flat)
        return " ".join(f"{float(v):.17g}" for v in flat)

    def data_array(name: str, value: np.ndarray) -> str:
        kind = "Int64" if np.issubdtype(np.asarray(value).dtype, np.integer) else "Float64"
        return f'<DataArray type="{kind}" Name="{name}" format="ascii">{values(value)}</DataArray>'

    points = np.column_stack((nodes, np.zeros(len(nodes))))
    offsets = np.arange(1, len(elems) + 1, dtype=np.int64) * 3
    types = np.full(len(elems), 5, dtype=np.int64)
    text = [
        '<?xml version="1.0"?>',
        '<VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">',
        '<UnstructuredGrid>',
        f'<Piece NumberOfPoints="{len(nodes)}" NumberOfCells="{len(elems)}">',
        '<PointData>',
        *(data_array(k, v) for k, v in point_fields.items()),
        '</PointData>', '<CellData>',
        *(data_array(k, v) for k, v in cell_fields.items()),
        '</CellData>', '<Points>',
        '<DataArray type="Float64" NumberOfComponents="3" format="ascii">'
        + values(points) + '</DataArray>',
        '</Points>', '<Cells>',
        '<DataArray type="Int64" Name="connectivity" format="ascii">'
        + values(elems) + '</DataArray>',
        '<DataArray type="Int64" Name="offsets" format="ascii">'
        + values(offsets) + '</DataArray>',
        '<DataArray type="UInt8" Name="types" format="ascii">'
        + values(types) + '</DataArray>',
        '</Cells>', '</Piece>', '</UnstructuredGrid>', '</VTKFile>',
    ]
    path.write_text("\n".join(text) + "\n")


def _write_crack_vtp(path: Path, network: Mapping[str, Any]) -> None:
    points: list[list[float]] = []
    lines: list[list[int]] = []
    status: list[int] = []
    generation: list[int] = []
    code = {"active": 1, "retired": 2, "coalesced": 3}
    for branch in network["branches"]:
        ids = []
        for xy in branch["path_m"]:
            ids.append(len(points)); points.append([float(xy[0]), float(xy[1]), 0.0])
        for left, right in zip(ids, ids[1:]):
            lines.append([left, right])
            status.append(code.get(branch["status"], 0))
            generation.append(int(branch["generation"]))
    offsets = np.arange(1, len(lines) + 1, dtype=np.int64) * 2
    flatten = lambda x: " ".join(str(v) for row in x for v in row)
    pvalues = " ".join(f"{v:.17g}" for row in points for v in row)
    text = f'''<?xml version="1.0"?>
<VTKFile type="PolyData" version="0.1" byte_order="LittleEndian">
<PolyData><Piece NumberOfPoints="{len(points)}" NumberOfVerts="0" NumberOfLines="{len(lines)}" NumberOfStrips="0" NumberOfPolys="0">
<PointData/><CellData>
<DataArray type="Int32" Name="branch_status" format="ascii">{' '.join(map(str, status))}</DataArray>
<DataArray type="Int32" Name="generation" format="ascii">{' '.join(map(str, generation))}</DataArray>
</CellData><Points><DataArray type="Float64" NumberOfComponents="3" format="ascii">{pvalues}</DataArray></Points>
<Lines><DataArray type="Int64" Name="connectivity" format="ascii">{flatten(lines)}</DataArray>
<DataArray type="Int64" Name="offsets" format="ascii">{' '.join(map(str, offsets))}</DataArray></Lines>
</Piece></PolyData></VTKFile>
'''
    path.write_text(text)


def _owner_arrays(runtime: MultiFrontRuntimeState) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {}
    for owner_id, region in sorted(runtime.process_regions.items()):
        state = runtime.process_engines[region.process_engine_id]
        engine = restore_complete_current_source_engine(state)
        mpz = engine.mpz
        prefix = "owner__" + _safe(owner_id) + "__"
        for name, value in sorted(mpz.__dict__.items()):
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufc":
                arrays[prefix + "mpz__" + _safe(name)] = np.asarray(value).copy()
        active_content = (
            np.asarray(mpz.retained_positive) - np.asarray(mpz.retained_negative)
            + float(mpz.cfg.mobile_shield_fraction)
            * (np.asarray(mpz.mobile_positive) - np.asarray(mpz.mobile_negative))
        )
        shielding_by_system_bin = np.asarray(mpz._signed_kernel.active_kernel) * active_content
        arrays[prefix + "signed_active_shielding_by_system_bin_Pa_sqrt_m"] = shielding_by_system_bin
        arrays[prefix + "signed_active_shielding_by_bin_Pa_sqrt_m"] = np.sum(
            shielding_by_system_bin, axis=0
        )
        arrays[prefix + "mobile_profile"] = np.asarray(mpz.mobile).copy()
        arrays[prefix + "retained_profile"] = np.asarray(mpz.retained).copy()
        arrays[prefix + "accumulated_slip_profile"] = np.asarray(mpz.accumulated_slip).copy()
        arrays[prefix + "wake_mobile_profile"] = np.asarray(mpz.wake_mobile).copy()
        arrays[prefix + "wake_retained_profile"] = np.asarray(mpz.wake_retained).copy()
        arrays[prefix + "owner_local_x_m"] = np.asarray(mpz.x).copy()
        arrays[prefix + "wake_x_m"] = np.asarray(mpz.wake_x).copy()
        backstress = np.asarray(
            getattr(mpz, "anisotropic_last_sigma_back_by_system_Pa",
                    getattr(mpz, "persistent_site_last_sigma_back_initial_Pa", [])),
            dtype=float,
        )
        arrays[prefix + "backstress_by_system_Pa"] = backstress.copy()
        diagnostics = mpz.diagnostics(engine.G, engine.nu, engine.b, engine.f.r0)
        metadata[owner_id] = {
            "member_front_ids": sorted(region.member_front_ids),
            "process_engine_id": region.process_engine_id,
            "source_state_id": region.source_state_id,
            "owner_local_family_coordinate_m": region.cumulative_process_advance_m,
            "unresolved_junction_ids": sorted(region.unresolved_junction_ids),
            "source_reservoir_id": region.source_reservoir_id,
            "process_update_count": state.update_count,
            "event_renewal_count": state.event_renewal_count,
            "active_ledgers": dict(state.active_ledgers),
            "wake_ledgers": dict(state.wake_ledgers),
            "signed_system_ledgers": dict(state.signed_system_ledgers),
            "engine_diagnostics": _jsonable(diagnostics),
            "source_multiplicity_per_system": diagnostics.get(
                "persistent_site_multiplicity_per_system"
            ),
            "effective_tip_radius_m": diagnostics.get("persistent_tip_radius_m"),
            "active_signed_shielding_Pa_sqrt_m": float(engine._active_shielding_signed()),
            "wake_signed_shielding_Pa_sqrt_m": float(engine._wake_shielding_signed()),
            "array_prefix": prefix,
        }
    metadata["archived_reservoirs"] = {
        key: value.to_dict() for key, value in sorted(runtime.reservoirs.items())
    }
    return arrays, metadata


def export_snapshot(
    root: str | Path, *, case: str, runtime: MultiFrontRuntimeState,
    accepted_fem_state: Any, accepted_stress_field: np.ndarray,
    physical_time_s: float, accepted_opening_m: float, step_count: int,
    mechanics_source_identity: str, selection_reasons: Iterable[str],
    accepted_checkpoint_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export one immutable accepted state and merge duplicate selection reasons."""
    root = Path(root)
    identity = canonical_hash({
        "accepted_state_id": runtime.accepted_state_id,
        "stress_field_state_id": runtime.stress_field_state_id,
        "step_count": int(step_count),
    })
    directory = root / f"step_{int(step_count):07d}__{identity[:12]}"
    metadata_path = directory / "metadata.json"
    reasons = sorted(set(str(v) for v in selection_reasons))
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        metadata["selection_reasons"] = sorted(set(metadata["selection_reasons"] + reasons))
        metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
        return metadata
    directory.mkdir(parents=True, exist_ok=False)
    state = accepted_fem_state
    mesh = state.mesh
    sigma = np.asarray(accepted_stress_field, dtype=float)
    if sigma.shape != (3, mesh.ne):
        raise ValueError("accepted stress tensor shape does not match the accepted mesh")
    u = np.asarray(state.displacement, dtype=float)
    ux, uy = u[0::2], u[1::2]
    eps_tot = np.einsum("eij,ej->ei", mesh.B_e, u[np.column_stack((
        2 * mesh.elems[:, 0], 2 * mesh.elems[:, 0] + 1,
        2 * mesh.elems[:, 1], 2 * mesh.elems[:, 1] + 1,
        2 * mesh.elems[:, 2], 2 * mesh.elems[:, 2] + 1,
    ))])
    sx, sy, txy = sigma
    avg = 0.5 * (sx + sy)
    radius = np.sqrt((0.5 * (sx - sy)) ** 2 + txy ** 2)
    szz = float(state.material.nu) * (sx + sy)
    von_mises = np.sqrt(
        0.5 * ((sx - sy) ** 2 + (sy - szz) ** 2 + (szz - sx) ** 2)
        + 3.0 * txy ** 2
    )
    ep = np.asarray(state.ep_gp, dtype=float)
    equivalent_ep = np.sqrt((2.0 / 3.0) * (ep[0] ** 2 + ep[1] ** 2 + 2.0 * ep[2] ** 2))
    inherited_damage = getattr(mesh, "element_damage_gp", None)
    element_damage = (
        np.asarray(inherited_damage, dtype=float).copy()
        if inherited_damage is not None else np.mean(np.asarray(state.damage)[mesh.elems], axis=1)
    )
    arrays: dict[str, np.ndarray] = {
        "nodes_m": np.asarray(mesh.nodes, dtype=float).copy(),
        "elements": np.asarray(mesh.elems, dtype=np.int64).copy(),
        "ux_m": ux.copy(), "uy_m": uy.copy(),
        "displacement_magnitude_m": np.hypot(ux, uy),
        "damage_nodal": np.asarray(state.damage, dtype=float).copy(),
        "element_damage": element_damage,
        "plastic_strain_xx": ep[0].copy(), "plastic_strain_yy": ep[1].copy(),
        "plastic_strain_xy": ep[2].copy(),
        "dislocation_density_m-2": np.asarray(state.rho_gp, dtype=float).copy(),
        "sigma_xx_Pa": sx.copy(), "sigma_yy_Pa": sy.copy(), "sigma_xy_Pa": txy.copy(),
        "strain_xx": eps_tot[:, 0].copy(), "strain_yy": eps_tot[:, 1].copy(),
        "engineering_shear_strain_xy": eps_tot[:, 2].copy(),
        "maximum_principal_stress_Pa": (avg + radius),
        "minimum_principal_stress_Pa": (avg - radius),
        "von_Mises_stress_Pa": von_mises,
        "equivalent_plastic_strain": equivalent_ep,
        "material_region_id": np.zeros(mesh.ne, dtype=np.int64),
        "mesh_region_id": np.zeros(mesh.ne, dtype=np.int64),
    }
    owner_data, owner_metadata = _owner_arrays(runtime)
    arrays.update(owner_data)
    npz_path = directory / "fields.npz"
    np.savez_compressed(npz_path, **arrays)
    network = runtime.crack_network.to_dict()
    network_path = directory / "crack_network.json"
    network_path.write_text(json.dumps(network, indent=2, sort_keys=True) + "\n")
    vtu_path = directory / "fields.vtu"
    crack_vtp_path = directory / "crack_network.vtp"
    _write_vtu(vtu_path, arrays); _write_crack_vtp(crack_vtp_path, network)
    growth = crack_growth_metrics(runtime.crack_network, initial_crack_length_m=0.5e-3)
    checkpoint_copy = None
    if accepted_checkpoint_path is not None and Path(accepted_checkpoint_path).is_file():
        source = Path(accepted_checkpoint_path)
        checkpoint_copy = directory / "accepted_checkpoint.v12.pkl"
        shutil.copyfile(source, checkpoint_copy)
    units = {
        "nodes_m": "m", "ux_m": "m", "uy_m": "m",
        "displacement_magnitude_m": "m", "damage_nodal": "1",
        "plastic_strain_*": "1", "dislocation_density_m-2": "m^-2",
        "sigma_*_Pa": "Pa", "strain_*": "1",
        "maximum_principal_stress_Pa": "Pa", "minimum_principal_stress_Pa": "Pa",
        "von_Mises_stress_Pa": "Pa", "equivalent_plastic_strain": "1",
        "owner_local_x_m": "m", "wake_x_m": "m",
        "mobile/retained/accumulated_slip": "signed line-content count",
        "backstress_by_system_Pa": "Pa",
        "signed_active_shielding_by_bin_Pa_sqrt_m": "Pa sqrt(m)",
    }
    metadata = {
        "schema": SCHEMA, "claim_label": CLAIM_LABEL, "case": case,
        "selection_reasons": reasons, "step_count": int(step_count),
        "physical_time_s": float(physical_time_s),
        "accepted_opening_m": float(accepted_opening_m),
        "accepted_state_id": runtime.accepted_state_id,
        "stress_field_state_id": runtime.stress_field_state_id,
        "runtime_registry_fingerprint": runtime.registry_fingerprint,
        "topology_fingerprint": runtime.topology_fingerprint,
        "mechanics_source_identity": mechanics_source_identity,
        "maximum_network_forward_reach_um": growth.max_forward_projected_extension_m * 1e6,
        "root_to_tip_extension_um": growth.max_root_to_tip_path_extension_m * 1e6,
        "cumulative_binary_branch_births": runtime.cumulative_branch_births,
        "active_front_ids": list(runtime.active_front_ids),
        "retired_front_ids": [b.branch_id for b in runtime.crack_network.branches if b.status == "retired"],
        "coalesced_front_ids": [b.branch_id for b in runtime.crack_network.branches if b.status == "coalesced"],
        "owner_by_front": dict(runtime.owner_by_front),
        "junctions": {k: v.to_dict() for k, v in runtime.junctions.items()},
        "owners": owner_metadata,
        "array_inventory": _array_inventory(arrays), "units": units,
        "region_identifier_note": (
            "The source mesh has one material and no categorical mesh-region field; "
            "both exported region arrays are exact zero-valued identifiers."
        ),
        "files": {
            "fields_npz": str(npz_path.resolve()), "fields_npz_sha256": sha256(npz_path),
            "crack_network_json": str(network_path.resolve()), "crack_network_sha256": sha256(network_path),
            "fields_vtu": str(vtu_path.resolve()), "fields_vtu_sha256": sha256(vtu_path),
            "crack_network_vtp": str(crack_vtp_path.resolve()), "crack_network_vtp_sha256": sha256(crack_vtp_path),
            "accepted_checkpoint_copy": None if checkpoint_copy is None else str(checkpoint_copy.resolve()),
            "accepted_checkpoint_copy_sha256": None if checkpoint_copy is None else sha256(checkpoint_copy),
        },
    }
    metadata_path.write_text(json.dumps(_jsonable(metadata), indent=2, sort_keys=True) + "\n")
    return metadata


def _owner_signature(runtime: MultiFrontRuntimeState) -> str:
    return canonical_hash({
        "owner_by_front": runtime.owner_by_front,
        "regions": {key: {
            "members": sorted(value.member_front_ids),
            "unresolved": sorted(value.unresolved_junction_ids),
            "engine": value.process_engine_id,
            "source_reservoir_id": value.source_reservoir_id,
        } for key, value in runtime.process_regions.items()},
    })


@dataclass
class SparseAcceptedStateObserver:
    case: str
    snapshot_root: Path
    checkpoint_path: Path
    original_writer: Any
    initial_births: int = 0
    previous_births: int = 0
    previous_owner_signature: str | None = None
    crossed_milestones: set[float] = field(default_factory=set)

    def __call__(self, runtime, path, **kwargs):
        current_growth = crack_growth_metrics(runtime.crack_network, initial_crack_length_m=0.5e-3)
        current_um = current_growth.max_forward_projected_extension_m * 1e6
        # At wrapper entry the atomic pointer still contains the immediately
        # preceding accepted state, so it is the exact nearest accepted state
        # below any newly crossed milestone.
        for milestone in MILESTONES_UM:
            if milestone in self.crossed_milestones or current_um < milestone:
                continue
            previous = Path(path)
            if previous.is_file() and current_um > milestone + 1e-10:
                from .multifront_checkpoint_v12 import load_accepted_boundary_checkpoint_v12
                restored = load_accepted_boundary_checkpoint_v12(previous)
                export_snapshot(
                    self.snapshot_root, case=self.case, runtime=restored.runtime,
                    accepted_fem_state=restored.accepted_fem_state,
                    accepted_stress_field=restored.accepted_stress_field,
                    physical_time_s=restored.physical_time_s,
                    accepted_opening_m=restored.accepted_opening_m,
                    step_count=restored.step_count,
                    mechanics_source_identity=restored.mechanics_source_identity,
                    selection_reasons=(f"milestone_at_or_below_{int(milestone)}um",),
                    accepted_checkpoint_path=previous,
                )
                self.crossed_milestones.add(milestone)
        record = self.original_writer(runtime, path, **kwargs)
        reasons = []
        if runtime.cumulative_branch_births > self.previous_births:
            for birth in range(self.previous_births + 1, runtime.cumulative_branch_births + 1):
                reasons.append(f"accepted_binary_branch_birth_{birth}")
        signature = _owner_signature(runtime)
        if self.previous_owner_signature is not None and signature != self.previous_owner_signature:
            reasons.append("process_owner_handoff_or_partition")
        for milestone in MILESTONES_UM:
            if milestone not in self.crossed_milestones and abs(current_um - milestone) <= 1e-10:
                reasons.append(f"milestone_at_or_below_{int(milestone)}um")
                self.crossed_milestones.add(milestone)
        if reasons:
            export_snapshot(
                self.snapshot_root, case=self.case, runtime=runtime,
                accepted_fem_state=kwargs["accepted_fem_state"],
                accepted_stress_field=kwargs["accepted_stress_field"],
                physical_time_s=kwargs["physical_time_s"],
                accepted_opening_m=kwargs["accepted_opening_m"],
                step_count=kwargs["step_count"],
                mechanics_source_identity=kwargs["mechanics_source_identity"],
                selection_reasons=reasons, accepted_checkpoint_path=path,
            )
        self.previous_births = runtime.cumulative_branch_births
        self.previous_owner_signature = signature
        return record


__all__ = [
    "CLAIM_LABEL", "MILESTONES_UM", "SCHEMA", "SparseAcceptedStateObserver",
    "export_snapshot", "sha256",
]
