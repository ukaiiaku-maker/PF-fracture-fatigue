"""Serializable fixed-crack FEM snapshots for physical signed-kernel generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import ElasticProperties, JIntegralConfig
from .mesh import BoundaryData, rebuild_tri_mesh
from .unit_slip_perturbation_v10212 import (
    INTERACTION_INTEGRAL_MODEL_ID,
    SlipRibbonPerturbation,
    equilibrated_base_state,
    interaction_response,
)

MODEL_ID = "v10.2.14_serialized_physical_fixed_crack_fem_state"
LEGACY_MODEL_IDS = {
    "v10.2.12_serialized_physical_fixed_crack_fem_state",
}
RESPONSE_COLUMNS = (
    "state_id",
    "r_eff_over_r0",
    "opening_strength_fraction",
    "crack_extension_m",
    "region",
    "system",
    "bin",
    "x_m",
    "burgers_sign",
    "delta_signed_line_content",
    "K_I_base_Pa_sqrt_m",
    "K_I_perturbed_Pa_sqrt_m",
    "K_II_base_Pa_sqrt_m",
    "K_II_perturbed_Pa_sqrt_m",
    "interaction_integral_schema",
    "ribbon_width_m",
    "mesh_area_ratio",
)


@dataclass(frozen=True)
class SnapshotMetadata:
    state_id: str
    r_eff_over_r0: float
    opening_strength_fraction: float
    crack_extension_m: float
    temperature_K: float
    Uy_top_m: float
    Uy_bot_m: float
    crack_tip_xy_m: tuple[float, float]
    crack_direction: tuple[float, float]
    interaction_ell_m: float
    exclude_radius_m: float
    active_x_m: tuple[float, ...]
    wake_x_m: tuple[float, ...]
    channel_directions: tuple[tuple[float, float], ...]
    channel_normals: tuple[tuple[float, float], ...]
    material: dict[str, float]
    engine_config: dict[str, Any]
    fem_tip_geometry_blunted: bool = False
    r_eff_is_analytical_tip_state: bool = True
    cohesive_network_present: bool = False
    crack_path_xy_m: tuple[tuple[float, float], ...] = ()
    displacement_state: str = "post_dirichlet_equilibrium"
    active_kernel_supported: bool = True
    wake_kernel_supported: bool = False

    def validate(self) -> "SnapshotMetadata":
        if not str(self.state_id).strip():
            raise ValueError("state_id must be nonempty")
        for name in (
            "r_eff_over_r0",
            "opening_strength_fraction",
            "crack_extension_m",
            "temperature_K",
            "Uy_top_m",
            "Uy_bot_m",
            "interaction_ell_m",
            "exclude_radius_m",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.r_eff_over_r0 <= 0.0:
            raise ValueError("r_eff_over_r0 must be positive")
        if not 0.0 <= self.opening_strength_fraction <= 1.0:
            raise ValueError("opening_strength_fraction must lie in [0,1]")
        if self.crack_extension_m < 0.0 or self.interaction_ell_m <= 0.0:
            raise ValueError("extension must be nonnegative and interaction ell positive")
        if self.exclude_radius_m < 0.0:
            raise ValueError("exclude radius must be nonnegative")
        directions = np.asarray(self.channel_directions, dtype=float)
        normals = np.asarray(self.channel_normals, dtype=float)
        if directions.ndim != 2 or directions.shape[1] != 2:
            raise ValueError("channel directions must have shape (n,2)")
        if normals.shape != directions.shape:
            raise ValueError("channel normals must match channel directions")
        if directions.shape[0] < 1:
            raise ValueError("at least one slip channel is required")
        for direction, normal in zip(directions, normals):
            if np.linalg.norm(direction) <= 0.0 or np.linalg.norm(normal) <= 0.0:
                raise ValueError("channel vectors must be nonzero")
            if abs(float(direction @ normal)) > 1.0e-6 * np.linalg.norm(direction) * np.linalg.norm(normal):
                raise ValueError("channel direction and normal must be orthogonal")
        path = np.asarray(self.crack_path_xy_m, dtype=float)
        if path.size and (path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2):
            raise ValueError("crack_path_xy_m must be empty or have shape (n>=2,2)")
        if path.size and not np.all(np.isfinite(path)):
            raise ValueError("crack_path_xy_m must be finite")
        if self.displacement_state not in {
            "post_dirichlet_equilibrium",
            "legacy_pre_solve_iterate_re_equilibrated_on_load",
        }:
            raise ValueError("invalid displacement_state")
        if not bool(self.active_kernel_supported):
            raise ValueError("v10.2.14 snapshots require active-kernel support")
        if bool(self.wake_kernel_supported):
            raise ValueError(
                "v10.2.14 scalar wake state cannot claim a physical 2-D wake kernel"
            )
        if self.cohesive_network_present:
            raise ValueError(
                "snapshot replay does not serialize cohesive-network state; "
                "collect sharp-front PF states or add an explicit cohesive serializer"
            )
        return self


def save_snapshot(
    root: str | Path,
    *,
    metadata: SnapshotMetadata,
    mesh,
    boundary,
    u: np.ndarray,
    ep_gp: np.ndarray,
    rho_gp: np.ndarray,
    d: np.ndarray,
    D: np.ndarray,
) -> dict[str, Any]:
    metadata = metadata.validate()
    root = Path(root)
    if root.exists():
        raise FileExistsError(f"refusing to overwrite snapshot {root}")
    root.mkdir(parents=True)
    arrays = root / "state_arrays.npz"
    np.savez_compressed(
        arrays,
        nodes=np.asarray(mesh.nodes, dtype=float),
        elems=np.asarray(mesh.elems, dtype=int),
        u=np.asarray(u, dtype=float),
        ep_gp=np.asarray(ep_gp, dtype=float),
        rho_gp=np.asarray(rho_gp, dtype=float),
        d=np.asarray(d, dtype=float),
        D=np.asarray(D, dtype=float),
        top_nodes=np.asarray(boundary.top_nodes, dtype=int),
        bot_nodes=np.asarray(boundary.bot_nodes, dtype=int),
        left_bot=np.asarray([boundary.left_bot], dtype=int),
        right_bot=np.asarray([boundary.right_bot], dtype=int),
        notch_nodes=np.asarray(boundary.notch_nodes, dtype=int),
    )
    payload = {
        "schema": MODEL_ID,
        **asdict(metadata),
        "arrays": arrays.name,
        "fixed_crack_geometry": True,
        "accepted_production_state_copied": True,
        "production_state_mutated": False,
        "active_kernel_supported": True,
        "wake_kernel_supported": False,
    }
    (root / "snapshot.json").write_text(json.dumps(payload, indent=2))
    return payload


def load_snapshot(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    metadata_path = root / "snapshot.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)
    payload = json.loads(metadata_path.read_text())
    schema = payload.get("schema")
    if schema not in {MODEL_ID, *LEGACY_MODEL_IDS}:
        raise ValueError("snapshot schema mismatch")
    legacy = schema in LEGACY_MODEL_IDS
    payload.setdefault("crack_path_xy_m", ())
    payload.setdefault(
        "displacement_state",
        "legacy_pre_solve_iterate_re_equilibrated_on_load" if legacy else "post_dirichlet_equilibrium",
    )
    payload.setdefault("active_kernel_supported", True)
    payload["wake_kernel_supported"] = False
    metadata_fields = {
        key: payload[key]
        for key in SnapshotMetadata.__dataclass_fields__
        if key in payload
    }
    for key in (
        "crack_tip_xy_m",
        "crack_direction",
        "active_x_m",
        "wake_x_m",
        "channel_directions",
        "channel_normals",
        "crack_path_xy_m",
    ):
        if key in metadata_fields:
            value = metadata_fields[key]
            if key in {"channel_directions", "channel_normals", "crack_path_xy_m"}:
                metadata_fields[key] = tuple(tuple(row) for row in value)
            else:
                metadata_fields[key] = tuple(value)
    metadata = SnapshotMetadata(**metadata_fields).validate()
    arrays = np.load(root / payload["arrays"], allow_pickle=False)
    mesh = rebuild_tri_mesh(
        arrays["nodes"], arrays["elems"], tip_centers=np.asarray(metadata.crack_tip_xy_m)
    )
    boundary = BoundaryData(
        top_nodes=np.asarray(arrays["top_nodes"], dtype=int),
        bot_nodes=np.asarray(arrays["bot_nodes"], dtype=int),
        left_bot=int(arrays["left_bot"][0]),
        right_bot=int(arrays["right_bot"][0]),
        notch_nodes=np.asarray(arrays["notch_nodes"], dtype=int),
    )
    mat = ElasticProperties(**metadata.material)
    return {
        "metadata": metadata,
        "mesh": mesh,
        "boundary": boundary,
        "u": np.asarray(arrays["u"], dtype=float),
        "ep_gp": np.asarray(arrays["ep_gp"], dtype=float),
        "rho_gp": np.asarray(arrays["rho_gp"], dtype=float),
        "d": np.asarray(arrays["d"], dtype=float),
        "D": np.asarray(arrays["D"], dtype=float),
        "mat": mat,
        "legacy_snapshot_schema": bool(legacy),
    }


def _unit(value: Iterable[float]) -> np.ndarray:
    result = np.asarray(tuple(value), dtype=float).reshape(2)
    norm = float(np.linalg.norm(result))
    if norm <= 0.0:
        raise ValueError("zero direction vector")
    return result / norm


def generate_signed_responses(
    snapshot_root: str | Path,
    *,
    out_csv: str | Path,
    magnitudes: Iterable[float] = (0.25, 0.5),
    ribbon_width_m: float | None = None,
    interaction_cfg: JIntegralConfig | None = None,
) -> dict[str, Any]:
    """Compatibility entry point; production uses resolved station generation."""
    data = load_snapshot(snapshot_root)
    meta: SnapshotMetadata = data["metadata"]
    out_csv = Path(out_csv)
    if out_csv.exists():
        raise FileExistsError(f"refusing to overwrite {out_csv}")
    values = sorted({float(value) for value in magnitudes})
    if len(values) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("at least two positive perturbation magnitudes are required")
    width = (
        max(2.0 * float(data["mesh"].hbar_tip), 10.0 * float(data["mat"].b))
        if ribbon_width_m is None
        else float(ribbon_width_m)
    )
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("ribbon width must be positive and finite")
    base = equilibrated_base_state(
        mesh=data["mesh"], boundary=data["boundary"], baseline_u=data["u"],
        baseline_ep_gp=data["ep_gp"], rho_gp=data["rho_gp"], d=data["d"],
        D=data["D"], mat=data["mat"], Uy_top=meta.Uy_top_m, Uy_bot=meta.Uy_bot_m,
    )
    tip = np.asarray(meta.crack_tip_xy_m, dtype=float)
    forward = _unit(meta.crack_direction)
    if meta.crack_path_xy_m:
        points = [np.asarray(row, dtype=float) for row in meta.crack_path_xy_m]
        crack_segments = [(points[i], points[i + 1]) for i in range(len(points) - 1)]
    else:
        domain_length = float(np.ptp(data["mesh"].nodes[:, 0]) + np.ptp(data["mesh"].nodes[:, 1]))
        crack_segments = [(tip - max(domain_length, meta.interaction_ell_m) * forward, tip)]
    rows: list[dict[str, Any]] = []
    channel_directions = [_unit(row) for row in meta.channel_directions]
    channel_normals = [_unit(row) for row in meta.channel_normals]
    for system, (slip, normal) in enumerate(zip(channel_directions, channel_normals)):
        oriented_slip = slip if float(slip @ forward) >= 0.0 else -slip
        for bin_index, x_m in enumerate(meta.active_x_m):
            distance = max(float(x_m), 4.0 * width)
            start = tip.copy()
            end = tip + distance * oriented_slip
            for sign in (-1, 1):
                for magnitude in values:
                    perturbation = SlipRibbonPerturbation(
                        system=system, region="active", bin_index=bin_index,
                        start_xy_m=start, end_xy_m=end,
                        slip_direction=oriented_slip, plane_normal=normal,
                        width_m=width, burgers_m=float(data["mat"].b),
                        signed_line_content=float(sign) * magnitude,
                    )
                    response = interaction_response(
                        mesh=data["mesh"], base_state=base,
                        baseline_ep_gp=data["ep_gp"], rho_gp=data["rho_gp"],
                        d=data["d"], D=data["D"], mat=data["mat"],
                        boundary=data["boundary"], Uy_top=meta.Uy_top_m,
                        Uy_bot=meta.Uy_bot_m, crack_tip=tip,
                        crack_direction=forward, interaction_ell_m=meta.interaction_ell_m,
                        perturbation=perturbation, interaction_cfg=interaction_cfg,
                        crack_segments=crack_segments, exclude_radius_m=meta.exclude_radius_m,
                    )
                    audit = response["perturbation"]
                    rows.append({
                        "state_id": meta.state_id,
                        "r_eff_over_r0": meta.r_eff_over_r0,
                        "opening_strength_fraction": meta.opening_strength_fraction,
                        "crack_extension_m": meta.crack_extension_m,
                        "region": "active", "system": system, "bin": bin_index,
                        "x_m": float(x_m), "burgers_sign": sign,
                        "delta_signed_line_content": float(sign) * magnitude,
                        "K_I_base_Pa_sqrt_m": response["K_I_base_Pa_sqrt_m"],
                        "K_I_perturbed_Pa_sqrt_m": response["K_I_perturbed_Pa_sqrt_m"],
                        "K_II_base_Pa_sqrt_m": response["K_II_base_Pa_sqrt_m"],
                        "K_II_perturbed_Pa_sqrt_m": response["K_II_perturbed_Pa_sqrt_m"],
                        "interaction_integral_schema": INTERACTION_INTEGRAL_MODEL_ID,
                        "ribbon_width_m": width,
                        "mesh_area_ratio": audit["mesh_area_ratio"],
                    })
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESPONSE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema": MODEL_ID,
        "snapshot": str(Path(snapshot_root).resolve()),
        "responses": str(out_csv.resolve()),
        "response_rows": len(rows),
        "active_kernel_mechanically_measured": True,
        "wake_kernel_mechanically_measured": False,
        "wake_shielding_supported": False,
        "production_parameterization_allowed": False,
    }
    out_csv.with_suffix(".audit.json").write_text(json.dumps(report, indent=2))
    return report


__all__ = [
    "MODEL_ID",
    "RESPONSE_COLUMNS",
    "SnapshotMetadata",
    "save_snapshot",
    "load_snapshot",
    "generate_signed_responses",
]
