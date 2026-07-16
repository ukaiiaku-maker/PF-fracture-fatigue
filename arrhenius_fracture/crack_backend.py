"""Sharp-tip crack-geometry backend for the v10 hazard solver.

This repository intentionally excludes cohesive-zone backends.  A completed
first-passage renewal is represented only by extending the binary sharp-wake
stiffness-kill path.  Hazard clocks, anisotropy, branching, MPZ state and FEM/J
mechanics remain in ``sharp_front.py``.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv

import numpy as np

from .mesh import TriMesh


@dataclass
class CrackAdvanceResult:
    mesh: TriMesh
    boundary: object
    damage: np.ndarray
    displacement: np.ndarray
    moved: float
    inserted: bool
    angle_error_deg: float = 0.0
    selected_edge_length: float = 0.0
    reason: str = "ok"
    elem_parent_map: np.ndarray | None = None


class SharpWakeBackend:
    """Mesh-preserving sharp crack represented by killed bulk stiffness."""

    name = "sharp_wake"

    def __init__(self) -> None:
        self.cohesive_network = None
        self.advance_log: list[dict] = []

    def advance(
        self,
        *,
        mesh: TriMesh,
        boundary,
        damage: np.ndarray,
        displacement: np.ndarray,
        p0: np.ndarray,
        p1: np.ndarray,
        kill_r: float,
        front_id: int = 0,
        **kwargs,
    ) -> CrackAdvanceResult:
        p0 = np.asarray(p0, dtype=float)
        p1 = np.asarray(p1, dtype=float)
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if not np.isfinite(length) or length <= 0.0:
            return CrackAdvanceResult(
                mesh, boundary, damage, displacement, 0.0, False,
                reason="zero_or_nonfinite_length",
            )

        centroids = mesh.nodes[mesh.elems].mean(axis=1)
        length2 = float(seg @ seg)
        t = np.clip(((centroids - p0[None, :]) @ seg) / max(length2, 1.0e-30), 0.0, 1.0)
        projection = p0[None, :] + t[:, None] * seg[None, :]
        distance2 = np.sum((centroids - projection) ** 2, axis=1)
        element_radius = np.sqrt(np.maximum(mesh.area_e, 1.0e-30))
        radius = np.maximum(float(kill_r), 0.7 * element_radius)
        selected = distance2 <= radius ** 2

        dnew = np.asarray(damage, dtype=float).copy()
        if np.any(selected):
            dnew[mesh.elems[selected]] = 1.0

        self.advance_log.append({
            "front_id": int(front_id),
            "event_index": int(len(self.advance_log) + 1),
            "x0": float(p0[0]),
            "y0": float(p0[1]),
            "x1": float(p1[0]),
            "y1": float(p1[1]),
            "length_m": length,
            "angle_error_deg": 0.0,
            "damage": 1.0,
            "reason": "ok",
            "geometry_update": "sharp_wake_stiffness_kill",
            "n_killed_elements": int(np.count_nonzero(selected)),
        })
        return CrackAdvanceResult(
            mesh=mesh,
            boundary=boundary,
            damage=dnew,
            displacement=np.asarray(displacement, dtype=float),
            moved=length,
            inserted=True,
            selected_edge_length=length,
            reason="ok",
        )

    def write_diagnostics(self, out_dir: str) -> None:
        if not self.advance_log:
            return
        path = Path(out_dir) / "sharp_wake_advance_log.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.advance_log[0])
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.advance_log)


def build_crack_backend(args, geom) -> SharpWakeBackend:
    """Build the only production geometry backend supported by v10."""
    kind = str(getattr(args, "crack_backend", "sharp_wake") or "sharp_wake").lower()
    if kind not in {"sharp", "sharp_wake", "legacy"}:
        raise ValueError(
            f"v10 sharp-front repository does not contain cohesive backend {kind!r}; "
            "use --crack-backend sharp_wake"
        )
    return SharpWakeBackend()
