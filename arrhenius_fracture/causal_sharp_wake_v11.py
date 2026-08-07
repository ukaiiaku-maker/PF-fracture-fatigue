"""Causal element-local sharp-wake discretization for v11 branching.

The physical crack graph is authoritative.  This module maps a committed graph
edge to P0 element damage without a radius, centroid band, or nodal-neighbour
dilation.  The advancing endpoint is half open: an element touched only at
``p1`` is left intact for the next event.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib

import numpy as np


CRACK_REPRESENTATION = "sharp_wake_causal_v11"


@dataclass(frozen=True)
class CausalSupportAudit:
    candidate_length_m: float
    newly_degraded_element_count: int
    newly_degraded_element_area_m2: float
    geometric_intersection_length_represented_m: float
    accepted_mechanical_fingerprint: str
    trial_mechanical_fingerprint: str
    selected_element_ids: tuple[int, ...]

    @property
    def mechanically_resolved(self) -> bool:
        return (
            self.newly_degraded_element_count > 0
            and self.trial_mechanical_fingerprint
            != self.accepted_mechanical_fingerprint
        )


def element_damage(mesh, nodal_damage: np.ndarray) -> np.ndarray:
    inherited = getattr(mesh, "element_damage_gp", None)
    if inherited is not None:
        return np.asarray(inherited, dtype=float).copy()
    return np.mean(np.asarray(nodal_damage, dtype=float)[mesh.elems], axis=1)


def mechanical_fingerprint(mesh, damage_gp: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(mesh.nodes, dtype=np.float64).tobytes())
    digest.update(np.ascontiguousarray(mesh.elems, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(damage_gp, dtype=np.float64).tobytes())
    return digest.hexdigest()


def _segment_triangle_parameter_interval(
    p0: np.ndarray, p1: np.ndarray, triangle: np.ndarray, *, tolerance: float,
) -> tuple[float, float] | None:
    """Return the segment parameter interval in a closed triangle.

    Barycentric coordinates are affine along the segment.  Clipping all three
    coordinates against ``lambda >= -tolerance`` is deterministic and scale
    independent.  A positive interval is required, so vertex-only contact is
    never support.  Mesh-edge coincidence selects the adjacent elements on both
    sides; this is the only deterministic P0 convention that is independent of
    element enumeration and it converges to zero width with nested refinement.
    """
    matrix = np.column_stack((triangle[1] - triangle[0], triangle[2] - triangle[0]))
    determinant = float(np.linalg.det(matrix))
    scale = max(float(np.linalg.norm(matrix, ord=np.inf)), np.finfo(float).tiny)
    if abs(determinant) <= 32.0 * np.finfo(float).eps * scale * scale:
        raise ValueError("degenerate triangle in causal sharp-wake support")
    inv = np.linalg.inv(matrix)

    def bary(point: np.ndarray) -> np.ndarray:
        uv = inv @ (point - triangle[0])
        return np.array((1.0 - uv[0] - uv[1], uv[0], uv[1]), dtype=float)

    a = bary(p0)
    b = bary(p1) - a
    lower, upper = 0.0, 1.0
    for intercept, slope in zip(a, b):
        if abs(float(slope)) <= np.finfo(float).eps:
            if float(intercept) < -tolerance:
                return None
            continue
        crossing = (-tolerance - float(intercept)) / float(slope)
        if slope > 0.0:
            lower = max(lower, crossing)
        else:
            upper = min(upper, crossing)
        if upper <= lower:
            return None
    lower = max(lower, 0.0)
    upper = min(upper, 1.0)
    return (lower, upper) if upper - lower > tolerance else None


def causal_segment_support(
    mesh, p0: np.ndarray, p1: np.ndarray, *, tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:
    """Select P0 elements intersected over positive length by ``[p0, p1)``."""
    start = np.asarray(p0, dtype=float)
    end = np.asarray(p1, dtype=float)
    length = float(np.linalg.norm(end - start))
    if not np.isfinite(length) or length <= 0.0:
        raise ValueError("causal crack segment must have positive finite length")
    selected: list[int] = []
    represented: list[float] = []
    endpoint_cut = max(0.0, 1.0 - float(tolerance))
    for element_id, connectivity in enumerate(np.asarray(mesh.elems, dtype=int)):
        interval = _segment_triangle_parameter_interval(
            start, end, np.asarray(mesh.nodes, dtype=float)[connectivity],
            tolerance=float(tolerance),
        )
        if interval is None:
            continue
        lo, hi = interval
        hi = min(hi, endpoint_cut)
        if hi - lo <= float(tolerance):
            continue
        selected.append(int(element_id))
        represented.append(length * (hi - lo))
    return np.asarray(selected, dtype=int), np.asarray(represented, dtype=float)


def apply_causal_segment(
    state, p0: np.ndarray, p1: np.ndarray, *, tolerance: float = 1.0e-12,
):
    """Return an isolated state with only intersected P0 elements degraded."""
    before = element_damage(state.mesh, state.damage)
    selected, represented = causal_segment_support(
        state.mesh, p0, p1, tolerance=tolerance,
    )
    after = before.copy()
    if selected.size:
        after[selected] = 1.0
    newly = selected[before[selected] < 1.0] if selected.size else selected
    new_mask = np.isin(selected, newly)
    # Nodal damage is a rendering/diagnostic projection only.  The FEM stiffness
    # law reads the P0 field installed on the mesh below.
    visual = np.asarray(state.damage, dtype=float).copy()
    if selected.size:
        visual[np.unique(state.mesh.elems[selected])] = 1.0
    mesh = replace(state.mesh, element_damage_gp=after)
    audit = CausalSupportAudit(
        candidate_length_m=float(np.linalg.norm(np.asarray(p1) - np.asarray(p0))),
        newly_degraded_element_count=int(newly.size),
        newly_degraded_element_area_m2=float(np.sum(mesh.area_e[newly])),
        geometric_intersection_length_represented_m=float(np.sum(represented[new_mask])),
        accepted_mechanical_fingerprint=mechanical_fingerprint(state.mesh, before),
        trial_mechanical_fingerprint=mechanical_fingerprint(mesh, after),
        selected_element_ids=tuple(int(value) for value in selected),
    )
    return replace(state, mesh=mesh, damage=visual), audit


__all__ = [
    "CRACK_REPRESENTATION", "CausalSupportAudit", "apply_causal_segment",
    "causal_segment_support", "element_damage", "mechanical_fingerprint",
]
