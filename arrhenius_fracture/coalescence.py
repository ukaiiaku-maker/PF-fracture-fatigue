"""Geometry-only crack-path coalescence utilities.

These routines do not alter driving forces or introduce attraction. They only
identify when a proposed sharp-front increment intersects an existing crack
polyline, permitting the caller to clip the increment and retire the impinging
tip.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np


def segment_intersection_first(
    p0: np.ndarray,
    p1: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    *,
    t_eps: float = 1.0e-8,
) -> Optional[tuple[float, np.ndarray]]:
    p0 = np.asarray(p0, float)
    p1 = np.asarray(p1, float)
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    r = p1 - p0
    s = b - a
    rr = float(r @ r)
    if rr <= 1.0e-30:
        return None

    cross_rs = float(r[0] * s[1] - r[1] * s[0])
    qp = a - p0
    scale = max(float(np.linalg.norm(r)), float(np.linalg.norm(s)), 1.0e-12)
    tol = 1.0e-10 * scale * scale

    if abs(cross_rs) > tol:
        t = float((qp[0] * s[1] - qp[1] * s[0]) / cross_rs)
        u = float((qp[0] * r[1] - qp[1] * r[0]) / cross_rs)
        if t > t_eps and t <= 1.0 + 1.0e-10 and u >= -1.0e-10 and u <= 1.0 + 1.0e-10:
            tt = min(max(t, 0.0), 1.0)
            return tt, p0 + tt * r
        return None

    cross_qr = float(qp[0] * r[1] - qp[1] * r[0])
    if abs(cross_qr) > tol:
        return None
    ta = float(((a - p0) @ r) / rr)
    tb = float(((b - p0) @ r) / rr)
    candidates = [value for value in (ta, tb) if value > t_eps and value <= 1.0 + 1.0e-10]
    if not candidates:
        return None
    tt = min(max(min(candidates), 0.0), 1.0)
    return tt, p0 + tt * r


def first_path_intersection(
    fronts: Iterable[dict],
    advancing_front: dict,
    p0: np.ndarray,
    p1: np.ndarray,
) -> Optional[dict]:
    best = None
    front_id = int(advancing_front.get("id", -1))
    for other in fronts:
        path = other.get("path", [])
        if len(path) < 2:
            continue
        for segment_index in range(len(path) - 1):
            if int(other.get("id", -2)) == front_id and segment_index == len(path) - 2:
                continue
            hit = segment_intersection_first(
                p0, p1, path[segment_index], path[segment_index + 1]
            )
            if hit is None:
                continue
            t, xy = hit
            if best is None or t < best["t"]:
                best = {
                    "t": float(t),
                    "xy": np.asarray(xy, float),
                    "target_front_id": int(other.get("id", -1)),
                    "target_segment_index": int(segment_index),
                }
    return best
