"""Robust initial-front direction selection for the v10.2.27 campaign.

The inherited v10.1.7.4 observer estimates the local front tangent from the
principal axis of nearby damaged elements.  At the unadvanced starter notch,
the damaged-tip cross-section can be longer transverse to the notch than along
it, causing the PCA axis to rotate by about 90 degrees.  The damaged-wake
centroid still lies behind the tip and therefore supplies the correct forward
orientation.

This overlay retains the inherited PCA direction when it agrees with the
wake-centroid direction.  If the two are strongly misaligned, it uses the
normalized wake-to-tip vector.  No stress, damage, crack-growth, or kinetic
parameter is changed.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import anisotropic_emission_v10174 as _base

MODEL_ID = "v10.2.27_wake_aligned_initial_front_direction"
DEFAULT_MINIMUM_ALIGNMENT_COSINE = math.cos(math.radians(45.0))

_ORIGINAL_INFER = _base.infer_front_direction
_ORIGINAL_BUILD = _base.build_front_drive
_INSTALLED = False
_LAST_SELECTION: dict[str, Any] = {}


def _unit(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(2)
    norm = float(np.linalg.norm(value))
    if not math.isfinite(norm) or norm <= 1.0e-30:
        return np.array([1.0, 0.0], dtype=float)
    return value / norm


def infer_front_direction(
    mesh,
    damage,
    tip_xy,
    radius_m: float,
    *,
    minimum_alignment_cosine: float = DEFAULT_MINIMUM_ALIGNMENT_COSINE,
) -> np.ndarray:
    """Return a PCA direction only when it agrees with the damaged-wake vector."""
    global _LAST_SELECTION

    tip = np.asarray(tip_xy, dtype=float).reshape(2)
    inherited = _unit(_ORIGINAL_INFER(mesh, damage, tip_xy, radius_m))
    centroids = np.asarray(mesh.nodes, dtype=float)[
        np.asarray(mesh.elems, dtype=int)
    ].mean(axis=1)
    damage_values = np.asarray(damage, dtype=float).reshape(-1)
    if damage_values.size == int(mesh.nn):
        element_damage = np.mean(
            damage_values[np.asarray(mesh.elems, dtype=int)], axis=1
        )
    elif damage_values.size == int(mesh.ne):
        element_damage = damage_values.copy()
    else:
        raise ValueError(
            f"damage field has {damage_values.size} entries; expected "
            f"mesh.nn={mesh.nn} or mesh.ne={mesh.ne}"
        )

    distance = np.linalg.norm(centroids - tip[None, :], axis=1)
    selected = (element_damage >= 0.5) & (
        distance <= max(4.0 * float(radius_m), 1.0e-12)
    )
    points = centroids[selected]
    weights = np.maximum(element_damage[selected], 1.0e-12)

    if points.shape[0] < 2:
        chosen = inherited
        method = "inherited_insufficient_wake_points"
        wake_direction = None
        alignment = None
    else:
        mean = np.average(points, axis=0, weights=weights)
        wake_to_tip = tip - mean
        wake_norm = float(np.linalg.norm(wake_to_tip))
        if not math.isfinite(wake_norm) or wake_norm <= 1.0e-30:
            chosen = inherited
            method = "inherited_degenerate_wake_centroid"
            wake_direction = None
            alignment = None
        else:
            wake_direction_array = _unit(wake_to_tip)
            if wake_direction_array[0] < 0.0:
                wake_direction_array = -wake_direction_array
            if float(inherited @ wake_direction_array) < 0.0:
                inherited = -inherited
            alignment_value = float(
                np.clip(inherited @ wake_direction_array, -1.0, 1.0)
            )
            threshold = float(minimum_alignment_cosine)
            if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
                raise ValueError("minimum alignment cosine must lie in [0,1]")
            if alignment_value < threshold:
                chosen = wake_direction_array
                method = "wake_to_tip_replaces_misaligned_pca"
            else:
                chosen = inherited
                method = "inherited_pca_wake_aligned"
            wake_direction = wake_direction_array.tolist()
            alignment = alignment_value

    chosen = _unit(chosen)
    if chosen[0] < 0.0:
        chosen = -chosen
    _LAST_SELECTION = {
        "model_id": MODEL_ID,
        "method": method,
        "minimum_alignment_cosine": float(minimum_alignment_cosine),
        "inherited_direction": inherited.tolist(),
        "wake_to_tip_direction": wake_direction,
        "inherited_wake_alignment_cosine": alignment,
        "selected_direction": chosen.tolist(),
        "selected_damaged_elements": int(points.shape[0]),
    }
    return chosen


def build_front_drive(*args, **kwargs):
    drive = _ORIGINAL_BUILD(*args, **kwargs)
    drive["front_direction_selection"] = dict(_LAST_SELECTION)
    return drive


def install_front_direction_fix() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    _base.infer_front_direction = infer_front_direction
    _base.build_front_drive = build_front_drive
    _INSTALLED = True


def audit_payload() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "installed": bool(_INSTALLED),
        "minimum_alignment_cosine": DEFAULT_MINIMUM_ALIGNMENT_COSINE,
        "last_selection": dict(_LAST_SELECTION),
    }


__all__ = [
    "MODEL_ID",
    "DEFAULT_MINIMUM_ALIGNMENT_COSINE",
    "infer_front_direction",
    "install_front_direction_fix",
    "audit_payload",
]
