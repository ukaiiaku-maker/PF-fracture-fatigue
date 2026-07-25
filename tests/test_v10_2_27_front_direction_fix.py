from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arrhenius_fracture import anisotropic_emission_v10174 as inherited
from arrhenius_fracture.anisotropic_front_direction_fix_v10227 import (
    DEFAULT_MINIMUM_ALIGNMENT_COSINE,
    infer_front_direction,
)


@dataclass
class _Mesh:
    nodes: np.ndarray
    elems: np.ndarray
    hbar_tip: float = 1.0e-7

    @property
    def nn(self) -> int:
        return int(self.nodes.shape[0])

    @property
    def ne(self) -> int:
        return int(self.elems.shape[0])


def _mesh_from_centroids(points: list[tuple[float, float]]) -> _Mesh:
    nodes = []
    elems = []
    scale = 1.0e-3
    for x, y in points:
        first = len(nodes)
        nodes.extend(
            [
                (x - scale, y - scale),
                (x + scale, y - scale),
                (x, y + 2.0 * scale),
            ]
        )
        elems.append((first, first + 1, first + 2))
    return _Mesh(np.asarray(nodes, dtype=float), np.asarray(elems, dtype=int))


def test_transverse_pca_is_replaced_by_wake_to_tip_direction():
    mesh = _mesh_from_centroids(
        [(0.9, -0.4), (0.9, -0.2), (0.9, 0.0), (0.9, 0.2), (0.9, 0.4)]
    )
    damage = np.ones(mesh.ne)
    tip = np.array([1.0, 0.0])

    inherited_direction = inherited.infer_front_direction(mesh, damage, tip, 1.0)
    fixed_direction = infer_front_direction(mesh, damage, tip, 1.0)

    assert abs(float(inherited_direction @ np.array([1.0, 0.0]))) < 0.1
    assert float(fixed_direction @ np.array([1.0, 0.0])) > 0.999


def test_wake_aligned_pca_is_preserved():
    mesh = _mesh_from_centroids(
        [(0.2, 0.0), (0.4, 0.01), (0.6, -0.01), (0.8, 0.0), (0.9, 0.0)]
    )
    damage = np.ones(mesh.ne)
    tip = np.array([1.0, 0.0])

    fixed_direction = infer_front_direction(mesh, damage, tip, 1.0)

    assert float(fixed_direction @ np.array([1.0, 0.0])) > 0.999
    assert 0.0 < DEFAULT_MINIMUM_ALIGNMENT_COSINE < 1.0


def test_campaign_and_capture_entries_install_fix():
    root = Path(__file__).resolve().parents[1]
    campaign = (
        root / "arrhenius_fracture" / "sharp_front_v10_2_27_audited.py"
    ).read_text()
    capture = (
        root / "arrhenius_fracture" / "sharp_front_v10_2_13_capture.py"
    ).read_text()

    assert "install_front_direction_fix()" in campaign
    assert "install_front_direction_fix()" in capture
