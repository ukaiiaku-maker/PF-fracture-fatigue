from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.physical_fem_station_responses_v10212 import (
    _ribbon_geometry,
)
from arrhenius_fracture.unit_slip_perturbation_v10212 import (
    SlipRibbonPerturbation,
    _clip_source_surface_overlap,
)


def _mesh():
    x = np.asarray([0.10, 0.30, 0.60, 0.90], dtype=float)
    nodes = np.column_stack([x, np.zeros_like(x)])
    elems = np.asarray([[i, i, i] for i in range(x.size)], dtype=int)
    return SimpleNamespace(
        nodes=nodes,
        elems=elems,
        area_e=np.ones(x.size),
        nn=x.size,
        ne=x.size,
        hbar_tip=0.05,
    )


def _perturbation():
    return SlipRibbonPerturbation(
        system=0,
        region="active",
        bin_index=0,
        start_xy_m=np.asarray([0.0, 0.0]),
        end_xy_m=np.asarray([1.0, 0.0]),
        slip_direction=np.asarray([1.0, 0.0]),
        plane_normal=np.asarray([0.0, 1.0]),
        width_m=0.20,
        burgers_m=2.5e-10,
        signed_line_content=0.5,
    )


def _increment():
    return np.ones((3, 4), dtype=float)


def test_source_side_killed_elements_are_clipped_not_rejected():
    corrected, audit = _clip_source_surface_overlap(
        _mesh(),
        np.asarray([1.0, 1.0, 0.0, 0.0]),
        _increment(),
        _perturbation(),
        maximum_damaged_area_fraction=0.05,
    )
    assert np.allclose(corrected[:, :2], 0.0)
    assert np.allclose(corrected[:, 2:], 1.0)
    assert audit["source_surface_clipping_applied"] is True
    assert audit["source_surface_clipped_elements"] == 2
    assert audit["source_surface_clipped_area_fraction"] == pytest.approx(0.5)
    assert audit["terminal_in_intact_material"] is True


def test_short_ribbon_clips_source_terminal_window_overlap_before_guarding_terminal():
    x = np.asarray([0.10, 0.30, 0.38], dtype=float)
    mesh = SimpleNamespace(
        nodes=np.column_stack([x, np.zeros_like(x)]),
        elems=np.asarray([[i, i, i] for i in range(x.size)], dtype=int),
        area_e=np.ones(x.size),
        nn=x.size,
        ne=x.size,
        hbar_tip=0.05,
    )
    perturbation = SlipRibbonPerturbation(
        system=0,
        region="active",
        bin_index=0,
        start_xy_m=np.asarray([0.0, 0.0]),
        end_xy_m=np.asarray([0.40, 0.0]),
        slip_direction=np.asarray([1.0, 0.0]),
        plane_normal=np.asarray([0.0, 1.0]),
        width_m=0.20,
        burgers_m=2.5e-10,
        signed_line_content=0.5,
    )
    corrected, audit = _clip_source_surface_overlap(
        mesh,
        np.asarray([1.0, 1.0, 0.0]),
        np.ones((3, 3), dtype=float),
        perturbation,
        maximum_damaged_area_fraction=0.05,
    )
    assert np.allclose(corrected[:, :2], 0.0)
    assert np.allclose(corrected[:, 2], 1.0)
    assert audit["terminal_in_intact_material"] is True


def test_first_active_station_is_extended_to_fem_resolved_terminal_distance():
    mesh = _mesh()
    start, end, audit = _ribbon_geometry(
        region="active",
        system=0,
        x_m=0.0,
        width_m=0.20,
        mesh=mesh,
        damage=np.zeros(mesh.nn),
        tip=np.asarray([0.0, 0.0]),
        forward=np.asarray([1.0, 0.0]),
        slip_direction=np.asarray([1.0, 0.0]),
    )
    assert np.linalg.norm(end - start) >= 0.75
    assert audit["fem_resolution_extension_applied"] is True
    assert audit["minimum_fem_resolved_ribbon_length_m"] == pytest.approx(0.8)


def test_interior_crack_crossing_remains_a_hard_error():
    with pytest.raises(ValueError, match="crosses stiffness-killed"):
        _clip_source_surface_overlap(
            _mesh(),
            np.asarray([0.0, 0.0, 1.0, 0.0]),
            _increment(),
            _perturbation(),
            maximum_damaged_area_fraction=0.05,
        )


def test_terminal_dislocation_in_killed_material_remains_a_hard_error():
    with pytest.raises(ValueError, match="terminal lies"):
        _clip_source_surface_overlap(
            _mesh(),
            np.asarray([0.0, 0.0, 0.0, 1.0]),
            _increment(),
            _perturbation(),
            maximum_damaged_area_fraction=0.05,
        )
