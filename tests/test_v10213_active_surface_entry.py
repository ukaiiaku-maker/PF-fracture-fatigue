from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.physical_fem_station_responses_v10212 import (
    _active_surface_entry,
    _ribbon_geometry,
)
from arrhenius_fracture.unit_slip_perturbation_v1026 import (
    SlipRibbonPerturbation,
    slip_ribbon_eigenstrain_increment,
)
from arrhenius_fracture.unit_slip_perturbation_v10212 import (
    _clip_source_surface_overlap,
)


def _line_mesh(x, *, hbar_tip=0.10):
    x = np.asarray(x, dtype=float)
    nodes = np.column_stack([x, np.zeros_like(x)])
    elems = np.asarray([[i, i, i] for i in range(x.size)], dtype=int)
    return SimpleNamespace(
        nodes=nodes,
        elems=elems,
        area_e=np.ones(x.size),
        nn=x.size,
        ne=x.size,
        hbar_tip=float(hbar_tip),
    )


def test_active_source_starts_after_contiguous_tip_damage_band():
    mesh = _line_mesh([0.20, 0.55, 0.90, 1.25, 1.60, 1.95, 2.40])
    damage = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
    start, audit = _active_surface_entry(
        mesh=mesh,
        damage=damage,
        tip=np.asarray([0.0, 0.0]),
        forward=np.asarray([1.0, 0.0]),
        ray=np.asarray([1.0, 0.0]),
        width_m=0.20,
    )
    assert start[0] == pytest.approx(1.95)
    assert audit["tip_connected_damage_extent_m"] > 1.60
    assert audit["tip_connected_damaged_centroid_count"] == 5
    assert audit["separated_damage_is_not_absorbed_into_source"] is True


def test_separated_damage_cluster_is_not_absorbed_into_source_entry():
    mesh = _line_mesh([0.20, 0.45, 0.80, 1.60, 1.90, 2.30])
    damage = np.asarray([1.0, 1.0, 0.0, 1.0, 0.0, 0.0])
    start, audit = _active_surface_entry(
        mesh=mesh,
        damage=damage,
        tip=np.asarray([0.0, 0.0]),
        forward=np.asarray([1.0, 0.0]),
        ray=np.asarray([1.0, 0.0]),
        width_m=0.20,
    )
    assert start[0] == pytest.approx(0.80)
    assert audit["tip_connected_damaged_centroid_count"] == 2
    assert audit["tip_connected_damage_extent_m"] < 1.0


def test_ribbon_geometry_uses_actual_entry_and_clipper_accepts_source_band_case():
    mesh = _line_mesh(
        [0.20, 0.55, 0.90, 1.25, 1.60, 1.95, 2.40, 2.80, 3.20],
        hbar_tip=0.10,
    )
    damage = np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    start, end, audit = _ribbon_geometry(
        region="active",
        system=0,
        x_m=0.0,
        width_m=0.20,
        mesh=mesh,
        damage=damage,
        tip=np.asarray([0.0, 0.0]),
        forward=np.asarray([1.0, 0.0]),
        slip_direction=np.asarray([1.0, 0.0]),
    )
    assert start[0] == pytest.approx(1.95)
    assert end[0] >= 2.80
    assert audit["nominal_tip_coordinate_used_as_ribbon_source"] is False

    perturbation = SlipRibbonPerturbation(
        system=0,
        region="active",
        bin_index=0,
        start_xy_m=start,
        end_xy_m=end,
        slip_direction=np.asarray([1.0, 0.0]),
        plane_normal=np.asarray([0.0, 1.0]),
        width_m=0.20,
        burgers_m=2.5e-10,
        signed_line_content=0.5,
    )
    increment, _ = slip_ribbon_eigenstrain_increment(mesh, perturbation)
    corrected, clipping = _clip_source_surface_overlap(
        mesh,
        damage,
        increment,
        perturbation,
        maximum_damaged_area_fraction=0.05,
    )
    assert np.any(np.abs(corrected) > 0.0)
    assert clipping["damaged_area_fraction"] == pytest.approx(0.0)
