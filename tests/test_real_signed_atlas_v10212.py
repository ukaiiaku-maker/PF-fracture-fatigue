import csv
import json
from pathlib import Path

import numpy as np

from arrhenius_fracture.mechanics_normalization_v10212 import (
    SourceGeometryAssumptions,
    derive_mechanical_normalization,
)
from arrhenius_fracture.mesh import BoundaryData, rebuild_tri_mesh
from arrhenius_fracture.physical_fem_capture_v10212 import load_capture_requests
from arrhenius_fracture.physical_fem_snapshot_v10212 import (
    SnapshotMetadata,
    load_snapshot,
    save_snapshot,
)


def test_mechanical_normalization_uses_packet_over_b_and_geometry_bounds():
    payload = {
        "front_config": {"L_pz": 1.0e-6},
        "mpz_config": {"n_systems": 2},
        "tip_config": {"packet_length_m": 2.5e-10},
        "b_m": 2.5e-10,
        "material_manifest": {"source_sites_per_system": 9000},
    }
    result = derive_mechanical_normalization(
        payload,
        assumptions=SourceGeometryAssumptions(
            minimum_spacing_b=10.0,
            maximum_spacing_b=100.0,
        ),
    )
    assert result["activation_to_line_content_by_system"] == [1.0, 1.0]
    assert result["source_capacity_bounds_per_system"] == [
        [40.0, 400.0],
        [40.0, 400.0],
    ]
    assert result["historical_source_sites_outside_mechanical_bounds"] is True
    assert result["out_of_plane_thickness_multiplier_applied"] is False
    assert result["fitted_to_toughness_or_fatigue"] is False


def _snapshot_metadata():
    return SnapshotMetadata(
        state_id="S000",
        r_eff_over_r0=1.25,
        opening_strength_fraction=0.5,
        crack_extension_m=0.0,
        temperature_K=700.0,
        Uy_top_m=1.0e-6,
        Uy_bot_m=-1.0e-6,
        crack_tip_xy_m=(0.5, 0.5),
        crack_direction=(1.0, 0.0),
        interaction_ell_m=0.1,
        exclude_radius_m=0.0,
        active_x_m=(0.05, 0.10),
        wake_x_m=(0.05,),
        channel_directions=((1.0, 0.0), (0.0, 1.0)),
        channel_normals=((0.0, 1.0), (-1.0, 0.0)),
        material={"E": 200.0e9, "nu": 0.3, "b": 2.5e-10, "Tm": 1800.0},
        engine_config={"front_config": {"L_pz": 1.0e-6}},
    )


def test_physical_snapshot_round_trip_preserves_fixed_crack_state(tmp_path):
    nodes = np.asarray([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    elems = np.asarray([[0, 1, 2], [0, 2, 3]])
    mesh = rebuild_tri_mesh(nodes, elems, tip_centers=np.asarray([0.5, 0.5]))
    boundary = BoundaryData(
        top_nodes=np.asarray([2, 3]),
        bot_nodes=np.asarray([0, 1]),
        left_bot=0,
        right_bot=1,
        notch_nodes=np.asarray([0]),
    )
    root = tmp_path / "snapshot"
    save_snapshot(
        root,
        metadata=_snapshot_metadata(),
        mesh=mesh,
        boundary=boundary,
        u=np.zeros(mesh.ndof),
        ep_gp=np.zeros((3, mesh.ne)),
        rho_gp=np.ones(mesh.ne),
        d=np.zeros(mesh.nn),
        D=np.eye(3),
    )
    loaded = load_snapshot(root)
    assert loaded["metadata"].state_id == "S000"
    assert loaded["metadata"].fem_tip_geometry_blunted is False
    assert loaded["metadata"].r_eff_is_analytical_tip_state is True
    assert np.array_equal(loaded["mesh"].elems, elems)
    assert np.array_equal(loaded["boundary"].top_nodes, np.asarray([2, 3]))
    assert loaded["D"].shape == (3, 3)


def test_capture_request_table_requires_explicit_tolerances(tmp_path):
    path = tmp_path / "states.csv"
    fields = [
        "state_id",
        "temperature_K",
        "r_eff_over_r0",
        "opening_strength_fraction",
        "crack_extension_m",
        "r_tolerance",
        "opening_tolerance",
        "extension_tolerance_m",
        "interaction_ell_m",
    ]
    row = {
        "state_id": "S0",
        "temperature_K": 700,
        "r_eff_over_r0": 1.0,
        "opening_strength_fraction": 0.5,
        "crack_extension_m": 0.0,
        "r_tolerance": 0.05,
        "opening_tolerance": 0.05,
        "extension_tolerance_m": 1.0e-6,
        "interaction_ell_m": 2.0e-6,
    }
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    requests = load_capture_requests(path)
    assert len(requests) == 1
    assert requests[0].interaction_ell_m == 2.0e-6
