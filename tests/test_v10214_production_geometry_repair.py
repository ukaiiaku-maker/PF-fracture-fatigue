import json
from pathlib import Path

import numpy as np

from arrhenius_fracture.config import make_emergent_config
from arrhenius_fracture.crystal import bcc_slip_traces
from arrhenius_fracture.mesh import make_tri_mesh
from arrhenius_fracture.unit_slip_perturbation_v10212 import (
    element_residual_stiffness_fraction,
)
from arrhenius_fracture.physical_fem_station_responses_v10212 import (
    _active_ribbon_geometry,
)


def _production_e000_geometry():
    cfg = make_emergent_config()
    cfg.mesh.tip_h_fine = 2.0e-7
    cfg.mesh.tip_ratio = 1.15
    mesh = make_tri_mesh(cfg.geometry, cfg.mesh, seed=42)
    x = mesh.nodes[:, 0]
    y = mesh.nodes[:, 1]
    d = np.zeros(mesh.nn)
    d[(x <= cfg.geometry.a0) & (np.abs(y) <= cfg.geometry.notch_half_thickness)] = 1.0
    return cfg, mesh, d


def test_production_e000_mesh_and_damage_distribution():
    cfg, mesh, d = _production_e000_geometry()
    assert mesh.nn == 1667
    assert mesh.ne == 3212
    assert np.isclose(mesh.hbar_tip, 2.815211e-7, rtol=5.0e-6)
    de = np.mean(d[mesh.elems], axis=1)
    values, counts = np.unique(de, return_counts=True)
    observed = {round(float(v), 12): int(c) for v, c in zip(values, counts)}
    assert observed == {
        0.0: 2314,
        round(1.0 / 3.0, 12): 64,
        round(2.0 / 3.0, 12): 76,
        1.0: 758,
    }
    residual = element_residual_stiffness_fraction(mesh, d)
    transition = (de > 0.0) & (de < 1.0)
    assert np.all(residual[transition] > 0.1)
    assert np.all(residual[de == 1.0] < 1.0e-3)


def test_both_45_degree_crystal_active_ribbons_keep_surface_source():
    cfg, mesh, d = _production_e000_geometry()
    tip = np.array([cfg.geometry.a0, 0.0])
    forward = np.array([1.0, 0.0])
    traces = bcc_slip_traces(45.0)
    assert len(traces) == 2
    for system, trace in enumerate(traces):
        start, end, audit = _active_ribbon_geometry(
            system=system,
            x_m=2.0e-6,
            width_m=2.0 * mesh.hbar_tip,
            mesh=mesh,
            damage=d,
            tip=tip,
            forward=forward,
            slip_direction=np.asarray(trace["t"], dtype=float),
            minimum_residual_stiffness_fraction=1.0e-3,
            stiffness_kappa=1.0e-6,
        )
        assert np.allclose(start, tip)
        assert not np.allclose(end, start)
        assert audit["source_is_physical_crack_surface_tip"] is True
        assert audit["source_relocated_into_intact_material"] is False
        assert audit["terminal_residual_stiffness_fraction"] >= 1.0e-3


def test_snapshot_schema_rejects_wake_kernel_claim(tmp_path: Path):
    from arrhenius_fracture.physical_fem_snapshot_v10212 import SnapshotMetadata

    payload = dict(
        state_id="E000",
        r_eff_over_r0=1.0,
        opening_strength_fraction=0.1,
        crack_extension_m=0.0,
        temperature_K=700.0,
        Uy_top_m=1.0e-7,
        Uy_bot_m=-1.0e-7,
        crack_tip_xy_m=(5.0e-4, 0.0),
        crack_direction=(1.0, 0.0),
        interaction_ell_m=2.0e-6,
        exclude_radius_m=3.0e-7,
        active_x_m=(2.5e-7, 7.5e-7),
        wake_x_m=(2.5e-7,),
        channel_directions=((1.0, 0.0), (0.0, 1.0)),
        channel_normals=((0.0, 1.0), (-1.0, 0.0)),
        material={"E": 410e9, "nu": 0.28, "b": 2.74e-10, "Tm": 3695.0},
        engine_config={},
    )
    SnapshotMetadata(**payload).validate()
    payload["wake_kernel_supported"] = True
    try:
        SnapshotMetadata(**payload).validate()
    except ValueError as exc:
        assert "wake kernel" in str(exc)
    else:
        raise AssertionError("wake-kernel claim should be rejected")
