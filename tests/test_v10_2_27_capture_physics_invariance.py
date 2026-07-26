from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.config import GeometryConfig
from arrhenius_fracture.frozen_measurement_reconstruction_v10227 import (
    FrozenMeasurementMeshConfig,
    _reapply_frozen_sharp_crack,
    _resolve_crack_path,
)
from arrhenius_fracture.kernel_configuration_v10227 import (
    MechanicalKernelConfiguration,
)

ROOT = Path(__file__).resolve().parents[1]


def test_capture_reconstruction_has_no_front_or_mpz_kinetic_entry_points():
    text = (
        ROOT
        / "arrhenius_fracture"
        / "frozen_measurement_reconstruction_v10227.py"
    ).read_text()
    assert "KineticMovingTipFrontEngine" not in text
    assert "fractional_moving_frame_advance" not in text
    assert ".mpz.advance(" not in text
    assert ".mpz.evolve(" not in text
    assert "engine.step(" not in text
    assert "update_plasticity(" not in text
    assert "assemble_mechanics(" in text
    assert "solve_dirichlet(" in text


def test_capture_entry_runs_the_audited_v10227_production_stack():
    text = (
        ROOT / "arrhenius_fracture" / "sharp_front_v10_2_13_capture.py"
    ).read_text()
    assert "AuditedPersistentSiteStateResolvedTipEngine" in text
    assert "install_backstress_complementarity_fix" in text
    assert "install_physical_front_width" in text
    assert "_run_current_paper_stack" in text
    assert "_paper._base.main(args)" in text
    assert "sharp_front_v10_1_7_5 as _transport" not in text
    assert "_transport.main(args)" not in text


def test_capture_entry_does_not_rewrite_production_physics_flags():
    text = (
        ROOT / "arrhenius_fracture" / "sharp_front_v10_2_13_capture.py"
    ).read_text()
    assert "_force_capture_modes" not in text
    assert 'args.extend(["--no-active-shielding"' not in text
    assert "physics_overrides=none" in text
    assert "production_parameterization_observed_not_modified" in text
    assert "_validate_single_front_capture" in text


def test_capture_hook_enforces_bitwise_production_state_invariance():
    text = (
        ROOT / "arrhenius_fracture" / "physical_fem_capture_v10212.py"
    ).read_text()
    assert "_engine_kinetic_state_digest" in text
    assert "kinetic_digest_before" in text
    assert "kinetic_digest_after" in text
    assert "capture-only endpoint reconstruction mutated" in text
    assert '"production_engine_state_bitwise_unchanged": True' in text
    assert '"measurement_reconstruction_called_mpz_advance": False' in text


def test_existing_moving_frame_and_mobile_kinetic_solver_remain_authoritative():
    moving = (
        ROOT / "arrhenius_fracture" / "fractional_moving_frame.py"
    ).read_text()
    kinetic = (
        ROOT / "arrhenius_fracture" / "kinetic_tip_cell.py"
    ).read_text()
    assert "def fractional_moving_frame_advance" in moving
    assert "self.mobile, crossed_m" in moving
    assert "self.retained, crossed_r" in moving
    assert "self.accumulated_slip, crossed_s" in moving
    assert "self.available_sites += refreshed" in moving
    assert "first = self._plastic_half_step(0.5 * h" in kinetic
    assert "advance = self.mpz.advance(da)" in kinetic
    assert "second = self._plastic_half_step(0.5 * h" in kinetic


def test_measurement_mesh_configuration_is_independent_of_production_spacing():
    configuration = MechanicalKernelConfiguration()
    measurement = FrozenMeasurementMeshConfig(
        specimen_length_x_m=configuration.specimen_length_x_m,
        specimen_length_y_m=configuration.specimen_length_y_m,
        initial_crack_length_m=configuration.initial_crack_length_m,
        notch_half_thickness_m=configuration.notch_half_thickness_m,
        mesh_nx=configuration.mesh_nx,
        mesh_ny=configuration.mesh_ny,
        tip_h_fine_m=configuration.measurement_tip_h_fine_m,
        tip_ratio=configuration.measurement_tip_ratio,
    ).validate()
    assert configuration.tip_h_fine_m == 1.0e-6
    assert measurement.tip_h_fine_m == configuration.measurement_tip_h_fine_m
    assert measurement.tip_h_fine_m < configuration.tip_h_fine_m
    assert measurement.kill_radius_floor_m == 0.0


def test_missing_polyline_is_synthesized_only_for_verified_straight_growth():
    geometry = GeometryConfig(Lx=2.0, Ly=2.0, a0=0.5, notch_half_thickness=0.08)
    direction = np.array([np.cos(np.pi / 6.0), np.sin(np.pi / 6.0)])
    tip = np.array([geometry.a0, 0.0]) + 0.2 * direction
    path, audit = _resolve_crack_path(
        (),
        geometry=geometry,
        crack_tip_xy_m=tip,
        crack_extension_m=0.2,
        trajectory_hbar_tip_m=1.0e-3,
    )
    assert len(path) == 2
    assert np.allclose(path[0], [geometry.a0, 0.0])
    assert np.allclose(path[-1], tip)
    assert audit["straight_single_front_path_synthesized"] is True
    assert audit["crack_path_source"] == (
        "verified_straight_single_front_tip_displacement"
    )

    with pytest.raises(RuntimeError, match="not provably straight"):
        _resolve_crack_path(
            (),
            geometry=geometry,
            crack_tip_xy_m=tip,
            crack_extension_m=0.25,
            trajectory_hbar_tip_m=1.0e-3,
        )


def test_reconstructed_crack_trace_never_kills_elements_ahead_of_tip():
    nodes = np.array(
        [
            [0.80, -0.01],
            [0.80, 0.01],
            [0.90, 0.00],
            [1.10, -0.01],
            [1.10, 0.01],
            [1.20, 0.00],
        ]
    )
    mesh = SimpleNamespace(
        nodes=nodes,
        elems=np.array([[0, 1, 2], [3, 4, 5]], dtype=int),
        area_e=np.array([1.0e-4, 1.0e-4]),
        hbar_tip=0.05,
        ne=2,
        nn=6,
    )
    geometry = GeometryConfig(Lx=2.0, Ly=2.0, a0=0.2, notch_half_thickness=0.01)
    path = (np.array([0.2, 0.0]), np.array([1.0, 0.0]))
    damage, audit = _reapply_frozen_sharp_crack(
        mesh,
        geometry=geometry,
        crack_tip_xy_m=np.array([1.0, 0.0]),
        crack_path_xy_m=path,
        kill_radius_floor_m=0.0,
    )
    assert np.all(damage[:3] == 1.0)
    assert np.all(damage[3:] == 0.0)
    assert audit["endpoint_caps_excluded"] is True
    assert audit["ahead_of_tip_killed_elements"] == 0
