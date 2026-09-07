import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

from arrhenius_fracture.analytical_monotonic_fracture_v10230 import (
    _fast_affine_compartment_advance,
)
from arrhenius_fracture.analytical_stationary_fatigue_v10230 import (
    AnalyticalControls,
    solve_hierarchy,
)
from scripts.analyze_v10_2_30_joint_fracture_fatigue_atlas import (
    load_candidate_rows,
    material_from_row,
)
from scripts.run_v10_2_30_canonical_temperature_fatigue_anchors import (
    build_jobs,
    prephysics_infrastructure_failure,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mechanical_transfer_temperature_anchors_v1"


def test_archive_affine_eigen_update_matches_dense_exponential():
    cases = [
        (
            np.array([[-3.0, 1.0], [2.0, -4.0]]),
            np.array([5.0, 0.0]), np.array([1.0, 2.0]), 0.3,
        ),
        (
            np.array([[-3.0, 1.0], [3.0, -1.0]]),
            np.array([8.0e17, 0.0]), np.array([1.0, 2.0]), 4.2,
        ),
    ]
    for matrix, source, initial, dt in cases:
        augmented = np.zeros((3, 3))
        augmented[:2, :2] = matrix
        augmented[:2, 2] = source
        expected = (expm(augmented * dt) @ np.r_[initial, 1.0])[:2]
        actual = _fast_affine_compartment_advance(
            matrix, source, initial, dt
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)
        assert np.all(actual >= 0.0)


def test_stationary_nonconvergence_returns_nan_residual_not_exception():
    row = load_candidate_rows().query("registry_role == 'peak_primary'").iloc[0]
    manifest = material_from_row(row)
    result = solve_hierarchy(
        manifest, row, 2.0, 0.1, 300.0, 1000.0,
        AnalyticalControls(max_iterations=0, n_phase=32),
    )
    assert not result["fixed_point_converged"]
    assert math.isnan(result["fixed_point_residual"])


def test_generated_transfer_temperature_bundle_passes_strict_verifier():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_v10_2_30_mechanical_transfer_temperature_anchors.py")],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_temperature_preflight_is_physics_gated_not_stationary_closure_promoted():
    data = __import__("pandas").read_csv(OUT / "bounded_temperature_fatigue_anchor_preflight.csv")
    assert set(data.n_bins) == {80}
    assert not data.fixed_point_converged.any()
    assert set(data.stationary_state_reduction_status) == {
        "UNAVAILABLE_NONCONVERGED_NOT_A_PHYSICAL_REJECTION"
    }
    assert data.asymptotic_gate_passed.sum() == 54
    assert data.accessibility_gate_passed.sum() == 40
    assert not data.current_production_row_complete.any()
    assert data.current_persistent_site_density_m2.isna().all()
    assert not data.preflight_launch_eligible.any()


def test_exact_canonical_rows_supply_the_production_eligible_temperature_matrix():
    data = __import__("pandas").read_csv(OUT / "canonical_temperature_fatigue_anchor_preflight.csv")
    assert len(data) == 36
    assert data.registry_role.nunique() == 4
    assert set(data.n_bins) == {80}
    assert data.current_production_row_complete.all()
    assert data.asymptotic_gate_passed.sum() == 36
    assert data.accessibility_gate_passed.sum() == 25
    assert data.preflight_launch_eligible.sum() == 36
    assert set(data[data.preflight_launch_eligible].production_row_gap) == {"NONE"}


def test_canonical_temperature_jobs_are_fresh_n80_and_use_Kmax_to_deltaK_mapping():
    jobs = build_jobs("test-head")
    assert len(jobs) == 36
    assert len({job["job_id"] for job in jobs}) == 36
    assert {job["seed"] for job in jobs} == {1720}
    assert {job["n_bins"] for job in jobs} == {80}
    assert {job["R"] for job in jobs} == {0.1}
    assert {job["frequency_Hz"] for job in jobs} == {1000.0}
    assert {job["target_extension_um"] for job in jobs} == {100.0}
    assert {job["maximum_cycles"] for job in jobs} == {1.0e12}
    assert all(job["fresh_virgin_required"] and not job["resume_permitted"] for job in jobs)
    assert all(
        math.isclose(job["DeltaK_MPa_sqrt_m"], 0.9 * job["Kmax_MPa_sqrt_m"])
        for job in jobs
    )
    assert not any("dbtt_intrinsic_control" in job["job_id"] for job in jobs)


def test_empty_prephysics_dev_fd_failure_is_not_a_numerical_trajectory(tmp_path):
    (tmp_path / "run.log").write_text("")
    (tmp_path / "high_cycle_run_manifest.json").write_text("{}")
    (tmp_path / "high_cycle_summary.json").write_text("{}")
    (tmp_path / "exit_code.txt").write_text("1\n")
    assert prephysics_infrastructure_failure(tmp_path)
    (tmp_path / "kinetic_tip_cell_audit_v101.json").write_text("{}")
    assert not prephysics_infrastructure_failure(tmp_path)
