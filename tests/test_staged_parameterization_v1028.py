import math

import numpy as np

from arrhenius_fracture.analytical_screen_v1028 import (
    AnalyticalControl,
    AnalyticalMechanics,
    DEFAULT_ANALYTICAL_TEMPERATURES_K,
    analytical_screen,
)
from arrhenius_fracture.material_manifest import (
    ExpFloorBarrier,
    MaterialManifest,
    TransportBarrier,
)
import arrhenius_fracture.staged_parameterization_v1028 as staged


class FakeDrive:
    n_systems = 2

    def validate_against_kernel_family(self, kernel):
        assert kernel.n_systems == 2

    def resolve(self, **coordinates):
        assert 0.0 <= coordinates["opening_strength_fraction"] <= 1.0
        return np.array([0.20, -0.10])


class FakeKernel:
    n_systems = 2
    activation_to_line_content = np.array([0.01, 0.01])

    def resolve(self, **coordinates):
        return np.array([[1.0e5, 0.5e5], [-0.5e5, -0.25e5]]), np.zeros((2, 0))


def manifest():
    cleavage = ExpFloorBarrier(
        G00_eV=0.65,
        gT_eV_per_K=0.0,
        sigc0_Pa=18.0e9,
        sT_Pa_per_K=0.0,
        alpha=2.0,
        exponent=2.0,
        floor_fraction=0.05,
        attempt_frequency_s=1.0e12,
    )
    emission = ExpFloorBarrier(
        G00_eV=0.75,
        gT_eV_per_K=-2.0e-4,
        sigc0_Pa=12.0e9,
        sT_Pa_per_K=0.0,
        alpha=2.0,
        exponent=2.0,
        floor_fraction=0.05,
        attempt_frequency_s=1.0e11,
    )
    transport = TransportBarrier(
        H0_eV=0.5,
        activation_entropy_kB=0.0,
        alpha=1.0,
        exponent=1.0,
        attempt_frequency_s=1.0e11,
    )
    return MaterialManifest(
        name="DBTT",
        candidate_id="TEST",
        cleavage=cleavage,
        emission=emission,
        peierls=transport,
        taylor=transport,
        taylor_corr_rho_c_m2=1.0e15,
        taylor_corr_scale=1.0,
        source_sites_per_system=10.0,
        encounter_efficiency=1.0,
        retained_recovery_rate_s=1.0,
        source_refresh_length_m=1.0e-6,
        c_blunt=1.0,
        max_K_shield_MPa_sqrt_m=0.0,
    )


def mechanics():
    return AnalyticalMechanics(
        r0_m=1.0e-6,
        sigma_cap_Pa=30.0e9,
        cleavage_hits=1.0,
        cleavage_tau_s=1.0e-6,
        source_bin_count=2,
    )


def test_analytical_temperature_grid_is_100K_increment():
    assert DEFAULT_ANALYTICAL_TEMPERATURES_K == tuple(float(T) for T in range(300, 1201, 100))


def test_analytical_screen_reports_all_temperatures():
    result = analytical_screen(
        manifest(), mechanics(),
        AnalyticalControl(Kmax_MPa_sqrt_m=40.0, dK_MPa_sqrt_m=0.5),
        FakeKernel(), FakeDrive(), target_class="DBTT",
    )
    assert result["screen_is_nonbinding"] is True
    assert len(result["details"]) == 10
    assert result["details"][0]["temperature_K"] == 300.0
    assert result["details"][-1]["temperature_K"] == 1200.0
    assert all("K_cleave_no_plastic_MPa_sqrt_m" in row for row in result["details"])


def _first_result(K):
    return {"status": "complete", "K_init_MPa_sqrt_m": float(K)}


def test_first_passage_uses_four_dbtt_and_three_weakt_temperatures():
    dbtt = {
        ("full", T): _first_result(K)
        for T, K in zip((300.0, 700.0, 900.0, 1200.0), (10.0, 11.0, 12.0, 13.0))
    }
    weak = {
        ("full", T): _first_result(K)
        for T, K in zip((300.0, 700.0, 1200.0), (10.0, 10.5, 10.8))
    }
    assert staged.score_first_passage_dbtt(dbtt)["first_passage_pass"] is True
    assert staged.score_first_passage_weakt(weak)["first_passage_pass"] is True


def test_first_passage_wrapper_requests_exactly_one_checkpoint(monkeypatch):
    seen = {}

    def fake_run(manifest, temperature, production, control, mode="full"):
        seen["target"] = control.target_extension_um
        return {
            "status": "complete",
            "K_init_MPa_sqrt_m": 10.0,
            "K_final_MPa_sqrt_m": 10.0,
        }

    monkeypatch.setattr(staged, "run_reduced_r_curve", fake_run)
    result = staged.run_first_passage(
        object(), 300.0, object(), staged.FirstPassageControl(), mode="full"
    )
    assert seen["target"] > 0.0
    assert seen["target"] < 1.0e-6
    assert result["exactly_one_checkpoint_requested"] is True


def test_two_d_plan_contains_full_curves_and_endpoint_ablations():
    rows = staged.two_d_validation_cases(["D1"], ["W1"], target_extension_um=100.0)
    dbtt = [row for row in rows if row["candidate_id"] == "D1"]
    weak = [row for row in rows if row["candidate_id"] == "W1"]
    assert len(dbtt) == 10  # four full plus three endpoint ablations at two T
    assert len(weak) == 9   # three full plus three endpoint ablations at two T
    assert all(row["target_extension_um"] == 100.0 for row in rows)
    assert {row["mode"] for row in rows} == {
        "full", "plasticity_off", "shielding_off", "backstress_off"
    }
