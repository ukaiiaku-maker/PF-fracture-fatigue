from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture import persistent_site_source_v10221 as source
from arrhenius_fracture.persistent_site_physical_width_v10222 import (
    install_physical_front_width,
    physical_source_geometry,
)
from arrhenius_fracture.persistent_site_source_v10221 import PersistentSiteConfig
from arrhenius_fracture.sharp_front_v10_2_22 import DEFAULT_REGISTRY, VALID_OPTIONS


def test_top_five_registry_is_exact_and_disables_legacy_closures():
    with DEFAULT_REGISTRY.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 5
    assert {row["option_key"]: row["candidate_id"] for row in rows} == VALID_OPTIONS
    expected_rho = {
        "v912_targeted_local_peak_013476_0368": 1.4115242646890916e16,
        "v912_targeted_local_peak_013476_0314": 1.4854598439174714e16,
        "v912_targeted_local_peak_013476_0162": 2.3391211664562664e16,
        "v912_targeted_local_peak_005518_0118": 2.3012695321899512e16,
        "v912_targeted_local_plateau_010759_0403": 1.3345163074387614e16,
    }
    for row in rows:
        assert float(row["rho_source0_m2"]) == pytest.approx(
            expected_rho[row["candidate_id"]]
        )
        for field in (
            "source_recovery_rate_s",
            "retained_recovery_rate_s",
            "recovery_nu0_s",
            "legacy_source_sites_active",
            "legacy_source_refresh_active",
            "explicit_recovery_active",
        ):
            assert float(row[field]) == 0.0


def _state(dx: float):
    return SimpleNamespace(
        dx=dx,
        length_m=50.0e-6,
        cfg=SimpleNamespace(forest_density_floor_m2=5.0e12),
        _persistent_site_cfg=PersistentSiteConfig(
            rho_site0_m2=1.4e16,
            reference_source_area_m2=25.0e-12,
            reference_front_width_m=10.0e-6,
            reference_density_m2=5.0e12,
            source_zone_length_m=2.0e-6,
            minimum_front_width_m=0.0,
            maximum_front_width_m=50.0e-6,
        ),
        _persistent_b=2.74e-10,
        _persistent_r0_m=1.0e-6,
        _persistent_active_arc_factor=2.5,
        blunted_radius=lambda r0, b: 1.2e-6,
    )


def test_front_width_is_independent_of_ahead_of_tip_grid(monkeypatch):
    monkeypatch.setattr(
        source,
        "_campaign_local_density_m2",
        lambda state: np.array([2.0e18, 0.0]),
    )
    coarse = physical_source_geometry(_state(0.625e-6))
    fine = physical_source_geometry(_state(0.100e-6))
    assert coarse["front_width_m"] == pytest.approx(fine["front_width_m"])
    assert coarse["multiplicity_per_system"] == pytest.approx(
        fine["multiplicity_per_system"]
    )
    assert coarse["front_width_m"] < 0.100e-6
    assert coarse["minimum_front_width_m"] == pytest.approx(2.74e-10)
    assert coarse["front_width_grid_independent"] is True


def test_physical_width_installer_replaces_active_geometry():
    original = source._source_geometry
    try:
        install_physical_front_width()
        assert source._source_geometry is physical_source_geometry
    finally:
        source._source_geometry = original


def test_screen_runner_declares_full_temperature_matrix_and_common_seed():
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts/run_v10_2_22_top5_dbtt_50um_screen.sh"
    completed = subprocess.run(
        ["bash", "-n", str(runner)], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    text = runner.read_text()
    assert 'TEMPS=${TEMPS:-"300 400 500 600 700 800 900 1000 1100 1200"}' in text
    assert "TARGET_EXT_UM=${TARGET_EXT_UM:-50}" in text
    assert "HAZARD_SEED=${HAZARD_SEED:-3621}" in text
    assert "BASE_SEED + T" not in text
    for option in VALID_OPTIONS:
        assert option in text
    assert "plot_v10_2_22_dbtt_rcurves.py" in text


def _write_case(root: Path, option: str, candidate: str, temperature: int) -> None:
    case = root / option / f"T{temperature}K_th45_seed3621"
    case.mkdir(parents=True)
    (case / "COMPLETE").write_text("\n")
    (case / "v10_2_22_parameter_selection.json").write_text(
        json.dumps(
            {
                "option_key": option,
                "candidate_id": candidate,
                "role": "synthetic smoke",
            }
        )
    )
    columns = [
        "KJ_Pa_sqrtm",
        "crack_extension_m",
        "da_block_m",
        "n_fire",
        "sigma_back_Pa",
        "mpz_available_site_fraction",
        "mpz_K_shield_Pa_sqrt_m",
    ]
    values = np.array(
        [
            [20.0e6, 5.0e-6, 5.0e-6, 1.0, 0.1e9, 1.0, 0.0],
            [25.0e6, 15.0e-6, 10.0e-6, 1.0, 0.2e9, 1.0, 1.0e3],
            [30.0e6, 55.0e-6, 40.0e-6, 1.0, 0.3e9, 1.0, 2.0e3],
        ]
    )
    np.savetxt(
        case / f"steps_{temperature:04d}K.csv",
        values,
        delimiter=",",
        header=",".join(columns),
        comments="",
    )


def test_rcurve_plotter_writes_case_candidate_and_temperature_plots(tmp_path):
    options = list(VALID_OPTIONS.items())[:2]
    for option, candidate in options:
        for temperature in (300, 400):
            _write_case(tmp_path, option, candidate, temperature)
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    completed = subprocess.run(
        [
            os.environ.get("PYTHON", "python"),
            str(root / "scripts/plot_v10_2_22_dbtt_rcurves.py"),
            "--outroot",
            str(tmp_path),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert (tmp_path / "v10_2_22_dbtt_50um_screen_summary.csv").exists()
    assert (tmp_path / "plots/by_temperature/K_vs_crack_extension_0300K.png").exists()
    for option, _ in options:
        assert (
            tmp_path / "plots/by_candidate" / f"{option}_K_vs_crack_extension.png"
        ).exists()
        assert (
            tmp_path
            / "plots/individual"
            / option
            / "K_vs_crack_extension_0300K.png"
        ).exists()
