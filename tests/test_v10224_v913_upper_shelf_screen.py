from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess

from arrhenius_fracture.sharp_front_v10_2_24 import (
    DEFAULT_REGISTRY,
    SELECTION_RECORD,
    VALID_OPTIONS,
)


ACTIVE_FIELDS = (
    "Tref_K",
    "cleave_G00_eV",
    "cleave_gT_eV_per_K",
    "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K",
    "cleave_exp_a",
    "cleave_exp_n",
    "cleave_floor_frac",
    "emit_G00_eV",
    "emit_gT_eV_per_K",
    "emit_sigc0_GPa",
    "emit_sT_GPa_per_K",
    "emit_exp_a",
    "emit_exp_n",
    "emit_floor_frac",
    "peierls_H0_eV",
    "peierls_activation_entropy_kB",
    "peierls_exp_a",
    "peierls_exp_n",
    "peierls_nu0_s",
    "taylor_H0_eV",
    "taylor_activation_entropy_kB",
    "taylor_exp_a",
    "taylor_exp_n",
    "taylor_nu0_s",
    "rho_source0_m2",
    "taylor_corr_rho_c_m2",
    "taylor_corr_scale",
    "c_blunt",
)
EXPECTED_ACTIVE_PAYLOAD_SHA256 = (
    "1a37e9e7dd812154455e0032fb9c7d0287b8eb0f810a1e8d2e50e976d580eb64"
)


def test_upper_shelf_selection_is_separate_and_reproducible() -> None:
    selection = json.loads(SELECTION_RECORD.read_text())
    assert selection["eligible_candidates"] == 32
    assert selection["eligibility"]["directional_dbtt_gain_min_MPa_sqrt_m"] == 5.0
    assert selection["eligibility"]["peak_like_1d"] is False
    assert selection["eligibility"]["peak_like_threshold_MPa_sqrt_m"] == 5.0
    assert selection["ranking"] == [
        "directional_dbtt_gain descending",
        "high_temperature_plateau descending",
        "candidate_id ascending",
    ]
    selected = selection["selected"]
    assert len(selected) == 10
    assert [row["shelf_rank"] for row in selected] == list(range(1, 11))
    gains = [row["directional_dbtt_gain_MPa_sqrt_m"] for row in selected]
    assert gains == sorted(gains, reverse=True)
    assert {row["candidate_id"] for row in selected} == set(VALID_OPTIONS.values())


def test_registry_has_exact_upper_shelf_top_ten_rows() -> None:
    with DEFAULT_REGISTRY.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {row["option_key"]: row["candidate_id"] for row in rows} == VALID_OPTIONS
    payload = []
    for row in rows:
        payload.append(
            {
                "candidate_id": row["candidate_id"],
                **{field: float(row[field]) for field in ACTIVE_FIELDS},
            }
        )
        for field in (
            "source_recovery_rate_s",
            "retained_recovery_rate_s",
            "source_refresh_length_um",
            "recovery_nu0_s",
            "legacy_source_sites_active",
            "legacy_source_refresh_active",
            "explicit_recovery_active",
        ):
            assert float(row[field]) == 0.0
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    assert hashlib.sha256(text.encode()).hexdigest() == EXPECTED_ACTIVE_PAYLOAD_SHA256


def test_upper_shelf_reference_has_complete_common_random_number_grid() -> None:
    reference = DEFAULT_REGISTRY.parent / "v10_2_24_v913_top10_upper_shelf_1d_reference.csv"
    with reference.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10
    assert {int(float(row["common_random_seed"])) for row in rows} == {3621}
    expected_temperatures = (
        700, 800, 900, 950, 1000, 1050, 1100, 1200, 1300, 1400,
    )
    for row in rows:
        assert row["selection_class"] == "directional_dbtt_upper_shelf_non_peak"
        assert float(row["y__directional_dbtt_gain"]) >= 5.0
        assert float(row["y__peak_prominence"]) < 5.0
        assert float(row["y__persistence_from_trajectory"]) >= 0.70
        for temperature in expected_temperatures:
            assert float(row[f"K25_T{temperature}K_MPa_sqrt_m"]) > 0.0
            assert float(row[f"K50_T{temperature}K_MPa_sqrt_m"]) > 0.0
    assert {row["candidate_id"] for row in rows} == set(VALID_OPTIONS.values())


def test_upper_shelf_runner_declares_exact_stochastic_matrix() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts/run_v10_2_24_top10_v913_upper_shelf_50um_screen.sh"
    completed = subprocess.run(
        ["bash", "-n", str(runner)], text=True, capture_output=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    text = runner.read_text()
    assert 'TEMPS=${TEMPS:-"700 800 900 950 1000 1050 1100 1200 1300 1400"}' in text
    assert "HAZARD_SEED=${HAZARD_SEED:-3621}" in text
    assert "TARGET_EXT_UM=${TARGET_EXT_UM:-50}" in text
    assert "CLEAVAGE_HAZARD_MODE=exponential" in text
    assert "CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled" in text
    assert "sharp_front_v10_2_24_audited" in text
    assert "compare_v10_2_24_upper_shelf_1d_2d.py" in text
    for option in VALID_OPTIONS:
        assert option in text
