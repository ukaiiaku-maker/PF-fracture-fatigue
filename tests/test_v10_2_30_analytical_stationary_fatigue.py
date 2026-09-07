import csv
import math
from pathlib import Path

import pytest

from arrhenius_fracture.analytical_stationary_fatigue_v10230 import (
    AnalyticalControls,
    cycle_opening,
    solve_hierarchy,
    waveform_K,
)
from arrhenius_fracture.material_manifest import MaterialManifest


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "runs/A_native_plus_8PT_fatigue_v1/A_native_plus_8PT_registry.csv"


def rows():
    with REGISTRY.open(newline="") as handle:
        return {r["option_key"]: r for r in csv.DictReader(handle)}


def manifest(option="A_NATIVE"):
    path = next((ROOT / f"runs/A_native_plus_8PT_fatigue_v1/developed/n80/{option}").glob("DK_*/selected_material_manifest_v10_2_22.csv"))
    return MaterialManifest.from_csv(path)


def test_waveform_preserves_kmax_and_clips_negative():
    K = waveform_K(18.0e6, -0.95, 8192)
    assert K.min() == pytest.approx(0.0)
    assert K.max() == pytest.approx(18.0e6, rel=1e-7)


def test_opening_only_reference_value_is_stable():
    result = cycle_opening(manifest(), 15.0, 0.1, 300.0, 1000.0, 1.0e-6, AnalyticalControls())
    assert result["da_dN"] == pytest.approx(3.9026982916e-7, rel=2e-7)


def test_exact_quadrature_has_expected_R_ordering():
    m = manifest(); c = AnalyticalControls()
    values = [cycle_opening(m, 18.0, R, 300.0, 1000.0, 1.0e-6, c)["da_dN"] for R in (-0.95, 0.1, 0.5)]
    assert values[0] < values[1] < values[2]


def test_emission_blunting_fixed_point_is_finite_and_reduces_opening():
    row = rows()["A_NATIVE"]
    result = solve_hierarchy(manifest(), row, 18.0, 0.1, 300.0, 1000.0)
    assert result["fixed_point_converged"]
    assert math.isfinite(result["stationary_net_source_slip"])
    assert result["r_eff_m"] > 1.0e-6
    assert 0.0 < result["A1_da_dN"] < result["A0_da_dN"]


def test_PT_changes_latent_partition_not_stationary_mean_opening():
    bank = rows()
    native = solve_hierarchy(manifest(), bank["A_NATIVE"], 18.0, 0.1, 300.0, 1000.0)
    pt03_id = next(k for k in bank if k.startswith("A_PT_03"))
    pt03 = solve_hierarchy(manifest(pt03_id), bank[pt03_id], 18.0, 0.1, 300.0, 1000.0)
    assert pt03["stationary_retained"] != pytest.approx(native["stationary_retained"])
    assert pt03["A2_da_dN"] == pytest.approx(native["A2_da_dN"], rel=1e-12)


def test_event_rate_is_not_capped_at_one_per_cycle():
    result = cycle_opening(manifest(), 60.0, 0.5, 300.0, 1.0, 1.0e-6, AnalyticalControls())
    assert result["q_open"] > 1.0
