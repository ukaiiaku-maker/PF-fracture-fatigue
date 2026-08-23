from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "physics_informed",
    ROOT / "scripts/analyze_v914_paris_physics_informed_continuation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_three_hit_low_exposure_has_cubic_rate_and_triple_log_sensitivity():
    row = {
        "Tref_K": 300.0, "cleave_G00_eV": 4.0, "cleave_gT_eV_per_K": 0.0,
        "cleave_sigc0_GPa": 8.0, "cleave_sT_GPa_per_K": 0.0,
        "cleave_exp_a": 0.05, "cleave_exp_n": 0.5, "cleave_floor_frac": 0.5,
    }
    common = {
        "r0_m": 1.0e-6, "cleavage_nu0_s": 1.0e12,
        "cleavage_correlation_time_s": 1.0e-6,
    }
    result = MODULE.surface(row, np.asarray([1.0]), common, 3.0)
    raw = result["raw_rate_s"][0]; tau = common["cleavage_correlation_time_s"]
    assert result["exposure"][0] < 1.0e-4
    assert result["production_rate_s"][0] == pytest.approx(raw**3 * tau**2 / 6.0, rel=2.0e-4)
    assert result["dln_production_dln_raw"][0] == pytest.approx(3.0, rel=2.0e-4)


def test_surface_barrier_is_positive_and_monotonically_decreases_with_load():
    row = {
        "Tref_K": 300.0, "cleave_G00_eV": 1.84, "cleave_gT_eV_per_K": 0.0,
        "cleave_sigc0_GPa": 2.10, "cleave_sT_GPa_per_K": 0.0,
        "cleave_exp_a": 0.42, "cleave_exp_n": 2.92, "cleave_floor_frac": 0.22,
    }
    loads = np.linspace(10.24, 12.96, 20)
    result = MODULE.surface(row, loads, {"r0_m": 1.0e-6, "cleavage_nu0_s": 1.0e12, "cleavage_correlation_time_s": 1.0e-6})
    assert np.all(result["Gstar_eV"] > 0.0)
    assert np.all(np.diff(result["Gstar_eV"]) < 0.0)
    assert np.all(result["dG_dsigma_eV_Pa"] < 0.0)
