from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_taylor_native_operator_consequence.py"


pytestmark = pytest.mark.external_evidence


def _module():
    spec = importlib.util.spec_from_file_location("taylor_native_consequence", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_checkpoint_reconstruction_uses_effective_native_geometry():
    state, provenance = _module()._load_state()
    geometry = provenance["geometry_ownership"]

    assert geometry["contradictory_alias_disposition"] == "rejected_stale_parser_alias"
    assert geometry["contradictory_alias_value_m"] == 2.0e-6
    assert state.length_m == pytest.approx(50.0e-6, rel=0.0, abs=1.0e-20)
    assert state.dx == pytest.approx(0.625e-6, rel=0.0, abs=2.0e-22)
    assert state.wake_length_m == pytest.approx(100.0e-6, rel=0.0, abs=2.0e-20)
    assert state.wake_dx == pytest.approx(state.dx, rel=0.0, abs=2.0e-22)
    assert state.mobile_positive.shape == (2, 80)
    assert state.wake_mobile_positive.shape == (2, 160)
    np.testing.assert_array_equal(
        state.x, (np.arange(80, dtype=float) + 0.5) * state.dx
    )
    np.testing.assert_array_equal(
        state.wake_x, (np.arange(160, dtype=float) + 0.5) * state.wake_dx
    )
    assert state.mobile_count + state.retained_count == 3.6783957316599154e-08


def test_checkpoint_reconstruction_preserves_native_rate_controls():
    state, provenance = _module()._load_state()
    config = provenance["native_config"]

    assert config["source_bin_count"] == 2
    assert config["blunting_length_m"] == 0.5e-6
    assert config["forest_density_floor_m2"] == 5.0e12
    assert config["jump_fraction"] == 1.0
    assert config["mobile_recovery_rate_s"] == 0.0
    assert config["mobile_shield_fraction"] == 0.0
    assert config["taylor_stress_fraction"] == 1.0 / np.sqrt(3.0)
    assert config["peierls_stress_fraction"] == 1.0 / np.sqrt(3.0)
    assert config["wake_shielding"] is False
    assert state.manifest.encounter_efficiency == pytest.approx(9.160246308716648)
