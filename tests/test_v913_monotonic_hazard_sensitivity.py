from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("mono", ROOT / "scripts/run_v913_monotonic_hazard_sensitivity.py")
mono = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(mono)


def test_operator_labels_exact_coupled_replay_not_instantaneous_proxy():
    source = (ROOT / "scripts/run_v913_monotonic_hazard_sensitivity.py").read_text()
    assert "EXACT_V913_FIXED_PEAK_COUPLED_TRAJECTORY_REPLAY_CENTERED_DIFFERENCE" in source
    assert "A_K_direct_frozen_path_per_MPa_sqrt_m" in source
    assert "A_K_state_correction_per_MPa_sqrt_m" in source


def test_sha256_is_content_sensitive(tmp_path):
    path = tmp_path / "value"
    path.write_text("a")
    first = mono.sha256(path)
    path.write_text("b")
    assert mono.sha256(path) != first
