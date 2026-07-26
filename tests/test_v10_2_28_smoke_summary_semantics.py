from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_v10_2_28_direct_kernel_smoke.sh"


def test_single_orientation_is_not_reported_as_a_verified_comparison():
    source = SCRIPT.read_text()
    assert '"orientation_fingerprint_comparison_applicable": comparison_applicable' in source
    assert 'comparison_applicable = len(by_theta) > 1' in source
    assert 'comparison_applicable and len(unique_by_theta) == len(by_theta)' in source
    assert 'len(by_theta) <= 1 or len(unique_by_theta) == len(by_theta)' not in source
