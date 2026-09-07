"""PX4.1 section 3: the PX4 developed-analysis canonical pairing key must
be seed-safe and must never collide across cohesive-strength or control-
mode variants -- a real gap in PX4's original by_pair key (protocol,
row_name, Kmax, R, frequency, hold, chemistry_factor only), which would
have silently let a second seed's rows overwrite the first's.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import build_part_x_px4_developed_analysis as analysis  # noqa: E402

BASE_ROW = {
    "seed": "1720", "protocol": "D1", "row_name": "COMPETING_REVERSIBLE",
    "config_hash": "deadbeef", "Kmax_Pa_sqrt_m": "18000000.0", "R": "-0.95",
    "frequency_Hz": "1000.0", "minimum_load_hold_s": "0.0", "chemistry_factor": "1.0",
    "K_rebond_max_target_Pa_sqrt_m": "900000.0", "physical_producer_sha": "abc123",
}


def test_different_seeds_do_not_collide():
    row_seed1 = dict(BASE_ROW, seed="1720")
    row_seed2 = dict(BASE_ROW, seed="1001723")
    assert analysis.canonical_analysis_key(row_seed1) != analysis.canonical_analysis_key(row_seed2)


def test_different_producer_sha_do_not_collide():
    row_a = dict(BASE_ROW, physical_producer_sha="abc123")
    row_b = dict(BASE_ROW, physical_producer_sha="def456")
    assert analysis.canonical_analysis_key(row_a) != analysis.canonical_analysis_key(row_b)


def test_different_control_mode_do_not_collide():
    key_dynamic = analysis.canonical_analysis_key(BASE_ROW, control_mode="dynamic")
    key_static = analysis.canonical_analysis_key(BASE_ROW, control_mode="prescribed_static")
    assert key_dynamic != key_static


def test_finite_and_zero_share_a_key_despite_different_K_rebond_max_target():
    """The pairing key must NOT include K_rebond_max_target_Pa_sqrt_m --
    a finite row's K_target (nonzero) and its matched zero row's K_target
    (always exactly 0.0) can never be equal, so a pairing key that required
    equality there would make every pair permanently unmatchable."""
    finite_row = dict(BASE_ROW, K_rebond_max_target_Pa_sqrt_m="900000.0")
    zero_row = dict(BASE_ROW, K_rebond_max_target_Pa_sqrt_m="0.0")
    assert analysis.canonical_analysis_key(finite_row) == analysis.canonical_analysis_key(zero_row)


def test_different_Kmax_R_frequency_hold_chemistry_do_not_collide():
    variants = [
        dict(BASE_ROW, Kmax_Pa_sqrt_m="21000000.0"),
        dict(BASE_ROW, R="-0.5"),
        dict(BASE_ROW, frequency_Hz="316.227766"),
        dict(BASE_ROW, minimum_load_hold_s="0.002"),
        dict(BASE_ROW, chemistry_factor="0.3"),
        dict(BASE_ROW, config_hash="feedface"),
        dict(BASE_ROW, protocol="D2"),
        dict(BASE_ROW, row_name="COMPETING_PERSISTENT"),
    ]
    keys = [analysis.canonical_analysis_key(v) for v in variants]
    base_key = analysis.canonical_analysis_key(BASE_ROW)
    assert base_key not in keys
    assert len(set(keys)) == len(keys), "distinct variants must not collide with each other either"
