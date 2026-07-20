from __future__ import annotations

import pytest

from arrhenius_fracture.sharp_front_v10_2_15 import (
    _force_stage3_validity_envelope,
    _option_value,
)


def _base_args():
    return [
        "--mode", "2d",
        "--crystal-aniso",
        "--crystal-compete",
        "--out", "unused",
    ]


def test_stage3_policy_forces_single_front_active_only(monkeypatch):
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "deterministic")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "fixed")
    monkeypatch.setenv("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")
    args = _base_args()
    _force_stage3_validity_envelope(args)
    assert _option_value(args, "--max-fronts") == "1"
    assert _option_value(args, "--mobile-shield-fraction") == "0.0"
    assert "--no-wake-shielding" in args
    assert "--wake-shielding" not in args
    assert _option_value(args, "--bulk-plasticity-mode") == "tip_only"
    assert _option_value(args, "--directional-j-mode") == "root_signed"
    assert _option_value(args, "--tip-source-model") == "continuum"


def test_stage3_policy_rejects_branching(monkeypatch):
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "deterministic")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "fixed")
    monkeypatch.setenv("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")
    with pytest.raises(SystemExit, match="branching disabled"):
        _force_stage3_validity_envelope(_base_args() + ["--crystal-branch"])


def test_stage3_policy_rejects_mobile_shielding(monkeypatch):
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "deterministic")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "fixed")
    monkeypatch.setenv("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")
    with pytest.raises(SystemExit, match="mobile-shield-fraction 0"):
        _force_stage3_validity_envelope(
            _base_args() + ["--mobile-shield-fraction", "0.5"]
        )
