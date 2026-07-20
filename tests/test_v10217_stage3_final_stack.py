from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import arrhenius_fracture.sharp_front_v10_2_17 as entry


def _family():
    return SimpleNamespace(
        states=[object(), object()],
        metadata={
            "production_parameterization_allowed": True,
            "constitutive_K_shield_cap_present": False,
            "active_kernel_mechanically_measured": True,
            "wake_kernel_mechanically_measured": False,
            "wake_shielding_supported": False,
        },
        audit_payload=lambda: {
            "schema": "v10.2.14_active_only_real_signed_2d_shielding_atlas",
            "active_physical_kernel_axes": ["cumulative_crack_path_extension_m"],
        },
    )


def _base_args(tmp_path: Path) -> list[str]:
    return [
        "--signed-kernel-family", "family.json",
        "--parameter-option", "ceramic_primary",
        "--out", str(tmp_path / "case"),
        "--mode", "2d",
        "--crystal-aniso",
        "--crystal-compete",
        "--no-wake-shielding",
        "--max-fronts", "1",
    ]


def _install(monkeypatch, family):
    monkeypatch.setattr(
        entry.ActiveOnlySigned2DShieldingKernelFamily,
        "from_json",
        classmethod(lambda cls, path: family),
    )
    monkeypatch.setattr(
        entry.StateResolvedSignedBurgersTipEngine,
        "configure_state_resolved_physics",
        classmethod(lambda cls, supplied, mode, **kwargs: None),
    )
    monkeypatch.setattr(
        entry.StateResolvedSignedBurgersTipEngine,
        "audit_payload",
        classmethod(lambda cls: {
            "state_resolved_signed_kernel_v10214": {
                "model_id": "v10.2.14_shared_active_only_signed_burgers_engine"
            }
        }),
    )
    monkeypatch.setattr(
        entry._transport, "normalize_transport_mode", lambda value: "validated_scalar"
    )


def test_final_signed_engine_and_stochastic_distance_are_installed(monkeypatch, tmp_path):
    family = _family()
    _install(monkeypatch, family)
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "exponential")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled")
    monkeypatch.setenv("CLEAVAGE_HAZARD_SEED", "2020")
    monkeypatch.setenv("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")
    monkeypatch.setenv("ANISOTROPIC_USE_AVALANCHE_BACKEND", "1")
    seen = {}

    def fake_main(args):
        seen["anisotropic"] = (
            entry._anisotropic.AnisotropicStochasticAvalancheTipEngine
            is entry.StateResolvedSignedBurgersTipEngine
        )
        seen["entry74"] = (
            entry._entry74.AnisotropicStochasticAvalancheTipEngine
            is entry.StateResolvedSignedBurgersTipEngine
        )
        seen["args"] = list(args)
        return "ok"

    monkeypatch.setattr(entry._transport, "main", fake_main)
    assert entry.main(_base_args(tmp_path)) == "ok"
    assert seen["anisotropic"] is True
    assert seen["entry74"] is True
    assert seen["args"][seen["args"].index("--mobile-shield-fraction") + 1] == "0.0"
    audit = json.loads(
        (tmp_path / "case" / "v10_2_17_final_signed_stochastic_stack.json").read_text()
    )
    assert audit["cleavage_hazard_mode"] == "exponential"
    assert audit["event_length_mode"] == "threshold_scaled"
    assert audit["cleavage_hazard_seed"] == 2020
    assert audit["constitutive_K_shield_cap_applied"] is False
    assert audit["signed_burgers_population_required"] is True
    assert audit["effective_opening_fixed_point_enabled"] is True
    assert audit["wake_shielding_enabled"] is False


def test_deterministic_or_fixed_modes_fail_closed(monkeypatch, tmp_path):
    family = _family()
    _install(monkeypatch, family)
    monkeypatch.setenv("CLEAVAGE_HAZARD_SEED", "2020")
    monkeypatch.setenv("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")
    monkeypatch.setenv("ANISOTROPIC_USE_AVALANCHE_BACKEND", "1")
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "deterministic")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "threshold_scaled")
    with pytest.raises(SystemExit, match="CLEAVAGE_HAZARD_MODE=exponential"):
        entry.main(_base_args(tmp_path))
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "exponential")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "fixed")
    with pytest.raises(SystemExit, match="CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled"):
        entry.main(_base_args(tmp_path))


def test_stage3_shell_launchers_parse():
    root = Path(__file__).resolve().parents[1]
    for relative in (
        "scripts/run_v10_2_17_stage3_monotonic_temperature_sweep.sh",
        "scripts/run_v10_2_17_stage3_overnight.sh",
        "scripts/freeze_v10_2_17_runtime_bundle.sh",
    ):
        completed = subprocess.run(
            ["bash", "-n", str(root / relative)],
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr


def test_runner_declares_final_stack_and_unique_seeds():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run_v10_2_17_stage3_monotonic_temperature_sweep.sh").read_text()
    assert "sharp_front_v10_2_17" in text
    assert "CLEAVAGE_HAZARD_MODE=exponential" in text
    assert "CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled" in text
    assert "CLEAVAGE_HAZARD_SEED" in text
    assert "signed-active-shielding" in text
    assert "state_resolved_signed_engine_v10214" in text


def test_overnight_reuses_local_family_without_legacy_mechanics_inputs():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/run_v10_2_17_stage3_overnight.sh").read_text()
    assert "runtime_inputs/v10_2_17/v10_2_14_active_only_campaign_family.json" in text
    assert "reusing signed family" in text
    assert "external_mechanics_inputs_required_for_this_launch" in text
    assert "stale editable import" in text
    assert "build_v10_2_14_campaign_ready_active_only_atlas_v2.py" in text
    assert "if [[ -f \"$FAMILY_JSON\" ]]" in text
