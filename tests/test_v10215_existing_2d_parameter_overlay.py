from __future__ import annotations

import csv
from pathlib import Path

from arrhenius_fracture import sharp_front_v10_2_15 as stage3


def _value(args: list[str], name: str) -> str | None:
    for index, token in enumerate(args):
        if token == name and index + 1 < len(args):
            return args[index + 1]
        if token.startswith(name + "="):
            return token.split("=", 1)[1]
    return None


def test_stage3_entry_is_parameter_overlay_only():
    text = Path("arrhenius_fracture/sharp_front_v10_2_15.py").read_text()
    assert "from . import sharp_front_v10_1_7_5 as _final_2d" in text
    assert "result = _final_2d.main(args)" in text
    assert '"parameter_overlay_only": True' in text
    assert '"tip_engine_replaced": False' in text
    assert '"source_lifecycle_replaced": False' in text
    assert '"transport_operator_replaced": False' in text
    assert '"shielding_law_replaced": False' in text
    assert "StateResolvedSignedBurgersTipEngine" not in text
    assert "ActiveOnlySigned2DShieldingKernelFamily" not in text
    assert "configure_state_resolved_physics" not in text
    assert "SIGNED_KERNEL_FAMILY_JSON" not in text


def test_real_registry_row_is_passed_to_final_2d_entry_only(monkeypatch, tmp_path):
    captured: dict[str, list[str]] = {}

    def fake_final(args):
        captured["args"] = list(args)
        return {"ok": True}

    monkeypatch.setattr(stage3._final_2d, "main", fake_final)
    monkeypatch.setenv("CLEAVAGE_HAZARD_MODE", "deterministic")
    monkeypatch.setenv("CLEAVAGE_EVENT_LENGTH_MODE", "fixed")
    monkeypatch.setenv("ANISOTROPIC_TRANSPORT_MODE", "validated_scalar")

    out = tmp_path / "case"
    result = stage3.main(
        [
            "--mode", "2d",
            "--parameter-option", "weakT_primary",
            "--temperatures", "700",
            "--crystal-aniso",
            "--crystal-compete",
            "--out", str(out),
        ]
    )
    assert result == {"ok": True}
    args = captured["args"]
    manifest = Path(_value(args, "--material-manifest") or "")
    assert manifest.is_file()
    with manifest.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert row["candidate_id"] == "weakT_restart00_candidate00"
    assert _value(args, "--mpz-length-um") == "100.0"
    assert _value(args, "--mpz-n-bins") == "200"
    assert _value(args, "--parameter-option") is None
    assert _value(args, "--parameter-registry") is None
    assert _value(args, "--signed-kernel-family") is None
    assert "--no-wake-shielding" in args
    assert _value(args, "--mobile-shield-fraction") == "0.0"
    audit = out / "v10_2_15_existing_2d_parameter_overlay.json"
    assert audit.is_file()


def test_overnight_execution_path_has_no_atlas_or_engine_substitution():
    overnight = Path("scripts/run_v10_2_15_stage3_overnight.sh").read_text()
    runner = Path("scripts/run_v10_2_15_stage3_monotonic_temperature_sweep.sh").read_text()
    combined = overnight + "\n" + runner
    forbidden = (
        "SIGNED_KERNEL_FAMILY_JSON",
        "LOAD_INVARIANCE_ROOT",
        "ENGINE_CONFIG",
        "build_v10_2_14_campaign_ready_active_only_atlas",
        "state_resolved_signed_engine_v10214",
        "runtime_grid_binding_v10215",
    )
    for token in forbidden:
        assert token not in combined
    assert "arrhenius_fracture.sharp_front_v10_1_7_5" in combined
    assert "--parameter-option" in runner
    assert "--parameter-registry" in runner
