from __future__ import annotations

from pathlib import Path


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
