from __future__ import annotations

from pathlib import Path


def test_default_kernel_builder_cannot_invoke_mechanics_only_capture():
    path = Path("scripts/build_v10_2_27_kernel_for_configuration.sh")
    text = path.read_text()

    assert "automatic mechanics-only kernel capture is disabled" in text
    assert "KERNEL_CAPTURE_COMMAND" in text
    assert "capture_v10_2_27_kernel_states_for_configuration.py" not in text


def test_production_runner_retains_stochastic_moving_pz_contract():
    path = Path("scripts/run_v10_2_27_paper_four_class_30deg_long_rcurves.sh")
    text = path.read_text()

    required = (
        "export CLEAVAGE_HAZARD_MODE=exponential",
        "export CLEAVAGE_EVENT_LENGTH_MODE=threshold_scaled",
        "--front-state-model moving_pz",
        "--tip-kinetics-mode moving_velocity",
        "--active-shielding",
        "--signed-active-shielding",
    )
    for token in required:
        assert token in text

    forbidden = (
        "CLEAVAGE_HAZARD_MODE=deterministic",
        "CLEAVAGE_EVENT_LENGTH_MODE=fixed",
    )
    for token in forbidden:
        assert token not in text
