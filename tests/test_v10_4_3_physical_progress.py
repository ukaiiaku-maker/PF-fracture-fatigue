from __future__ import annotations

from pathlib import Path

from arrhenius_fracture.plastic_flow_physical_progress_v1043 import (
    transform_source,
)


def _source() -> str:
    return Path("arrhenius_fracture/sharp_front.py").read_text()


def test_physical_progress_transform_compiles() -> None:
    transformed = transform_source(_source())
    compile(transformed, "sharp_front.py[v10.4.3-physical-progress-test]", "exec")


def test_requested_steps_are_nominal_progress_not_accepted_rows() -> None:
    transformed = transform_source(_source())

    assert "while step < args.steps:" not in transformed
    assert (
        "while nominal_progress_v1043 < (\n"
        "            nominal_progress_target_v1043"
    ) in transformed
    assert "nominal_progress_v1043 += float(trial_frac)" in transformed
    assert "adaptive substep exceeded nominal loading horizon" in transformed

    loop = transformed.index("while nominal_progress_v1043 <")
    cap = transformed.index("remaining_nominal_fraction_v1043", loop)
    accept = transformed.index("nominal_progress_v1043 += float(trial_frac)", cap)
    row = transformed.index("rows.append((step, Uapp", accept)
    assert loop < cap < accept < row


def test_final_remainder_and_output_are_progress_aware() -> None:
    transformed = transform_source(_source())

    assert (
        "_v1043_min_trial_frac = min(\n"
        "                        float(args.stagger_min_dt_fraction),"
    ) in transformed
    assert "nominal_progress_start,nominal_progress" in transformed
    assert "float(nominal_progress_step_start_v1043)" in transformed
    assert "float(nominal_progress_v1043)))" in transformed

    # Progress columns are appended after the existing final MPZ field so
    # legacy positional readers retain all prior offsets.
    wake = transformed.index("mpz_wake_retained_total,'")
    progress_header = transformed.index(
        "nominal_progress_start,nominal_progress", wake
    )
    assert wake < progress_header


def test_terminal_window_and_horizon_use_nominal_physical_span() -> None:
    transformed = transform_source(_source())

    required = [
        "nominal_progress_start",
        "nominal_progress_end",
        "classification_window_nominal_increment_span",
        "window_first_nominal_progress",
        "window_last_nominal_progress",
        "remaining_loading_horizon = max(\n        float(remaining_steps)",
        "nominal_progress_target_v1043\n                        - nominal_progress_v1043",
    ]
    missing = [token for token in required if token not in transformed]
    assert not missing, missing

    assert "plastic_flow_window = deque(maxlen=plastic_flow_window_size)" not in transformed
    assert "plastic_flow_window = deque()" in transformed
