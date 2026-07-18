"""v10.2.1 entry point for prescribed fixed-DeltaK stochastic fatigue.

This stage retains the v10.2 moving-MPZ fatigue reintegration and prescribes the
local cyclic K waveform exactly.  The elastic FEM is held at a nonzero probe load
and supplies evolving geometry, directional J information, and normalized tensor
shape.  Full cyclic displacement feedback is intentionally deferred until the
bulk cyclic-plasticity model is promoted and validated.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sys

from .fixed_deltaK_v1021 import (
    MODEL_ID,
    fixed_deltaK_audit_payload,
    install_fixed_deltaK_waveform,
)
from . import sharp_front_v10_1_7_3 as _avalanche
from . import sharp_front_v10_2_0 as _fatigue


def _pop_value(args: list[str], name: str) -> str:
    prefix = name + "="
    for index, token in enumerate(list(args)):
        if token.startswith(prefix):
            del args[index]
            return token[len(prefix):]
        if token == name:
            if index + 1 >= len(args):
                raise SystemExit(f"{name} requires a value")
            value = args[index + 1]
            del args[index:index + 2]
            return value
    raise SystemExit(f"v10.2.1 requires {name}")


def _option_value(args: list[str], name: str, default: str | None = None) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return default


def _ensure_toggle(args: list[str], positive: str, negative: str) -> None:
    if positive in args:
        return
    args[:] = [token for token in args if token != negative]
    args.append(positive)


@contextmanager
def _allow_right_censored_stochastic_summary():
    """Make zero-event fixed-DeltaK cases valid right-censored observations."""
    original = _avalanche._rewrite_summary_event_semantics

    def tolerant(args: list[str]) -> None:
        out = _option_value(args, "--out")
        if not out:
            return original(args)
        root = Path(out)
        geometry_path = root / "stochastic_avalanche_geometry_events.json"
        summary_path = root / "summary.json"
        if geometry_path.is_file() and summary_path.is_file():
            geometry = json.loads(geometry_path.read_text())
            if isinstance(geometry, list) and len(geometry) == 0:
                summary = json.loads(summary_path.read_text())
                if isinstance(summary, list) and summary:
                    summary[0].update({
                        "n_geometry_events": 0,
                        "n_equivalent_checkpoints_exact": 0.0,
                        "n_equivalent_checkpoints_rounded": 0,
                        "geometry_path_length_m": 0.0,
                        "geometry_projected_extension_m": 0.0,
                        "fatigue_censor_status": "right_censored_no_event",
                    })
                    summary_path.write_text(json.dumps(summary, indent=2))
                return
        return original(args)

    _avalanche._rewrite_summary_event_semantics = tolerant
    try:
        yield
    finally:
        _avalanche._rewrite_summary_event_semantics = original


def _write_audit(args: list[str], target_deltaK: float) -> None:
    out = _option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    payload = fixed_deltaK_audit_payload()
    R = float(_option_value(args, "--R", "0.1") or 0.1)
    target_Kmax = target_deltaK / (1.0 - R)
    geometry_path = root / "stochastic_avalanche_geometry_events.json"
    events = []
    if geometry_path.is_file():
        data = json.loads(geometry_path.read_text())
        if isinstance(data, list):
            events = data
    payload.update({
        "schema": MODEL_ID,
        "target_deltaK_MPa_sqrt_m": float(target_deltaK),
        "target_Kmax_MPa_sqrt_m": float(target_Kmax),
        "R": R,
        "frequency_Hz": float(_option_value(args, "--frequency-Hz", "1000") or 1000.0),
        "fatigue_control_mode": "prescribed_fixed_local_deltaK",
        "fem_loading_mode": "held_nonzero_shape_probe",
        "cyclic_mechanics_enabled": False,
        "full_displacement_feedback_enabled": False,
        "stochastic_geometry_events": len(events),
        "censor_status": "propagated" if events else "right_censored_no_event",
    })
    target_pa = float(target_deltaK) * 1.0e6
    error_pa = float(payload.get("maximum_abs_target_error_Pa_sqrt_m", float("inf")))
    payload["fixed_deltaK_exact_within_relative_1e-12"] = bool(
        error_pa <= max(1.0e-6, 1.0e-12 * target_pa)
    )
    (root / "v10_2_1_fixed_deltaK_control.json").write_text(
        json.dumps(payload, indent=2)
    )


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    target_deltaK = float(_pop_value(args, "--target-deltaK-MPa-sqrt-m"))
    if target_deltaK <= 0.0:
        raise SystemExit("--target-deltaK-MPa-sqrt-m must be positive")
    if "--no-cyclic-mechanics" not in args:
        raise SystemExit(
            "v10.2.1 prescribed fixed-DeltaK mode requires --no-cyclic-mechanics; "
            "full cyclic displacement feedback is a later validation stage"
        )
    _ensure_toggle(args, "--fatigue-hold-load", "--no-fatigue-hold-load")

    R = float(_option_value(args, "--R", "0.1") or 0.1)
    if not 0.0 <= R < 1.0:
        raise SystemExit("v10.2.1 fixed-DeltaK mode requires 0 <= R < 1")
    print(
        "  v10.2.1 fixed-DeltaK fatigue: "
        f"DeltaK={target_deltaK:g} MPa*sqrt(m) "
        f"Kmax={target_deltaK/(1.0-R):g} MPa*sqrt(m) R={R:g} "
        "control=prescribed_local_K FEM=held_shape_probe"
    )

    with install_fixed_deltaK_waveform(target_deltaK):
        with _allow_right_censored_stochastic_summary():
            result = _fatigue.main(args)
    _write_audit(args, target_deltaK)
    return result


if __name__ == "__main__":
    main()
