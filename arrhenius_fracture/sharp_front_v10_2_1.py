"""v10.2.1 entry point for prescribed fixed-DeltaK stochastic fatigue.

This stage retains the v10.2 moving-MPZ fatigue reintegration and prescribes the
local cyclic K waveform exactly. The elastic FEM is held at a nonzero probe load
and supplies evolving geometry, directional J information, and normalized tensor
shape. Full cyclic displacement feedback is intentionally deferred until the
bulk cyclic-plasticity model is promoted and validated.
"""
from __future__ import annotations

import builtins
from contextlib import contextmanager
import json
from pathlib import Path
import sys

from .fixed_deltaK_v1021 import (
    MODEL_ID,
    fixed_deltaK_audit_payload,
    install_fixed_deltaK_waveform,
)


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
    """Make zero-event fixed-DeltaK cases valid right-censored observations.

    Import the avalanche execution stack lazily. Older campaign entry points
    intentionally replace the public continuum-engine symbol at import time;
    importing them during pytest collection contaminates unrelated source-model
    tests. Runtime entry-point imports are safe because each simulation is a
    dedicated process and the replacement is part of the selected v10 campaign.
    """
    from . import sharp_front_v10_1_7_3 as avalanche

    original = avalanche._rewrite_summary_event_semantics

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

    avalanche._rewrite_summary_event_semantics = tolerant
    try:
        yield
    finally:
        avalanche._rewrite_summary_event_semantics = original


@contextmanager
def _fixed_deltaK_console_semantics(target_deltaK: float, R: float):
    """Relabel inherited probe-K output while the legacy driver is executing."""
    original_print = builtins.print
    target_Kmax = target_deltaK / (1.0 - R)

    def labelled_print(*items, **kwargs):
        if len(items) == 1 and isinstance(items[0], str):
            text = items[0]
            if text.startswith("  FATIGUE MODE:"):
                text = (
                    "  FATIGUE MODE: prescribed fixed local "
                    f"DeltaK={target_deltaK:g} MPa*sqrt(m), "
                    f"Kmax={target_Kmax:g} MPa*sqrt(m), R={R:g}; "
                    "FEM KJ is a held geometry/tensor probe only."
                )
            elif text.startswith("  [T="):
                text = text.replace(" KJ=", " KJ_probe=")
            elif text.startswith("  == T="):
                text = text.replace("Kc_first=", "KJ_probe_first=")
                text = text.replace(", mode=", ", inherited_mode=")
            elif "toughness_vs_temperature.png" in text:
                return
            return original_print(text, **kwargs)
        return original_print(*items, **kwargs)

    builtins.print = labelled_print
    try:
        yield
    finally:
        builtins.print = original_print


def _normalize_output_semantics(root: Path, target_deltaK: float, R: float) -> dict:
    """Prevent the held FEM probe K from being reported as fatigue toughness."""
    target_Kmax = target_deltaK / (1.0 - R)
    target_Kmin = R * target_Kmax
    target_deltaK_pa = target_deltaK * 1.0e6
    target_Kmax_pa = target_Kmax * 1.0e6
    target_Kmin_pa = target_Kmin * 1.0e6

    summary_path = root / "summary.json"
    probe_first = None
    summary_rewritten = False
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
        if isinstance(summary, list):
            for row in summary:
                if not isinstance(row, dict):
                    continue
                old = row.get("Kc_first_MPa_sqrt_m")
                if probe_first is None and old is not None:
                    probe_first = float(old)
                row["KJ_probe_at_first_event_MPa_sqrt_m"] = old
                row["Kc_first_MPa_sqrt_m"] = None
                row["Kc_first_interpretation"] = "not_applicable_fixed_deltaK_fatigue"
                row["fatigue_DeltaK_MPa_sqrt_m"] = float(target_deltaK)
                row["fatigue_Kmax_MPa_sqrt_m"] = float(target_Kmax)
                row["fatigue_Kmin_MPa_sqrt_m"] = float(target_Kmin)
                row["probe_KJ_is_fatigue_driving_K"] = False
                advances = int(row.get("n_advances", 0) or 0)
                row["mode"] = "fatigue_propagated" if advances > 0 else "fatigue_right_censored"
            summary_path.write_text(json.dumps(summary, indent=2))
            summary_rewritten = True

    steps_rewritten = 0
    for path in sorted(root.glob("steps_*K.csv")):
        lines = path.read_text().splitlines()
        if not lines:
            continue
        header = [part.strip() for part in lines[0].lstrip("# ").split(",")]
        if "KJ_Pa_sqrtm" in header:
            header[header.index("KJ_Pa_sqrtm")] = "KJ_probe_Pa_sqrtm"
        additions = [
            "fatigue_DeltaK_target_Pa_sqrtm",
            "fatigue_Kmax_target_Pa_sqrtm",
            "fatigue_Kmin_target_Pa_sqrtm",
        ]
        if all(name in header for name in additions):
            continue
        header.extend(additions)
        rewritten = [",".join(header)]
        suffix = f",{target_deltaK_pa:.17g},{target_Kmax_pa:.17g},{target_Kmin_pa:.17g}"
        rewritten.extend(line + suffix for line in lines[1:] if line.strip())
        path.write_text("\n".join(rewritten) + "\n")
        steps_rewritten += 1

    inherited_plot = root / "toughness_vs_temperature.png"
    plot_removed = inherited_plot.is_file()
    if plot_removed:
        inherited_plot.unlink()

    return {
        "probe_KJ_at_first_event_MPa_sqrt_m": probe_first,
        "summary_Kc_first_suppressed": summary_rewritten,
        "steps_probe_KJ_columns_rewritten": steps_rewritten,
        "inherited_toughness_plot_removed": plot_removed,
    }


def _write_audit(args: list[str], target_deltaK: float) -> dict:
    out = _option_value(args, "--out")
    if not out:
        return {}
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    R = float(_option_value(args, "--R", "0.1") or 0.1)
    target_Kmax = target_deltaK / (1.0 - R)
    target_Kmin = R * target_Kmax
    semantics = _normalize_output_semantics(root, target_deltaK, R)

    payload = fixed_deltaK_audit_payload()
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
        "target_Kmin_MPa_sqrt_m": float(target_Kmin),
        "R": R,
        "frequency_Hz": float(_option_value(args, "--frequency-Hz", "1000") or 1000.0),
        "fatigue_control_mode": "prescribed_fixed_local_deltaK",
        "fem_loading_mode": "held_nonzero_shape_probe",
        "cyclic_mechanics_enabled": False,
        "full_displacement_feedback_enabled": False,
        "probe_KJ_is_fatigue_driving_K": False,
        "stochastic_geometry_events": len(events),
        "censor_status": "propagated" if events else "right_censored_no_event",
        **semantics,
    })
    target_pa = float(target_deltaK) * 1.0e6
    error_pa = float(payload.get("maximum_abs_target_error_Pa_sqrt_m", float("inf")))
    payload["fixed_deltaK_exact_within_relative_1e-12"] = bool(
        error_pa <= max(1.0e-6, 1.0e-12 * target_pa)
    )
    (root / "v10_2_1_fixed_deltaK_control.json").write_text(
        json.dumps(payload, indent=2)
    )
    return payload


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
    target_Kmax = target_deltaK / (1.0 - R)
    print(
        "  v10.2.1 fixed-DeltaK fatigue: "
        f"DeltaK={target_deltaK:g} MPa*sqrt(m) "
        f"Kmax={target_Kmax:g} MPa*sqrt(m) R={R:g} "
        "control=prescribed_local_K FEM=held_shape_probe"
    )

    # Import the campaign execution stack only when running the entry point.
    # Several inherited modules intentionally patch engine symbols at import
    # time; keeping these imports out of module scope prevents pytest collection
    # from changing unrelated source-model tests.
    from . import sharp_front_v10_2_0 as fatigue

    with install_fixed_deltaK_waveform(target_deltaK):
        with _allow_right_censored_stochastic_summary():
            with _fixed_deltaK_console_semantics(target_deltaK, R):
                result = fatigue.main(args)
    audit = _write_audit(args, target_deltaK)
    print(
        "  fixed-DeltaK semantics: "
        f"DeltaK={target_deltaK:g}, Kmax={target_Kmax:g} MPa*sqrt(m); "
        f"probe_KJ_first={audit.get('probe_KJ_at_first_event_MPa_sqrt_m')} MPa*sqrt(m) "
        "(probe only, not toughness)."
    )
    return result


if __name__ == "__main__":
    main()
