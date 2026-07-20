"""Permit valid zero-event monotonic runs in the v10.1.7.3 summary adapter.

A short initialization run, or a genuinely right-censored production run, may
complete with no cleavage renewal and therefore an empty geometry-event list.
That is a physical result, not a post-processing failure. Non-empty event lists
continue to use the original strict event-semantics implementation.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from . import sharp_front_v10_1_7_3 as _entry73

_ORIGINAL_REWRITE = _entry73._rewrite_summary_event_semantics


def _rewrite_summary_event_semantics_allow_zero(args: list[str]) -> None:
    out = _entry73._option_value(args, "--out")
    if not out:
        return
    root = Path(out)
    geometry_path = root / "stochastic_avalanche_geometry_events.json"
    summary_path = root / "summary.json"
    if not geometry_path.is_file() or not summary_path.is_file():
        return

    geometry = json.loads(geometry_path.read_text())
    summary = json.loads(summary_path.read_text())
    if not isinstance(geometry, list):
        raise RuntimeError("stochastic avalanche geometry diagnostics must be a list")
    if not isinstance(summary, list) or not summary:
        raise RuntimeError("summary.json contains no rows")
    if geometry:
        _ORIGINAL_REWRITE(args)
        return

    da_value = _entry73._option_value(args, "--da-phys")
    if da_value is None:
        raise RuntimeError("zero-event summary requires --da-phys")
    nominal_checkpoint_m = float(da_value)
    if not math.isfinite(nominal_checkpoint_m) or nominal_checkpoint_m <= 0.0:
        raise RuntimeError("zero-event summary requires positive finite --da-phys")

    row = summary[0]
    row.update(
        {
            "n_geometry_events": 0,
            "n_equivalent_checkpoints_exact": 0.0,
            "n_equivalent_checkpoints_rounded": 0,
            "nominal_checkpoint_length_m": nominal_checkpoint_m,
            "geometry_path_length_m": 0.0,
            "geometry_projected_extension_m": 0.0,
            "n_advances_semantics": "rounded_path_length_over_nominal_checkpoint",
            "n_geometry_events_semantics": (
                "accepted_cleavage_renewals_and_geometry_commits"
            ),
            "geometry_event_status": "no_accepted_events",
            "zero_event_run_is_valid": True,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))


def install_zero_event_summary_support() -> None:
    if _entry73._rewrite_summary_event_semantics is not _rewrite_summary_event_semantics_allow_zero:
        _entry73._rewrite_summary_event_semantics = _rewrite_summary_event_semantics_allow_zero


install_zero_event_summary_support()

__all__ = ["install_zero_event_summary_support"]
