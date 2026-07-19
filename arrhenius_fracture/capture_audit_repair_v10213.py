"""Side-effect-free output repairs for v10.2.13 capture runs."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import statistics

MODEL_ID = "v10.2.13_capture_output_audit_repair"


def repair_capture_audits(root: str | Path) -> None:
    root = Path(root)
    paths = {
        "driver": root / "v10_1_driver_modes.json",
        "source": root / "v10_1_1_source_model.json",
        "transport": root / "v10_1_7_5_transport_mode.json",
    }
    for label, path in paths.items():
        if not path.is_file():
            continue
        payload = json.loads(path.read_text())
        payload.update(
            {
                "v10_2_13_capture_mode": True,
                "active_shielding_enabled": False,
                "wake_shielding": False,
                "capture_unsigned_shielding_disabled": True,
                "manifest_K_shield_cap_enabled": False,
                "campaign_active_shielding_cap_preserved": False,
                "constitutive_K_shield_cap_applied": False,
                "local_strength_sigma_cap_is_not_Kshield_cap": True,
            }
        )
        if label == "source":
            payload["cleavage_shielding_bound"] = "none_capture_shielding_disabled"
        path.write_text(json.dumps(payload, indent=2))


def repair_multitemperature_geometry_summary(root: str | Path) -> dict:
    root = Path(root)
    summary_path = root / "summary.json"
    geometry_path = root / "stochastic_avalanche_geometry_events.json"
    if not summary_path.is_file() or not geometry_path.is_file():
        return {"repaired": False, "reason": "summary_or_geometry_missing"}
    summary = json.loads(summary_path.read_text())
    geometry = json.loads(geometry_path.read_text())
    if not isinstance(summary, list) or not summary:
        raise RuntimeError("summary.json contains no rows")
    if not isinstance(geometry, list) or not geometry:
        return {"repaired": False, "reason": "geometry_events_empty"}

    event_fields = (
        "n_geometry_events",
        "n_equivalent_checkpoints_exact",
        "n_equivalent_checkpoints_rounded",
        "nominal_checkpoint_length_m",
        "geometry_path_length_m",
        "geometry_projected_extension_m",
        "n_advances_semantics",
        "n_geometry_events_semantics",
        "geometry_diagnostics_temperature_K",
    )
    for row in summary:
        for key in event_fields:
            row.pop(key, None)

    run_args_path = root / "run_args.json"
    last_temperature = None
    if run_args_path.is_file():
        run_args = json.loads(run_args_path.read_text())
        temperatures = run_args.get("temperatures", [])
        if temperatures:
            last_temperature = float(temperatures[-1])
    if last_temperature is None:
        last_temperature = float(summary[-1].get("T"))
    matches = [
        row
        for row in summary
        if math.isclose(float(row.get("T")), last_temperature)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "cannot associate final geometry diagnostics with exactly one temperature"
        )
    row = matches[0]
    lengths = [
        max(float(item.get("event_advance_m", 0.0)), 0.0) for item in geometry
    ]
    if not all(math.isfinite(value) and value > 0.0 for value in lengths):
        raise RuntimeError("geometry diagnostics contain a nonpositive event length")
    fixed = [
        float(item.get("requested_fixed_length_m", 0.0))
        for item in geometry
        if math.isfinite(float(item.get("requested_fixed_length_m", 0.0)))
        and float(item.get("requested_fixed_length_m", 0.0)) > 0.0
    ]
    if not fixed:
        raise RuntimeError("geometry diagnostics contain no nominal checkpoint length")
    nominal = float(statistics.median(fixed))
    path_length = float(sum(lengths))
    projected = float(geometry[-1]["x1"]) - float(geometry[0]["x0"])
    row.update(
        {
            "n_geometry_events": len(geometry),
            "n_equivalent_checkpoints_exact": path_length / nominal,
            "n_equivalent_checkpoints_rounded": int(round(path_length / nominal)),
            "nominal_checkpoint_length_m": nominal,
            "geometry_path_length_m": path_length,
            "geometry_projected_extension_m": projected,
            "n_advances_semantics": "rounded_path_length_over_nominal_checkpoint",
            "n_geometry_events_semantics": (
                "accepted_cleavage_renewals_and_geometry_commits"
            ),
            "geometry_diagnostics_temperature_K": last_temperature,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2))

    suffix = f"_{int(round(last_temperature))}K"
    copied = []
    for source_name in (
        "stochastic_avalanche_geometry_events.json",
        "sharp_wake_advance_log.csv",
    ):
        source = root / source_name
        if source.is_file():
            destination = root / f"{source.stem}{suffix}{source.suffix}"
            shutil.copy2(source, destination)
            copied.append(str(destination))
    audit = {
        "schema": MODEL_ID,
        "repaired": True,
        "geometry_diagnostics_temperature_K": last_temperature,
        "earlier_temperature_geometry_diagnostics_available": len(summary) == 1,
        "last_temperature_files": copied,
        "summary_rows": len(summary),
    }
    (root / "v10_2_13_geometry_summary_repair.json").write_text(
        json.dumps(audit, indent=2)
    )
    return audit


__all__ = [
    "MODEL_ID",
    "repair_capture_audits",
    "repair_multitemperature_geometry_summary",
]
