"""v10.2.13 physical FEM capture with extension-only matching and audit repair."""
from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
import statistics
import sys

from . import anisotropic_emission_v10174 as _anisotropic
from . import fem as _fem
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_1_7_5 as _transport
from .physical_fem_capture_v10213 import (
    MODEL_ID,
    PhysicalFEMCapture,
    load_extension_capture_requests,
)


def _pop_value(args: list[str], option: str) -> str | None:
    prefix = option + "="
    for index, token in enumerate(list(args)):
        if token.startswith(prefix):
            value = token[len(prefix):]
            del args[index]
            return value
        if token == option:
            if index + 1 >= len(args):
                raise SystemExit(f"{option} requires a value")
            value = args[index + 1]
            del args[index:index + 2]
            return value
    return None


def _option_value(args: list[str], option: str) -> str | None:
    prefix = option + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == option and index + 1 < len(args):
            return args[index + 1]
    return None


def _remove_option(args: list[str], option: str, takes_value: bool = False) -> None:
    prefix = option + "="
    index = 0
    while index < len(args):
        token = args[index]
        if token.startswith(prefix):
            del args[index]
            continue
        if token == option:
            del args[index]
            if takes_value and index < len(args):
                del args[index]
            continue
        index += 1


def _force_capture_modes(args: list[str]) -> None:
    for option in (
        "--active-shielding",
        "--no-active-shielding",
        "--wake-shielding",
        "--no-wake-shielding",
        "--crystal-branch",
    ):
        _remove_option(args, option)
    _remove_option(args, "--max-fronts", takes_value=True)
    args.extend(["--no-active-shielding", "--no-wake-shielding", "--max-fronts", "1"])


def _repair_capture_audits(root: Path) -> None:
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


def _repair_multitemperature_geometry_summary(root: Path) -> dict:
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
    matches = [row for row in summary if math.isclose(float(row.get("T")), last_temperature)]
    if len(matches) != 1:
        raise RuntimeError(
            "cannot associate final geometry diagnostics with exactly one temperature"
        )
    row = matches[0]
    lengths = [max(float(item.get("event_advance_m", 0.0)), 0.0) for item in geometry]
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
            "n_geometry_events_semantics": "accepted_cleavage_renewals_and_geometry_commits",
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
        "schema": "v10.2.13_multitemperature_geometry_summary_repair",
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


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    state_table = _pop_value(args, "--atlas-state-table")
    outroot = _pop_value(args, "--atlas-outroot")
    minimum_resolution = float(
        _pop_value(args, "--minimum-elements-per-process-zone") or 3.0
    )
    trajectory_only = "--atlas-trajectory-only" in args
    if trajectory_only:
        args.remove("--atlas-trajectory-only")
    allow_incomplete = "--allow-incomplete-atlas-capture" in args
    if allow_incomplete:
        args.remove("--allow-incomplete-atlas-capture")
    if not outroot:
        raise SystemExit("v10.2.13 capture requires --atlas-outroot PATH")
    if trajectory_only:
        if state_table:
            raise SystemExit(
                "--atlas-trajectory-only must not be combined with --atlas-state-table"
            )
        requests = []
        allow_incomplete = True
    else:
        if not state_table:
            raise SystemExit(
                "snapshot capture requires --atlas-state-table PATH; use "
                "--atlas-trajectory-only for discovery"
            )
        requests = load_extension_capture_requests(state_table)
    capture = PhysicalFEMCapture(
        requests,
        outroot,
        minimum_elements_per_process_zone=minimum_resolution,
    )
    _force_capture_modes(args)

    engine_type = _anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_step = engine_type.step
    original_factory = _entry74.wrap_assemble_mechanics
    original_solve = _fem.solve_dirichlet
    _entry74.wrap_assemble_mechanics = capture.wrap_assemble_factory(original_factory)
    _fem.solve_dirichlet = capture.wrap_solve_dirichlet(original_solve)
    engine_type.step = capture.wrap_engine_step(original_step)
    try:
        print(
            "  v10.2.13 physical FEM atlas capture: "
            f"mode={'trajectory_only' if trajectory_only else 'extension_snapshot_capture'} "
            f"requests={len(requests)} unsigned_shielding=disabled "
            f"minimum_Lpz_over_h={minimum_resolution:g} "
            "opening=validation_only parameterization=blocked"
        )
        result = _transport.main(args)
        mechanics_root_value = _option_value(args, "--out")
        mechanics_root = Path(mechanics_root_value) if mechanics_root_value else None
        repair = None
        if mechanics_root is not None:
            _repair_capture_audits(mechanics_root)
            repair = _repair_multitemperature_geometry_summary(mechanics_root)
        audit = capture.finalize(require_complete=not allow_incomplete)
        root = Path(outroot)
        (root / "v10_2_13_capture_entry.json").write_text(
            json.dumps(
                {
                    "schema": MODEL_ID,
                    "capture_mode": (
                        "trajectory_only" if trajectory_only else "extension_snapshot_capture"
                    ),
                    "state_table": (
                        str(Path(state_table).resolve()) if state_table else None
                    ),
                    "atlas_outroot": str(root.resolve()),
                    "minimum_elements_per_process_zone": minimum_resolution,
                    "allow_incomplete": allow_incomplete,
                    "capture": audit,
                    "mechanics_output_repair": repair,
                    "next_step": (
                        "select frozen crack-path extensions and validate load invariance"
                        if trajectory_only
                        else "run scripts/evaluate_v10_2_13_frozen_geometry_load_invariance.py for every snapshot"
                    ),
                },
                indent=2,
            )
        )
        return result
    finally:
        engine_type.step = original_step
        _entry74.wrap_assemble_mechanics = original_factory
        _fem.solve_dirichlet = original_solve


if __name__ == "__main__":
    main()
