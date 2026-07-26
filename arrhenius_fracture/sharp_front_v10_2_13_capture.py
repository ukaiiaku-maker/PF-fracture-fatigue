"""v10.2.13 physical FEM capture with extension-only matching and audit repair.

The capture entry is a transparent observer.  It does not alter hazard modes,
shielding flags, moving-process-zone kinetics, source laws, or material
parameters.  The fixed-extension atlas remains single-front only, so unsupported
branch topology is rejected rather than silently rewritten.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import anisotropic_emission_v10174 as _anisotropic
from . import fem as _fem
from . import physical_fem_capture_v10212 as _capture_base
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_1_7_5 as _transport
from .anisotropic_front_direction_fix_v10227 import install_front_direction_fix
from .capture_audit_repair_v10213 import (
    repair_capture_audits,
    repair_multitemperature_geometry_summary,
)
from .frozen_measurement_reconstruction_v10227 import FrozenMeasurementMeshConfig
from .geometry_override_v10227 import (
    install_geometry_override,
    restore_geometry_override,
)
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


def _has_flag(args: list[str], option: str) -> bool:
    return option in args or any(token.startswith(option + "=") for token in args)


def _validate_single_front_capture(args: list[str]) -> None:
    max_fronts_raw = _option_value(args, "--max-fronts")
    maximum_fronts = 1 if max_fronts_raw is None else int(max_fronts_raw)
    if _has_flag(args, "--crystal-branch") or maximum_fronts != 1:
        raise SystemExit(
            "extension-indexed signed-kernel capture is single-front only; "
            "branching requires a topology_cached or direct_fem provider"
        )


def _measurement_mesh_config(args: list[str]) -> FrozenMeasurementMeshConfig | None:
    names = {
        "specimen_length_x_m": "--atlas-specimen-length-x",
        "specimen_length_y_m": "--atlas-specimen-length-y",
        "initial_crack_length_m": "--atlas-initial-crack-length",
        "notch_half_thickness_m": "--atlas-notch-half-thickness",
        "mesh_nx": "--atlas-measurement-mesh-nx",
        "mesh_ny": "--atlas-measurement-mesh-ny",
        "tip_h_fine_m": "--atlas-measurement-tip-h-fine",
        "tip_ratio": "--atlas-measurement-tip-ratio",
    }
    values = {name: _pop_value(args, option) for name, option in names.items()}
    supplied = [name for name, value in values.items() if value is not None]
    if not supplied:
        return None
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise SystemExit(
            "capture-only measurement mesh requires a complete configuration; missing="
            + ",".join(missing)
        )
    return FrozenMeasurementMeshConfig(
        specimen_length_x_m=float(values["specimen_length_x_m"]),
        specimen_length_y_m=float(values["specimen_length_y_m"]),
        initial_crack_length_m=float(values["initial_crack_length_m"]),
        notch_half_thickness_m=float(values["notch_half_thickness_m"]),
        mesh_nx=int(values["mesh_nx"]),
        mesh_ny=int(values["mesh_ny"]),
        tip_h_fine_m=float(values["tip_h_fine_m"]),
        tip_ratio=float(values["tip_ratio"]),
    ).validate()


def _transparent_engine_payload(original):
    def wrapped(engine):
        payload = original(engine)
        payload.update({
            "capture_loading_path": "accepted_production_state_observer",
            "capture_physics_overrides": [],
            "production_parameterization_observed_not_modified": True,
            "active_shielding_observed": bool(
                getattr(getattr(engine, "tip_cfg", None), "active_shielding", False)
            ),
            "signed_active_shielding_observed": bool(
                getattr(getattr(engine, "tip_cfg", None), "signed_active_shielding", False)
            ),
            "wake_shielding_observed": bool(
                getattr(getattr(getattr(engine, "mpz", None), "cfg", None), "wake_shielding", False)
            ),
            "tip_kinetics_mode_observed": (
                "moving_velocity"
                if bool(getattr(engine, "kinetic_tip_cell_active", False))
                else "legacy_checkpoint"
            ),
            "moving_process_zone_advection_observed": bool(
                hasattr(getattr(engine, "mpz", None), "advance")
            ),
        })
        return payload

    return wrapped


def main(argv=None):
    install_front_direction_fix()
    install_geometry_override()
    args = list(sys.argv[1:] if argv is None else argv)
    state_table = _pop_value(args, "--atlas-state-table")
    outroot = _pop_value(args, "--atlas-outroot")
    minimum_resolution = float(
        _pop_value(args, "--minimum-elements-per-process-zone") or 3.0
    )
    measurement_config = _measurement_mesh_config(args)
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

    _validate_single_front_capture(args)
    capture = PhysicalFEMCapture(
        requests,
        outroot,
        minimum_elements_per_process_zone=minimum_resolution,
        measurement_mesh_config=measurement_config,
    )

    engine_type = _anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_step = engine_type.step
    original_factory = _entry74.wrap_assemble_mechanics
    original_solve = _fem.solve_dirichlet
    original_engine_payload = _capture_base._engine_payload
    _entry74.wrap_assemble_mechanics = capture.wrap_assemble_factory(original_factory)
    _fem.solve_dirichlet = capture.wrap_solve_dirichlet(original_solve)
    engine_type.step = capture.wrap_engine_step(original_step)
    _capture_base._engine_payload = _transparent_engine_payload(original_engine_payload)
    try:
        print(
            "  v10.2.27 physical FEM atlas capture: "
            f"mode={'trajectory_only' if trajectory_only else 'extension_snapshot_capture'} "
            f"requests={len(requests)} minimum_Lpz_over_h={minimum_resolution:g} "
            f"measurement_clone={'enabled' if measurement_config is not None else 'disabled'} "
            "physics_overrides=none production_kinetics=unchanged "
            "parameterization=observed_not_modified"
        )
        result = _transport.main(args)
        mechanics_root_value = _option_value(args, "--out")
        mechanics_root = Path(mechanics_root_value) if mechanics_root_value else None
        repair = None
        if mechanics_root is not None:
            repair_capture_audits(mechanics_root)
            repair = repair_multitemperature_geometry_summary(mechanics_root)
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
                    "measurement_mesh_config": (
                        None if measurement_config is None else measurement_config.as_dict()
                    ),
                    "capture_physics_overrides": [],
                    "production_parameterization_observed_not_modified": True,
                    "production_moving_process_zone_physics_preserved": True,
                    "measurement_reconstruction_is_capture_only": True,
                    "allow_incomplete": allow_incomplete,
                    "capture": audit,
                    "mechanics_output_repair": repair,
                    "next_step": (
                        "select frozen crack-path extensions and validate load invariance"
                        if trajectory_only
                        else "run load-invariance evaluation for every snapshot"
                    ),
                },
                indent=2,
            )
        )
        return result
    finally:
        _capture_base._engine_payload = original_engine_payload
        engine_type.step = original_step
        _entry74.wrap_assemble_mechanics = original_factory
        _fem.solve_dirichlet = original_solve
        restore_geometry_override()


if __name__ == "__main__":
    main()
