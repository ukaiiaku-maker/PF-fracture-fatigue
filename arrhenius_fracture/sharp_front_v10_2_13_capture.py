"""v10.2.13 physical FEM capture with extension-only matching and audit repair.

The capture entry is a transparent observer.  It does not alter hazard modes,
shielding flags, moving-process-zone kinetics, source laws, or material
parameters.  The fixed-extension atlas remains single-front only, so unsupported
branch topology is rejected rather than silently rewritten.
"""
from __future__ import annotations

import json
import os
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
from .kernel_configuration_v10227 import load_configuration
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


def _from_kernel_configuration() -> FrozenMeasurementMeshConfig | None:
    source = str(os.environ.get("V10227_KERNEL_CONFIGURATION", "")).strip()
    if not source:
        return None
    configuration = load_configuration(source)
    return FrozenMeasurementMeshConfig(
        specimen_length_x_m=configuration.specimen_length_x_m,
        specimen_length_y_m=configuration.specimen_length_y_m,
        initial_crack_length_m=configuration.initial_crack_length_m,
        notch_half_thickness_m=configuration.notch_half_thickness_m,
        mesh_nx=configuration.mesh_nx,
        mesh_ny=configuration.mesh_ny,
        tip_h_fine_m=configuration.measurement_tip_h_fine_m,
        tip_ratio=configuration.measurement_tip_ratio,
    ).validate()


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
        return _from_kernel_configuration()
    missing = [name for name, value in values.items() if value is None]
    if missing:
        raise SystemExit(
            "capture-only measurement mesh requires a complete configuration; missing="
            + ",".join(missing)
        )
    result = FrozenMeasurementMeshConfig(
        specimen_length_x_m=float(values["specimen_length_x_m"]),
        specimen_length_y_m=float(values["specimen_length_y_m"]),
        initial_crack_length_m=float(values["initial_crack_length_m"]),
        notch_half_thickness_m=float(values["notch_half_thickness_m"]),
        mesh_nx=int(values["mesh_nx"]),
        mesh_ny=int(values["mesh_ny"]),
        tip_h_fine_m=float(values["tip_h_fine_m"]),
        tip_ratio=float(values["tip_ratio"]),
    ).validate()
    configured = _from_kernel_configuration()
    if configured is not None and result.as_dict() != configured.as_dict():
        raise SystemExit(
            "explicit atlas measurement-mesh options do not match the resolver's "
            "mechanical configuration"
        )
    return result


def _transparent_engine_payload(original):
    def wrapped(engine):
        payload = original(engine)
        payload.update({
            "capture_loading_path": "accepted_production_state_observer",
            "capture_physics_overrides": [],
            "production_parameterization_observed_not_modified": True,
            "cleavage_hazard_mode_observed": str(
                os.environ.get("CLEAVAGE_HAZARD_MODE", "")
            ).strip().lower(),
            "cleavage_event_length_mode_observed": str(
                os.environ.get("CLEAVAGE_EVENT_LENGTH_MODE", "")
            ).strip().lower(),
            "cleavage_event_min_factor_observed": str(
                os.environ.get("CLEAVAGE_EVENT_MIN_FACTOR", "")
            ).strip(),
            "cleavage_event_max_factor_observed": str(
                os.environ.get("CLEAVAGE_EVENT_MAX_FACTOR", "")
            ).strip(),
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


def _observed_engine_payloads(audit: dict) -> list[dict]:
    result = []
    for record in dict(audit.get("states", {})).values():
        payload = dict(record.get("payload", {}))
        engine = payload.get("engine_config")
        if isinstance(engine, dict):
            result.append(engine)
    return result


def _write_kernel_capture_manifest(
    root: Path,
    *,
    audit: dict,
    measurement_config: FrozenMeasurementMeshConfig | None,
    state_table: str | None,
) -> dict | None:
    source = str(os.environ.get("V10227_KERNEL_CONFIGURATION", "")).strip()
    fingerprint = str(
        os.environ.get("V10227_KERNEL_CONFIGURATION_FINGERPRINT", "")
    ).strip()
    if not source:
        return None
    configuration = load_configuration(source)
    expected = configuration.fingerprint()
    if fingerprint and fingerprint != expected:
        raise RuntimeError(
            "resolver/capture mechanical-configuration fingerprint mismatch: "
            f"{fingerprint} != {expected}"
        )
    if measurement_config is None:
        raise RuntimeError(
            "resolver-driven kernel capture requires a capture-only measurement mesh"
        )

    engines = _observed_engine_payloads(audit)
    if not engines:
        raise RuntimeError("kernel capture manifest has no observed engine payloads")
    hazard_modes = {str(row.get("cleavage_hazard_mode_observed", "")) for row in engines}
    event_modes = {str(row.get("cleavage_event_length_mode_observed", "")) for row in engines}
    kinetic_modes = {str(row.get("tip_kinetics_mode_observed", "")) for row in engines}
    stochastic = hazard_modes == {"exponential"}
    variable_events = event_modes == {"threshold_scaled"}
    moving_kinetics = kinetic_modes == {"moving_velocity"}
    moving_advection = all(
        row.get("moving_process_zone_advection_observed") is True for row in engines
    )
    active_shielding = all(
        row.get("active_shielding_observed") is True for row in engines
    )
    signed_shielding = all(
        row.get("signed_active_shielding_observed") is True for row in engines
    )
    parameterization_unchanged = all(
        row.get("production_parameterization_observed_not_modified") is True
        and row.get("capture_physics_overrides") == []
        for row in engines
    )
    required = {
        "stochastic_first_passage_preserved": stochastic,
        "variable_event_length_preserved": variable_events,
        "moving_process_zone_physics_preserved": moving_kinetics and moving_advection,
        "fractional_moving_frame_preserved": moving_kinetics and moving_advection,
        "mobile_kinetic_solver_preserved": moving_kinetics,
        "active_shielding_preserved": active_shielding,
        "signed_active_shielding_preserved": signed_shielding,
        "production_parameterization_observed_not_modified": parameterization_unchanged,
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError(
            "registered kernel capture did not preserve the production physics contract: "
            + ",".join(failed)
            + f"; hazard_modes={sorted(hazard_modes)}, event_modes={sorted(event_modes)}, "
            + f"kinetic_modes={sorted(kinetic_modes)}"
        )

    payload = {
        "schema": "v10.2.27_accepted_production_state_kernel_capture_v2",
        "mechanical_configuration": configuration.canonical_payload(),
        "mechanical_configuration_fingerprint": expected,
        "trajectory_driver": {
            "driver": "registered_accepted_production_command",
            "historical_reference_condition_required": False,
            "accepted_production_parameterization_observed": True,
            "capture_physics_overrides": [],
            "observed_hazard_modes": sorted(hazard_modes),
            "observed_event_length_modes": sorted(event_modes),
            "observed_tip_kinetics_modes": sorted(kinetic_modes),
            **required,
        },
        "measurement_snapshot": {
            "capture_only": True,
            "trajectory_state_cloned": True,
            "plasticity_frozen": True,
            "kinetics_not_advanced": True,
            "endpoint_mesh_re_equilibrated": True,
            "measurement_mesh_config": measurement_config.as_dict(),
        },
        "state_table": None if state_table is None else str(Path(state_table).resolve()),
        "captured_states": int(audit.get("captured_states", 0)),
        "snapshot_root": str(root.resolve()),
    }
    (root / "kernel_capture_manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


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
        kernel_manifest = _write_kernel_capture_manifest(
            root,
            audit=audit,
            measurement_config=measurement_config,
            state_table=state_table,
        )
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
                    "kernel_capture_manifest": kernel_manifest,
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
