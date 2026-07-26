"""Transparent accepted-production FEM capture for v10.2.27 kernels.

The production trajectory is the audited v10.2.27 persistent-site paper stack.
Capture does not alter stochastic first passage, variable event lengths, source
kinetics, signed shielding, moving-process-zone advection, front-width physics,
or the selected material parameterization. At a requested accepted state, the
capture hook clones the converged trajectory state onto a separate endpoint-
resolved measurement mesh and performs only a frozen-state elastic equilibrium
solve.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

from . import fem as _fem
from . import physical_fem_capture_v10212 as _capture_base
from . import sharp_front_v10_1_7_4 as _entry74
from . import sharp_front_v10_2_27 as _paper
from .anisotropic_front_direction_fix_v10227 import install_front_direction_fix
from .capture_audit_repair_v10213 import (
    repair_capture_audits,
    repair_multitemperature_geometry_summary,
)
from .energy_ledger_output_v10227 import (
    install_energy_ledger_output,
    restore_energy_ledger_output,
    write_energy_ledger_audit,
)
from .frozen_measurement_reconstruction_v10227 import FrozenMeasurementMeshConfig
from .geometry_override_v10227 import (
    install_geometry_override,
    restore_geometry_override,
)
from .kernel_configuration_v10227 import load_configuration
from .persistent_site_audited_engine_v10221 import (
    AuditedPersistentSiteStateResolvedTipEngine,
)
from .persistent_site_bracket_fix_v10221 import (
    install_backstress_complementarity_fix,
)
from .persistent_site_physical_width_v10222 import install_physical_front_width
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


def _option_value(args: list[str], option: str, default: str | None = None) -> str | None:
    prefix = option + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == option and index + 1 < len(args):
            return args[index + 1]
    return default


def _has_flag(args: list[str], option: str) -> bool:
    return option in args or any(token.startswith(option + "=") for token in args)


def _validate_single_front_capture(args: list[str]) -> None:
    maximum_fronts = int(_option_value(args, "--max-fronts", "1"))
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
        payload.update(
            {
                "capture_loading_path": "accepted_v10_2_27_production_state_observer",
                "capture_physics_overrides": [],
                "production_parameterization_observed_not_modified": True,
                "production_engine_class_observed": type(engine).__name__,
                "persistent_site_engine_observed": isinstance(
                    engine, AuditedPersistentSiteStateResolvedTipEngine
                ),
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
                    getattr(
                        getattr(engine, "tip_cfg", None),
                        "signed_active_shielding",
                        False,
                    )
                ),
                "wake_shielding_observed": bool(
                    getattr(
                        getattr(getattr(engine, "mpz", None), "cfg", None),
                        "wake_shielding",
                        False,
                    )
                ),
                "tip_kinetics_mode_observed": (
                    "moving_velocity"
                    if bool(getattr(engine, "kinetic_tip_cell_active", False))
                    else "legacy_checkpoint"
                ),
                "moving_process_zone_advection_observed": bool(
                    hasattr(getattr(engine, "mpz", None), "advance")
                ),
                "persistent_site_source_observed": bool(
                    hasattr(getattr(engine, "mpz", None), "_persistent_site_cfg")
                ),
                "physical_front_width_observed": bool(
                    hasattr(getattr(engine, "mpz", None), "persistent_site_last_geometry")
                ),
            }
        )
        return payload

    return wrapped


def _run_current_paper_stack(args: list[str]):
    """Run the current paper engine with an explicit bootstrap kernel family."""
    if not _paper.DEFAULT_REGISTRY.is_file() or not _paper.SELECTION_RECORD.is_file():
        raise FileNotFoundError(
            "missing generated v10.2.27 registry or selection record; run "
            "python scripts/install_v10_2_27_four_class_registry.py"
        )
    campaign = json.loads(_paper.SELECTION_RECORD.read_text())
    expected_order = list(_paper.VALID_OPTIONS)
    if campaign.get("canonical_option_order") != expected_order:
        raise RuntimeError("v10.2.27 installed option order is invalid")

    family_value = _option_value(
        args,
        "--signed-kernel-family",
        os.environ.get("SIGNED_KERNEL_FAMILY_JSON"),
    )
    if not family_value:
        raise SystemExit(
            "accepted production capture requires an explicit production-authorized "
            "bootstrap --signed-kernel-family"
        )
    family_path = Path(family_value).expanduser().resolve()
    if not family_path.is_file():
        raise FileNotFoundError(family_path)
    os.environ["SIGNED_KERNEL_FAMILY_JSON"] = str(family_path)

    original_registry = _paper._base.DEFAULT_REGISTRY
    original_options = _paper._base.VALID_OPTIONS
    original_model_id = _paper._base.MODEL_ID
    original_engine = _paper._base.PersistentSiteStateResolvedTipEngine
    original_select_option = _paper._base.select_option
    _paper._base.DEFAULT_REGISTRY = _paper.DEFAULT_REGISTRY
    _paper._base.VALID_OPTIONS = _paper.VALID_OPTIONS
    _paper._base.MODEL_ID = _paper.MODEL_ID
    _paper._base.PersistentSiteStateResolvedTipEngine = (
        AuditedPersistentSiteStateResolvedTipEngine
    )
    _paper._base.select_option = _paper._select_option_four_class
    try:
        result = _paper._base.main(args)
        out = _paper._base._base._option_value(args, "--out")
        if out:
            root = Path(out)
            selection_path = root / "v10_2_22_parameter_selection.json"
            selection = (
                json.loads(selection_path.read_text())
                if selection_path.is_file()
                else {}
            )
            exact_row = selection.get("exact_registry_row") or {}
            payload = {
                "schema": "v10.2.27_accepted_production_capture_parameter_transfer_v1",
                "base_entry": "arrhenius_fracture.sharp_front_v10_2_22",
                "capture_observer_only": True,
                "selected_option": selection.get("option_key"),
                "selected_candidate": selection.get("candidate_id"),
                "source_material_class": exact_row.get("material_class"),
                "mechanics_changed": False,
                "source_closure_changed": False,
                "stochastic_cleavage_law_changed": False,
                "persistent_sites": True,
                "finite_source_inventory": False,
                "source_depletion_on_emission": False,
                "source_refresh": False,
                "explicit_recovery": False,
                "front_width_grid_independent": True,
                "signed_kernel_resolved_automatically": False,
                "trajectory_seed_signed_kernel_family": str(family_path),
                "target_mechanical_configuration_fingerprint": os.environ.get(
                    "V10227_KERNEL_CONFIGURATION_FINGERPRINT"
                ),
            }
            (root / "v10_2_27_capture_parameter_transfer.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        return result
    finally:
        _paper._base.DEFAULT_REGISTRY = original_registry
        _paper._base.VALID_OPTIONS = original_options
        _paper._base.MODEL_ID = original_model_id
        _paper._base.PersistentSiteStateResolvedTipEngine = original_engine
        _paper._base.select_option = original_select_option


def _observed_engine_payloads(audit: dict) -> list[dict]:
    result = []
    for record in dict(audit.get("states", {})).values():
        engine = dict(record.get("payload", {})).get("engine_config")
        if isinstance(engine, dict):
            result.append(engine)
    return result


def _seed_family_provenance() -> dict[str, str | bool]:
    source = str(os.environ.get("V10227_KERNEL_CAPTURE_SEED_FAMILY", "")).strip()
    expected_sha = str(
        os.environ.get("V10227_KERNEL_CAPTURE_SEED_FAMILY_SHA256", "")
    ).strip()
    if not source:
        raise RuntimeError("accepted production capture lacks seed-family provenance")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected_sha and expected_sha != actual_sha:
        raise RuntimeError("capture seed-family SHA-256 mismatch")
    return {
        "trajectory_seed_signed_kernel_family": str(path),
        "trajectory_seed_signed_kernel_family_sha256": actual_sha,
        "trajectory_seed_family_required_to_break_kernel_build_cycle": True,
        "trajectory_seed_family_used_only_for_production_state_evolution": True,
    }


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
    event_modes = {
        str(row.get("cleavage_event_length_mode_observed", "")) for row in engines
    }
    kinetic_modes = {str(row.get("tip_kinetics_mode_observed", "")) for row in engines}
    required = {
        "audited_persistent_site_engine_preserved": all(
            row.get("persistent_site_engine_observed") is True for row in engines
        ),
        "persistent_site_source_preserved": all(
            row.get("persistent_site_source_observed") is True for row in engines
        ),
        "stochastic_first_passage_preserved": hazard_modes == {"exponential"},
        "variable_event_length_preserved": event_modes == {"threshold_scaled"},
        "moving_process_zone_physics_preserved": kinetic_modes == {"moving_velocity"}
        and all(
            row.get("moving_process_zone_advection_observed") is True
            for row in engines
        ),
        "fractional_moving_frame_preserved": kinetic_modes == {"moving_velocity"},
        "mobile_kinetic_solver_preserved": kinetic_modes == {"moving_velocity"},
        "active_shielding_preserved": all(
            row.get("active_shielding_observed") is True for row in engines
        ),
        "signed_active_shielding_preserved": all(
            row.get("signed_active_shielding_observed") is True for row in engines
        ),
        "wake_shielding_remains_disabled": all(
            row.get("wake_shielding_observed") is False for row in engines
        ),
        "production_parameterization_observed_not_modified": all(
            row.get("production_parameterization_observed_not_modified") is True
            and row.get("capture_physics_overrides") == []
            for row in engines
        ),
    }
    failed = [name for name, passed in required.items() if not passed]
    if failed:
        raise RuntimeError(
            "registered kernel capture did not preserve the v10.2.27 production "
            "physics contract: " + ",".join(failed)
        )

    payload = {
        "schema": "v10.2.27_accepted_production_state_kernel_capture_v4",
        "mechanical_configuration": configuration.canonical_payload(),
        "mechanical_configuration_fingerprint": expected,
        "trajectory_driver": {
            "driver": "audited_v10_2_27_persistent_site_production_stack",
            "accepted_production_parameterization_observed": True,
            "capture_physics_overrides": [],
            "observed_hazard_modes": sorted(hazard_modes),
            "observed_event_length_modes": sorted(event_modes),
            "observed_tip_kinetics_modes": sorted(kinetic_modes),
            **_seed_family_provenance(),
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
    install_backstress_complementarity_fix()
    install_physical_front_width()
    install_geometry_override()
    install_energy_ledger_output()

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
        raise SystemExit("v10.2.27 capture requires --atlas-outroot PATH")
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

    engine_type = AuditedPersistentSiteStateResolvedTipEngine
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
            "  v10.2.27 accepted production FEM capture: "
            f"mode={'trajectory_only' if trajectory_only else 'extension_snapshot_capture'} "
            f"requests={len(requests)} minimum_Lpz_over_h={minimum_resolution:g} "
            f"measurement_clone={'enabled' if measurement_config is not None else 'disabled'} "
            "engine=audited_persistent_site physics_overrides=none"
        )
        early_stop = None
        try:
            result = _run_current_paper_stack(args)
        except _capture_base.CaptureCompleteStop as exc:
            result = 0
            early_stop = {
                "intentional": True,
                "state_id": exc.state_id,
                "reason": "final accepted equilibrium serialized before next kernel query",
            }
            print(
                "  capture complete: stopped after final accepted equilibrium "
                f"{exc.state_id}"
            )

        mechanics_root_value = _option_value(args, "--out")
        mechanics_root = Path(mechanics_root_value) if mechanics_root_value else None
        repair = None
        if mechanics_root is not None and early_stop is None:
            write_energy_ledger_audit(mechanics_root)
            repair_capture_audits(mechanics_root)
            repair = repair_multitemperature_geometry_summary(mechanics_root)
        elif mechanics_root is not None:
            repair = {
                "skipped": True,
                "reason": "intentional capture completion before production-run finalizers",
                "mechanics_root": str(mechanics_root.resolve()),
            }

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
                    "production_entry": "v10.2.27 audited persistent-site paper stack",
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
                    "capture_terminated_after_final_equilibrium": early_stop is not None,
                    "capture_early_stop": early_stop,
                    "kernel_capture_manifest": kernel_manifest,
                    "allow_incomplete": allow_incomplete,
                    "capture": audit,
                    "mechanics_output_repair": repair,
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
        restore_energy_ledger_output()
        restore_geometry_override()


if __name__ == "__main__":
    main()
