"""Fail-closed workflow utilities for v10.2.10 mechanics artifacts.

This module does not fabricate FEM responses.  It defines the state-request and
review contracts around the existing signed interaction-integral and tensor-probe
builders so that raw physical data, authorization, and parameterization readiness
remain separately auditable.
"""
from __future__ import annotations

from dataclasses import dataclass
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

MODEL_ID = "v10.2.10_physical_artifact_workflow"
STATE_AXES = (
    "r_eff_over_r0",
    "opening_strength_fraction",
    "crack_extension_m",
)
STATE_TABLE_COLUMNS = ("state_id", *STATE_AXES, "engine_template")
SIGNED_RESPONSE_COLUMNS = (
    "state_id",
    *STATE_AXES,
    "region",
    "system",
    "bin",
    "x_m",
    "burgers_sign",
    "delta_signed_line_content",
    "K_I_base_Pa_sqrt_m",
    "K_I_perturbed_Pa_sqrt_m",
    "K_II_base_Pa_sqrt_m",
    "K_II_perturbed_Pa_sqrt_m",
)
TENSOR_RESPONSE_COLUMNS = (
    "state_id",
    *STATE_AXES,
    "system",
    "sigma_local_Pa",
    "tau_signed_Pa",
    "probe_reliable",
)
MECHANICAL_NORMALIZATION_SOURCES = {
    "2d_unit_slip_to_line_content",
    "plastic_distortion_burgers_integral",
    "process_zone_geometry_and_line_spacing",
    "front_thickness_source_geometry",
}
REVIEW_CHECKS = (
    "signed_linearity_passed",
    "multi_amplitude_linearity_passed",
    "interaction_integral_contour_stability_passed",
    "mesh_and_ribbon_convergence_passed",
    "state_envelope_coverage_passed",
    "source_normalization_reviewed",
    "tensor_probe_repeatability_passed",
    "replay_gate_passed",
)


@dataclass(frozen=True)
class MechanicalStateRequest:
    state_id: str
    r_eff_over_r0: float
    opening_strength_fraction: float
    crack_extension_m: float
    engine_template: Path

    def coordinates(self) -> tuple[float, float, float]:
        return (
            float(self.r_eff_over_r0),
            float(self.opening_strength_fraction),
            float(self.crack_extension_m),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id,
            "r_eff_over_r0": float(self.r_eff_over_r0),
            "opening_strength_fraction": float(self.opening_strength_fraction),
            "crack_extension_m": float(self.crack_extension_m),
            "engine_template": str(self.engine_template),
        }


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"non-finite {name}")
    return result


def inspect_engine_template(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"engine template is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"engine template must contain a JSON object: {path}")
    required = ("front_config", "mpz_config", "tip_config", "anisotropic_config")
    missing = [name for name in required if not isinstance(payload.get(name), dict)]
    if missing:
        raise ValueError(f"engine template {path} lacks dictionary fields {missing}")
    return {
        "path": str(path.resolve()),
        "sigma_cap": payload["front_config"].get("sigma_cap"),
        "mpz_bins": payload["mpz_config"].get("n_bins"),
        "transport_mode": payload.get("transport_mode"),
        "state_class": payload.get("state_class"),
        "engine_class": payload.get("engine_class"),
    }


def load_state_requests(path: str | Path) -> list[MechanicalStateRequest]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"state request table is missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("state request table is empty")
    missing = sorted(set(STATE_TABLE_COLUMNS).difference(rows[0]))
    if missing:
        raise ValueError(f"state request table is missing columns {missing}")
    result: list[MechanicalStateRequest] = []
    seen: set[str] = set()
    for row in rows:
        state_id = str(row["state_id"]).strip()
        if not state_id or state_id in seen:
            raise ValueError(f"state IDs must be nonempty and unique: {state_id!r}")
        seen.add(state_id)
        r_ratio = _finite_float(row["r_eff_over_r0"], "r_eff_over_r0")
        opening = _finite_float(
            row["opening_strength_fraction"], "opening_strength_fraction"
        )
        extension = _finite_float(row["crack_extension_m"], "crack_extension_m")
        if r_ratio <= 0.0:
            raise ValueError("r_eff_over_r0 must be positive")
        if not 0.0 < opening <= 1.0:
            raise ValueError("opening_strength_fraction must lie in (0, 1]")
        if extension < 0.0:
            raise ValueError("crack_extension_m must be non-negative")
        engine = Path(row["engine_template"]).expanduser()
        inspect_engine_template(engine)
        result.append(
            MechanicalStateRequest(
                state_id=state_id,
                r_eff_over_r0=r_ratio,
                opening_strength_fraction=opening,
                crack_extension_m=extension,
                engine_template=engine.resolve(),
            )
        )
    return result


def state_coverage(
    states: Iterable[MechanicalStateRequest],
    *,
    minimum_distinct_r: int = 2,
    minimum_distinct_opening: int = 3,
    minimum_distinct_extension: int = 2,
) -> dict[str, Any]:
    states = list(states)
    distinct = {
        axis: len(
            {
                round(float(getattr(state, axis)), 14)
                for state in states
            }
        )
        for axis in STATE_AXES
    }
    required = {
        "r_eff_over_r0": int(minimum_distinct_r),
        "opening_strength_fraction": int(minimum_distinct_opening),
        "crack_extension_m": int(minimum_distinct_extension),
    }
    return {
        "distinct_values": distinct,
        "required_distinct_values": required,
        "coverage_passed": all(distinct[key] >= required[key] for key in STATE_AXES),
    }


def write_collection_plan(
    states: Iterable[MechanicalStateRequest],
    *,
    outroot: str | Path,
    n_systems: int,
    active_bins: int,
    wake_bins: int,
    perturbation_magnitudes: Iterable[float],
) -> dict[str, Any]:
    states = list(states)
    if n_systems < 1 or active_bins < 1 or wake_bins < 0:
        raise ValueError("system/bin counts are invalid")
    magnitudes = sorted({_finite_float(value, "perturbation magnitude") for value in perturbation_magnitudes})
    if len(magnitudes) < 2 or any(value <= 0.0 for value in magnitudes):
        raise ValueError("at least two positive perturbation magnitudes are required")
    coverage = state_coverage(states)
    outroot = Path(outroot)
    if outroot.exists():
        raise FileExistsError(f"refusing to overwrite output: {outroot}")
    outroot.mkdir(parents=True)

    jobs_path = outroot / "mechanical_state_jobs.csv"
    fields = [
        "job_id",
        *STATE_TABLE_COLUMNS,
        "job_type",
        "region",
        "system",
        "bin",
        "burgers_sign",
        "requested_abs_line_content",
        "status",
    ]
    rows: list[dict[str, Any]] = []
    for state in states:
        base = state.as_dict()
        for system in range(int(n_systems)):
            rows.append(
                {
                    "job_id": f"{state.state_id}:tensor:s{system}",
                    **base,
                    "job_type": "tensor_probe",
                    "region": "",
                    "system": system,
                    "bin": "",
                    "burgers_sign": "",
                    "requested_abs_line_content": "",
                    "status": "pending_physical_fem",
                }
            )
            for region, n_bins in (("active", active_bins), ("wake", wake_bins)):
                for bin_index in range(int(n_bins)):
                    for sign in (-1, 1):
                        for magnitude in magnitudes:
                            rows.append(
                                {
                                    "job_id": (
                                        f"{state.state_id}:slip:{region}:s{system}:"
                                        f"b{bin_index}:{sign:+d}:{magnitude:g}"
                                    ),
                                    **base,
                                    "job_type": "signed_slip_perturbation",
                                    "region": region,
                                    "system": system,
                                    "bin": bin_index,
                                    "burgers_sign": sign,
                                    "requested_abs_line_content": magnitude,
                                    "status": "pending_physical_fem",
                                }
                            )
    with jobs_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    signed_template = outroot / "signed_interaction_integral_responses.csv"
    tensor_template = outroot / "tensor_probe_responses.csv"
    for path, columns in (
        (signed_template, SIGNED_RESPONSE_COLUMNS),
        (tensor_template, TENSOR_RESPONSE_COLUMNS),
    ):
        with path.open("w", newline="") as handle:
            csv.writer(handle).writerow(columns)

    engine_templates = {
        str(state.engine_template): inspect_engine_template(state.engine_template)
        for state in states
    }
    payload = {
        "schema": MODEL_ID,
        "stage": "collection_plan",
        "physical_fem_responses_generated": False,
        "automatic_authorization": False,
        "n_states": len(states),
        "n_systems": int(n_systems),
        "active_bins": int(active_bins),
        "wake_bins": int(wake_bins),
        "perturbation_magnitudes": magnitudes,
        "state_coverage": coverage,
        "engine_templates": list(engine_templates.values()),
        "job_manifest": str(jobs_path),
        "signed_response_template": str(signed_template),
        "tensor_response_template": str(tensor_template),
        "required_next_step": (
            "run the physical FEM states and replace the header-only response "
            "templates with measured signed interaction-integral and tensor-probe rows"
        ),
    }
    (outroot / "physical_artifact_plan.json").write_text(json.dumps(payload, indent=2))
    return payload


def _read_csv(path: str | Path, required: Iterable[str], label: str) -> list[dict[str, str]]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"{label} is empty")
    missing = sorted(set(required).difference(rows[0]))
    if missing:
        raise ValueError(f"{label} is missing columns {missing}")
    return rows


def inspect_raw_artifacts(
    *,
    signed_responses: str | Path,
    tensor_responses: str | Path,
    normalization: str | Path,
) -> dict[str, Any]:
    signed = _read_csv(signed_responses, SIGNED_RESPONSE_COLUMNS, "signed response table")
    tensor = _read_csv(tensor_responses, TENSOR_RESPONSE_COLUMNS, "tensor response table")
    normalization_path = Path(normalization)
    if not normalization_path.is_file():
        raise FileNotFoundError(f"normalization artifact is missing: {normalization_path}")
    norm = json.loads(normalization_path.read_text())
    if not isinstance(norm, dict):
        raise ValueError("normalization artifact must contain a JSON object")
    required_norm = {
        "normalization_source",
        "activation_to_line_content_by_system",
        "source_capacity_bounds_per_system",
    }
    missing = sorted(required_norm.difference(norm))
    if missing:
        raise ValueError(f"normalization artifact is missing {missing}")
    if norm["normalization_source"] not in MECHANICAL_NORMALIZATION_SOURCES:
        raise ValueError("normalization source is not mechanically authorized")
    if bool(norm.get("fitted_to_toughness_or_fatigue", False)):
        raise ValueError("normalization must not be fitted to fracture/fatigue output")

    signed_states = {row["state_id"] for row in signed}
    tensor_states = {row["state_id"] for row in tensor}
    if signed_states != tensor_states:
        raise ValueError("signed and tensor response tables have different state sets")
    coordinates: dict[str, tuple[float, float, float]] = {}
    for row in signed + tensor:
        state_id = row["state_id"]
        coord = tuple(_finite_float(row[name], name) for name in STATE_AXES)
        old = coordinates.setdefault(state_id, coord)
        if old != coord:
            raise ValueError(f"state coordinates differ for {state_id}")
    distinct = {
        axis: len({round(coord[index], 14) for coord in coordinates.values()})
        for index, axis in enumerate(STATE_AXES)
    }
    return {
        "schema": MODEL_ID,
        "stage": "raw_artifact_preflight",
        "signed_rows": len(signed),
        "tensor_rows": len(tensor),
        "state_count": len(coordinates),
        "distinct_state_coordinates": distinct,
        "normalization_source": norm["normalization_source"],
        "raw_inputs_present": True,
        "production_parameterization_allowed": False,
    }


def validate_review_approval(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"independent review approval is missing: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("review approval must contain a JSON object")
    if payload.get("schema") != "v10.2.10_independent_mechanics_review":
        raise ValueError("review approval has the wrong schema")
    missing = [name for name in REVIEW_CHECKS if payload.get(name) is not True]
    if missing:
        raise ValueError(f"review approval has incomplete checks: {missing}")
    reviewer = str(payload.get("reviewer", "")).strip()
    reviewed_utc = str(payload.get("reviewed_utc", "")).strip()
    if not reviewer or not reviewed_utc:
        raise ValueError("review approval requires reviewer and reviewed_utc")
    return payload


def readiness_report(
    *,
    kernel_family: str | Path,
    drive_family: str | Path,
    engine_template: str | Path,
) -> dict[str, Any]:
    kernel_path = Path(kernel_family)
    drive_path = Path(drive_family)
    engine = inspect_engine_template(engine_template)
    if not kernel_path.is_file() or not drive_path.is_file():
        missing = [str(path) for path in (kernel_path, drive_path) if not path.is_file()]
        raise FileNotFoundError(f"mechanical artifact files are missing: {missing}")
    kernel = json.loads(kernel_path.read_text())
    drive = json.loads(drive_path.read_text())
    if not isinstance(kernel, dict) or not isinstance(drive, dict):
        raise ValueError("kernel and drive artifacts must contain JSON objects")
    kernel_authorized = bool(kernel.get("production_parameterization_allowed", False))
    drive_authorized = bool(drive.get("production_parameterization_allowed", False))
    kernel_states = kernel.get("states", [])
    drive_states = drive.get("states", [])
    state_match = {
        str(row.get("state_id")) for row in kernel_states if isinstance(row, dict)
    } == {
        str(row.get("state_id")) for row in drive_states if isinstance(row, dict)
    }
    ready = bool(kernel_authorized and drive_authorized and state_match)
    return {
        "schema": MODEL_ID,
        "stage": "parameterization_readiness",
        "kernel_family": str(kernel_path.resolve()),
        "drive_family": str(drive_path.resolve()),
        "engine_template": engine,
        "kernel_authorized": kernel_authorized,
        "drive_authorized": drive_authorized,
        "kernel_drive_state_match": state_match,
        "kernel_state_count": len(kernel_states),
        "drive_state_count": len(drive_states),
        "ready_for_stage_1": ready,
    }


__all__ = [
    "MODEL_ID",
    "MechanicalStateRequest",
    "REVIEW_CHECKS",
    "SIGNED_RESPONSE_COLUMNS",
    "STATE_AXES",
    "STATE_TABLE_COLUMNS",
    "TENSOR_RESPONSE_COLUMNS",
    "inspect_engine_template",
    "inspect_raw_artifacts",
    "load_state_requests",
    "readiness_report",
    "state_coverage",
    "validate_review_approval",
    "write_collection_plan",
]
