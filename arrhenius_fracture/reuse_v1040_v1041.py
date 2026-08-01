"""Audited selective reuse of v10.4.0 cases in v10.4.1 campaigns.

A v10.4.0 result is never accepted merely because it has a COMPLETE marker.
For each case, this module reconstructs the exact selected-row bulk model and
computes a conservative upper bound on the accumulated equivalent-plastic-
strain difference introduced by the v10.4.1 detailed-balance correction over
that case's recorded density range and elapsed loading time.

Accepted cases are materialized into a fresh v10.4.1 output root as a directory
of immutable links to the original result plus explicit reuse provenance. The
v10.4.1 scheduler verifies that provenance and the hashes of the linked source
files before skipping the case.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
from types import SimpleNamespace
from typing import Any, Iterable

import numpy as np

from .bulk_plasticity_manifest_v104 import BulkManifestParameters
from .emission_derived_plasticity import (
    EmissionDerivedPeierlsTaylorModel,
    ExpFloorSurface,
    config_from_dislocation_config,
)
from .thermodynamic_net_slip_v1041 import detailed_balance_rate_s

SCHEMA = "v10.4.1_selective_reuse_audit_v1"
CASE_SCHEMA = "v10.4.1_reused_v10.4.0_case_v1"
SOURCE_MODEL = "v10.4.0_one_way_arrhenius_bulk_slip"
TARGET_MODEL = "v10.4.1_detailed_balance_forward_minus_reverse"

REQUIRED_SOURCE_FILES = (
    "COMPLETE",
    "stage3_case_status.json",
    "v10_2_27_case_contract.json",
    "v10_2_27_paper_four_class_parameter_transfer.json",
    "command.sh",
    "v10_2_30_hazard_energy_gate_audit.json",
    "stochastic_avalanche_geometry_events.json",
    "v10_4_bulk_peierls_taylor_coupling_audit.json",
    "v10_4_bulk_coupled_model_audit.json",
)

_MATERIALIZED_OVERRIDES = {
    "v10_2_27_case_contract.json",
    "v10_4_bulk_coupled_model_audit.json",
    "v10_4_1_bulk_detailed_balance_audit.json",
    "v10_4_1_reuse_audit.json",
    "RUN_FAILED",
}

_STEP_RE = re.compile(r"\bstep\s+(\d+)\b")


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _command_value(command: str, option: str) -> str | None:
    tokens = shlex.split(command)
    prefix = option + "="
    for index, token in enumerate(tokens):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == option and index + 1 < len(tokens):
            return tokens[index + 1]
    return None


def _last_logged_step(path: Path) -> int:
    maximum = -1
    with path.open(errors="replace") as stream:
        for line in stream:
            match = _STEP_RE.search(line)
            if match:
                maximum = max(maximum, int(match.group(1)))
    return maximum


def _rho_grid(rho_min: float, rho_max: float, count: int) -> np.ndarray:
    lower = max(float(rho_min), 1.0e6)
    upper = max(float(rho_max), lower)
    if count <= 1 or math.isclose(lower, upper, rel_tol=1.0e-14):
        return np.array([lower], dtype=float)
    return np.geomspace(lower, upper, count)


def _stress_grid(max_stress_Pa: float, count: int) -> np.ndarray:
    upper = max(float(max_stress_Pa), 1.0)
    if count <= 2:
        return np.array([0.0, upper], dtype=float)
    positive = np.geomspace(1.0, upper, count - 1)
    return np.concatenate(([0.0], positive))


def _model_from_exact_row(row: dict[str, str]) -> EmissionDerivedPeierlsTaylorModel:
    parameters = BulkManifestParameters.from_row(row)
    cfg = SimpleNamespace()
    parameters.configure(cfg)
    return EmissionDerivedPeierlsTaylorModel(config_from_dislocation_config(cfg))


def max_rate_correction_bound(
    exact_row: dict[str, str],
    *,
    temperature_K: float,
    rho_min_m2: float,
    rho_max_m2: float,
    max_stress_Pa: float,
    stress_points: int,
    rho_points: int,
    b_m: float = 2.74e-10,
) -> dict[str, float]:
    """Return a numerical upper bound on |epsdot_v1040-epsdot_v1041|.

    The grid spans zero through a deliberately high equivalent stress and the
    complete density interval recorded by the source case. The detailed-balance
    reverse rate is largest at low stress, so inclusion of the exact zero-stress
    point is essential.
    """
    model = _model_from_exact_row(exact_row)
    stress = _stress_grid(max_stress_Pa, stress_points)
    rho = _rho_grid(rho_min_m2, rho_max_m2, rho_points)
    stress_mesh, rho_mesh = np.meshgrid(stress, rho, indexing="ij")
    stress_flat = stress_mesh.ravel()
    rho_flat = rho_mesh.ravel()

    original_method = ExpFloorSurface.rate_s
    try:
        ExpFloorSurface.rate_s = original_method
        old = model.rates(stress_flat, rho_flat, temperature_K, b_m)
        ExpFloorSurface.rate_s = detailed_balance_rate_s
        new = model.rates(stress_flat, rho_flat, temperature_K, b_m)
    finally:
        ExpFloorSurface.rate_s = original_method

    old_rate = np.asarray(old["equivalent_plastic_rate_s"], dtype=float)
    new_rate = np.asarray(new["equivalent_plastic_rate_s"], dtype=float)
    difference = np.abs(old_rate - new_rate)
    index = int(np.nanargmax(difference))
    return {
        "maximum_absolute_equivalent_rate_difference_s": float(difference[index]),
        "stress_at_maximum_difference_Pa": float(stress_flat[index]),
        "rho_at_maximum_difference_m2": float(rho_flat[index]),
        "old_rate_at_maximum_difference_s": float(old_rate[index]),
        "new_rate_at_maximum_difference_s": float(new_rate[index]),
        "zero_stress_old_rate_max_s": float(
            np.nanmax(old_rate[stress_flat == 0.0])
        ),
        "zero_stress_new_rate_max_s": float(
            np.nanmax(new_rate[stress_flat == 0.0])
        ),
    }


def _required_hashes(case_root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in REQUIRED_SOURCE_FILES:
        path = case_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"required source-case file is missing: {path}")
        hashes[relative] = sha256_file(path)
    return hashes


def _seed_rows(source_root: Path) -> list[dict[str, str]]:
    path = source_root / "v10_2_27_case_seed_map.csv"
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def _case_relpath(row: dict[str, str], theta_deg: float) -> Path:
    temperature = float(row["temperature_K"])
    return Path(row["option"]) / (
        f"T{temperature:g}K_th{float(theta_deg):g}_seed{int(row['seed'])}"
    )


def audit_case(
    case_root: Path,
    *,
    option: str,
    temperature_K: float,
    seed: int,
    expected_target_um: float,
    expected_theta_deg: float,
    expected_rate_factor: float,
    source_commit: str,
    target_commit: str,
    max_cumulative_strain_difference: float,
    max_stress_Pa: float,
    stress_points: int,
    rho_points: int,
) -> dict[str, Any]:
    reasons: list[str] = []
    record: dict[str, Any] = {
        "schema": CASE_SCHEMA,
        "source_case": str(case_root.resolve()),
        "option": option,
        "temperature_K": float(temperature_K),
        "seed": int(seed),
        "source_commit": source_commit,
        "target_commit": target_commit,
        "source_model": SOURCE_MODEL,
        "target_model": TARGET_MODEL,
        "acceptance_tolerance_cumulative_equivalent_strain": float(
            max_cumulative_strain_difference
        ),
    }

    try:
        hashes = _required_hashes(case_root)
        record["source_required_file_sha256"] = hashes
    except Exception as exc:
        reasons.append(str(exc))
        record["decision"] = "rerun"
        record["reasons"] = reasons
        return record

    if (case_root / "RUN_FAILED").exists():
        reasons.append("RUN_FAILED exists")

    try:
        contract = _json(case_root / "v10_2_27_case_contract.json")
        status = _json(case_root / "stage3_case_status.json")
        bulk_audit = _json(case_root / "v10_4_bulk_peierls_taylor_coupling_audit.json")
        model_audit = _json(case_root / "v10_4_bulk_coupled_model_audit.json")
        command = (case_root / "command.sh").read_text()

        checks = {
            "option": contract.get("option") == option,
            "temperature": math.isclose(
                float(contract.get("temperature_K", float("nan"))),
                temperature_K,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "seed": int(contract.get("seed", -1)) == seed,
            "target": math.isclose(
                float(contract.get("target_extension_um", float("nan"))),
                expected_target_um,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "theta": math.isclose(
                float(contract.get("theta_deg", float("nan"))),
                expected_theta_deg,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "rate": math.isclose(
                float(contract.get("loading_rate_factor", float("nan"))),
                expected_rate_factor,
                rel_tol=0.0,
                abs_tol=1.0e-12,
            ),
            "complete": status.get("complete") is True,
            "bulk_mode": model_audit.get("bulk_plasticity_mode") == "full_field",
            "old_one_way_model": (
                model_audit.get("zero_stress_net_plastic_rate_exactly_zero")
                is not True
            ),
        }
        failed = [name for name, passed in checks.items() if not passed]
        if failed:
            reasons.append("contract/status checks failed: " + ", ".join(failed))

        runtime = bulk_audit.get("runtime_diagnostics", {})
        if runtime.get("local_plastic_work_nonnegative") is not True:
            reasons.append("source case has negative accepted plastic work")
        rho_min = float(runtime.get("minimum_bulk_density_m2", float("nan")))
        rho_max = float(runtime.get("maximum_bulk_density_m2", float("nan")))
        if not (math.isfinite(rho_min) and math.isfinite(rho_max) and rho_min > 0.0):
            reasons.append("source bulk-density range is unavailable")

        exact_row = bulk_audit.get("exact_registry_row")
        if not isinstance(exact_row, dict):
            reasons.append("source exact registry row is unavailable")

        logged_step = _last_logged_step(case_root / "run.log")
        if logged_step < 0:
            reasons.append("no logged step could be recovered from run.log")

        dt_token = _command_value(command, "--dt")
        print_token = _command_value(command, "--print-every")
        dt_s = float(dt_token) if dt_token is not None else float("nan")
        print_every = int(print_token) if print_token is not None else 200
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            reasons.append("case time step could not be recovered from command.sh")

        if not reasons:
            correction = max_rate_correction_bound(
                exact_row,
                temperature_K=temperature_K,
                rho_min_m2=rho_min,
                rho_max_m2=rho_max,
                max_stress_Pa=max_stress_Pa,
                stress_points=stress_points,
                rho_points=rho_points,
            )
            conservative_steps = logged_step + max(print_every, 1)
            duration_s = conservative_steps * dt_s
            cumulative_bound = (
                correction["maximum_absolute_equivalent_rate_difference_s"]
                * duration_s
            )
            record.update(
                {
                    "maximum_logged_step": logged_step,
                    "conservative_step_count": conservative_steps,
                    "nominal_dt_s": dt_s,
                    "conservative_duration_s": duration_s,
                    "recorded_rho_min_m2": rho_min,
                    "recorded_rho_max_m2": rho_max,
                    "stress_grid_max_Pa": max_stress_Pa,
                    "stress_grid_points": stress_points,
                    "rho_grid_points": rho_points,
                    "rate_correction": correction,
                    "upper_bound_cumulative_equivalent_strain_difference": (
                        cumulative_bound
                    ),
                }
            )
            if cumulative_bound > max_cumulative_strain_difference:
                reasons.append(
                    "detailed-balance correction bound exceeds tolerance: "
                    f"{cumulative_bound:.6e} > "
                    f"{max_cumulative_strain_difference:.6e}"
                )
    except Exception as exc:
        reasons.append(f"audit exception: {type(exc).__name__}: {exc}")

    record["decision"] = "reuse" if not reasons else "rerun"
    record["reasons"] = reasons
    record["approved"] = not reasons
    return record


def audit_campaign(
    source_root: str | Path,
    *,
    output_json: str | Path,
    source_commit: str,
    target_commit: str,
    target_extension_um: float = 1000.0,
    theta_deg: float = 0.0,
    loading_rate_factor: float = 1.0,
    max_cumulative_strain_difference: float = 1.0e-6,
    max_stress_GPa: float = 50.0,
    stress_points: int = 401,
    rho_points: int = 65,
) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve()
    output = Path(output_json).expanduser().resolve()
    rows = _seed_rows(source)
    records: list[dict[str, Any]] = []
    for row in rows:
        relative = _case_relpath(row, theta_deg)
        case_root = source / relative
        if not case_root.is_dir():
            continue
        records.append(
            audit_case(
                case_root,
                option=row["option"],
                temperature_K=float(row["temperature_K"]),
                seed=int(row["seed"]),
                expected_target_um=target_extension_um,
                expected_theta_deg=theta_deg,
                expected_rate_factor=loading_rate_factor,
                source_commit=source_commit,
                target_commit=target_commit,
                max_cumulative_strain_difference=max_cumulative_strain_difference,
                max_stress_Pa=max_stress_GPa * 1.0e9,
                stress_points=stress_points,
                rho_points=rho_points,
            )
        )

    accepted = [record for record in records if record.get("approved") is True]
    rejected = [record for record in records if record.get("approved") is not True]
    payload = {
        "schema": SCHEMA,
        "source_campaign_root": str(source),
        "source_commit": source_commit,
        "target_commit": target_commit,
        "source_model": SOURCE_MODEL,
        "target_model": TARGET_MODEL,
        "target_extension_um": float(target_extension_um),
        "theta_deg": float(theta_deg),
        "loading_rate_factor": float(loading_rate_factor),
        "max_cumulative_equivalent_strain_difference": float(
            max_cumulative_strain_difference
        ),
        "max_stress_GPa": float(max_stress_GPa),
        "stress_points": int(stress_points),
        "rho_points": int(rho_points),
        "audited_existing_case_count": len(records),
        "approved_reuse_case_count": len(accepted),
        "rerun_case_count_among_existing": len(rejected),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="") as stream:
        fieldnames = [
            "decision",
            "option",
            "temperature_K",
            "seed",
            "upper_bound_cumulative_equivalent_strain_difference",
            "acceptance_tolerance_cumulative_equivalent_strain",
            "source_case",
            "reasons",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: (
                        "; ".join(record.get(key, []))
                        if key == "reasons"
                        else record.get(key, "")
                    )
                    for key in fieldnames
                }
            )
    return payload


def _verify_record_hashes(record: dict[str, Any]) -> None:
    source = Path(record["source_case"]).resolve()
    hashes = record.get("source_required_file_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("reuse record contains no source-file hashes")
    for relative, expected in hashes.items():
        path = source / relative
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"source-case file hash changed: {path}")


def _link_source_contents(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        if item.name in _MATERIALIZED_OVERRIDES:
            continue
        target = destination / item.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"materialized target already exists: {target}")
        target.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def materialize_reuse_cases(
    audit_json: str | Path,
    destination_root: str | Path,
) -> dict[str, Any]:
    audit_path = Path(audit_json).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    audit = _json(audit_path)
    if audit.get("schema") != SCHEMA:
        raise ValueError(f"unexpected reuse-audit schema: {audit.get('schema')}")
    destination.mkdir(parents=True, exist_ok=True)

    materialized: list[dict[str, Any]] = []
    for record in audit.get("records", []):
        if record.get("approved") is not True:
            continue
        _verify_record_hashes(record)
        source = Path(record["source_case"]).resolve()
        option = str(record["option"])
        temperature = float(record["temperature_K"])
        seed = int(record["seed"])
        theta = float(audit["theta_deg"])
        case_root = destination / option / (
            f"T{temperature:g}K_th{theta:g}_seed{seed}"
        )
        if case_root.exists():
            existing = case_root / "v10_4_1_reuse_audit.json"
            if existing.is_file() and _json(existing).get("source_case") == str(source):
                materialized.append(record)
                continue
            raise FileExistsError(f"destination case already exists: {case_root}")
        case_root.mkdir(parents=True)
        _link_source_contents(source, case_root)

        contract = _json(source / "v10_2_27_case_contract.json")
        contract.update(
            {
                "bulk_net_slip_model": "detailed_balance_forward_minus_reverse",
                "zero_stress_net_plastic_rate_exactly_zero": True,
                "v10_4_0_outputs_physics_compatible": False,
                "case_execution_mode": "audited_v10_4_0_reuse",
                "source_case": str(source),
                "reuse_audit": str(audit_path),
            }
        )
        (case_root / "v10_2_27_case_contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n"
        )

        model_audit = _json(source / "v10_4_bulk_coupled_model_audit.json")
        model_audit.update(
            {
                "execution_mode": "audited_v10_4_0_reuse",
                "source_one_way_arrhenius_rate_used_as_net_slip": True,
                "target_bulk_net_slip_model": (
                    "detailed_balance_forward_minus_reverse"
                ),
                "v10_4_0_outputs_physics_compatible": False,
                "reuse_acceptance_bound_cumulative_equivalent_strain": (
                    record[
                        "upper_bound_cumulative_equivalent_strain_difference"
                    ]
                ),
                "reuse_acceptance_tolerance_cumulative_equivalent_strain": (
                    record[
                        "acceptance_tolerance_cumulative_equivalent_strain"
                    ]
                ),
            }
        )
        (case_root / "v10_4_bulk_coupled_model_audit.json").write_text(
            json.dumps(model_audit, indent=2, sort_keys=True) + "\n"
        )

        case_reuse = dict(record)
        case_reuse.update(
            {
                "schema": CASE_SCHEMA,
                "approved": True,
                "materialized_case": str(case_root),
                "campaign_reuse_audit": str(audit_path),
            }
        )
        (case_root / "v10_4_1_reuse_audit.json").write_text(
            json.dumps(case_reuse, indent=2, sort_keys=True) + "\n"
        )
        materialized.append(record)

    manifest = {
        "schema": "v10.4.1_materialized_reuse_manifest_v1",
        "reuse_audit": str(audit_path),
        "destination_root": str(destination),
        "materialized_case_count": len(materialized),
        "materialized_cases": [
            {
                "option": record["option"],
                "temperature_K": record["temperature_K"],
                "seed": record["seed"],
                "source_case": record["source_case"],
            }
            for record in materialized
        ],
    }
    (destination / "v10_4_1_materialized_reuse_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def verify_materialized_reuse(case_root: str | Path) -> dict[str, Any]:
    root = Path(case_root).resolve()
    reuse = _json(root / "v10_4_1_reuse_audit.json")
    if reuse.get("schema") != CASE_SCHEMA or reuse.get("approved") is not True:
        raise ValueError("case reuse audit is not approved")
    bound = float(
        reuse.get(
            "upper_bound_cumulative_equivalent_strain_difference",
            float("inf"),
        )
    )
    tolerance = float(
        reuse.get(
            "acceptance_tolerance_cumulative_equivalent_strain",
            float("nan"),
        )
    )
    if not (math.isfinite(bound) and math.isfinite(tolerance) and bound <= tolerance):
        raise ValueError("case reuse correction bound exceeds its tolerance")
    source = Path(reuse["source_case"]).resolve()
    hashes = reuse.get("source_required_file_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("case reuse audit has no source hashes")
    for relative, expected in hashes.items():
        source_path = source / relative
        materialized_path = root / relative
        if sha256_file(source_path) != expected:
            raise ValueError(f"source hash changed: {source_path}")
        if relative not in _MATERIALIZED_OVERRIDES:
            if sha256_file(materialized_path) != expected:
                raise ValueError(f"materialized hash mismatch: {materialized_path}")
    return reuse


__all__ = [
    "CASE_SCHEMA",
    "SCHEMA",
    "audit_campaign",
    "audit_case",
    "materialize_reuse_cases",
    "max_rate_correction_bound",
    "sha256_file",
    "verify_materialized_reuse",
]
