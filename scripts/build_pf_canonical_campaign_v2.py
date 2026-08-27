#!/usr/bin/env python3
"""Build and verify the immutable 288-condition canonical PF V2 campaign.

This is a planning/status utility.  It reads existing case products and
qualified deterministic kernel families, but it never starts a stochastic
trajectory and never removes a case directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_pf_canonical_fracture_campaign as runner


SCHEMA = "pf_canonical_fracture_campaign_v2"
PLAN_VERSION = "V2"
PHYSICAL_SOURCE_COMMIT = "9e884fb0b0845da621d2612bdf1042e481b8df49"
FROZEN_V1_LAUNCHER_COMMIT = "998665899d15f818203d1742528462f21b99f7ed"
PLAN_PARENT_COMMIT = "04680ae5449a80110f7ea4e930d8fdecf88f77ca"
TARGET_EXTENSION_UM = 1000.0
SAFETY_MARGIN_UM = 20.0
REQUIRED_FAMILY_EXTENSION_UM = TARGET_EXTENSION_UM + SAFETY_MARGIN_UM
NOMINAL_DU_M = 2.0e-7
CLASSES = tuple(runner.EXPECTED_CLASSES)
TEMPERATURES_K = (300, 600, 800, 900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300)
THETAS_DEG = (0, 15, 30, 45)
RATES = (
    ("rate0p01x", 0.01, 840.0, 2.380952380952381e-10),
    ("rate1x", 1.0, 8.4, 2.3809523809523807e-8),
    ("rate100x", 100.0, 0.084, 2.3809523809523808e-6),
)
BASE_SEED = 3621
CLASS_SEED_STRIDE = 1_000_000
TEMPERATURE_SEED_STRIDE = 1009
MATERIAL_HASHES = {
    "Peak": "937644f63e8f44982523ea11fce962bc28fe38d347cfc3d37f898af070073283",
    "DBTT": "4ef4cda0fcdaedd2b8bad4330cd749772594f85b3df7af51ee955592f32256e6",
    "weak-T": "5689ee29ac72f27c7259cbb6a60f3175ad4327cd045a3e6f7935884c66f3e368",
    "ceramic-like": "fee4d08d19a0576b72157f76b5ef910739be3826354c63467e7ddb2249ba896b",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def git(*args: str, root: Path) -> str:
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty canonical table: {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def copy_text_normalized(source: Path, target: Path) -> None:
    """Preserve text content while making committed line endings deterministic."""
    normalized = source.read_text().replace("\r\n", "\n")
    if not normalized.endswith("\n"):
        normalized += "\n"
    if target.exists() and target.read_text().replace("\r\n", "\n") != normalized:
        raise RuntimeError(f"preserved text differs; refusing overwrite: {target}")
    target.write_text(normalized)


def seed_for(material_class: str, temperature_K: int) -> int:
    return (
        BASE_SEED
        + CLASSES.index(material_class) * CLASS_SEED_STRIDE
        + TEMPERATURES_K.index(temperature_K) * TEMPERATURE_SEED_STRIDE
    )


def slug(material_class: str) -> str:
    return material_class.replace("-", "").lower()


def physical_condition_id(
    material_class: str, temperature_K: int, theta_deg: int, rate_tag: str
) -> str:
    return (
        f"PFV2__{slug(material_class)}__T{temperature_K:04d}K"
        f"__theta{theta_deg:g}__{rate_tag}"
    )


def old_case_id(
    material_class: str, temperature_K: int, theta_deg: int, rate_tag: str
) -> str:
    matrix = (
        "canonical_single_crack_theta"
        if theta_deg in {15, 30}
        else "canonical_strain_rate"
    )
    return (
        f"{matrix}__{slug(material_class)}__T{temperature_K:04d}K"
        f"__theta{theta_deg:g}__{rate_tag}__seed{seed_for(material_class, temperature_K)}"
    )


def new_theta0_case_id(
    material_class: str, temperature_K: int, rate_tag: str
) -> str:
    group = "canonical_theta0_shared" if rate_tag == "rate1x" else "canonical_theta0_rate"
    return (
        f"{group}__{slug(material_class)}__T{temperature_K:04d}K"
        f"__theta0__{rate_tag}__seed{seed_for(material_class, temperature_K)}"
    )


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def process_owners(path: Path, process_table: str) -> list[int]:
    result: list[int] = []
    needle = str(path.resolve())
    for line in process_table.splitlines():
        if needle not in line:
            continue
        words = line.strip().split(maxsplit=1)
        if words and words[0].isdigit():
            result.append(int(words[0]))
    return sorted(set(result))


def case_is_complete(case_root: Path, temperature_K: int) -> bool:
    result = read_json(case_root / "canonical_case_result.json")
    return (
        result.get("status") == "COMPLETE"
        and int(result.get("returncode", -1)) == 0
        and runner.completed(case_root, float(temperature_K), TARGET_EXTENSION_UM)
    )


def verify_complete_case(
    case_root: Path,
    material_class: str,
    temperature_K: int,
    theta_deg: int,
    rate_tag: str,
    family_sha256: str,
) -> dict[str, Any]:
    result_path = case_root / "canonical_case_result.json"
    result = read_json(result_path)
    expected = {
        "candidate_id": runner.EXPECTED_CLASSES[material_class],
        "material_class": material_class,
        "temperature_K": float(temperature_K),
        "theta_deg": float(theta_deg),
        "rate_tag": rate_tag,
        "seed": seed_for(material_class, temperature_K),
        "target_extension_um": TARGET_EXTENSION_UM,
    }
    mismatches = {
        key: {"expected": value, "actual": result.get(key)}
        for key, value in expected.items()
        if result.get(key) != value
    }
    if not case_is_complete(case_root, temperature_K):
        mismatches["target_status"] = {"expected": "COMPLETE_AT_1000_UM", "actual": result.get("status")}
    if result.get("kernel_family_sha256") != family_sha256:
        mismatches["kernel_family_sha256"] = {
            "expected": family_sha256,
            "actual": result.get("kernel_family_sha256"),
        }
    observer_manifest_path = case_root / "canonical_pf_state_observer_v2_manifest.json"
    observer_path = case_root / "canonical_pf_state_observer_v2.json.zst"
    observer = read_json(observer_manifest_path)
    observer_ok = (
        observer.get("records_exactly_equal_across_sources") is True
        and observer.get("source_files_removed_only_after_verification") is True
        and observer_path.is_file()
        and observer.get("canonical_compressed_sha256") == sha256(observer_path)
    )
    if not observer_ok:
        mismatches["observer_consolidation"] = {"expected": "VERIFIED", "actual": observer}
    if mismatches:
        raise RuntimeError(f"preserved canonical case failed verification: {case_root}: {mismatches}")
    return {
        "result_sha256": sha256(result_path),
        "observer_manifest_sha256": sha256(observer_manifest_path),
        "observer_artifact_sha256": sha256(observer_path),
        "observer_consolidation_verified": True,
    }


def family_records(kernel_cache: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for theta in THETAS_DEG:
        family_path, family = runner.family_for_theta(
            kernel_cache, float(theta), REQUIRED_FAMILY_EXTENSION_UM
        )
        validation_path = family_path.parent / "direct_kernel_validation_manifest.json"
        validation = read_json(validation_path)
        configuration = family["mechanical_configuration"]
        maximum_um = 1.0e6 * float(family["cumulative_crack_path_extension_levels_m"][-1])
        interpolation = family.get("interpolation", {})
        record = {
            "theta_deg": theta,
            "family_path": str(family_path),
            "configuration_fingerprint": family["mechanical_configuration_fingerprint"],
            "physics_fingerprint": validation.get("family_physics_fingerprint", ""),
            "family_sha256": sha256(family_path),
            "validation_manifest_sha256": sha256(validation_path),
            "coordinate_definition": "laboratory_x_projected_extension_along_horizontal_forward_100_cleavage_trace",
            "prescribed_crack_path_policy": configuration["extra"]["prescribed_crack_path_policy"],
            "nominal_crack_angle_deg": float(configuration["nominal_crack_angle_deg"]),
            "forward_cosine": 1.0,
            "valid_extension_min_um": 0.0,
            "valid_extension_max_um": maximum_um,
            "required_extension_um": REQUIRED_FAMILY_EXTENSION_UM,
            "target_extension_um": TARGET_EXTENSION_UM,
            "established_safety_margin_um": SAFETY_MARGIN_UM,
            "interpolation_method": interpolation.get("method"),
            "interpolation_neighbors": interpolation.get("neighbors"),
            "interpolation_power": interpolation.get("power"),
            "interpolation_envelope_relative_tolerance": interpolation.get("envelope_relative_tolerance"),
            "interpolation_uncertainty": "bounded_by_recorded_family_envelope_relative_tolerance",
            "extrapolation_allowed": bool(interpolation.get("extrapolation_allowed", True)),
            "production_parameterization_allowed": bool(family.get("production_parameterization_allowed")),
            "source_qualified": bool(validation.get("passed")),
        }
        if not record["physics_fingerprint"]:
            raise RuntimeError(f"family physics fingerprint absent: {validation_path}")
        if record["extrapolation_allowed"]:
            raise RuntimeError(f"family permits extrapolation: {family_path}")
        if not record["production_parameterization_allowed"] or not record["source_qualified"]:
            raise RuntimeError(f"family is not source-qualified: {family_path}")
        records.append(record)
    for key in ("configuration_fingerprint", "physics_fingerprint", "family_sha256"):
        values = [row[key] for row in records]
        if len(values) != len(set(values)):
            raise RuntimeError(f"an angle-specific family was reused under another theta: {key}")
    return records


def pin_family_files(
    records: list[dict[str, Any]], pinned_cache: Path
) -> None:
    """Copy exact qualified family inputs into the repository fail-closed."""
    pinned_cache.mkdir(parents=True, exist_ok=True)
    required_names = (
        "family.json",
        "direct_kernel_validation_manifest.json",
        "mechanical_configuration.json",
        "kernel_build_manifest.json",
        "family_extended_assembly_audit.json",
    )
    for record in records:
        source_root = Path(record["family_path"]).parent
        target_root = pinned_cache / record["configuration_fingerprint"]
        target_root.mkdir(parents=True, exist_ok=True)
        for name in required_names:
            source = source_root / name
            target = target_root / name
            if name == "mechanical_configuration.json" and not source.is_file():
                embedded = read_json(source_root / "family.json").get("mechanical_configuration")
                if not isinstance(embedded, dict):
                    raise RuntimeError(f"embedded mechanical configuration absent: {source_root}")
                if target.exists() and read_json(target) != embedded:
                    raise RuntimeError(f"pinned mechanical configuration differs: {target}")
                if not target.exists():
                    write_json(target, embedded)
                continue
            if not source.is_file():
                raise RuntimeError(f"required family provenance file absent: {source}")
            if target.exists() and sha256(target) != sha256(source):
                raise RuntimeError(f"pinned family file differs; refusing overwrite: {target}")
            if not target.exists():
                shutil.copyfile(source, target)


def paused_theta45_audit(
    run_root: Path, process_table: str, family_sha256: str
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rate_tag, factor, dt_s, opening_rate in RATES:
        for material_class in CLASSES:
            for temperature_K in TEMPERATURES_K:
                identifier = old_case_id(material_class, temperature_K, 45, rate_tag)
                case_root = run_root / identifier
                result_path = case_root / "canonical_case_result.json"
                result = read_json(result_path)
                complete = case_is_complete(case_root, temperature_K)
                exists = case_root.is_dir()
                status = "COMPLETE" if complete else ("INTERRUPTED" if exists else "PENDING")
                if rate_tag == "rate1x":
                    canonical_status = {
                        "COMPLETE": "CANONICAL_REUSE",
                        "INTERRUPTED": "CANONICAL_RESTART_CLEAN",
                        "PENDING": "CANONICAL_PENDING",
                    }[status]
                else:
                    canonical_status = {
                        "COMPLETE": "SUPPLEMENTAL_CURRENT_SOURCE_NONCANONICAL",
                        "INTERRUPTED": "CANCEL_SUPERSEDED_INCOMPLETE",
                        "PENDING": "CANCEL_SUPERSEDED_UNSTARTED",
                    }[status]
                if complete:
                    verify_complete_case(
                        case_root, material_class, temperature_K, 45, rate_tag,
                        family_sha256,
                    )
                owners = process_owners(case_root, process_table) if exists else []
                rows.append({
                    "case_id": identifier,
                    "material_class": material_class,
                    "candidate_id": runner.EXPECTED_CLASSES[material_class],
                    "temperature_K": temperature_K,
                    "theta_deg": 45,
                    "rate_tag": rate_tag,
                    "loading_rate_factor": factor,
                    "nominal_dU_m": NOMINAL_DU_M,
                    "nominal_dt_s": dt_s,
                    "nominal_opening_rate_m_per_s": opening_rate,
                    "seed": seed_for(material_class, temperature_K),
                    "status": status,
                    "returncode": result.get("returncode", ""),
                    "result_sha256": sha256(result_path) if result_path.is_file() else "",
                    "output_path": str(case_root),
                    "output_directory_exists": exists,
                    "active_process_owned": bool(owners),
                    "active_process_pids": ";".join(map(str, owners)),
                    "canonical_status": canonical_status,
                    "complete_result_preserved": complete,
                    "cleanup_eligible_after_lock": canonical_status == "CANCEL_SUPERSEDED_INCOMPLETE" and not owners,
                })
    return rows


def plan_rows(
    run_root: Path,
    paused: list[dict[str, Any]],
    families: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    family_by_theta = {int(row["theta_deg"]): row for row in families}
    paused_by_key = {
        (row["material_class"], int(row["temperature_K"]), row["rate_tag"]): row
        for row in paused
    }
    conditions: list[tuple[int, str, float, float, float]] = []
    conditions.extend((theta, "rate1x", 1.0, 8.4, RATES[1][3]) for theta in THETAS_DEG)
    conditions.extend((0, rate_tag, factor, dt_s, opening_rate)
                      for rate_tag, factor, dt_s, opening_rate in (RATES[0], RATES[2]))
    rows: list[dict[str, Any]] = []
    for theta, rate_tag, factor, dt_s, opening_rate in conditions:
        for material_class in CLASSES:
            for temperature_K in TEMPERATURES_K:
                if theta in {15, 30, 45}:
                    identifier = old_case_id(material_class, temperature_K, theta, rate_tag)
                else:
                    identifier = new_theta0_case_id(material_class, temperature_K, rate_tag)
                case_root = run_root / identifier
                if theta in {15, 30}:
                    verification = verify_complete_case(
                        case_root, material_class, temperature_K, theta, rate_tag,
                        family_by_theta[theta]["family_sha256"],
                    )
                    status = "CANONICAL_REUSE_COMPLETE"
                    result_hash = verification["result_sha256"]
                    observer_hash = verification["observer_artifact_sha256"]
                elif theta == 45:
                    audit = paused_by_key[(material_class, temperature_K, rate_tag)]
                    status = (
                        "CANONICAL_REUSE_COMPLETE"
                        if audit["canonical_status"] == "CANONICAL_REUSE"
                        else audit["canonical_status"]
                    )
                    result_hash = audit["result_sha256"]
                    observer_hash = ""
                else:
                    complete = case_is_complete(case_root, temperature_K)
                    if complete:
                        verification = verify_complete_case(
                            case_root, material_class, temperature_K, theta, rate_tag,
                            family_by_theta[theta]["family_sha256"],
                        )
                        status = "CANONICAL_REUSE_COMPLETE"
                        result_hash = verification["result_sha256"]
                        observer_hash = verification["observer_artifact_sha256"]
                    else:
                        status = "CANONICAL_RESTART_CLEAN" if case_root.is_dir() else "CANONICAL_PENDING"
                        result_hash = ""
                        observer_hash = ""
                rows.append({
                    "campaign_plan_version": PLAN_VERSION,
                    "case_id": identifier,
                    "physical_condition_id": physical_condition_id(
                        material_class, temperature_K, theta, rate_tag
                    ),
                    "material_class": material_class,
                    "candidate_id": runner.EXPECTED_CLASSES[material_class],
                    "full_material_sha256": MATERIAL_HASHES[material_class],
                    "temperature_K": temperature_K,
                    "theta_deg": theta,
                    "rate_tag": rate_tag,
                    "loading_rate_factor": factor,
                    "nominal_dU_m": NOMINAL_DU_M,
                    "nominal_dt_s": dt_s,
                    "nominal_opening_rate_m_per_s": opening_rate,
                    "seed": seed_for(material_class, temperature_K),
                    "target_projected_extension_um": TARGET_EXTENSION_UM,
                    "is_orientation_matrix_case": rate_tag == "rate1x",
                    "is_rate_matrix_case": theta == 0,
                    "canonical_execution_status": status,
                    "output_path": str(case_root),
                    "result_sha256": result_hash,
                    "observer_artifact_sha256": observer_hash,
                    "kernel_configuration_fingerprint": family_by_theta[theta]["configuration_fingerprint"],
                    "kernel_physics_fingerprint": family_by_theta[theta]["physics_fingerprint"],
                    "kernel_family_sha256": family_by_theta[theta]["family_sha256"],
                })
    rows.sort(key=lambda row: (
        int(row["theta_deg"]), float(row["loading_rate_factor"]),
        CLASSES.index(row["material_class"]), int(row["temperature_K"]),
    ))
    validate_plan(rows)
    return rows


def validate_plan(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 288:
        raise RuntimeError(f"V2 unique count is {len(rows)}, expected 288")
    physical_ids = [row["physical_condition_id"] for row in rows]
    if len(set(physical_ids)) != 288:
        raise RuntimeError("V2 plan contains duplicate physical conditions")
    orientation = [row for row in rows if row["is_orientation_matrix_case"]]
    rate = [row for row in rows if row["is_rate_matrix_case"]]
    shared = [row for row in rows if row["is_orientation_matrix_case"] and row["is_rate_matrix_case"]]
    if (len(orientation), len(rate), len(shared)) != (192, 144, 48):
        raise RuntimeError("V2 analysis membership counts do not close")
    theta0_rates: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rate:
        theta0_rates.setdefault((row["material_class"], int(row["temperature_K"])), []).append(row)
    for key, group in theta0_rates.items():
        if {row["rate_tag"] for row in group} != {item[0] for item in RATES}:
            raise RuntimeError(f"theta0 rate contract incomplete for {key}")
        if len({int(row["seed"]) for row in group}) != 1:
            raise RuntimeError(f"common random numbers violated for {key}")
    preserved = [row for row in rows if int(row["theta_deg"]) in {15, 30}]
    if len(preserved) != 96 or any(
        row["canonical_execution_status"] != "CANONICAL_REUSE_COMPLETE"
        for row in preserved
    ):
        raise RuntimeError("completed theta15/theta30 cases are not frozen for reuse")


def seed_rows() -> list[dict[str, Any]]:
    return [
        {
            "material_class": material_class,
            "candidate_id": runner.EXPECTED_CLASSES[material_class],
            "temperature_K": temperature_K,
            "seed": seed_for(material_class, temperature_K),
            "theta0_rate0p01x_seed": seed_for(material_class, temperature_K),
            "theta0_rate1x_seed": seed_for(material_class, temperature_K),
            "theta0_rate100x_seed": seed_for(material_class, temperature_K),
            "orientation_theta0_seed": seed_for(material_class, temperature_K),
            "orientation_theta15_seed": seed_for(material_class, temperature_K),
            "orientation_theta30_seed": seed_for(material_class, temperature_K),
            "orientation_theta45_seed": seed_for(material_class, temperature_K),
            "seed_formula": "3621+class_index*1000000+temperature_index*1009",
        }
        for material_class in CLASSES
        for temperature_K in TEMPERATURES_K
    ]


def counts_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def source_hashes(root: Path) -> dict[str, str]:
    relatives = (
        "scripts/build_pf_canonical_campaign_v2.py",
        "scripts/run_pf_canonical_fracture_campaign.py",
        "scripts/analyze_pf_canonical_fracture_campaign.py",
        "scripts/consolidate_pf_canonical_observer_artifacts.py",
        "scripts/generate_pf_canonical_angle_provider_maps.py",
        "arrhenius_fracture/sharp_front_v10_2_22.py",
        "arrhenius_fracture/persistent_site_physical_width_v10222.py",
        "arrhenius_fracture/persistent_site_source_v10221.py",
        "arrhenius_fracture/stochastic_avalanche_tip.py",
        "arrhenius_fracture/fem.py",
        "arrhenius_fracture/anisotropic_emission_v10174.py",
    )
    return {relative: sha256(root / relative) for relative in relatives}


def report_text(
    rows: list[dict[str, Any]], paused: list[dict[str, Any]], families: list[dict[str, Any]],
    plan_fingerprint: str,
) -> str:
    supplemental = [r for r in paused if r["canonical_status"] == "SUPPLEMENTAL_CURRENT_SOURCE_NONCANONICAL"]
    interrupted = [r for r in paused if r["canonical_status"] == "CANCEL_SUPERSEDED_INCOMPLETE"]
    return f"""# Canonical PF fracture run plan V2

This immutable design supersedes the earlier 240-case V1 matrix without
overwriting it. No stochastic trajectory was launched while producing this
plan.

## Primary design

- Unique canonical conditions: **{len(rows)}**.
- Orientation membership: **192**, theta = 0/15/30/45 degrees at rate1x.
- Rate membership: **144**, theta = 0 degrees at rate0p01x/rate1x/rate100x.
- Shared theta0/rate1 membership: **48**, stored once with both flags true.
- Fixed loading increment: dU = 2e-7 m; dt = 840/8.4/0.084 s.
- Common random numbers: identical seed across the three theta0 rates for each
  fixed material class and temperature.
- Scientific plan fingerprint: `{plan_fingerprint}`.

## Preserved and reclassified products

- Theta15/theta30 complete products: **96/96**, verified and frozen for reuse.
- Reusable complete theta45/rate1 products: **{sum(r['canonical_status'] == 'CANONICAL_REUSE' for r in paused)}**.
- Complete theta45 extreme-rate supplemental products: **{len(supplemental)}**.
- Interrupted theta45 extreme-rate directories marked for fail-closed cancellation: **{len(interrupted)}**.
- Supplemental runs are excluded from the primary rate analysis.

## Mechanics/source coordinates

All four angle-specific families use the horizontal
`forward_100_cleavage_trace`. Theta rotates cubic elasticity and slip/source
coordinates, not the prescribed crack line; the laboratory-x forward cosine
is therefore 1.0 for theta 0/15/30/45. Every family covers at least
{REQUIRED_FAMILY_EXTENSION_UM:g} micrometres (1000 target plus the established
20 micrometre maximum-event safety margin). Family interpolation is bounded
by its recorded envelope and extrapolation is forbidden.

## Execution gate

This document is a plan/lock record only. Stage A may select only incomplete
theta45/rate1 rows. Stage B may select only theta0 rows. Completed theta15/30
rows cannot be selected by either V2 production stage.
"""


def zip_independence_text(zip_path: Path, runtime_hits: list[str]) -> str:
    status = "NOT_READY_TO_DELETE"
    text = f"""# Legacy ZIP independence test for campaign V2

Runtime verdict: **PASS**. The 288-row V2 plan, family resolution, seed/rate
validation, current-status regeneration, completed-case verification, and
pending-case dry-run validation do not read `{zip_path}`.

Deletion verdict: **{status}**.

The ZIP is no longer a runtime dependency, but it remains the retained
historical record for legacy theta/rate products not proven to have a second
content-equivalent archive. Runtime independence alone is insufficient to
authorize destructive deletion. Keep the ZIP until a complete member-level
replacement archive is verified.

Runtime ZIP-reference findings: {len(runtime_hits)}. Any listed references are
fail-closed audit/history references, not V2 launcher dependencies.
""" + "".join(f"\n- `{hit}`" for hit in runtime_hits)
    return text.rstrip() + "\n"


def build(args: argparse.Namespace) -> dict[str, Any]:
    root = args.repo.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.process_table is None:
        process_table = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    else:
        process_table = args.process_table.read_text()
    source_families = family_records(args.kernel_cache.resolve())
    pinned_cache = root / "runtime_inputs/pf_canonical_kernel_families_v2"
    pin_family_files(source_families, pinned_cache)
    families = family_records(pinned_cache)
    for record in families:
        record["family_path"] = str(Path(record["family_path"]).relative_to(root))
    family_by_theta = {int(row["theta_deg"]): row for row in families}
    paused = paused_theta45_audit(
        args.run_root.resolve(), process_table, family_by_theta[45]["family_sha256"]
    )
    rows = plan_rows(args.run_root.resolve(), paused, families)
    plan_fingerprint = canonical_sha256(rows)
    with args.v1_plan.open(newline="") as stream:
        v1_rows = list(csv.DictReader(stream))
    v1_fingerprint = canonical_sha256(v1_rows)
    if v1_fingerprint == plan_fingerprint:
        raise RuntimeError("V2 scientific fingerprint did not change from V1")

    v1_copy = out / "pf_canonical_fracture_run_plan_v1.csv"
    copy_text_normalized(args.v1_plan, v1_copy)
    v1_storage_source = args.v1_plan.parent / "pf_storage_reclaimed.csv"
    v1_storage_copy = out / "pf_storage_reclaimed_v1.csv"
    if v1_storage_source.is_file():
        copy_text_normalized(v1_storage_source, v1_storage_copy)

    plan_csv = out / "pf_canonical_fracture_run_plan_v2.csv"
    plan_json = out / "pf_canonical_fracture_run_plan_v2.json"
    write_csv(plan_csv, rows)
    write_json(plan_json, {
        "schema": SCHEMA,
        "campaign_plan_version": PLAN_VERSION,
        "scientific_fingerprint_sha256": plan_fingerprint,
        "superseded_v1_scientific_fingerprint_sha256": v1_fingerprint,
        "scientific_fingerprint_changed_from_v1": True,
        "unique_case_count": len(rows),
        "orientation_membership_count": sum(r["is_orientation_matrix_case"] for r in rows),
        "rate_membership_count": sum(r["is_rate_matrix_case"] for r in rows),
        "shared_theta0_rate1_membership_count": sum(
            r["is_orientation_matrix_case"] and r["is_rate_matrix_case"] for r in rows
        ),
        "rows": rows,
    })
    (out / "pf_canonical_fracture_run_plan_v2.sha256").write_text(
        f"{sha256(plan_csv)}  {plan_csv.name}\n"
        f"{sha256(plan_json)}  {plan_json.name}\n"
        f"{plan_fingerprint}  SCIENTIFIC_FINGERPRINT\n"
    )
    (out / "PF_CANONICAL_FRACTURE_RUN_PLAN_V2.md").write_text(
        report_text(rows, paused, families, plan_fingerprint)
    )
    write_csv(out / "pf_canonical_seed_map_v2.csv", seed_rows())
    write_csv(out / "pf_canonical_angle_family_lock_v2.csv", families)
    write_json(out / "pf_canonical_angle_family_lock_v2.json", {
        "schema": "pf_canonical_angle_family_lock_v2",
        "families": families,
        "all_source_qualified": all(row["source_qualified"] for row in families),
        "all_cover_target_plus_margin": all(
            row["valid_extension_max_um"] >= REQUIRED_FAMILY_EXTENSION_UM
            for row in families
        ),
        "no_extrapolation": all(not row["extrapolation_allowed"] for row in families),
    })
    write_csv(out / "pf_theta45_paused_stage_audit_v2.csv", paused)
    write_json(out / "pf_theta45_paused_stage_audit_v2.json", {
        "schema": "pf_theta45_paused_stage_audit_v2",
        "counts_by_status": counts_by(paused, "status"),
        "counts_by_canonical_status": counts_by(paused, "canonical_status"),
        "rows": paused,
    })
    supplemental = [row for row in paused if row["canonical_status"] == "SUPPLEMENTAL_CURRENT_SOURCE_NONCANONICAL"]
    cancelled = [row for row in paused if row["canonical_status"].startswith("CANCEL_SUPERSEDED")]
    write_csv(out / "pf_theta45_supplemental_manifest_v2.csv", supplemental)
    write_json(out / "pf_theta45_supplemental_manifest_v2.json", {
        "schema": "pf_theta45_supplemental_manifest_v2",
        "principal_rate_analysis_membership": False,
        "complete_case_count": len(supplemental),
        "rows": supplemental,
    })
    write_csv(out / "pf_theta45_cancellation_manifest_v2.csv", cancelled)
    write_json(out / "pf_theta45_cancellation_manifest_v2.json", {
        "schema": "pf_theta45_cancellation_manifest_v2",
        "interrupted_directory_count": sum(r["status"] == "INTERRUPTED" for r in cancelled),
        "unstarted_count": sum(r["status"] == "PENDING" for r in cancelled),
        "directories_deleted": False,
        "delete_only_after_lock_push": True,
        "rows": cancelled,
    })

    runtime_sources = (
        root / "scripts/run_pf_canonical_fracture_campaign.py",
        root / "scripts/analyze_pf_canonical_fracture_campaign.py",
        out / "PF_CANONICAL_FRACTURE_RUN_PLAN_V2.md",
    )
    runtime_hits = []
    for path in runtime_sources:
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if "1_backupdatatouse.zip" in line or "Archive.zip" in line:
                runtime_hits.append(f"{path.relative_to(root)}:{number}")
    if runtime_hits:
        raise RuntimeError(f"V2 runtime depends on the legacy ZIP: {runtime_hits}")
    (out / "READY_TO_DELETE_1_BACKUPDATATOUSE_V2.md").write_text(
        zip_independence_text(args.legacy_zip.resolve(), runtime_hits)
    )

    copied_registry = out / "pf_v2_four_class_registry.csv"
    copy_text_normalized(args.registry, copied_registry)

    pending = [r for r in rows if r["canonical_execution_status"] != "CANONICAL_REUSE_COMPLETE"]
    registry_rows = runner.load_registry(args.pf_transfer_registry)
    dry_runs = []
    for row in pending:
        family_path, _ = runner.family_for_theta(
            pinned_cache, float(row["theta_deg"]), REQUIRED_FAMILY_EXTENSION_UM
        )
        command = runner.build_command(
            {key: str(value) for key, value in row.items()},
            registry_rows[row["material_class"]]["option_key"], family_path,
            Path(row["output_path"]), TARGET_EXTENSION_UM, 20,
        )
        dry_runs.append({
            "case_id": row["case_id"], "validated": True,
            "command_sha256": canonical_sha256(command),
            "family_sha256": sha256(family_path),
        })
    completed_sample = [
        row for row in rows
        if int(row["theta_deg"]) in {15, 30}
        and int(row["temperature_K"]) in {300, 1100}
        and row["material_class"] in {"Peak", "ceramic-like"}
    ]
    independence_path = out / "pf_canonical_zip_independence_v2.json"
    previous_independence = read_json(independence_path)
    unavailable_test_passed = (
        not args.legacy_zip.exists()
        or bool(previous_independence.get("legacy_zip_temporarily_unavailable_test_passed"))
    )
    independence = {
        "schema": "pf_canonical_zip_independence_v2",
        "legacy_zip_path": str(args.legacy_zip.resolve()),
        "legacy_zip_accessed": False,
        "legacy_zip_present_during_last_regeneration": args.legacy_zip.exists(),
        "legacy_zip_temporarily_unavailable_test_passed": unavailable_test_passed,
        "all_288_rows_validated": len(rows) == 288,
        "pending_dry_run_count": len(dry_runs),
        "pending_dry_runs": dry_runs,
        "completed_theta15_theta30_status_regeneration_count": len(completed_sample),
        "completed_theta15_theta30_status_regeneration": [
            {"case_id": row["case_id"], "result_sha256": row["result_sha256"],
             "observer_artifact_sha256": row["observer_artifact_sha256"]}
            for row in completed_sample
        ],
        "all_four_family_lookups_passed": len(families) == 4,
        "seed_and_rate_contracts_passed": True,
        "supplemental_manifest_verified": len(supplemental) == sum(
            r["canonical_status"] == "SUPPLEMENTAL_CURRENT_SOURCE_NONCANONICAL"
            for r in paused
        ),
        "runtime_zip_reference_count": len(runtime_hits),
        "runtime_independence_passed": True,
        "historical_archive_deletion_ready": False,
    }
    write_json(independence_path, independence)

    hashes = source_hashes(root)
    lock: dict[str, Any] = {
        "schema": "pf_canonical_campaign_lock_v2",
        "branch": git("branch", "--show-current", root=root),
        "plan_producer_parent_commit": PLAN_PARENT_COMMIT,
        "qualified_physical_source_commit": PHYSICAL_SOURCE_COMMIT,
        "frozen_v1_launcher_commit": FROZEN_V1_LAUNCHER_COMMIT,
        "campaign_plan_version": PLAN_VERSION,
        "scientific_fingerprint_sha256": plan_fingerprint,
        "superseded_v1_scientific_fingerprint_sha256": v1_fingerprint,
        "scientific_fingerprint_changed_from_v1": True,
        "expected_unique_cases": 288,
        "orientation_analysis_membership": 192,
        "rate_analysis_membership": 144,
        "shared_theta0_rate1_membership": 48,
        "completed_canonical_cases": sum(r["canonical_execution_status"] == "CANONICAL_REUSE_COMPLETE" for r in rows),
        "pending_canonical_cases": sum(r["canonical_execution_status"] != "CANONICAL_REUSE_COMPLETE" for r in rows),
        "completed_theta15_theta30_cases": 96,
        "supplemental_completed_theta45_extreme_rate_cases": len(supplemental),
        "cancelled_superseded_cases": len(cancelled),
        "cancelled_interrupted_directories": sum(r["status"] == "INTERRUPTED" for r in cancelled),
        "current_case_directory_count": sum(
            row["output_directory_exists"] for row in paused
        ) + 96,
        "verified_complete_current_case_count": 96 + len(supplemental),
        "verified_consolidated_observer_case_count": 96 + len(supplemental),
        "current_storage_status_partition_closes": (
            96 + len(supplemental)
            + sum(row["status"] == "INTERRUPTED" for row in cancelled)
            == sum(row["output_directory_exists"] for row in paused) + 96
        ),
        "case_status_counts": counts_by(rows, "canonical_execution_status"),
        "material_candidates": dict(runner.EXPECTED_CLASSES),
        "material_hashes": MATERIAL_HASHES,
        "registry_path": str(args.registry.resolve()),
        "registry_sha256": sha256(args.registry),
        "committed_registry_sha256": sha256(copied_registry),
        "superseded_v1_plan_source_sha256": sha256(args.v1_plan),
        "superseded_v1_plan_sha256": sha256(v1_copy),
        "pf_transfer_registry_path": str(args.pf_transfer_registry.resolve()),
        "pf_transfer_registry_sha256": sha256(args.pf_transfer_registry),
        "plan_files": {
            path.name: sha256(path)
            for path in (plan_csv, plan_json, out / "pf_canonical_fracture_run_plan_v2.sha256",
                         out / "PF_CANONICAL_FRACTURE_RUN_PLAN_V2.md")
        },
        "family_locks": families,
        "pinned_family_cache_repository_path": str(pinned_cache.relative_to(root)),
        "source_launcher_observer_analyzer_sha256": hashes,
        "zip_runtime_independence": True,
        "zip_safe_to_delete": False,
        "stochastic_cases_launched_by_producer": 0,
    }
    lock["campaign_lock_fingerprint_sha256"] = canonical_sha256(lock)
    write_json(out / "pf_canonical_campaign_lock_v2.json", lock)
    return lock


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--kernel-cache", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--pf-transfer-registry", type=Path, required=True)
    parser.add_argument("--v1-plan", type=Path, required=True)
    parser.add_argument("--legacy-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--process-table", type=Path)
    args = parser.parse_args()
    lock = build(args)
    print(json.dumps({
        key: lock[key]
        for key in (
            "campaign_lock_fingerprint_sha256", "expected_unique_cases",
            "completed_canonical_cases", "pending_canonical_cases",
            "supplemental_completed_theta45_extreme_rate_cases",
            "cancelled_superseded_cases",
        )
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
