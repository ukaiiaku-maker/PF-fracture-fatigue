"""Materialize only directionally compatible v10.4.1 fracture cases.

v10.4.2 defines forward crack-driving work as ``max(J_signed, 0)`` under the
fixed domain-integral convention. Older cases used a first-nonzero sign latch.
A completed older fracture case is reusable only when its root-front history up
to first passage already satisfies the corrected positive-signed-J relation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "v10.4.2_reused_v10.4.1_complete_fracture_case_v2"
MANIFEST_SCHEMA = "v10.4.2_materialized_v10.4.1_complete_cases_v2"

REQUIRED = (
    "COMPLETE",
    "stage3_case_status.json",
    "v10_2_27_case_contract.json",
    "v10_2_27_paper_four_class_parameter_transfer.json",
    "command.sh",
    "v10_2_30_hazard_energy_gate_audit.json",
    "v10_4_bulk_peierls_taylor_coupling_audit.json",
    "v10_4_bulk_coupled_model_audit.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _single_front_table(root: Path) -> Path:
    paths = sorted(root.glob("fronts_*K.csv"))
    if len(paths) != 1:
        raise ValueError(
            f"expected one root-front diagnostic table in {root}; found {len(paths)}"
        )
    return paths[0]


def audit_positive_directional_J_history(case_root: str | Path) -> dict[str, Any]:
    """Verify corrected directional J through the first accepted crack event."""
    root = Path(case_root).expanduser().resolve()
    path = _single_front_table(root)
    table = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        dtype=float,
        encoding=None,
        autostrip=True,
    )
    names = table.dtype.names or ()
    required = {
        "step",
        "front_id",
        "n_fire",
        "J_signed_trial",
        "J_effective_trial",
        "J_sign_ref",
    }
    missing = sorted(required.difference(names))
    if missing:
        raise ValueError(f"directional-J audit fields missing from {path}: {missing}")

    rows = np.atleast_1d(table)
    root_rows = rows[np.isclose(np.asarray(rows["front_id"], dtype=float), 0.0)]
    if root_rows.size == 0:
        raise ValueError(f"no root-front rows in {path}")
    order = np.argsort(np.asarray(root_rows["step"], dtype=float))
    root_rows = root_rows[order]

    fired = np.flatnonzero(np.asarray(root_rows["n_fire"], dtype=float) > 0.0)
    if fired.size == 0:
        raise ValueError("completed fracture case has no root first-passage event")
    first_index = int(fired[0])
    prefix = root_rows[: first_index + 1]

    raw = np.asarray(prefix["J_signed_trial"], dtype=float)
    effective = np.asarray(prefix["J_effective_trial"], dtype=float)
    sign_ref = np.asarray(prefix["J_sign_ref"], dtype=float)
    expected = np.maximum(raw, 0.0)

    finite = np.isfinite(raw) & np.isfinite(effective) & np.isfinite(sign_ref)
    if not np.all(finite):
        raise ValueError("non-finite root directional-J value before first passage")

    scale = max(
        float(np.max(np.abs(raw))) if raw.size else 0.0,
        float(np.max(np.abs(effective))) if effective.size else 0.0,
        1.0,
    )
    atol = 1.0e-10 * scale
    compatible_rows = np.isclose(effective, expected, rtol=1.0e-8, atol=atol)
    max_error = float(np.max(np.abs(effective - expected)))

    first_raw = float(raw[-1])
    first_effective = float(effective[-1])
    first_sign_ref = float(sign_ref[-1])
    compatible = bool(
        np.all(compatible_rows)
        and first_raw > 0.0
        and first_effective > 0.0
        and np.isclose(first_sign_ref, 1.0, rtol=0.0, atol=1.0e-12)
    )

    audit = {
        "schema": "v10.4.2_positive_directional_J_reuse_audit_v1",
        "front_table": str(path),
        "front_table_sha256": sha256_file(path),
        "root_rows_through_first_passage": int(prefix.size),
        "first_passage_step": int(round(float(prefix["step"][-1]))),
        "first_passage_J_signed_J_per_m2": first_raw,
        "first_passage_J_effective_J_per_m2": first_effective,
        "first_passage_J_sign_ref": first_sign_ref,
        "maximum_pre_first_passage_relation_error_J_per_m2": max_error,
        "relation_tolerance_J_per_m2": atol,
        "required_relation": "J_effective=max(J_signed,0)",
        "all_pre_first_passage_rows_compatible": bool(np.all(compatible_rows)),
        "compatible": compatible,
    }
    if not compatible:
        raise ValueError(
            "source fracture case is incompatible with positive signed directional J: "
            f"step={audit['first_passage_step']} raw={first_raw:.9g} "
            f"effective={first_effective:.9g} sign_ref={first_sign_ref:.9g} "
            f"max_relation_error={max_error:.9g}"
        )
    return audit


def verify_source_case(case_root: str | Path) -> dict[str, Any]:
    root = Path(case_root).expanduser().resolve()
    if (root / "RUN_FAILED").exists():
        raise ValueError(f"source case has RUN_FAILED: {root}")
    for name in REQUIRED:
        if not (root / name).is_file():
            raise FileNotFoundError(f"required completed-case file missing: {root / name}")

    status = _json(root / "stage3_case_status.json")
    if status.get("complete") is not True:
        raise ValueError("source case did not complete target crack extension")
    if status.get("status") != "complete_target_extension":
        raise ValueError("source case status is not complete_target_extension")

    model = _json(root / "v10_4_bulk_coupled_model_audit.json")
    if model.get("bulk_plasticity_mode") != "full_field":
        raise ValueError("source case is not full-field bulk plasticity")
    if model.get("v10_4_0_outputs_physics_compatible") is not False:
        raise ValueError("source case compatibility provenance is incomplete")

    detailed = root / "v10_4_1_bulk_detailed_balance_audit.json"
    reuse = root / "v10_4_1_reuse_audit.json"
    if detailed.is_file():
        payload = _json(detailed)
        if model.get("zero_stress_net_plastic_rate_exactly_zero") is not True:
            raise ValueError("native v10.4.1 model audit lacks zero-stress gate")
        if payload.get("zero_stress_net_plastic_rate_exactly_zero") is not True:
            raise ValueError("detailed-balance audit failed zero-stress gate")
        if payload.get("one_way_arrhenius_rate_used_as_net_slip") is not False:
            raise ValueError("native v10.4.1 case used one-way rate as net slip")
        execution = "native_v10.4.1"
    elif reuse.is_file():
        from .reuse_v1040_v1041 import verify_materialized_reuse

        admitted = verify_materialized_reuse(root)
        if admitted.get("target_model") != (
            "v10.4.1_detailed_balance_forward_minus_reverse"
        ):
            raise ValueError("v10.4.1 reuse target model mismatch")
        execution = "audited_v10.4.0_reuse_admitted_by_v10.4.1"
    else:
        raise ValueError("source case has neither native nor reused v10.4.1 audit")

    directional = audit_positive_directional_J_history(root)
    return {
        "source_case": str(root),
        "source_execution_mode": execution,
        "status": status,
        "directional_J_audit": directional,
        "source_required_file_sha256": {
            name: sha256_file(root / name) for name in REQUIRED
        },
    }


def _link_contents(source: Path, destination: Path) -> None:
    for item in source.iterdir():
        if item.name in {"RUN_FAILED", "v10_4_2_reuse_audit.json"}:
            continue
        target = destination / item.name
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"reuse destination exists: {target}")
        target.symlink_to(item.resolve(), target_is_directory=item.is_dir())


def materialize_completed_cases(
    source_root: str | Path,
    destination_root: str | Path,
    *,
    source_commit: str,
    target_commit: str,
) -> dict[str, Any]:
    source = Path(source_root).expanduser().resolve()
    destination = Path(destination_root).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for complete in sorted(source.glob("*/T*K_th*_seed*/COMPLETE")):
        case = complete.parent.resolve()
        try:
            verified = verify_source_case(case)
        except Exception as exc:
            rejected.append({
                "source_case": str(case),
                "approved": False,
                "reason": f"{type(exc).__name__}: {exc}",
            })
            continue

        relative = case.relative_to(source)
        target_case = destination / relative
        if target_case.exists():
            audit_path = target_case / "v10_4_2_reuse_audit.json"
            if audit_path.is_file() and _json(audit_path).get("source_case") == str(case):
                records.append(_json(audit_path))
                continue
            raise FileExistsError(f"destination case already exists: {target_case}")
        target_case.mkdir(parents=True)
        _link_contents(case, target_case)
        record = {
            "schema": SCHEMA,
            "approved": True,
            "physics_compatibility_basis": (
                "source case completed target extension and its complete root-front "
                "history through first passage already satisfies "
                "J_effective=max(J_signed,0) under the v10.4.2 convention"
            ),
            "source_commit": source_commit,
            "target_commit": target_commit,
            "source_case": str(case),
            "materialized_case": str(target_case),
            "source_execution_mode": verified["source_execution_mode"],
            "source_required_file_sha256": verified[
                "source_required_file_sha256"
            ],
            "directional_J_audit": verified["directional_J_audit"],
            "directional_J_positive_convention_compatible": True,
            "fracture_measure_unchanged_for_this_case": True,
            "constitutive_acceptance_unchanged_from_v10_4_1": True,
            "terminal_logic_was_not_reached": True,
            "target_extension_complete": True,
        }
        (target_case / "v10_4_2_reuse_audit.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n"
        )
        records.append(record)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "source_campaign_root": str(source),
        "destination_campaign_root": str(destination),
        "source_commit": source_commit,
        "target_commit": target_commit,
        "materialized_case_count": len(records),
        "rejected_case_count": len(rejected),
        "records": records,
        "rejected_records": rejected,
    }
    (destination / "v10_4_2_materialized_reuse_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def verify_materialized_case(case_root: str | Path) -> dict[str, Any]:
    root = Path(case_root).expanduser().resolve()
    audit = _json(root / "v10_4_2_reuse_audit.json")
    if audit.get("schema") != SCHEMA or audit.get("approved") is not True:
        raise ValueError("v10.4.2 reuse audit is not approved")
    if audit.get("directional_J_positive_convention_compatible") is not True:
        raise ValueError("positive directional-J compatibility gate failed")
    if audit.get("fracture_measure_unchanged_for_this_case") is not True:
        raise ValueError("fracture compatibility gate failed")
    if audit.get("constitutive_acceptance_unchanged_from_v10_4_1") is not True:
        raise ValueError("constitutive compatibility gate failed")
    if audit.get("target_extension_complete") is not True:
        raise ValueError("only completed fracture cases may be reused")

    source = Path(audit["source_case"]).resolve()
    hashes = audit.get("source_required_file_sha256", {})
    if not isinstance(hashes, dict) or not hashes:
        raise ValueError("v10.4.2 reuse audit has no hashes")
    for name, expected in hashes.items():
        source_path = source / name
        target_path = root / name
        if sha256_file(source_path) != expected:
            raise ValueError(f"source file changed: {source_path}")
        if sha256_file(target_path) != expected:
            raise ValueError(f"materialized file changed: {target_path}")

    directional = audit.get("directional_J_audit", {})
    front_path = Path(directional.get("front_table", ""))
    expected_front_hash = directional.get("front_table_sha256")
    if not front_path.is_file() or sha256_file(front_path) != expected_front_hash:
        raise ValueError("source directional-J front table changed")
    target_front = root / front_path.name
    if not target_front.is_file() or sha256_file(target_front) != expected_front_hash:
        raise ValueError("materialized directional-J front table changed")
    return audit


__all__ = [
    "MANIFEST_SCHEMA",
    "SCHEMA",
    "audit_positive_directional_J_history",
    "materialize_completed_cases",
    "verify_materialized_case",
    "verify_source_case",
]
