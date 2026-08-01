"""Materialize completed v10.4.1 fracture cases for a v10.4.2 campaign.

v10.4.2 changes only terminal classification and diagnostic output for cases
that never reach sharp-tip first passage. A v10.4.1 case that already reached
the requested crack extension under the identical detailed-balance constitutive
law is therefore physics-compatible and must not be recomputed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "v10.4.2_reused_v10.4.1_complete_fracture_case_v1"
MANIFEST_SCHEMA = "v10.4.2_materialized_v10.4.1_complete_cases_v1"

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
    if model.get("zero_stress_net_plastic_rate_exactly_zero") is not True:
        raise ValueError("source case does not use detailed-balance net slip")
    if model.get("v10_4_0_outputs_physics_compatible") is not False:
        raise ValueError("source case detailed-balance provenance is incomplete")

    detailed = root / "v10_4_1_bulk_detailed_balance_audit.json"
    reuse = root / "v10_4_1_reuse_audit.json"
    if detailed.is_file():
        payload = _json(detailed)
        if payload.get("zero_stress_net_plastic_rate_exactly_zero") is not True:
            raise ValueError("detailed-balance audit failed zero-stress gate")
        execution = "native_v10.4.1"
    elif reuse.is_file():
        from .reuse_v1040_v1041 import verify_materialized_reuse

        verify_materialized_reuse(root)
        execution = "audited_v10.4.0_reuse_admitted_by_v10.4.1"
    else:
        raise ValueError("source case has neither native nor reused v10.4.1 audit")

    return {
        "source_case": str(root),
        "source_execution_mode": execution,
        "status": status,
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
    for complete in sorted(source.glob("*/T*K_th*_seed*/COMPLETE")):
        case = complete.parent.resolve()
        verified = verify_source_case(case)
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
                "v10.4.2 changes only no-first-passage terminal classification and "
                "diagnostics; this source case already completed the requested "
                "sharp-fracture crack extension under v10.4.1 detailed-balance physics"
            ),
            "source_commit": source_commit,
            "target_commit": target_commit,
            "source_case": str(case),
            "materialized_case": str(target_case),
            "source_execution_mode": verified["source_execution_mode"],
            "source_required_file_sha256": verified[
                "source_required_file_sha256"
            ],
            "fracture_measure_unchanged": True,
            "detailed_balance_constitutive_law_unchanged": True,
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
        "records": records,
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
    if audit.get("fracture_measure_unchanged") is not True:
        raise ValueError("fracture compatibility gate failed")
    if audit.get("detailed_balance_constitutive_law_unchanged") is not True:
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
    return audit


__all__ = [
    "MANIFEST_SCHEMA",
    "SCHEMA",
    "materialize_completed_cases",
    "verify_materialized_case",
    "verify_source_case",
]
