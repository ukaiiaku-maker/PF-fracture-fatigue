#!/usr/bin/env python3
"""Validate or restore the tracked v10.2.27 theta=30° load-invariance ZIP."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT_NAME = "v10_2_27_active_load_invariance_700K_theta30_frontfix_E000_E1200_v2"
STATES = ("E000", "E200", "E500", "E800", "E1000", "E1200")
EXPECTED_EXTENSIONS_M = {
    "E000": 0.0,
    "E200": 2.0e-4,
    "E500": 5.0e-4,
    "E800": 8.0e-4,
    "E1000": 1.0e-3,
    "E1200": 1.2e-3,
}
LOADS = (("0p5", 0.5), ("1", 1.0), ("1p5", 1.5))
EXPECTED_SHA256 = "e71d9dcae52129a175100fa56f3f0445105536598bc21d160624c75b5b52b919"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    REPO_ROOT
    / "artifacts"
    / "v10_2_27_theta30_frontfix"
    / f"{ROOT_NAME}.zip"
)
DEFAULT_OUTPUT = REPO_ROOT / "runs" / ROOT_NAME


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_members(archive: zipfile.ZipFile) -> list[str]:
    names = archive.namelist()
    for name in names:
        member = Path(name)
        if member.is_absolute() or ".." in member.parts:
            raise ValueError(f"unsafe archive member: {name}")
    return names


def validate_extracted(root: Path) -> dict[str, object]:
    state_records: list[dict[str, object]] = []

    for state in STATES:
        state_root = root / state
        report_path = state_root / "frozen_geometry_load_invariance.json"
        sweep_path = state_root / "active_frozen_geometry_load_sweep_responses.csv"
        if not report_path.is_file() or not sweep_path.is_file():
            raise FileNotFoundError(f"incomplete load-invariance state: {state_root}")

        report = json.loads(report_path.read_text())
        if report.get("schema") != "v10.2.14_active_frozen_geometry_load_invariance":
            raise ValueError(f"load-invariance schema mismatch for {state}")
        if str(report.get("parent_state_id")) != state:
            raise ValueError(f"parent-state mismatch for {state}")
        extension = float(report.get("cumulative_crack_path_extension_m", float("nan")))
        if abs(extension - EXPECTED_EXTENSIONS_M[state]) > 2.5e-6:
            raise ValueError(f"extension mismatch for {state}: {extension}")
        if [float(value) for value in report.get("load_scales", [])] != [0.5, 1.0, 1.5]:
            raise ValueError(f"load-scale list mismatch for {state}")
        if report.get("load_invariance_passed") is not True:
            raise ValueError(f"load invariance did not pass for {state}")
        if report.get("active_kernel_mechanically_measured") is not True:
            raise ValueError(f"active kernel is not mechanically measured for {state}")
        if report.get("wake_shielding_supported") is not False:
            raise ValueError(f"wake shielding is unexpectedly enabled for {state}")

        load_records: list[dict[str, object]] = []
        for label, load_scale in LOADS:
            csv_path = state_root / f"active_station_responses_load_{label}.csv"
            audit_path = state_root / f"active_station_responses_load_{label}.audit.json"
            if not csv_path.is_file() or not audit_path.is_file():
                raise FileNotFoundError(f"missing load {load_scale} artifacts for {state}")
            rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
            audit = json.loads(audit_path.read_text())
            expected_state_id = f"{state}__load_{label}"
            if len(rows) != 320 or int(audit.get("response_rows", -1)) != 320:
                raise ValueError(f"response-row mismatch for {state}, load {load_scale}")
            if str(audit.get("state_id")) != expected_state_id:
                raise ValueError(f"audit state-id mismatch for {state}, load {load_scale}")
            if audit.get("physical_fem_responses_generated") is not True:
                raise ValueError(f"nonphysical FEM responses for {state}, load {load_scale}")
            if audit.get("responses_are_measured_stations_not_full_grid") is not True:
                raise ValueError(f"responses are not measured stations for {state}, load {load_scale}")
            if any(str(row.get("state_id")) != expected_state_id for row in rows):
                raise ValueError(f"CSV state-id mismatch for {state}, load {load_scale}")
            if any(abs(float(row["crack_extension_m"]) - extension) > 1.0e-12 for row in rows):
                raise ValueError(f"CSV extension mismatch for {state}, load {load_scale}")
            load_records.append(
                {
                    "load_scale": load_scale,
                    "response_rows": len(rows),
                    "response_csv_sha256": sha256(csv_path),
                    "response_audit_sha256": sha256(audit_path),
                }
            )

        state_records.append(
            {
                "state_id": state,
                "crack_extension_m": extension,
                "load_invariance_report_sha256": sha256(report_path),
                "load_sweep_csv_sha256": sha256(sweep_path),
                "loads": load_records,
            }
        )

    return {
        "schema": "v10.2.27_tracked_theta30_load_invariance_archive_v1",
        "root_name": ROOT_NAME,
        "state_count": len(state_records),
        "states": state_records,
        "load_invariance_passed": True,
        "active_kernel_mechanically_measured": True,
        "wake_shielding_supported": False,
    }


def validate_archive(archive_path: Path) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    digest = sha256(archive_path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"archive SHA mismatch: {digest} != {EXPECTED_SHA256}")

    with tempfile.TemporaryDirectory(prefix="v10227_load_invariance_check_") as tmp:
        extracted = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            safe_members(archive)
            archive.extractall(extracted)
        root = extracted / ROOT_NAME
        if not root.is_dir():
            raise FileNotFoundError(root)
        audit = validate_extracted(root)

    audit["archive"] = str(archive_path)
    audit["archive_sha256"] = digest
    return audit


def restore(archive_path: Path, output: Path, force: bool) -> dict[str, object]:
    audit = validate_archive(archive_path)
    if output.exists() and not force:
        existing = validate_extracted(output)
        existing["archive"] = str(archive_path)
        existing["archive_sha256"] = audit["archive_sha256"]
        existing["output"] = str(output)
        existing["reused_existing_output"] = True
        return existing

    with tempfile.TemporaryDirectory(prefix="v10227_load_invariance_restore_") as tmp:
        extracted = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extracted)
        source = extracted / ROOT_NAME
        validate_extracted(source)
        temporary_output = output.with_name(output.name + ".restoring")
        if temporary_output.exists():
            shutil.rmtree(temporary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, temporary_output)
        validate_extracted(temporary_output)
        if output.exists():
            shutil.rmtree(output)
        temporary_output.replace(output)

    audit["output"] = str(output)
    audit["reused_existing_output"] = False
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    archive = args.archive.expanduser().resolve()
    output = args.output.expanduser().resolve()
    audit = validate_archive(archive) if args.check_only else restore(archive, output, args.force)
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
