#!/usr/bin/env python3
"""Validate or restore the tracked v10.2.27 theta=30° snapshot archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


ROOT_NAME = "v10_2_27_frozen_geometry_snapshots_700K_theta30_frontfix_E000_E1200_v2"
STATES = ("E000", "E200", "E500", "E800", "E1000", "E1200")
EXPECTED_EXTENSIONS_M = {
    "E000": 0.0,
    "E200": 2.0e-4,
    "E500": 5.0e-4,
    "E800": 8.0e-4,
    "E1000": 1.0e-3,
    "E1200": 1.2e-3,
}
EXPECTED_TEMPERATURE_K = 700.0
EXPECTED_THETA_DEG = 30.0
EXPECTED_SHA256 = "8a4bc221447aa98e8b56b3a1797f42224b6c1bda4da124c37ab2448fd8e4b5ae"
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
    required_root_files = (
        "capture_complete.json",
        "capture_manifest.json",
        "v10_2_13_capture_entry.json",
        "reachable_physical_state_trace.csv",
        "reachable_physical_state_trace.json",
    )
    for name in required_root_files:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)

    completion = json.loads((root / "capture_complete.json").read_text())
    if int(completion.get("requested_states", -1)) != len(STATES):
        raise ValueError("snapshot archive does not request six states")
    if int(completion.get("captured_states", -1)) != len(STATES):
        raise ValueError("snapshot archive does not contain six completed states")
    if completion.get("pending_state_ids") not in ([], None):
        raise ValueError("snapshot archive contains pending state IDs")

    state_records: list[dict[str, object]] = []
    for state in STATES:
        state_root = root / state
        metadata_path = state_root / "snapshot.json"
        arrays_path = state_root / "state_arrays.npz"
        if not metadata_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(f"incomplete snapshot state: {state_root}")
        metadata = json.loads(metadata_path.read_text())
        state_id = str(metadata.get("state_id"))
        if state_id != state:
            raise ValueError(f"state-id mismatch: {state_id} != {state}")
        temperature = float(metadata.get("temperature_K", float("nan")))
        theta = float(
            metadata.get("engine_config", {})
            .get("anisotropic_config", {})
            .get("crystal_theta_deg", float("nan"))
        )
        extension = float(metadata.get("crack_extension_m", float("nan")))
        if abs(temperature - EXPECTED_TEMPERATURE_K) > 1.0e-8:
            raise ValueError(f"temperature mismatch for {state}: {temperature}")
        if abs(theta - EXPECTED_THETA_DEG) > 1.0e-10:
            raise ValueError(f"orientation mismatch for {state}: {theta}")
        if abs(extension - EXPECTED_EXTENSIONS_M[state]) > 2.5e-6:
            raise ValueError(f"extension mismatch for {state}: {extension}")
        state_records.append(
            {
                "state_id": state,
                "temperature_K": temperature,
                "theta_deg": theta,
                "crack_extension_m": extension,
                "snapshot_json_sha256": sha256(metadata_path),
                "state_arrays_sha256": sha256(arrays_path),
            }
        )

    return {
        "schema": "v10.2.27_tracked_theta30_snapshot_archive_v1",
        "root_name": ROOT_NAME,
        "state_count": len(state_records),
        "states": state_records,
        "capture_complete": True,
    }


def validate_archive(archive_path: Path) -> dict[str, object]:
    if not archive_path.is_file():
        raise FileNotFoundError(archive_path)
    digest = sha256(archive_path)
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"archive SHA mismatch: {digest} != {EXPECTED_SHA256}")
    with tempfile.TemporaryDirectory(prefix="v10227_snapshot_check_") as tmp:
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

    with tempfile.TemporaryDirectory(prefix="v10227_snapshot_restore_") as tmp:
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
