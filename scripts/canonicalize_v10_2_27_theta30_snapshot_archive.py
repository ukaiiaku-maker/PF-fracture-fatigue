#!/usr/bin/env python3
"""Create the canonical tracked v10.2.27 theta=30° six-state snapshot ZIP."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
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
EXPECTED_CANONICAL_SHA256 = (
    "8a4bc221447aa98e8b56b3a1797f42224b6c1bda4da124c37ab2448fd8e4b5ae"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "v10_2_27_theta30_frontfix"
    / f"{ROOT_NAME}.zip"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_root(root: Path) -> None:
    complete = root / "capture_complete.json"
    manifest = root / "capture_manifest.json"
    entry = root / "v10_2_13_capture_entry.json"
    trace_csv = root / "reachable_physical_state_trace.csv"
    trace_json = root / "reachable_physical_state_trace.json"
    for required in (complete, manifest, entry, trace_csv, trace_json):
        if not required.is_file():
            raise FileNotFoundError(required)

    completion = json.loads(complete.read_text())
    if int(completion.get("requested_states", -1)) != len(STATES):
        raise ValueError("capture requested-state count is not six")
    if int(completion.get("captured_states", -1)) != len(STATES):
        raise ValueError("capture did not complete all six states")
    if completion.get("pending_state_ids") not in ([], None):
        raise ValueError("capture still contains pending states")

    for state in STATES:
        state_root = root / state
        metadata_path = state_root / "snapshot.json"
        arrays_path = state_root / "state_arrays.npz"
        if not metadata_path.is_file() or not arrays_path.is_file():
            raise FileNotFoundError(f"incomplete state artifact: {state_root}")
        metadata = json.loads(metadata_path.read_text())
        if str(metadata.get("state_id")) != state:
            raise ValueError(f"state-id mismatch for {state}")
        temperature = float(metadata.get("temperature_K", float("nan")))
        if abs(temperature - EXPECTED_TEMPERATURE_K) > 1.0e-8:
            raise ValueError(f"temperature mismatch for {state}: {temperature}")
        theta = float(
            metadata.get("engine_config", {})
            .get("anisotropic_config", {})
            .get("crystal_theta_deg", float("nan"))
        )
        if abs(theta - EXPECTED_THETA_DEG) > 1.0e-10:
            raise ValueError(f"orientation mismatch for {state}: {theta}")
        extension = float(metadata.get("crack_extension_m", float("nan")))
        if abs(extension - EXPECTED_EXTENSIONS_M[state]) > 2.5e-6:
            raise ValueError(f"extension mismatch for {state}: {extension}")


def locate_root(extracted: Path) -> Path:
    direct = extracted / ROOT_NAME
    if direct.is_dir():
        return direct
    matches = [path for path in extracted.rglob(ROOT_NAME) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {ROOT_NAME} directory; found {matches}")
    return matches[0]


def canonical_files(root: Path) -> list[Path]:
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name == ".DS_Store" or "__MACOSX" in path.parts:
            continue
        files.append(path)
    return files


def write_canonical_zip(root: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in canonical_files(root):
            member = Path(ROOT_NAME) / path.relative_to(root)
            info = zipfile.ZipInfo(str(member), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100644 & 0xFFFF) << 16
            archive.writestr(
                info,
                path.read_bytes(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )
    digest = sha256(temporary)
    if digest != EXPECTED_CANONICAL_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            f"canonical archive SHA mismatch: {digest} != {EXPECTED_CANONICAL_SHA256}"
        )
    temporary.replace(output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_zip", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    source = args.source_zip.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    with tempfile.TemporaryDirectory(prefix="v10227_snapshot_canonicalize_") as tmp:
        extracted = Path(tmp)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(extracted)
        root = locate_root(extracted)
        validate_root(root)
        write_canonical_zip(root, output)

    print(
        json.dumps(
            {
                "source": str(source),
                "source_sha256": sha256(source),
                "canonical_archive": str(output),
                "canonical_sha256": sha256(output),
                "states": list(STATES),
                "temperature_K": EXPECTED_TEMPERATURE_K,
                "theta_deg": EXPECTED_THETA_DEG,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
