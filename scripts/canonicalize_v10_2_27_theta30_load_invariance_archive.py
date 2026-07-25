#!/usr/bin/env python3
"""Create the canonical tracked v10.2.27 theta=30° load-invariance ZIP."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
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
EXPECTED_CANONICAL_SHA256 = (
    "e71d9dcae52129a175100fa56f3f0445105536598bc21d160624c75b5b52b919"
)
DEFAULT_OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "artifacts"
    / "v10_2_27_theta30_frontfix"
    / f"{ROOT_NAME}.zip"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def locate_root(extracted: Path) -> Path:
    direct = extracted / ROOT_NAME
    if direct.is_dir():
        return direct
    matches = [path for path in extracted.rglob(ROOT_NAME) if path.is_dir()]
    if len(matches) != 1:
        raise RuntimeError(f"expected one {ROOT_NAME} directory; found {matches}")
    return matches[0]


def required_state_files() -> tuple[str, ...]:
    return (
        "frozen_geometry_load_invariance.json",
        "active_frozen_geometry_load_sweep_responses.csv",
        "active_station_responses_load_0p5.csv",
        "active_station_responses_load_0p5.audit.json",
        "active_station_responses_load_1.csv",
        "active_station_responses_load_1.audit.json",
        "active_station_responses_load_1p5.csv",
        "active_station_responses_load_1p5.audit.json",
    )


def validate_root(root: Path) -> dict[str, object]:
    state_records: list[dict[str, object]] = []

    for state in STATES:
        state_root = root / state
        for name in required_state_files():
            path = state_root / name
            if not path.is_file():
                raise FileNotFoundError(path)

        report_path = state_root / "frozen_geometry_load_invariance.json"
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
        residual = float(report.get("minimum_residual_stiffness_fraction", float("nan")))
        if abs(residual - 0.001) > 1.0e-15:
            raise ValueError(f"residual-stiffness fraction mismatch for {state}: {residual}")

        generated = report.get("generated_load_cases", [])
        generated_by_scale = {float(item["load_scale"]): item for item in generated}
        if set(generated_by_scale) != {0.5, 1.0, 1.5}:
            raise ValueError(f"generated load cases mismatch for {state}")

        load_records: list[dict[str, object]] = []
        for label, load_scale in LOADS:
            csv_path = state_root / f"active_station_responses_load_{label}.csv"
            audit_path = state_root / f"active_station_responses_load_{label}.audit.json"
            rows = list(csv.DictReader(io.StringIO(csv_path.read_text())))
            audit = json.loads(audit_path.read_text())
            expected_state_id = f"{state}__load_{label}"

            if len(rows) != 320:
                raise ValueError(f"response-row count mismatch for {state}, load {load_scale}")
            if int(generated_by_scale[load_scale].get("response_rows", -1)) != 320:
                raise ValueError(f"generated-case row count mismatch for {state}, load {load_scale}")
            if str(audit.get("state_id")) != expected_state_id:
                raise ValueError(f"audit state-id mismatch for {state}, load {load_scale}")
            if int(audit.get("response_rows", -1)) != 320:
                raise ValueError(f"audit response-row mismatch for {state}, load {load_scale}")
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
                "load_sweep_csv_sha256": sha256(
                    state_root / "active_frozen_geometry_load_sweep_responses.csv"
                ),
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


def canonical_files(root: Path) -> list[Path]:
    return [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != ".DS_Store" and "__MACOSX" not in path.parts
    ]


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

    with tempfile.TemporaryDirectory(prefix="v10227_load_invariance_canonicalize_") as tmp:
        extracted = Path(tmp)
        with zipfile.ZipFile(source) as archive:
            archive.extractall(extracted)
        root = locate_root(extracted)
        audit = validate_root(root)
        write_canonical_zip(root, output)

    audit.update(
        {
            "source": str(source),
            "source_sha256": sha256(source),
            "canonical_archive": str(output),
            "canonical_sha256": sha256(output),
        }
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
