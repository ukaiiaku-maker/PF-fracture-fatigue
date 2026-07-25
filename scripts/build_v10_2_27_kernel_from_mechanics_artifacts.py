#!/usr/bin/env python3
"""Build a v10.2.27 signed kernel from portable mechanics artifacts.

The builder accepts either extracted snapshot/load-invariance roots or ZIP
archives. Mechanics normalization is derived from the serialized engine
configuration in the snapshots; no separately preserved normalization file is
required.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import (  # noqa: E402
    load_configuration,
)
from arrhenius_fracture.kernel_registry_v10227 import (  # noqa: E402
    family_physics_fingerprint,
    sha256_file,
    validate_family,
)
from arrhenius_fracture.mechanics_normalization_v10212 import (  # noqa: E402
    SourceGeometryAssumptions,
    derive_mechanical_normalization,
)


def _safe_extract(archive_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            member = Path(name)
            if member.is_absolute() or ".." in member.parts:
                raise ValueError(f"unsafe ZIP member: {name}")
        archive.extractall(destination)


def _snapshot_root(search: Path) -> Path:
    candidates = []
    for complete in search.rglob("capture_complete.json"):
        root = complete.parent
        count = sum(
            (path / "snapshot.json").is_file()
            for path in root.iterdir()
            if path.is_dir()
        )
        if count >= 2:
            candidates.append(root)
    if search.joinpath("capture_complete.json").is_file():
        candidates.append(search)
    unique = sorted({path.resolve() for path in candidates})
    if len(unique) != 1:
        raise RuntimeError(f"expected one snapshot root; found {unique}")
    return unique[0]


def _load_root(search: Path) -> Path:
    candidates: set[Path] = set()
    for report in search.rglob("frozen_geometry_load_invariance.json"):
        candidates.add(report.parent.parent.resolve())
    qualified = [
        root
        for root in candidates
        if sum(
            (path / "frozen_geometry_load_invariance.json").is_file()
            for path in root.iterdir()
            if path.is_dir()
        )
        >= 2
    ]
    if len(qualified) != 1:
        raise RuntimeError(f"expected one load-invariance root; found {qualified}")
    return qualified[0]


def _materialize(
    *,
    root: Path | None,
    archive: Path | None,
    workspace: Path,
    kind: str,
) -> Path:
    if (root is None) == (archive is None):
        raise ValueError(f"provide exactly one {kind} root or archive")
    if root is not None:
        resolved = root.expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(resolved)
        return resolved
    source = archive.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    destination = workspace / kind
    destination.mkdir(parents=True, exist_ok=True)
    _safe_extract(source, destination)
    return _snapshot_root(destination) if kind == "snapshots" else _load_root(destination)


def _state_records(snapshot_root: Path, load_root: Path) -> list[dict[str, Any]]:
    snapshots: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(snapshot_root.glob("*/snapshot.json")):
        payload = json.loads(metadata_path.read_text())
        state_id = str(payload.get("state_id", metadata_path.parent.name))
        snapshots[state_id] = {
            "metadata": payload,
            "metadata_path": metadata_path,
            "arrays_path": metadata_path.parent / "state_arrays.npz",
        }
    reports: dict[str, dict[str, Any]] = {}
    for report_path in sorted(load_root.glob("*/frozen_geometry_load_invariance.json")):
        payload = json.loads(report_path.read_text())
        state_id = str(payload.get("parent_state_id", report_path.parent.name))
        reports[state_id] = {"payload": payload, "path": report_path}
    common = sorted(set(snapshots) & set(reports))
    if len(common) < 2:
        raise ValueError("at least two matching snapshot/load-invariance states are required")
    if set(snapshots) != set(reports):
        raise ValueError(
            "snapshot/load-invariance state sets differ: "
            f"snapshots={sorted(snapshots)}, reports={sorted(reports)}"
        )

    rows = []
    theta_values = set()
    for state_id in common:
        snapshot = snapshots[state_id]
        report = reports[state_id]
        metadata = snapshot["metadata"]
        if not snapshot["arrays_path"].is_file():
            raise FileNotFoundError(snapshot["arrays_path"])
        if report["payload"].get("load_invariance_passed") is not True:
            raise ValueError(f"load invariance did not pass for {state_id}")
        theta = float(
            metadata.get("engine_config", {})
            .get("anisotropic_config", {})
            .get("crystal_theta_deg", float("nan"))
        )
        theta_values.add(round(theta, 12))
        snapshot_extension = float(metadata["crack_extension_m"])
        report_extension = float(
            report["payload"]["cumulative_crack_path_extension_m"]
        )
        if abs(snapshot_extension - report_extension) > 2.5e-6:
            raise ValueError(f"snapshot/report extension mismatch for {state_id}")
        response = report["path"].parent / "active_station_responses_load_1.csv"
        audit = response.with_suffix(".audit.json")
        if not response.is_file() or not audit.is_file():
            raise FileNotFoundError(
                f"missing load=1 response or audit for {state_id}"
            )
        rows.append(
            {
                "state_id": state_id,
                "extension_m": report_extension,
                "snapshot_json": snapshot["metadata_path"],
                "state_arrays": snapshot["arrays_path"],
                "response": response,
                "response_audit": audit,
                "report": report["path"],
                "engine_config": metadata.get("engine_config", {}),
            }
        )
    if len(theta_values) != 1:
        raise ValueError(f"snapshot states contain multiple theta values: {theta_values}")
    rows.sort(key=lambda row: row["extension_m"])
    return rows


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command, cwd=ROOT, text=True, capture_output=True, check=False
    )
    if completed.stdout:
        print(completed.stdout, end="", file=sys.stderr)
    if completed.stderr:
        print(completed.stderr, end="", file=sys.stderr)
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed ({completed.returncode}): {' '.join(command)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--snapshot-archive", type=Path)
    parser.add_argument("--load-invariance-root", type=Path)
    parser.add_argument("--load-invariance-archive", type=Path)
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--family-out", type=Path)
    parser.add_argument("--target-extension-um", type=float, required=True)
    parser.add_argument("--theta-deg", type=float, required=True)
    parser.add_argument("--da-phys-um", type=float, default=5.0)
    parser.add_argument("--event-minimum-factor", type=float, default=0.5)
    parser.add_argument("--event-maximum-factor", type=float, default=4.0)
    parser.add_argument("--margin-events", type=float, default=1.0)
    parser.add_argument("--minimum-source-spacing-b", type=float, default=10.0)
    parser.add_argument("--maximum-source-spacing-b", type=float, default=100.0)
    parser.add_argument("--source-region-length-um", type=float)
    args = parser.parse_args()

    configuration = load_configuration(args.mechanical_config)
    if configuration.branching_mode != "single_front" or configuration.maximum_fronts != 1:
        raise SystemExit(
            "fixed-path active-only atlas construction is valid only for "
            "single_front, maximum_fronts=1 configurations"
        )
    if abs(configuration.theta_deg - float(args.theta_deg)) > 1.0e-10:
        raise SystemExit("mechanical configuration theta does not match requested theta")

    outroot = args.outroot.expanduser().resolve()
    outroot.mkdir(parents=True, exist_ok=True)
    family_out = (
        args.family_out.expanduser().resolve()
        if args.family_out is not None
        else outroot / "family.json"
    )
    if family_out.exists():
        audit = validate_family(family_out)
        print(json.dumps({"reused": True, **audit}, indent=2, sort_keys=True))
        return 0

    with tempfile.TemporaryDirectory(prefix="v10227_kernel_artifacts_") as temp:
        workspace = Path(temp)
        snapshot_root = _materialize(
            root=args.snapshot_root,
            archive=args.snapshot_archive,
            workspace=workspace,
            kind="snapshots",
        )
        load_root = _materialize(
            root=args.load_invariance_root,
            archive=args.load_invariance_archive,
            workspace=workspace,
            kind="load_invariance",
        )
        states = _state_records(snapshot_root, load_root)
        observed_theta = float(
            states[0]["engine_config"]
            .get("anisotropic_config", {})
            .get("crystal_theta_deg", float("nan"))
        )
        if abs(observed_theta - float(args.theta_deg)) > 1.0e-10:
            raise SystemExit(
                f"mechanics artifact theta mismatch: {observed_theta} != {args.theta_deg}"
            )

        assumptions = SourceGeometryAssumptions(
            minimum_spacing_b=args.minimum_source_spacing_b,
            maximum_spacing_b=args.maximum_source_spacing_b,
            source_region_length_m=(
                None
                if args.source_region_length_um is None
                else float(args.source_region_length_um) * 1.0e-6
            ),
        )
        normalization_payload = derive_mechanical_normalization(
            states[0]["engine_config"], assumptions=assumptions
        )
        normalization_path = outroot / "mechanics_normalization.json"
        normalization_path.write_text(
            json.dumps(normalization_payload, indent=2, sort_keys=True) + "\n"
        )

        portable_report_root = outroot / "portable_load_invariance_reports"
        portable_report_root.mkdir(parents=True, exist_ok=True)
        portable_reports: list[Path] = []
        for row in states:
            report_payload = json.loads(row["report"].read_text())
            generated = []
            for case in report_payload.get("generated_load_cases", []):
                patched = dict(case)
                try:
                    load_scale = float(patched.get("load_scale"))
                except (TypeError, ValueError):
                    load_scale = float("nan")
                if math.isfinite(load_scale):
                    label = {0.5: "0p5", 1.0: "1", 1.5: "1p5"}.get(load_scale)
                    if label is not None:
                        local_response = (
                            row["report"].parent
                            / f"active_station_responses_load_{label}.csv"
                        )
                        local_audit = local_response.with_suffix(".audit.json")
                        if local_response.is_file():
                            patched["responses"] = str(local_response.resolve())
                        if local_audit.is_file() and "audit" in patched:
                            patched["audit"] = str(local_audit.resolve())
                generated.append(patched)
            report_payload["generated_load_cases"] = generated
            report_payload["portable_path_relocation"] = {
                "original_report": str(row["report"].resolve()),
                "policy": "state_and_load_identity_with_local_integrity_audits",
            }
            portable = portable_report_root / f"{row['state_id']}.json"
            portable.write_text(
                json.dumps(report_payload, indent=2, sort_keys=True) + "\n"
            )
            portable_reports.append(portable)

        command = [
            sys.executable,
            str(ROOT / "scripts" / "build_v10_2_27_extended_active_only_atlas.py"),
            "--normalization",
            str(normalization_path),
            "--out",
            str(family_out),
            "--minimum-max-extension-um",
            f"{1.0e6 * max(row['extension_m'] for row in states):.17g}",
        ]
        for row in states:
            command.extend(["--responses", str(row["response"])])
        for report in portable_reports:
            command.extend(["--load-invariance", str(report)])
        _run(command)

        coverage_path = outroot / "coverage_audit.json"
        _run(
            [
                sys.executable,
                str(ROOT / "scripts" / "check_v10_2_27_signed_kernel_coverage.py"),
                "--family",
                str(family_out),
                "--target-extension-um",
                f"{args.target_extension_um:.17g}",
                "--theta-deg",
                f"{args.theta_deg:.17g}",
                "--da-phys-um",
                f"{args.da_phys_um:.17g}",
                "--event-minimum-factor",
                f"{args.event_minimum_factor:.17g}",
                "--event-maximum-factor",
                f"{args.event_maximum_factor:.17g}",
                "--margin-events",
                f"{args.margin_events:.17g}",
                "--output",
                str(coverage_path),
            ]
        )

        audit = validate_family(family_out)
        manifest = {
            "schema": "v10.2.27_portable_mechanics_artifact_kernel_build_v1",
            "configuration": configuration.canonical_payload(),
            "configuration_fingerprint": configuration.fingerprint(),
            "family": str(family_out),
            "family_sha256": audit["file_sha256"],
            "family_physics_fingerprint": family_physics_fingerprint(family_out),
            "normalization": str(normalization_path),
            "normalization_sha256": sha256_file(normalization_path),
            "coverage_audit": str(coverage_path),
            "coverage_audit_sha256": sha256_file(coverage_path),
            "snapshot_source": str(
                args.snapshot_archive.expanduser().resolve()
                if args.snapshot_archive is not None
                else args.snapshot_root.expanduser().resolve()
            ),
            "load_invariance_source": str(
                args.load_invariance_archive.expanduser().resolve()
                if args.load_invariance_archive is not None
                else args.load_invariance_root.expanduser().resolve()
            ),
            "states": [
                {
                    "state_id": row["state_id"],
                    "extension_m": row["extension_m"],
                    "snapshot_json_sha256": sha256_file(row["snapshot_json"]),
                    "state_arrays_sha256": sha256_file(row["state_arrays"]),
                    "response_sha256": sha256_file(row["response"]),
                    "response_audit_sha256": sha256_file(row["response_audit"]),
                    "load_invariance_report_sha256": sha256_file(row["report"]),
                }
                for row in states
            ],
        }
        manifest_path = outroot / "kernel_build_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )

    print(json.dumps({"reused": False, **manifest}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
