from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile


TEMPS = (700, 900, 1200, 1400)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def reference(path: Path, family: str, candidate: str, values: tuple[float, ...]) -> None:
    rank_name = "search_rank" if family == "peak" else "shelf_rank"
    row: dict[str, object] = {rank_name: 1, "candidate_id": candidate}
    for temperature, value in zip(TEMPS, values):
        row[f"K50_T{temperature}K_MPa_sqrt_m"] = value
    write_csv(path, [row])


def run_root(path: Path, candidate: str, values: tuple[float, ...]) -> None:
    rows = []
    for temperature, value in zip(TEMPS, values):
        rows.append(
            {
                "option_key": f"option_{candidate}",
                "candidate_id": candidate,
                "temperature_K": temperature,
                "K_50um_MPa_sqrt_m": value,
            }
        )
        status = path / candidate / f"T{temperature}K" / "stage3_case_status.json"
        status.parent.mkdir(parents=True, exist_ok=True)
        status.write_text(json.dumps({"complete": True}) + "\n")
    write_csv(path / "v10_2_22_dbtt_50um_screen_summary.csv", rows)


def invoke(
    root: Path,
    peak_root: Path,
    shelf_root: Path,
    peak_reference: Path,
    shelf_reference: Path,
    out: Path,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analyze_v10223_v10224_transfer.py"),
            "--peak-root",
            str(peak_root),
            "--shelf-root",
            str(shelf_root),
            "--peak-reference",
            str(peak_reference),
            "--shelf-reference",
            str(shelf_reference),
            "--out-dir",
            str(out),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_transfer_analyzer_direct_and_compact_modes(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    peak_root = tmp_path / "peak"
    shelf_root = tmp_path / "upper_shelf"
    peak_reference = tmp_path / "peak_reference.csv"
    shelf_reference = tmp_path / "shelf_reference.csv"

    reference(peak_reference, "peak", "peak_candidate", (30.0, 50.0, 35.0, 34.0))
    reference(shelf_reference, "upper_shelf", "shelf_candidate", (25.0, 33.0, 42.0, 42.0))
    run_root(peak_root, "peak_candidate", (31.0, 52.0, 36.0, 35.0))
    run_root(shelf_root, "shelf_candidate", (24.0, 34.0, 43.0, 43.0))

    direct_out = tmp_path / "direct_out"
    completed = invoke(
        root,
        peak_root,
        shelf_root,
        peak_reference,
        shelf_reference,
        direct_out,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (direct_out / "analysis_report.md").is_file()
    assert (direct_out / "analysis_summary.json").is_file()
    assert (direct_out / "candidate_transfer_metrics.csv").is_file()
    assert (direct_out / "temperature_transfer_points.csv").is_file()
    assert (direct_out / "v10223_v10224_transfer_analysis_bundle.zip").is_file()

    rows = list(csv.DictReader((direct_out / "candidate_transfer_metrics.csv").open()))
    peak = next(row for row in rows if row["family"] == "peak")
    shelf = next(row for row in rows if row["family"] == "upper_shelf")
    assert peak["peak_like_2d"] == "True"
    assert peak["intended_topology_retained"] == "True"
    assert shelf["classic_upper_shelf_2d"] == "True"
    assert shelf["intended_topology_retained"] == "True"

    archive = tmp_path / "compact.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for label, family_root in (("peak", peak_root), ("upper_shelf", shelf_root)):
            for path in family_root.rglob("*"):
                if path.is_file():
                    bundle.write(path, Path(label) / path.relative_to(family_root))

    compact_out = tmp_path / "compact_out"
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    compact = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/analyze_v10223_v10224_transfer.py"),
            "--compact-zip",
            str(archive),
            "--peak-reference",
            str(peak_reference),
            "--shelf-reference",
            str(shelf_reference),
            "--out-dir",
            str(compact_out),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert compact.returncode == 0, compact.stdout + "\n" + compact.stderr
    assert (compact_out / "analysis_report.md").is_file()
    assert (compact_out / "v10223_v10224_transfer_analysis_bundle.zip").is_file()
