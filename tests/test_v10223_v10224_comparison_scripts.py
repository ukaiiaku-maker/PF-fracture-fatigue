from __future__ import annotations

import csv
import os
from pathlib import Path
import subprocess
import sys


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _two_d(path: Path, candidate: str, option: str) -> None:
    _write(
        path,
        [
            {
                "option_key": option,
                "candidate_id": candidate,
                "temperature_K": 700,
                "K_50um_MPa_sqrt_m": 31.0,
            },
            {
                "option_key": option,
                "candidate_id": candidate,
                "temperature_K": 800,
                "K_50um_MPa_sqrt_m": 42.0,
            },
        ],
    )


def _run(root: Path, script: str, two: Path, one: Path, out: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env["MPLBACKEND"] = "Agg"
    return subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / script),
            "--two-d-summary",
            str(two),
            "--one-d-reference",
            str(one),
            "--out-dir",
            str(out),
        ],
        cwd=root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )


def test_peak_comparison_runs_without_pandas(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    candidate = "v913_zeroD_sobol_test_peak"
    two = tmp_path / "two.csv"
    one = tmp_path / "one.csv"
    out = tmp_path / "peak_out"
    _two_d(two, candidate, "peak_option")
    _write(
        one,
        [
            {
                "search_rank": 1,
                "candidate_id": candidate,
                "y__peak_temperature_K": 800,
                "y__peak_prominence": 10.0,
                "K50_T700K_MPa_sqrt_m": 30.0,
                "K50_T800K_MPa_sqrt_m": 40.0,
            }
        ],
    )
    script = root / "scripts/compare_v10_2_23_top10_1d_2d.py"
    assert "pandas" not in script.read_text()
    completed = _run(root, script.name, two, one, out)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (out / "v10_2_23_top10_1d_2d_K50_comparison.csv").is_file()
    assert (out / "v10_2_23_top10_1d_2d_candidate_errors.csv").is_file()
    assert (out / f"{candidate}_K50_1d_vs_2d.png").is_file()


def test_upper_shelf_comparison_runs_without_pandas(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    candidate = "v913_zeroD_sobol_test_shelf"
    two = tmp_path / "two.csv"
    one = tmp_path / "one.csv"
    out = tmp_path / "shelf_out"
    _two_d(two, candidate, "shelf_option")
    _write(
        one,
        [
            {
                "shelf_rank": 1,
                "candidate_id": candidate,
                "y__directional_dbtt_gain": 12.0,
                "y__high_temperature_plateau": 42.0,
                "y__peak_prominence": 0.5,
                "K50_T700K_MPa_sqrt_m": 30.0,
                "K50_T800K_MPa_sqrt_m": 40.0,
            }
        ],
    )
    script = root / "scripts/compare_v10_2_24_upper_shelf_1d_2d.py"
    assert "pandas" not in script.read_text()
    completed = _run(root, script.name, two, one, out)
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (out / "v10_2_24_upper_shelf_1d_2d_K50_comparison.csv").is_file()
    assert (out / "v10_2_24_upper_shelf_1d_2d_candidate_errors.csv").is_file()
    assert (out / f"{candidate}_K50_1d_vs_2d.png").is_file()
