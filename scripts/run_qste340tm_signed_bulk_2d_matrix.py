#!/usr/bin/env python3
"""Execute the four-R native 2-D signed-bulk transfer matrix."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run_qste340tm_main_result_full_2d.py"
FIT = ROOT / (
    "literature_data/fatigue_recovered_review/model_conditional/"
    "emission_owned_blunting/resolved_increment_1d/"
    "signed_bulk_joint_refit_20261028/fit.json"
)
DEFAULT_OUT = ROOT / (
    "literature_data/fatigue_recovered_review/model_conditional/"
    "emission_owned_blunting/native_signed_bulk_matrix_20261028"
)
CASES = (
    (0.1, 18.0), (0.1, 24.0), (0.1, 32.0), (0.1, 40.0),
    (0.3, 17.0), (0.3, 23.0), (0.3, 30.0), (0.3, 36.0),
    (0.5, 15.5), (0.5, 19.0), (0.5, 23.0), (0.5, 27.0),
    (0.7, 12.0), (0.7, 14.0), (0.7, 16.0), (0.7, 18.0),
)


def label(r_value: float, delta_k: float) -> str:
    return f"R_{r_value:.1f}_DK_{delta_k:g}".replace(".", "p")


def run_case(case: tuple[float, float], outroot: Path, python: str,
             loading_control: str, fit: Path, events: int) -> dict:
    r_value, delta_k = case
    case_root = outroot / label(r_value, delta_k)
    csv = case_root / f"R_{r_value:.1f}".replace(".", "p") / "steps_0298K.csv"
    if csv.exists():
        return {"R": r_value, "DeltaK": delta_k, "status": "reused", "csv": str(csv)}
    case_root.mkdir(parents=True, exist_ok=True)
    command = [
        python, str(RUNNER), "--out", str(case_root),
        "--delta-k", str(delta_k), "--R", str(r_value),
        "--cycles", "5000", "--steps", "5000", "--block-cycles", "1",
        "--events", str(events), "--compact", "--material-fit", str(fit),
        "--loading-control", loading_control,
        "--plasticity-scope", "tip_plus_bulk",
        "--blunting-pathway", "source_plus_forward",
        "--blunting-ledger-mode", "analytical_equivalent",
        "--mpz-length-um", "1", "--normalize-passage-by-source-capacity",
        "--incremental-opening-advection", "--opening-advection-step-um", "0.25",
        "--opening-advection-state-owner", "signed_gnd",
        "--signed-gnd-tip-absorption", "--tip-h-fine-m", "2.5e-7",
    ]
    started = time.time()
    with (case_root / "execution.log").open("w", encoding="utf-8") as log:
        process = subprocess.run(
            command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True
        )
    return {
        "R": r_value, "DeltaK": delta_k,
        "status": "completed" if process.returncode == 0 else "failed",
        "returncode": process.returncode, "elapsed_s": time.time() - started,
        "csv": str(csv), "command": command,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--fit", type=Path, default=FIT)
    parser.add_argument("--events", type=int, default=3)
    parser.add_argument("--case", action="append", default=[], metavar="R:DELTAK",
                        help="Run only an exact matrix case; may be repeated.")
    parser.add_argument("--R", type=float, nargs="+", default=None,
                        help="Optional load-ratio subset; completed cases remain reusable by the full launcher.")
    parser.add_argument("--loading-control",
                        choices=("fixed_local_kmax", "fixed_reaction", "fixed_displacement"),
                        default="fixed_local_kmax")
    args = parser.parse_args(argv)
    cases = CASES if args.R is None else tuple(
        case for case in CASES
        if any(abs(case[0] - requested) < 1e-12 for requested in args.R)
    )
    if args.case:
        requested = tuple(tuple(map(float, value.split(":"))) for value in args.case)
        invalid = [case for case in requested if case not in CASES]
        if invalid:
            raise ValueError(f"requested cases are outside the declared matrix: {invalid}")
        cases = requested
    args.out.mkdir(parents=True, exist_ok=True)
    receipt = {
        "schema": "qste340tm-native-signed-bulk-matrix/1",
        "fit": str(args.fit), "loading_control": args.loading_control,
        "target_events": int(args.events),
        "cases": [], "status": "running",
    }
    receipt_name = (
        "MATRIX.json" if args.R is None else
        "MATRIX_" + "_".join(f"R{value:g}".replace(".", "p") for value in args.R) + ".json"
    )
    path = args.out / receipt_name
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    with ThreadPoolExecutor(max_workers=max(int(args.workers), 1)) as pool:
        futures = {
            pool.submit(
                run_case, case, args.out, args.python, args.loading_control,
                args.fit, args.events,
            ): case
            for case in cases
        }
        for future in as_completed(futures):
            row = future.result()
            receipt["cases"].append(row)
            receipt["cases"].sort(key=lambda item: (item["R"], item["DeltaK"]))
            path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            print(row, flush=True)
    receipt["status"] = (
        "completed" if all(row["status"] in {"completed", "reused"}
                           for row in receipt["cases"]) else "incomplete"
    )
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return 0 if receipt["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())

