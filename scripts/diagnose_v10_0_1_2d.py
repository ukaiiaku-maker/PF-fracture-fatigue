#!/usr/bin/env python3
from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
from pathlib import Path

from arrhenius_fracture import sharp_front_v10_1


def run_case(root: Path, bulk: str, jmode: str) -> dict:
    out = root / f"{bulk}_{jmode}"
    buf = io.StringIO()
    error = None
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            sharp_front_v10_1.main([
                "--mode", "2d", "--material-class", "ceramic",
                "--temperatures", "700",
                "--bulk-plasticity-mode", bulk,
                "--directional-j-mode", jmode,
                "--steps", "12", "--nx", "16", "--ny", "32",
                "--dU", "2e-6", "--dt", "0.01", "--n-stagger", "1",
                "--tip-h-fine", "2e-6", "--tip-ratio", "1.2",
                "--da-phys", "5e-6", "--target-crack-extension-um", "10",
                "--mpz-length-um", "100", "--mpz-n-bins", "200",
                "--wake-length-um", "100", "--wake-n-bins", "0",
                "--wake-shielding", "--crystal-aniso", "--crystal-compete",
                "--crystal-theta-deg", "45", "--crystal-material", "w",
                "--j-decomposition", "cluster", "--max-fronts", "1",
                "--print-every", "1", "--save-snapshots", "0", "--no-plots",
                "--out", str(out),
            ])
    except BaseException as exc:  # diagnostic must still emit the other cases
        error = f"{type(exc).__name__}: {exc}"
    text = buf.getvalue()
    kj = [float(v) for v in re.findall(r"KJ=\s*([0-9.eE+-]+)", text)]
    sig = [float(v) for v in re.findall(r"sig_tip=\s*([0-9.eE+-]+)GPa", text)]
    return {
        "bulk_mode": bulk,
        "j_mode": jmode,
        "error": error,
        "KJ_MPa_sqrt_m": kj,
        "sigma_tip_GPa": sig,
        "first_positive_index": next((i for i, v in enumerate(kj) if v > 0.0), None),
        "zero_after_positive": bool(kj and any(v == 0.0 for v in kj[(next((i for i, x in enumerate(kj) if x > 0.0), len(kj))):])),
        "stdout_tail": text.splitlines()[-20:],
    }


def main():
    root = Path(tempfile.mkdtemp(prefix="v10_0_1_drive_"))
    payload = {
        "schema": "v10_0_1_four_way_2d_driving_audit",
        "cases": [
            run_case(root, "tip_only", "abs_forward"),
            run_case(root, "tip_only", "root_signed"),
            run_case(root, "full_field", "abs_forward"),
            run_case(root, "full_field", "root_signed"),
        ],
    }
    path = Path("v10_0_1_2d_driving_audit.json")
    path.write_text(json.dumps(payload, indent=2))
    print(path)


if __name__ == "__main__":
    main()
