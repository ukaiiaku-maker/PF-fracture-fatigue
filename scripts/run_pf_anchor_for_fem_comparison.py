#!/usr/bin/env python3
"""Run the current PF anchor matched to the 1000 K FEM/CZM condition."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import SharedOptions


SOURCE = (
    ROOT / "integration_evidence" / "current_science_pilots_6abbf10" /
    "pf" / "Peak_900K" / "accommodation_enabled" / "shared_execution_request.json"
)
DEFAULT_OUTPUT = (
    ROOT / "integration_evidence" / "active_fracture_comparisons_0a5d55e" /
    "pf_peak1000_theta0_seed8666_anchor"
)


def replace_argument(args: list[str], flag: str, value: str) -> None:
    args[args.index(flag) + 1] = value


def configured_request() -> tuple[SharedOptions, list[str]]:
    request = json.loads(SOURCE.read_text())
    native = list(request["native_args"])
    replace_argument(native, "--temperatures", "1000")
    replace_argument(native, "--target-crack-extension-um", "10")
    configured = dict(request["options"])
    configured["random_seed"] = 8666
    configured["output"] = "selected_fields"
    return SharedOptions(**configured), native


def matrix() -> dict:
    options, native = configured_request()
    return {
        "schema": "pf-anchor-for-current-fem-comparison/1",
        "source_request": str(SOURCE),
        "source_request_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "material_option": "v913_paper_peak01_0242980_persistent_sites",
        "temperature_K": 1000.0,
        "crystal_theta_deg": 0.0,
        "hazard_seed": 8666,
        "nominal_dU_m": 2e-7,
        "nominal_dt_s": 8.4,
        "target_crack_extension_um": 10.0,
        "options": options.to_dict(),
        "native_args": native,
        "intentional_backend_difference": "PF sharp-wake versus FEM adaptive-CZM",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    parsed = parser.parse_args(argv)
    plan = matrix()
    if parsed.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    output = parsed.out.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing PF anchor: {output}")
    output.mkdir(parents=True)
    (output / "RUN_MATRIX.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    options, native = configured_request()
    execute_shared(options, native, output, source_policy="qualified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
