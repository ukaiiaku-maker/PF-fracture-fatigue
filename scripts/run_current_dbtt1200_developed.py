#!/usr/bin/env python3
"""Run the developed 1200 K DBTT PF path from the qualified short-pilot inputs."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import SharedOptions


SOURCE = (
    ROOT / "integration_evidence" / "current_science_pilots_6abbf10" /
    "pf" / "DBTT_1200K" / "accommodation_enabled" /
    "shared_execution_request.json"
)
DEFAULT_OUTPUT = (
    ROOT / "integration_evidence" / "current_science_pilots_6abbf10" /
    "pf" / "DBTT_1200K" / "developed_1000um_current_v2_run3"
)


def replace_argument(args: list[str], flag: str, value: str) -> None:
    index = args.index(flag)
    args[index + 1] = value


def main() -> int:
    pilot = json.loads(SOURCE.read_text())
    native_args = list(pilot["native_args"])
    replace_argument(native_args, "--steps", "2000000")
    replace_argument(native_args, "--target-crack-extension-um", "1000")
    replace_argument(native_args, "--print-every", "400")
    options = dict(pilot["options"])
    options["output"] = "selected_fields"
    output = Path(os.environ.get(
        "CURRENT_DBTT1200_OUTPUT", str(DEFAULT_OUTPUT)
    )).resolve()
    execute_shared(
        SharedOptions(**options), native_args, output,
        source_policy="qualified",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
