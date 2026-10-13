#!/usr/bin/env python3
"""Qualify the F1 persistent-source path and its fresh-process restart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import OutputProfile, SharedOptions


SOURCE_REQUEST = (
    ROOT / "integration_evidence" / "final_fracture_campaign" /
    "baseline_dbtt_950K_accommodation_enabled" / "shared_execution_request.json"
)
DEFAULT_ROOT = ROOT / "integration_evidence" / "f1_active_native_qualification"
INITIAL_STEPS = 48
FINAL_STEPS = 64


def _replace(args: list[str], flag: str, value: str) -> None:
    args[args.index(flag) + 1] = value


def _configuration(steps: int) -> tuple[SharedOptions, list[str]]:
    request = json.loads(SOURCE_REQUEST.read_text())
    options = dict(request["options"])
    options["output"] = OutputProfile.PUBLICATION_RESTART.value
    native = list(request["native_args"])
    _replace(native, "--steps", str(steps))
    return SharedOptions.from_dict(options), native


def _run(output: Path, steps: int, restart_from: Path | None = None) -> None:
    if output.exists():
        raise SystemExit(f"refusing to overwrite qualification output: {output}")
    output.mkdir(parents=True)
    options, native = _configuration(steps)
    execute_shared(
        options, native, output, source_policy="qualified",
        restart_from=restart_from,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=("initial", "resume", "uninterrupted", "all"), default="all", nargs="?",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    values = parser.parse_args(argv)
    root = values.root.resolve()
    initial = root / "initial_to_step48"
    resumed = root / "fresh_process_restart_to_step64"
    uninterrupted = root / "uninterrupted_to_step64"
    if values.phase in {"initial", "all"}:
        _run(initial, INITIAL_STEPS)
    if values.phase in {"resume", "all"}:
        checkpoint = initial / "checkpoint"
        if not (checkpoint / "run_state_checkpoint.json").is_file():
            raise SystemExit(f"complete outer checkpoint is missing: {checkpoint}")
        _run(resumed, FINAL_STEPS, checkpoint)
    if values.phase in {"uninterrupted", "all"}:
        _run(uninterrupted, FINAL_STEPS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
