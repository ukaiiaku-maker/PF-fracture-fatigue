#!/usr/bin/env python3
"""Restart one accepted hot-path state at three adaptive resolution floors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.run_state_checkpoint_v10230 import MODEL_ID
from arrhenius_fracture.shared_execution import execute_shared
from arrhenius_fracture.shared_options import SharedOptions


SOURCE_RUN = (
    ROOT / "integration_evidence" / "current_science_pilots_6abbf10" /
    "pf" / "DBTT_1200K" / "developed_1000um_current_v2_run3"
)
SOURCE_REQUEST = SOURCE_RUN / "shared_execution_request.json"
GENERATION = "85f8ef062b9a298f1f2a92fb"
SOURCE_GENERATION = SOURCE_RUN / "checkpoint" / "run_state_generations" / GENERATION
EXPECTED_SHA256 = {
    "outer.json": "420c3b2a68a0fc40dad1c65adda05fe9f6561e7e057b575b6edab2f8d77b03b2",
    "kinetic.json": "ee9303a01b339fdf895217456eb61ee5532bc0eab5eacaf756db68022d617731",
    "state.npz": "fd58ea62bd11f5c3c5d68828bdfc7dd412b17f944b76126892fd6d3453889fe6",
}
FLOORS = (1e-8, 1e-9, 1e-10)
DEFAULT_OUTPUT = (
    ROOT / "integration_evidence" / "active_fracture_comparisons_0a5d55e" /
    "hot_late_floor_refinement"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_argument(args: list[str], flag: str, value: str) -> None:
    if flag in args:
        args[args.index(flag) + 1] = value
    else:
        args.extend([flag, value])


def native_args_for_floor(source: list[str], floor: float) -> list[str]:
    args = list(source)
    replace_argument(args, "--steps", "1710")
    # The accepted family is measured through 1030 um.  The resolver reserves
    # one 30 um maximum stochastic event, making 1000 um the qualified target.
    replace_argument(args, "--target-crack-extension-um", "1000")
    replace_argument(args, "--print-every", "1")
    replace_argument(args, "--adaptive-min-frac", f"{floor:.0e}")
    return args


def prepare_restart_root(root: Path) -> Path:
    restart = root / "restart_input_generation_85f8ef"
    destination = restart / "run_state_generations" / GENERATION
    destination.mkdir(parents=True)
    observed = {name: file_sha256(SOURCE_GENERATION / name) for name in EXPECTED_SHA256}
    if observed != EXPECTED_SHA256:
        raise RuntimeError(f"accepted checkpoint digest mismatch: {observed}")
    for name in EXPECTED_SHA256:
        shutil.copy2(SOURCE_GENERATION / name, destination / name)
    manifest = {
        "schema": MODEL_ID,
        "generation": GENERATION,
        "created_unix_s": 0.0,
        "files": EXPECTED_SHA256,
        "source": str(SOURCE_GENERATION),
        "purpose": "immutable late pre-event input for adaptive-floor refinement",
    }
    (restart / "run_state_checkpoint.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return restart


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--plan-only", action="store_true")
    parsed = parser.parse_args(argv)
    output = parsed.out.resolve()
    request = json.loads(SOURCE_REQUEST.read_text())
    plan = {
        "schema": "hot-late-adaptive-floor-refinement/1",
        "source_request": str(SOURCE_REQUEST),
        "source_request_sha256": file_sha256(SOURCE_REQUEST),
        "source_generation": GENERATION,
        "source_generation_sha256": EXPECTED_SHA256,
        "source_step": 1701,
        "source_events": 202,
        "source_crack_extension_um": 998.1919262543385,
        "floors": list(FLOORS),
        "common_endpoint": {"maximum_step": 1710, "target_crack_extension_um": 1000.0},
        "invariants": {
            "nominal_dU_m": 2e-7,
            "nominal_dt_s": 8.4,
            "nominal_loading_rate_m_per_s": 2e-7 / 8.4,
            "hazard_mode": "exponential",
            "event_length_mode": "threshold_scaled",
            "event_length_factor_bounds": [0.5, 4.0],
            "hazard_seed": 1733,
        },
    }
    if parsed.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if output.exists():
        raise SystemExit(f"refusing to overwrite existing refinement root: {output}")
    output.mkdir(parents=True)
    (output / "RUN_MATRIX.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    restart = prepare_restart_root(output)
    os.environ.update({
        "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5",
        "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
    })
    options = SharedOptions(**request["options"])
    for floor in FLOORS:
        label = f"floor_{floor:.0e}".replace("-", "m")
        execute_shared(
            options,
            native_args_for_floor(request["native_args"], floor),
            output / label,
            source_policy="qualified",
            restart_from=restart,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
