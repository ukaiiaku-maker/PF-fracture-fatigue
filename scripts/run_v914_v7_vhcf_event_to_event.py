#!/usr/bin/env python3
"""Run intrinsic reverse-glide v7 directly to a VHCF horizon or growth target."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_V914 = Path(
    os.environ.get(
        "V914_ROOT",
        "/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search",
    )
)
for _path in (str(ROOT / "scripts"), str(DEFAULT_V914)):
    while _path in sys.path:
        sys.path.remove(_path)
sys.path.insert(0, str(DEFAULT_V914))
sys.path.insert(0, str(ROOT / "scripts"))

from arrhenius_fracture import fatigue_v914 as base
from arrhenius_fracture.emergent_gnd_campaign_v913 import candidate_from_registry_row
from arrhenius_fracture.endurance_knee_v914 import physics_for_row
from arrhenius_fracture.emergent_gnd_types_v913 import CommonPhysics
from v914_adaptive_feedback_v6 import AdaptiveFeedbackControls
from v914_intrinsic_reverse_glide_v7 import IntrinsicReverseGlideState
from v914_signed_fatigue_loading import SignedFatigueLoading
from v914_v7_adaptive_block_accelerator import AdaptiveBlockControls
from v914_v7_vhcf_event_engine import (
    ENGINE_ID,
    VHCFRunControls,
    run_v7_vhcf_event_to_event,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--physics", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--deltaK", type=float, required=True)
    p.add_argument("--R", type=float, required=True)
    p.add_argument("--frequency-Hz", type=float, default=1000.0)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--seed", type=int, default=1720)
    p.add_argument("--n-bins", type=int, default=640)
    p.add_argument("--coupled-substeps", type=int, default=4)
    p.add_argument("--base-phase-intervals", type=int, default=256)
    p.add_argument("--state-rtol", type=float, default=0.0025)
    p.add_argument("--tip-radius-rtol", type=float, default=0.001)
    p.add_argument("--hazard-rtol", type=float, default=0.01)
    p.add_argument("--max-refinement-depth", type=int, default=18)

    p.add_argument("--minimum-exact-cycles", type=int, default=4)
    p.add_argument("--readiness-rtol", type=float, default=0.05)
    p.add_argument("--readiness-consecutive-passes", type=int, default=2)
    p.add_argument("--initial-block-stride", type=int, default=4)
    p.add_argument(
        "--maximum-block-stride",
        type=int,
        default=1 << 47,
        help="Power-of-two maximum. 2^47 exceeds 1e14 cycles.",
    )
    p.add_argument("--block-state-rtol", type=float, default=0.03)
    p.add_argument("--block-hazard-rtol", type=float, default=0.03)
    p.add_argument("--max-projection-correction", type=float, default=0.10)

    p.add_argument("--maximum-physical-cycles", type=int, default=10**14)
    p.add_argument("--maximum-cycle-map-evaluations", type=int, default=4096)
    p.add_argument("--heartbeat-map-evaluations", type=int, default=12)
    p.add_argument("--event-guard-stride", type=int, default=4)
    p.add_argument("--phase-localization-tolerance", type=float, default=1.0e-13)
    p.add_argument("--target-extension-um", type=float, default=100.0)
    p.add_argument("--base-event-length-nm", type=float, default=None)

    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--restart-from", type=Path, default=None)
    p.add_argument("--expected-head", default=None)
    p.add_argument("--out", type=Path, required=True)
    return p.parse_args()


def load_common(path: Path) -> CommonPhysics:
    values = json.loads(path.read_text())
    values = values.get("common_physics", values)
    names = {field.name for field in fields(CommonPhysics)}
    selected = {key: value for key, value in values.items() if key in names}
    for name in (
        "emission_signs",
        "emission_schmid_factors",
        "shielding_orientation_factors",
        "forest_interaction_matrix",
        "gnd_stress_projection_matrix",
        "activation_to_line_content_per_system",
        "emission_geometry_extension_m",
        "emission_geometry_factors",
    ):
        if name in selected:
            selected[name] = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in selected[name]
            )
    result = CommonPhysics(**selected)
    result.validate()
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    names: list[str] = []
    for row in rows:
        for key in row:
            if key not in names:
                names.append(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        for row in rows:
            clean = {}
            for key in names:
                value = row.get(key, "")
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value, sort_keys=True, allow_nan=True)
                clean[key] = value
            writer.writerow(clean)


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).strip()
    if args.expected_head and head != args.expected_head:
        raise SystemExit(f"HEAD mismatch: expected {args.expected_head}, found {head}")
    if args.expected_head and dirty:
        raise SystemExit("authoritative launch requires a clean worktree")

    rows = list(csv.DictReader(args.registry.open()))
    row = next((r for r in rows if r["candidate_id"] == args.candidate), None)
    if row is None:
        raise ValueError(f"candidate missing from registry: {args.candidate}")
    candidate = candidate_from_registry_row(row)
    common = replace(load_common(args.physics), n_bins=int(args.n_bins))
    physics = physics_for_row(common, row)
    loading = SignedFatigueLoading(
        args.deltaK,
        R=args.R,
        frequency_Hz=args.frequency_Hz,
        temperature_K=args.temperature_K,
        phase_steps=max(int(args.base_phase_intervals), 2),
    )
    loading.validate()
    IntrinsicReverseGlideState.coupled_operator_substeps = int(args.coupled_substeps)

    cycle_controls = AdaptiveFeedbackControls(
        state_rtol=args.state_rtol,
        tip_radius_rtol=args.tip_radius_rtol,
        hazard_rtol=args.hazard_rtol,
        base_phase_intervals=args.base_phase_intervals,
        max_refinement_depth=args.max_refinement_depth,
    )
    block_controls = AdaptiveBlockControls(
        minimum_exact_cycles=args.minimum_exact_cycles,
        readiness_relative_tolerance=args.readiness_rtol,
        readiness_consecutive_passes=args.readiness_consecutive_passes,
        initial_block_stride=args.initial_block_stride,
        maximum_block_stride=args.maximum_block_stride,
        block_state_rtol=args.block_state_rtol,
        block_hazard_rtol=args.block_hazard_rtol,
        max_projection_constraint_correction=args.max_projection_correction,
    )
    run_controls = VHCFRunControls(
        maximum_physical_cycles=args.maximum_physical_cycles,
        maximum_cycle_map_evaluations=args.maximum_cycle_map_evaluations,
        heartbeat_cycle_map_evaluations=args.heartbeat_map_evaluations,
        event_guard_stride=args.event_guard_stride,
        phase_localization_tolerance=args.phase_localization_tolerance,
    )
    cycle_controls.validate()
    block_controls.validate()
    run_controls.validate()

    numerics = base.FatigueNumerics()
    numerics = replace(
        numerics,
        target_extension_m=float(args.target_extension_um) * 1.0e-6,
    )
    if args.base_event_length_nm is not None:
        numerics = replace(
            numerics,
            base_event_length_m=float(args.base_event_length_nm) * 1.0e-9,
        )
    numerics.validate()

    checkpoint = args.checkpoint or (args.out / "checkpoint.json")
    result = run_v7_vhcf_event_to_event(
        candidate,
        physics,
        loading,
        seed=args.seed,
        cycle_controls=cycle_controls,
        block_controls=block_controls,
        run_controls=run_controls,
        numerics=numerics,
        checkpoint_path=checkpoint,
        restart_from=args.restart_from,
        contract_metadata={
            "repository_head": head,
            "registry_sha256": digest(args.registry),
            "physics_sha256": digest(args.physics),
        },
    )

    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    write_csv(args.out / "events.csv", result["events"])
    write_csv(args.out / "anchor_history.csv", result["anchor_history"])
    write_csv(args.out / "block_history.csv", result["block_history"])

    contract = {
        "schema": "v914_v7_vhcf_event_to_event_run_contract_v1",
        "engine_id": ENGINE_ID,
        "candidate": args.candidate,
        "R": args.R,
        "deltaK_MPa_sqrt_m": args.deltaK,
        "frequency_Hz": args.frequency_Hz,
        "temperature_K": args.temperature_K,
        "seed": args.seed,
        "maximum_physical_cycles": args.maximum_physical_cycles,
        "target_extension_um": args.target_extension_um,
        "maximum_block_stride": args.maximum_block_stride,
        "repository_branch": branch,
        "repository_head": head,
        "repository_clean": not bool(dirty),
        "registry_sha256": digest(args.registry),
        "physics_sha256": digest(args.physics),
        "within_cycle_law": "shared adaptive v7 advance_v7_cycle",
        "stochastic_law": "unchanged exponential first-passage threshold",
        "event_length_law": "unchanged v9.14 event_length_factor",
        "projective_blocks_may_cross_event": False,
    }
    (args.out / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )

    print(
        args.candidate,
        f"R={args.R:g}",
        f"status={result['status']}",
        f"cycles={result['completed_physical_cycles']}/{result['maximum_physical_cycles']}",
        f"events={result['event_count']}",
        f"extension_um={1e6*result['cumulative_extension_m']:.6g}",
        f"max_stride={result['maximum_accepted_stride']}",
        f"maps={result['total_cycle_map_evaluations']}",
        f"cycles_per_map={result['physical_cycles_per_total_cycle_map_evaluation']:.6g}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
