#!/usr/bin/env python3
"""Run the adaptive block-growth v7 accelerator without an independent exact baseline."""
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

from arrhenius_fracture.emergent_gnd_campaign_v913 import candidate_from_registry_row
from arrhenius_fracture.endurance_knee_v914 import physics_for_row
from arrhenius_fracture.emergent_gnd_types_v913 import CommonPhysics
from v914_adaptive_feedback_v6 import AdaptiveFeedbackControls
from v914_intrinsic_reverse_glide_v7 import IntrinsicReverseGlideState
from v914_signed_fatigue_loading import SignedFatigueLoading
from v914_v7_adaptive_block_accelerator import (
    ACCELERATOR_ID,
    AdaptiveBlockControls,
    run_adaptive_block_multicycle,
)
from v914_v7_cycle_map import CYCLE_MAP_ID
from v914_v7_projective_state import PROJECTOR_ID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--physics", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--deltaK", type=float, required=True)
    p.add_argument("--R", type=float, required=True)
    p.add_argument("--frequency-Hz", type=float, default=1000.0)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--n-bins", type=int, default=640)
    p.add_argument("--coupled-substeps", type=int, default=4)
    p.add_argument("--base-phase-intervals", type=int, default=256)
    p.add_argument("--state-rtol", type=float, default=0.0025)
    p.add_argument("--tip-radius-rtol", type=float, default=0.001)
    p.add_argument("--hazard-rtol", type=float, default=0.01)
    p.add_argument("--max-refinement-depth", type=int, default=18)
    p.add_argument("--cycles", type=int, default=40)
    p.add_argument("--minimum-exact-cycles", type=int, default=4)
    p.add_argument("--readiness-rtol", type=float, default=0.05)
    p.add_argument("--readiness-consecutive-passes", type=int, default=2)
    p.add_argument("--initial-block-stride", type=int, default=4)
    p.add_argument("--maximum-block-stride", type=int, default=32)
    p.add_argument("--block-state-rtol", type=float, default=0.03)
    p.add_argument("--block-hazard-rtol", type=float, default=0.03)
    p.add_argument("--max-projection-correction", type=float, default=0.10)
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
        writer.writerows(rows)


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
    initial = IntrinsicReverseGlideState(candidate, physics)
    cycle_controls = AdaptiveFeedbackControls(
        state_rtol=args.state_rtol,
        tip_radius_rtol=args.tip_radius_rtol,
        hazard_rtol=args.hazard_rtol,
        base_phase_intervals=args.base_phase_intervals,
        max_refinement_depth=args.max_refinement_depth,
    )
    cycle_controls.validate()
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
    block_controls.validate()

    final_state, anchor_history, block_history, metadata = (
        run_adaptive_block_multicycle(
            initial,
            loading,
            cycle_controls,
            args.cycles,
            block_controls=block_controls,
        )
    )

    write_csv(args.out / "anchor_history.csv", anchor_history)
    write_csv(args.out / "block_history.csv", block_history)
    write_csv(args.out / "readiness_history.csv", metadata["readiness_history"])

    result = {
        "schema": "v914_v7_adaptive_block_growth_v1",
        "cycle_map_id": CYCLE_MAP_ID,
        "accelerator_id": ACCELERATOR_ID,
        "projector_id": PROJECTOR_ID,
        "cycles": args.cycles,
        "metadata": metadata,
        "anchor_history": anchor_history,
        "block_history": block_history,
        "final_reversibility": final_state.reversibility_diagnostics(),
    }
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    contract = {
        "schema": "v914_v7_adaptive_block_growth_contract_v1",
        "candidate": args.candidate,
        "R": args.R,
        "deltaK_MPa_sqrt_m": args.deltaK,
        "frequency_Hz": args.frequency_Hz,
        "temperature_K": args.temperature_K,
        "n_bins": physics.n_bins,
        "coupled_substeps": args.coupled_substeps,
        "adaptive_cycle_controls": vars(cycle_controls),
        "adaptive_block_controls": vars(block_controls),
        "repository_branch": branch,
        "repository_head": head,
        "repository_clean": not bool(dirty),
        "registry_sha256": digest(args.registry),
        "physics_sha256": digest(args.physics),
        "cycle_map_id": CYCLE_MAP_ID,
        "accelerator_id": ACCELERATOR_ID,
        "projector_id": PROJECTOR_ID,
        "accelerator_within_cycle_law": "same_advance_v7_cycle",
        "accepted_solution": "fine_two_half_block_path",
        "coarse_path_private": True,
    }
    (args.out / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )

    print(
        args.candidate,
        f"R={args.R:g}",
        f"cycles={args.cycles}",
        f"promotion={metadata['promotion_cycle']}",
        f"blocks={metadata['accepted_block_count']}",
        f"rejected={metadata['rejected_block_count']}",
        f"max_stride={metadata['maximum_accepted_stride']}",
        f"map_evals={metadata['total_cycle_map_evaluations']}",
        f"actual_speedup={metadata['actual_cycle_map_evaluation_speedup']:.6g}",
        f"state_err_max={metadata['maximum_accepted_state_error']:.6g}",
        f"hazard_err_max={metadata['maximum_accepted_hazard_error']:.6g}",
        f"Hcum={metadata['cumulative_hazard_action']:.6g}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
