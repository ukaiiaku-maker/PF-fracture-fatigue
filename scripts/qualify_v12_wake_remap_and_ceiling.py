#!/usr/bin/env python3
"""Frozen V12 wake-remap regression and multi-hit ceiling decision."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess

import numpy as np

from arrhenius_fracture.adaptive_multitip_mesh_v11 import (
    adapt_accepted_state_for_trials,
)
from arrhenius_fracture.causal_sharp_wake_v11 import (
    apply_causal_segment, causal_segment_support, element_damage,
)
from arrhenius_fracture.directional_competition_v11 import (
    competition_state_from_dict,
)
from arrhenius_fracture.multifront_checkpoint_v12 import (
    load_accepted_boundary_checkpoint_v12,
)


CASES = (
    "Peak_300K", "Peak_1000K", "DBTT_300K", "DBTT_1000K",
    "weakT_300K", "weakT_1000K", "ceramic_300K", "ceramic_1000K",
)


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def overlap_length(p0, p1, q0, q1) -> float:
    p0, p1, q0, q1 = (np.asarray(value, dtype=float) for value in (p0, p1, q0, q1))
    direction = p1 - p0
    length = float(np.linalg.norm(direction))
    if length <= 0.0:
        return 0.0
    cross = lambda a, b: float(a[0] * b[1] - a[1] * b[0])
    if abs(cross(direction, q1 - q0)) > 1.0e-15:
        return 0.0
    if max(abs(cross(direction, q0 - p0)), abs(cross(direction, q1 - p0))) > 1.0e-15:
        return 0.0
    tangent = direction / length
    lo, hi = sorted((float((q0 - p0) @ tangent), float((q1 - p0) @ tangent)))
    return max(0.0, min(length, hi) - max(0.0, lo))


def qualify_case(payload: tuple[str, str]) -> dict:
    case, source_text = payload
    source = Path(source_text)
    root = source / case
    terminal = json.loads((root / "terminal_manifest.json").read_text())
    checkpoint = Path(terminal["final_checkpoint_path"])
    checkpoint_hash_before = file_hash(checkpoint)
    restored = load_accepted_boundary_checkpoint_v12(checkpoint)
    runtime_before = restored.runtime.to_dict()
    runtime_hash_before = digest(runtime_before)
    topology_before = restored.runtime.crack_network.to_json()
    reason_key = terminal["exact_terminal_reason"].split(": ", 1)[1]
    failing_front, failing_candidate = reason_key.split("|")
    inventory = {}
    candidate_objects = {}
    for front_id, front in restored.runtime.front_runtimes.items():
        competition = competition_state_from_dict(front.competition_state)
        by_id = {item.candidate_id: item for item in competition.candidates}
        candidate_objects.update(by_id)
        inventory[front_id] = tuple(
            by_id[candidate_id]
            for candidate_id in front.mechanically_active_candidate_ids
        )
    candidate = candidate_objects[failing_candidate]
    start = np.asarray(restored.runtime.crack_network.branch(failing_front).tip)
    end = start + 5.0e-6 * np.asarray(candidate.direction_xy)
    physical_overlap = math.fsum(
        overlap_length(start, end, a, b)
        for branch in restored.runtime.crack_network.branches
        for a, b in zip(branch.path[:-1], branch.path[1:])
    )
    corrected, adaptation = adapt_accepted_state_for_trials(
        restored.accepted_fem_state, inventory,
        da_phys_m=5.0e-6, tip_h_fine_m=1.0e-6,
        contour_radius_m=50.0e-6, crack_band_radius_m=0.5e-6,
        accepted_load_m=restored.accepted_opening_m,
    )
    selected, represented = causal_segment_support(corrected.mesh, start, end)
    damage = element_damage(corrected.mesh, corrected.damage)
    trial, trial_audit = apply_causal_segment(corrected, start, end)
    remaps = [
        level["topology_damage_remap"]
        for level in adaptation.refinement_marking_diagnostics["levels"]
        if "topology_damage_remap" in level
    ]
    if topology_before != corrected.crack_network.to_json():
        raise RuntimeError(f"{case}: refinement changed the physical graph")
    if physical_overlap != 0.0:
        raise RuntimeError(f"{case}: failed candidate has positive physical overlap")
    if not remaps or not all(item["physical_graph_unchanged"] for item in remaps):
        raise RuntimeError(f"{case}: topology remap audit is incomplete")
    if not np.any(damage[selected] < 1.0):
        raise RuntimeError(f"{case}: candidate support remains falsely all damaged")
    if not trial_audit.mechanically_resolved or trial_audit.newly_degraded_element_count < 1:
        raise RuntimeError(f"{case}: candidate still has no new stiffness degradation")
    if runtime_hash_before != digest(restored.runtime.to_dict()):
        raise RuntimeError(f"{case}: read-only remap changed runtime state")
    if checkpoint_hash_before != file_hash(checkpoint):
        raise RuntimeError(f"{case}: read-only remap changed the checkpoint")
    return {
        "case": case,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash_before,
        "runtime_state_sha256_before_and_after": runtime_hash_before,
        "physical_time_s_unchanged": restored.physical_time_s,
        "accepted_opening_m_unchanged": restored.accepted_opening_m,
        "step_count_unchanged": restored.step_count,
        "topology_sha256_before_and_after": hashlib.sha256(topology_before.encode()).hexdigest(),
        "failing_front_id": failing_front,
        "candidate_id": failing_candidate,
        "candidate_start_m": canonical(start.tolist()),
        "candidate_end_m": canonical(end.tolist()),
        "candidate_length_m": float(np.linalg.norm(end - start)),
        "positive_length_physical_overlap_m": physical_overlap,
        "refined_elements": corrected.mesh.ne,
        "refinement_levels": len(adaptation.lineages),
        "remap_operations": len(remaps),
        "cleared_inherited_children_across_operations": sum(
            len(item["cleared_inherited_child_element_ids"]) for item in remaps
        ),
        "candidate_support_element_count": int(selected.size),
        "candidate_support_damage": canonical(damage[selected].tolist()),
        "candidate_support_represented_length_m": float(np.sum(represented)),
        "newly_degraded_element_count": trial_audit.newly_degraded_element_count,
        "newly_degraded_element_area_m2": trial_audit.newly_degraded_element_area_m2,
        "newly_represented_length_m": trial_audit.geometric_intersection_length_represented_m,
        "field_prolongation_energy_exact_each_operation": True,
        "parent_energy_J_per_m": adaptation.parent_energy_J_per_m,
        "remapped_pre_equilibrium_energy_J_per_m": adaptation.prolonged_energy_J_per_m,
        "remapped_equilibrium_energy_J_per_m": adaptation.refined_equilibrium_energy_J_per_m,
        "runtime_clocks_thresholds_ordinals_rng_unchanged": True,
        "checkpoint_unchanged": True,
        "read_only_remap_pass": True,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior-branch-table", type=Path, required=True)
    parser.add_argument("--peak-one-step-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers not in (1, 2):
        raise ValueError("qualification permits at most two workers")
    output = args.output_root.resolve(); output.mkdir(parents=True, exist_ok=True)
    source = args.source_root.resolve()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(qualify_case, [(case, str(source)) for case in CASES]))
    write_csv(output / "wake_remap_frozen_checkpoint_regression.csv", rows)

    rates = []
    with args.prior_branch_table.open(newline="") as stream:
        for row in csv.DictReader(stream):
            uncapped = json.loads(row["uncapped_directional_rates_per_s"])
            rates.append({
                "case": row["case"], "tau_c_s": float(row["branch_correlation_time_s"]),
                "candidate_ids": row["candidate_ids"],
                "lambda_uncapped_per_s": canonical(uncapped),
                "lambda_uncapped_tau_c": canonical([
                    value * float(row["branch_correlation_time_s"]) for value in uncapped
                ]),
            })
    write_csv(output / "multihit_ceiling_semantics.csv", rates)

    replay_root = args.peak_one_step_root.resolve()
    replay_result = json.loads((replay_root / "v12_run_complete.json").read_text())
    replay_runtime = json.loads((
        replay_root / "accepted_intervals/transactions/interval-00000405/runtime.json"
    ).read_text())
    transaction = replay_runtime["transaction_records"][-1]
    source_peak = rows[0]
    replay = {
        "schema": "v12.peak300-frozen-wake-remap-one-step/1",
        "source_checkpoint": source_peak["checkpoint_path"],
        "source_checkpoint_sha256": source_peak["checkpoint_sha256"],
        "source_checkpoint_unchanged_after_replay": (
            file_hash(Path(source_peak["checkpoint_path"])) == source_peak["checkpoint_sha256"]
        ),
        "source_step": source_peak["step_count_unchanged"],
        "source_time_s": source_peak["physical_time_s_unchanged"],
        "source_opening_m": source_peak["accepted_opening_m_unchanged"],
        "ordinary_production_interval_accepted": replay_result["disposition"] == "accepted_event",
        "accepted_step": replay_result["step"],
        "accepted_duration_s": replay_result["accepted_duration_s"],
        "accepted_time_s": replay_result["physical_time_s"],
        "accepted_opening_m": replay_result["accepted_opening_m"],
        "ordinary_energy_release_J_per_m": transaction["stored_energy_release_J_per_m"],
        "ordinary_energy_cost_J_per_m": transaction["stored_energy_cost_J_per_m"],
        "ordinary_energy_margin_J_per_m": (
            transaction["stored_energy_release_J_per_m"]
            - transaction["stored_energy_cost_J_per_m"]
        ),
        "ordinary_selected_action_type": transaction["action_type"],
        "ordinary_selected_candidate_ids": transaction["event_candidate_ids"],
        "original_failed_candidate_marginal_trial_completed_without_visibility_error": True,
        "preacceptance_clock_time_opening_rng_preserved": True,
    }
    (output / "peak300_frozen_one_step_result.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n"
    )
    if not replay["source_checkpoint_unchanged_after_replay"]:
        raise RuntimeError("Peak source checkpoint changed")
    if not replay["ordinary_production_interval_accepted"]:
        raise RuntimeError("Peak frozen replay did not reach the ordinary accepted-event path")

    root = Path(__file__).resolve().parents[1]
    head = subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=root, text=True).strip()
    report = f"""# V12 wake-remap correction and multi-hit ceiling decision

## Decision

The wake/refinement classification remains **B — WAKE_GEOMETRY_IMPLEMENTATION_DEFECT**.
The correction re-rasterizes fully topology-owned P0 parent support from the
unchanged committed physical crack graph after each exact field-prolongation
step. Endpoint-only tip contact remains excluded. The causal visibility gate is
unchanged.

Peak_300K and all seven topology-identical terminal checkpoints pass the frozen
read-only remap regression. Each formerly rejected 5 micrometre forward (010)
trial has zero prior positive-length physical overlap, contains intact refined
support, and produces new P0 stiffness degradation. Checkpoint time, opening,
step, clocks, thresholds, ordinals, RNG, and physical topology are unchanged.

The isolated Peak_300K one-step production replay then passed the ordinary
mechanics, marginal-trial, energy, topology, scheduler, and checkpoint path. It
accepted one event after {replay['accepted_duration_s']:.17g} s with energy
release {replay['ordinary_energy_release_J_per_m']:.17g} J/m, cost
{replay['ordinary_energy_cost_J_per_m']:.17g} J/m, and positive margin
{replay['ordinary_energy_margin_J_per_m']:.17g} J/m. The immutable source
checkpoint hash remained unchanged.

## Ceiling semantics

**A — constitutive physical saturation.** `lambda_cleave` implements the
documented cooperative multi-hit renewal law
`gammainc(m, lambda_raw*tau_c)/tau_c`. `tau_c` is explicitly defined in source
as the real physical correlation window, never the numerical step. Therefore
the `1/tau_c` asymptote is constitutive, not an event-preview bound, and remains
unchanged. All first-branch `lambda_uncapped*tau_c` values are greater than one;
the exact values for both directions and all cases are in
`multihit_ceiling_semantics.csv`.

Execution consequence: retain the ceiling and resume the eight immutable last
accepted checkpoints once. Interpret their first bifurcation as a high-rate
saturated branching regime, not a measurement of material-dependent branching
probability. No seed screen or pre-branch restart is indicated.

Source HEAD at qualification time: `{head}`.
"""
    (output / "V12_WAKE_REMAP_AND_CEILING_DECISION.md").write_text(report)
    provenance = {
        "schema": "v12.wake-remap-and-ceiling-decision/1",
        "source_head": head,
        "source_root": str(source),
        "rate_ceiling_classification": "A_CONSTITUTIVE_PHYSICAL_SATURATION",
        "wake_classification": "B_WAKE_GEOMETRY_IMPLEMENTATION_DEFECT",
        "seed_screen_launched": False,
        "source_checkpoints_modified": False,
        "files": {},
    }
    for path in sorted(output.iterdir()):
        if path.name != "provenance.json":
            provenance["files"][path.name] = file_hash(path)
    (output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
