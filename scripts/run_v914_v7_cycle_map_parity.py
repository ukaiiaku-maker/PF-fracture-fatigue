#!/usr/bin/env python3
"""Run bounded v7 shear, exact multi-cycle, or projective-accelerator probes."""
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
from v914_v7_cycle_map import (
    CYCLE_MAP_ID,
    advance_v7_cycle,
    minimum_shear_sample,
)
from v914_v7_multicycle_accelerator import (
    ACCELERATOR_ID as SCAFFOLD_ACCELERATOR_ID,
    run_accelerator_anchor_path,
    run_exact_multicycle,
)
from v914_v7_projective_accelerator import (
    ACCELERATOR_ID as PROJECTIVE_ACCELERATOR_ID,
    ProjectiveAcceleratorControls,
    compare_projective_to_exact,
    run_projective_multicycle,
)
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
    p.add_argument(
        "--mode", choices=("shear", "multicycle", "projective"), required=True
    )
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--verify-anchor-parity", action="store_true")
    p.add_argument("--warmup-cycles", type=int, default=4)
    p.add_argument("--block-stride", type=int, default=2)
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
    controls = AdaptiveFeedbackControls(
        state_rtol=args.state_rtol,
        tip_radius_rtol=args.tip_radius_rtol,
        hazard_rtol=args.hazard_rtol,
        base_phase_intervals=args.base_phase_intervals,
        max_refinement_depth=args.max_refinement_depth,
    )
    controls.validate()

    active_accelerator_id = SCAFFOLD_ACCELERATOR_ID
    cycle_skipping_enabled = False
    projective_controls = None

    if args.mode == "shear":
        final_state, hazard, telemetry = advance_v7_cycle(initial, loading, controls)
        samples = telemetry.pop("samples")
        shear = minimum_shear_sample(samples)
        result = {
            "schema": "v914_v7_shear_audit_v1",
            "cycle_map_id": CYCLE_MAP_ID,
            "hazard_action": hazard,
            "minimum_shear_sample": shear,
            "reversibility": final_state.reversibility_diagnostics(),
            "telemetry": telemetry,
        }
        write_csv(args.out / "adaptive_phase_history.csv", samples)
        print(
            args.candidate,
            f"R={args.R:g}",
            f"phase={shear['phase']:.9g}",
            f"tau_app={shear['tau_applied_projected_GPa']:.6g}GPa",
            f"tau_gnd={shear['tau_gnd_projected_GPa']:.6g}GPa",
            f"tau_eff={shear['tau_eff_projected_GPa']:.6g}GPa",
            f"return={result['reversibility'].get('reversible_physical_return_fraction_of_emitted',0.0):.6g}",
        )

    elif args.mode == "multicycle":
        final_state, exact, telemetry = run_exact_multicycle(
            initial, loading, controls, args.cycles
        )
        write_csv(args.out / "exact_cycle_history.csv", exact)
        write_csv(args.out / "cycle_telemetry.csv", telemetry)
        parity = None
        if args.verify_anchor_parity:
            _, anchor, _ = run_accelerator_anchor_path(
                initial, loading, controls, args.cycles, anchor_stride=1
            )
            if len(anchor) != len(exact):
                raise RuntimeError("anchor parity history length mismatch")
            maxdiff = 0.0
            for a, b in zip(exact, anchor):
                for key in (
                    "hazard_action",
                    "shielding_MPa_sqrt_m",
                    "mobile_line_content",
                    "retained_line_content",
                    "returned_source_slip",
                    "tip_radius_m",
                ):
                    maxdiff = max(maxdiff, abs(float(a[key]) - float(b[key])))
            parity = {
                "anchor_stride": 1,
                "max_absolute_difference": maxdiff,
                "pass": bool(maxdiff == 0.0),
            }
            if not parity["pass"]:
                raise RuntimeError(f"exact/accelerator anchor parity failed: {parity}")
        result = {
            "schema": "v914_v7_multicycle_cycle_map_probe_v1",
            "cycle_map_id": CYCLE_MAP_ID,
            "accelerator_id": SCAFFOLD_ACCELERATOR_ID,
            "cycles": args.cycles,
            "exact_history": exact,
            "telemetry": telemetry,
            "anchor_parity": parity,
            "final_reversibility": final_state.reversibility_diagnostics(),
            "cycle_skipping_enabled": False,
        }
        print(
            args.candidate,
            f"R={args.R:g}",
            f"cycles={args.cycles}",
            f"hazard_last={exact[-1]['hazard_action']:.6g}",
            f"rtip_last_um={1e6*exact[-1]['tip_radius_m']:.6g}",
            f"return_last={exact[-1]['physical_return_fraction']:.6g}",
            f"parity={None if parity is None else parity['pass']}",
        )

    else:
        active_accelerator_id = PROJECTIVE_ACCELERATOR_ID
        cycle_skipping_enabled = True
        projective_controls = ProjectiveAcceleratorControls(
            warmup_cycles=int(args.warmup_cycles),
            block_stride=int(args.block_stride),
            max_projection_constraint_correction=float(
                args.max_projection_correction
            ),
        )
        projective_controls.validate()

        # Independent exact reference and accelerated calculation.  They share
        # the cycle law, not mutable state objects.
        exact_final, exact, exact_telemetry = run_exact_multicycle(
            initial, loading, controls, args.cycles
        )
        accel_final, accelerated, accel_telemetry, accel_metadata = (
            run_projective_multicycle(
                initial,
                loading,
                controls,
                args.cycles,
                accelerator_controls=projective_controls,
            )
        )
        comparison = compare_projective_to_exact(
            exact_final,
            exact,
            accel_final,
            accelerated,
            warmup_cycles=projective_controls.warmup_cycles,
        )
        write_csv(args.out / "exact_cycle_history.csv", exact)
        write_csv(args.out / "accelerated_cycle_history.csv", accelerated)
        write_csv(args.out / "exact_cycle_telemetry.csv", exact_telemetry)
        write_csv(args.out / "accelerated_cycle_telemetry.csv", accel_telemetry)
        result = {
            "schema": "v914_v7_projective_accelerator_qualification_v1",
            "cycle_map_id": CYCLE_MAP_ID,
            "accelerator_id": PROJECTIVE_ACCELERATOR_ID,
            "projector_id": PROJECTOR_ID,
            "cycles": args.cycles,
            "accelerator_metadata": accel_metadata,
            "comparison": comparison,
            "exact_history": exact,
            "accelerated_history": accelerated,
            "exact_final_reversibility": exact_final.reversibility_diagnostics(),
            "accelerated_final_reversibility": accel_final.reversibility_diagnostics(),
            "cycle_skipping_enabled": True,
        }
        print(
            args.candidate,
            f"R={args.R:g}",
            f"cycles={args.cycles}",
            f"resolved={accel_metadata['resolved_cycle_count']}",
            f"skipped={accel_metadata['projected_cycle_count']}",
            f"ideal_speedup={accel_metadata['ideal_cycle_map_speedup']:.4g}",
            f"Hpost_err={comparison['post_warmup_cumulative_hazard_relative_error']:.6g}",
            f"fullstate_max={max(comparison['final_full_state_relative_norm_error'].values(), default=0.0):.6g}",
            f"qualification={comparison['pass']}",
        )

    contract = {
        "schema": "v914_v7_cycle_map_contract_v2",
        "mode": args.mode,
        "candidate": args.candidate,
        "R": args.R,
        "deltaK_MPa_sqrt_m": args.deltaK,
        "frequency_Hz": args.frequency_Hz,
        "temperature_K": args.temperature_K,
        "n_bins": physics.n_bins,
        "coupled_substeps": args.coupled_substeps,
        "adaptive_controls": vars(controls),
        "repository_branch": branch,
        "repository_head": head,
        "repository_clean": not bool(dirty),
        "registry_sha256": digest(args.registry),
        "physics_sha256": digest(args.physics),
        "cycle_map_id": CYCLE_MAP_ID,
        "accelerator_id": active_accelerator_id,
        "accelerator_within_cycle_law": "same_advance_v7_cycle",
        "cycle_skipping_enabled": cycle_skipping_enabled,
        "projective_controls": (
            None if projective_controls is None else vars(projective_controls)
        ),
        "projector_id": PROJECTOR_ID if projective_controls is not None else None,
        "skipped_cycle_hazard_rule": (
            "log_bridge_between_resolved_anchor_hazards"
            if projective_controls is not None
            else None
        ),
        "crack_extension_during_projection_allowed": False,
        "physics_parameters_changed_for_acceleration": False,
    }
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    (args.out / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
