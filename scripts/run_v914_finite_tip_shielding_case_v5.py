#!/usr/bin/env python3
"""Run matched sharp and finite-tip reversible fatigue shielding audits."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
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
from arrhenius_fracture import fatigue_v914 as base

from v914_signed_fatigue_loading import SignedFatigueLoading
from v914_minimal_reversible_explicit_v4 import run_minimal_reversible_explicit as run_v4
from v914_minimal_reversible_explicit_v5 import (
    run_finite_tip_floor_explicit,
    run_finite_tip_shift_explicit,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--physics", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--deltaK", type=float, required=True)
    p.add_argument(
        "--shield-mode",
        choices=("sharp-v4", "radius-floor-v5", "radius-shift-v5"),
        required=True,
    )
    p.add_argument("--R", type=float, default=-0.95)
    p.add_argument("--frequency-Hz", type=float, default=1000.0)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--phase-steps", type=int, default=1024)
    p.add_argument("--n-bins", type=int, default=None)
    p.add_argument("--coupled-substeps", type=int, default=4)
    p.add_argument("--target-um", type=float, default=100.0)
    p.add_argument("--maximum-cycles", type=int, default=1)
    p.add_argument("--seed", type=int, default=1720)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--expected-head", default=None)
    return p.parse_args()


def load_common(path: Path) -> CommonPhysics:
    values = json.loads(path.read_text())
    values = values.get("common_physics", values)
    names = {field.name for field in fields(CommonPhysics)}
    selected = {key: value for key, value in values.items() if key in names}
    tuple_names = (
        "emission_signs",
        "emission_schmid_factors",
        "shielding_orientation_factors",
        "forest_interaction_matrix",
        "gnd_stress_projection_matrix",
        "activation_to_line_content_per_system",
        "emission_geometry_extension_m",
        "emission_geometry_factors",
    )
    for name in tuple_names:
        if name in selected:
            selected[name] = tuple(
                tuple(item) if isinstance(item, list) else item
                for item in selected[name]
            )
    return CommonPhysics(**selected)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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

    common = load_common(args.physics)
    if args.n_bins is not None:
        from dataclasses import replace
        common = replace(common, n_bins=int(args.n_bins))
    common.validate()
    physics = physics_for_row(common, row)

    loading = SignedFatigueLoading(
        args.deltaK,
        R=args.R,
        frequency_Hz=args.frequency_Hz,
        temperature_K=args.temperature_K,
        phase_steps=args.phase_steps,
    )
    loading.validate()

    numerics = base.FatigueNumerics(
        maximum_cycles=float(args.maximum_cycles),
        target_extension_m=args.target_um * 1e-6,
        maximum_explicit_cycles=max(args.maximum_cycles + 2, 4096),
    )

    if args.shield_mode == "sharp-v4":
        run = run_v4
    elif args.shield_mode == "radius-floor-v5":
        run = run_finite_tip_floor_explicit
    else:
        run = run_finite_tip_shift_explicit

    # Coupled-operator resolution is numerical, not a CommonPhysics field.
    # The explicit integrator constructs its state internally, so use the
    # authoritative environment-class contract inherited from v9.12 stiff.
    old = os.environ.get("MPZ_V912_COUPLED_OPERATOR_SUBSTEPS")
    os.environ["MPZ_V912_COUPLED_OPERATOR_SUBSTEPS"] = str(args.coupled_substeps)
    try:
        # The class variable was imported before the environment override above;
        # rebind it on all audit state classes through the module globals used by
        # each wrapper.
        import v914_minimal_reversible_state_v4 as s4
        import v914_finite_tip_shielding_state_v5 as s5
        s4.MinimalReversibleEmergentGNDState.coupled_operator_substeps = int(args.coupled_substeps)
        s5.FiniteTipFloorReversibleState.coupled_operator_substeps = int(args.coupled_substeps)
        s5.FiniteTipShiftReversibleState.coupled_operator_substeps = int(args.coupled_substeps)

        result = run(
            candidate,
            physics,
            loading,
            seed=args.seed,
            numerics=numerics,
            checkpoint_path=None,
            restart_from=None,
            maximum_physical_cycles=args.maximum_cycles,
            checkpoint_each_phase=False,
            checkpoint_cycle_interval=None,
            state_history_cycle_interval=1,
        )
    finally:
        if old is None:
            os.environ.pop("MPZ_V912_COUPLED_OPERATOR_SUBSTEPS", None)
        else:
            os.environ["MPZ_V912_COUPLED_OPERATOR_SUBSTEPS"] = old

    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    contract = {
        "schema": "v914_finite_tip_shielding_audit_contract_v1",
        "shield_mode": args.shield_mode,
        "candidate": args.candidate,
        "deltaK_MPa_sqrt_m": args.deltaK,
        "R": args.R,
        "frequency_Hz": args.frequency_Hz,
        "temperature_K": args.temperature_K,
        "phase_steps": args.phase_steps,
        "n_bins": physics.n_bins,
        "coupled_substeps": args.coupled_substeps,
        "maximum_cycles": args.maximum_cycles,
        "seed": args.seed,
        "repository_branch": branch,
        "repository_head": head,
        "repository_clean": not bool(dirty),
        "registry_sha256": digest(args.registry),
        "physics_sha256": digest(args.physics),
        "finite_tip_radius_source": "existing_dynamic_tip_radius_m",
        "fitted_shielding_parameter": False,
        "transport_storage_physics_changed_from_v4": False,
        "return_semantics_changed_from_v4": False,
    }
    (args.out / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )
    print(
        args.candidate,
        args.shield_mode,
        result["status"],
        result["final_cycles"],
        result["final_extension_m"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
