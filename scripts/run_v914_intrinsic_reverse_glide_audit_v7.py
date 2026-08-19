#!/usr/bin/env python3
"""Run one-cycle adaptive intrinsic reverse-glide v7 audits.

This runner preserves the v6 adaptive feedback integrator and v5 finite-tip
shielding.  The only constitutive change is the v7 mobile-glide drive:

    tau_glide = tau_signed_applied(finite tip) + tau_GND.

Nonlocal K_shield remains in the opening/cleavage channel but is not subtracted
again from mobile transport.  No crack-closure function, return fraction, or
new kinetic coefficient is introduced.
"""
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

import numpy as np

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

from v914_signed_fatigue_loading import SignedFatigueLoading
from v914_intrinsic_reverse_glide_v7 import IntrinsicReverseGlideState
from v914_adaptive_feedback_v6 import (
    AdaptiveFeedbackControls,
    adaptive_one_cycle,
    state_observables,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--physics", type=Path, required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--deltaK", type=float, required=True)
    p.add_argument("--R", type=float, default=-0.95)
    p.add_argument("--frequency-Hz", type=float, default=1000.0)
    p.add_argument("--temperature-K", type=float, default=300.0)
    p.add_argument("--n-bins", type=int, default=640)
    p.add_argument("--coupled-substeps", type=int, default=4)
    p.add_argument("--base-phase-intervals", type=int, default=256)
    p.add_argument("--state-rtol", type=float, default=0.0025)
    p.add_argument("--tip-radius-rtol", type=float, default=0.001)
    p.add_argument("--hazard-rtol", type=float, default=0.01)
    p.add_argument("--max-refinement-depth", type=int, default=18)
    p.add_argument("--expected-head", default=None)
    p.add_argument("--out", type=Path, required=True)
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
    result = CommonPhysics(**selected)
    result.validate()
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    names: list[str] = []
    seen = set()
    for record in records:
        for key in record:
            if key not in seen:
                names.append(key)
                seen.add(key)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(records)


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
    common.validate()
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
        state_rtol=float(args.state_rtol),
        tip_radius_rtol=float(args.tip_radius_rtol),
        hazard_rtol=float(args.hazard_rtol),
        base_phase_intervals=int(args.base_phase_intervals),
        max_refinement_depth=int(args.max_refinement_depth),
    )
    controls.validate()

    final_state, hazard, telemetry = adaptive_one_cycle(initial, loading, controls)
    final_obs = state_observables(final_state)
    reversibility = final_state.reversibility_diagnostics()
    samples = telemetry.pop("samples")

    ktr = np.asarray(
        [sample["transport_K_signed_MPa_sqrt_m"] for sample in samples], dtype=float
    )
    ksh = np.asarray(
        [sample["shielding_MPa_sqrt_m"] for sample in samples], dtype=float
    )
    depths = np.asarray([sample["depth"] for sample in samples], dtype=int)

    # Additional v7 diagnostics: the transport-K compatibility field is now the
    # signed applied loading coordinate, while transport shielding is carried
    # only through the local GND stress field.
    final_rates = final_state.local_rates(
        loading.K_at_phase(1.0), loading.temperature_K
    )
    final_tau = np.asarray(final_rates["reversible_tau_transport_eff_Pa"], dtype=float)

    result = {
        "schema": "v914_intrinsic_reverse_glide_audit_v7_result_v1",
        "status": "adaptive_one_cycle_complete",
        "state_model": final_state.integration_metadata()["model_id"],
        "hazard_action": float(hazard),
        "final": final_obs,
        "reversibility": reversibility,
        "transport_K_min_MPa_sqrt_m": float(np.min(ktr)),
        "transport_K_max_MPa_sqrt_m": float(np.max(ktr)),
        "max_abs_shielding_MPa_sqrt_m": float(np.max(np.abs(ksh))),
        "final_transport_tau_min_Pa": float(np.min(final_tau)),
        "final_transport_tau_max_Pa": float(np.max(final_tau)),
        "maximum_sample_depth": int(np.max(depths)),
        "telemetry": telemetry,
        "sample_count": len(samples),
    }
    (args.out / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=True) + "\n"
    )
    write_csv(args.out / "adaptive_phase_history.csv", samples)

    contract = {
        "schema": "v914_intrinsic_reverse_glide_audit_v7_contract_v1",
        "candidate": args.candidate,
        "deltaK_MPa_sqrt_m": args.deltaK,
        "R": args.R,
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
        "finite_tip_shielding": "radius_floor_v5",
        "mobile_glide_drive": "signed_applied_finite_tip_plus_local_gnd",
        "mobile_glide_subtracts_K_shield": False,
        "peierls_forward_reverse_parameters_shared": True,
        "physical_return_fraction_parameter": False,
        "crack_closure_law": False,
        "persistent_unsigned_backstress_promoted_to_kinematic": False,
        "stochastic_fracture_events_enabled": False,
        "emission_physics_changed_from_v6": False,
        "cleavage_physics_changed_from_v6": False,
        "surface_return_semantics_changed_from_v6": False,
    }
    (args.out / "run_contract.json").write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n"
    )

    print(
        args.candidate,
        "intrinsic-reverse-glide-v7",
        "adaptive_one_cycle_complete",
        f"tol={controls.state_rtol:g}",
        f"Ksh_end={final_obs['shielding_MPa_sqrt_m']:.8g}",
        f"Kapp_min={result['transport_K_min_MPa_sqrt_m']:.8g}",
        f"return={reversibility.get('reversible_physical_return_fraction_of_emitted', 0.0):.8g}",
        f"accepted={telemetry['accepted_intervals']}",
        f"refined={telemetry['refined_intervals']}",
        f"depth={telemetry['maximum_depth_reached']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
