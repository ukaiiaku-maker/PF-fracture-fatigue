#!/usr/bin/env python3
"""Run the v10.2.3 shared-state reduced model without a FEM solve."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from arrhenius_fracture.reduced_shared_state_v1023 import (
    DEFAULT_THETA45_DRIVE_FACTORS,
    SharedReducedConfig,
    fallback_registry,
    load_manifest,
    replay_shared_state,
    run_monotonic_shared_front,
    write_shared_result,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--operation", choices=["monotonic", "replay"], default="monotonic")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate")
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--temperature-K", type=float)
    parser.add_argument("--schedule", type=Path)
    parser.add_argument(
        "--mode",
        choices=["full", "plasticity_off", "shielding_off", "backstress_off"],
        default="full",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--Kdot-MPa-sqrt-m-s", type=float, default=0.005)
    parser.add_argument("--Kmax-MPa-sqrt-m", type=float, default=80.0)
    parser.add_argument("--max-dK-step-MPa-sqrt-m", type=float, default=0.01)
    parser.add_argument("--target-extension-um", type=float, default=5.0)
    parser.add_argument("--checkpoint-da-um", type=float, default=5.0)
    parser.add_argument("--r0-m", type=float, default=1.0e-6)
    parser.add_argument("--mpz-length-um", type=float, default=100.0)
    parser.add_argument("--mpz-n-bins", type=int, default=200)
    parser.add_argument("--wake-length-um", type=float, default=100.0)
    parser.add_argument("--wake-n-bins", type=int, default=0)
    parser.add_argument("--source-bin-count", type=int, default=2)
    parser.add_argument("--blunting-length-um", type=float, default=0.5)
    parser.add_argument("--shielding-core-m", type=float, default=2.5e-10)
    parser.add_argument("--forest-density-floor-m2", type=float, default=5.0e12)
    parser.add_argument("--mobile-shield-fraction", type=float, default=1.0)
    parser.add_argument("--G-Pa", type=float, default=160.0e9)
    parser.add_argument("--poisson", type=float, default=0.28)
    parser.add_argument("--b-m", type=float, default=2.74e-10)
    parser.add_argument("--cleavage-hits", type=float, default=3.0)
    parser.add_argument("--cleavage-tau-s", type=float, default=1.0e-6)
    parser.add_argument(
        "--drive-factors",
        type=float,
        nargs=2,
        metavar=("F0", "F1"),
        default=list(DEFAULT_THETA45_DRIVE_FACTORS),
    )
    parser.add_argument("--crystal-theta-deg", type=float, default=45.0)
    parser.add_argument(
        "--transport-mode",
        choices=["validated_scalar", "channel_resolved"],
        default="validated_scalar",
    )
    parser.add_argument("--max-action-substep", type=float, default=0.01)
    parser.add_argument("--max-translation-substep-m", type=float, default=5.0e-8)
    parser.add_argument("--max-internal-steps", type=int, default=20000)
    parser.add_argument("--max-outer-steps", type=int, default=2_000_000)
    parser.add_argument("--write-fallback-registry", action="store_true")
    return parser


def _config(args: argparse.Namespace) -> SharedReducedConfig:
    return SharedReducedConfig(
        Kdot_MPa_sqrt_m_s=args.Kdot_MPa_sqrt_m_s,
        Kmax_MPa_sqrt_m=args.Kmax_MPa_sqrt_m,
        max_dK_step_MPa_sqrt_m=args.max_dK_step_MPa_sqrt_m,
        target_extension_um=args.target_extension_um,
        checkpoint_da_um=args.checkpoint_da_um,
        r0_m=args.r0_m,
        mpz_length_um=args.mpz_length_um,
        mpz_n_bins=args.mpz_n_bins,
        wake_length_um=args.wake_length_um,
        wake_n_bins=args.wake_n_bins,
        source_bin_count=args.source_bin_count,
        blunting_length_um=args.blunting_length_um,
        shielding_core_m=args.shielding_core_m,
        forest_density_floor_m2=args.forest_density_floor_m2,
        mobile_shield_fraction=args.mobile_shield_fraction,
        G_Pa=args.G_Pa,
        poisson=args.poisson,
        b_m=args.b_m,
        cleavage_hits=args.cleavage_hits,
        cleavage_tau_s=args.cleavage_tau_s,
        drive_factors=tuple(args.drive_factors),
        crystal_theta_deg=args.crystal_theta_deg,
        transport_mode=args.transport_mode,
        max_action_substep=args.max_action_substep,
        max_translation_substep_m=args.max_translation_substep_m,
        max_internal_steps=args.max_internal_steps,
        max_outer_steps=args.max_outer_steps,
    ).validate()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.out.exists():
        raise SystemExit(
            f"output already exists: {args.out}\n"
            "Use a new versioned path; this runner does not overwrite results."
        )
    if args.operation == "monotonic" and args.temperature_K is None:
        raise SystemExit("--temperature-K is required for monotonic operation")
    if args.operation == "replay" and args.schedule is None:
        raise SystemExit("--schedule is required for replay operation")
    if args.operation == "replay" and not args.schedule.is_file():
        raise SystemExit(f"replay schedule not found: {args.schedule}")

    manifest = load_manifest(
        candidate_id=args.candidate,
        manifest_path=args.manifest,
    )
    cfg = _config(args)
    args.out.mkdir(parents=True, exist_ok=False)

    if args.operation == "monotonic":
        result = run_monotonic_shared_front(
            manifest,
            float(args.temperature_K),
            cfg,
            mode=args.mode,
        )
    else:
        result = replay_shared_state(
            manifest,
            args.schedule,
            cfg,
            mode=args.mode,
        )
    write_shared_result(result, args.out)
    if args.write_fallback_registry:
        (args.out / "fallback_parameterizations.json").write_text(
            json.dumps(fallback_registry(), indent=2)
        )

    display = {
        "schema": result["schema"],
        "operation": args.operation,
        "candidate_id": result["candidate_id"],
        "mode": result["mode"],
        "temperature_K": result.get("temperature_K"),
        "status": result.get("status", "replayed"),
        "K_first_MPa_sqrt_m": result.get("K_first_MPa_sqrt_m"),
        "n_schedule_rows": result.get("n_schedule_rows"),
        "raw_equals_effective_when_shielding_active": result[
            "raw_equals_effective_when_shielding_active"
        ],
        "legacy_manifest_cap_used_in_kinetics": False,
        "out": str(args.out.resolve()),
    }
    print(json.dumps(display, indent=2), flush=True)
    if not result["raw_equals_effective_when_shielding_active"]:
        return 3
    if args.operation == "monotonic" and result.get("status") != "complete":
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
