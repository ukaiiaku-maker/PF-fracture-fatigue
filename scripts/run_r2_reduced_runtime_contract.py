#!/usr/bin/env python3
"""Execute and attest the R2 persistent/reversible reduced runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from arrhenius_fracture.persistent_site_high_cycle_checkpoint_v10230 import (
    capture_ledgers, restore_checkpoint, write_checkpoint,
)
from arrhenius_fracture.persistent_site_high_cycle_state_v10230 import (
    serialize_active_state,
)
from arrhenius_fracture.reduced_persistent_reference_r2 import (
    advance_signed, build_restored_engine, runtime_receipt,
)
from arrhenius_fracture.reduced_shared_state_v1023 import SharedReducedConfig


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--temperature-K", type=float, default=1200.0)
    parser.add_argument("--forward-K-MPa-sqrt-m", type=float, default=10.0)
    parser.add_argument("--reverse-K-MPa-sqrt-m", type=float, default=-10.0)
    parser.add_argument("--reload-K-MPa-sqrt-m", type=float, default=20.0)
    parser.add_argument("--duration-s", type=float, default=1.0e-3)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"output already exists: {args.out}")
    args.out.mkdir(parents=True)
    cfg = SharedReducedConfig(
        mpz_length_um=50.0, mpz_n_bins=80, wake_length_um=50.0,
        max_internal_steps=100000,
    )
    engine = build_restored_engine(cfg)
    initial = runtime_receipt(engine, stage="initialized")
    forward = advance_signed(
        engine, K_Pa_sqrt_m=args.forward_K_MPa_sqrt_m * 1.0e6,
        temperature_K=args.temperature_K, dt_s=args.duration_s,
    )
    after_forward = runtime_receipt(engine, stage="accepted_forward_step")
    before = serialize_active_state(engine)
    ledgers_before = capture_ledgers(engine)
    checkpoint = args.out / "checkpoint"
    old = os.environ.get("V10230_HIGH_CYCLE_CHECKPOINT_DIR")
    os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = str(checkpoint)
    try:
        write_checkpoint(
            engine, temperature_K=args.temperature_K,
            reason="r2_runtime_contract", metadata={"model": engine._r2_model_id},
        )
    finally:
        if old is None:
            os.environ.pop("V10230_HIGH_CYCLE_CHECKPOINT_DIR", None)
        else:
            os.environ["V10230_HIGH_CYCLE_CHECKPOINT_DIR"] = old

    restarted = build_restored_engine(cfg)
    kinetic_payload = json.loads(
        (checkpoint / "high_cycle_live_checkpoint.json").read_text()
    )
    geometry = kinetic_payload["geometry_signature"]
    restarted.n_adv = int(geometry[0])
    restarted.a_adv = float(geometry[1])
    restarted.micro_advance_total_m = float(geometry[2])
    restarted.checkpoint_advance_total_m = float(geometry[3])
    restarted.mpz.advance_total_m = float(geometry[4])
    restore_checkpoint(restarted, checkpoint)
    after_restart = runtime_receipt(restarted, stage="fresh_process_restart")
    restored = serialize_active_state(restarted)
    ledgers_after = capture_ledgers(restarted)
    exact = bool(
        np.array_equal(before.vector, restored.vector)
        and before.fields == restored.fields
        and before.geometry_signature == restored.geometry_signature
        and ledgers_before == ledgers_after
    )
    reverse = advance_signed(
        restarted, K_Pa_sqrt_m=args.reverse_K_MPa_sqrt_m * 1.0e6,
        temperature_K=args.temperature_K, dt_s=args.duration_s,
    )
    after_reverse = runtime_receipt(restarted, stage="accepted_reverse_step")
    reload_result = advance_signed(
        restarted, K_Pa_sqrt_m=args.reload_K_MPa_sqrt_m * 1.0e6,
        temperature_K=args.temperature_K, dt_s=args.duration_s,
    )
    after_reload = runtime_receipt(restarted, stage="accepted_reload_step")
    payload = {
        "schema": "r2-reduced-runtime-contract-execution/1",
        "configuration": vars(args) | {"out": str(args.out.resolve())},
        "checkpoint_exact": exact,
        "forward": forward,
        "reverse": reverse,
        "reload": reload_result,
        "return_chain": {
            "negative_applied_load": args.reverse_K_MPa_sqrt_m < 0.0,
            "negative_local_transport_drive": float(reverse.get("reversible_mobile_velocity_m_s", 0.0)) < 0.0,
            "actual_reverse_motion": float(reverse.get("dN_returned", 0.0)) > 0.0,
            "surface_arrival": float(reverse.get("dN_physical_returned", 0.0)) > 0.0,
            "completed_physical_return": float(restarted.mpz.cumulative_gross_return_activity) > 0.0,
        },
        "receipts": [initial, after_forward, after_restart, after_reverse, after_reload],
    }
    (args.out / "r2_runtime_contract.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n"
    )
    print(json.dumps({
        "checkpoint_exact": exact,
        "return_chain": payload["return_chain"],
        "forward_source_activity": after_forward["owning_arrays"]["accumulated_slip_positive"]["sum"],
        "out": str(args.out.resolve()),
    }, indent=2))
    return 0 if exact and all(payload["return_chain"].values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
