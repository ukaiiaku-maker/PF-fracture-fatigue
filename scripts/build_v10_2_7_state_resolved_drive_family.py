#!/usr/bin/env python3
"""Build the v10.2.7 signed emission-drive family from 2-D tensor probes."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from arrhenius_fracture.signed_kernel_family_v1026 import (
    STATE_AXES,
    StateResolvedSignedShieldingKernelFamily,
)
from arrhenius_fracture.state_resolved_drive_family_v1027 import SCHEMA


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--kernel-family", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--relative-repeat-tolerance", type=float, default=0.03)
    parser.add_argument("--authorize-production-parameterization", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    kernel = StateResolvedSignedShieldingKernelFamily.from_json(args.kernel_family)
    with args.responses.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "state_id",
        *STATE_AXES,
        "system",
        "sigma_local_Pa",
        "tau_signed_Pa",
        "probe_reliable",
    }
    if not rows:
        raise SystemExit("empty 2-D tensor-probe response table")
    missing = sorted(required.difference(rows[0]))
    if missing:
        raise SystemExit(f"drive response table is missing {missing}")

    groups: dict[tuple[str, int], list[float]] = defaultdict(list)
    coordinates: dict[str, tuple[float, float, float]] = {}
    metadata: dict[str, dict[str, str]] = {}
    for row in rows:
        state_id = str(row["state_id"])
        reliable = str(row["probe_reliable"]).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not reliable:
            raise SystemExit(f"unreliable 2-D tensor probe for state {state_id}")
        system = int(row["system"])
        sigma = float(row["sigma_local_Pa"])
        tau = float(row["tau_signed_Pa"])
        if not math.isfinite(sigma) or sigma <= 0.0 or not math.isfinite(tau):
            raise SystemExit(f"invalid tensor-probe values for state {state_id}")
        factor = tau / sigma
        if abs(factor) > 1.0 + 1.0e-9:
            raise SystemExit(
                f"normalized signed shear exceeds unity for state {state_id}, system {system}"
            )
        coord = tuple(float(row[name]) for name in STATE_AXES)
        old = coordinates.setdefault(state_id, coord)
        if not np.allclose(old, coord, rtol=1.0e-12, atol=1.0e-15):
            raise SystemExit(f"inconsistent coordinates for state {state_id}")
        groups[(state_id, system)].append(factor)
        metadata.setdefault(
            state_id,
            {
                key: value
                for key, value in row.items()
                if key not in required
            },
        )

    kernel_by_id = {state.state_id: state.coordinates for state in kernel.states}
    if set(coordinates) != set(kernel_by_id):
        raise SystemExit(
            "drive response states must exactly match the signed shielding-kernel states"
        )
    states = []
    repeat_checks = []
    for state_id in sorted(coordinates):
        coord = np.asarray(coordinates[state_id], dtype=float)
        if not np.allclose(coord, kernel_by_id[state_id], rtol=1.0e-12, atol=1.0e-15):
            raise SystemExit(f"kernel/drive coordinates differ for state {state_id}")
        factors = []
        for system in range(kernel.n_systems):
            values = np.asarray(groups.get((state_id, system), []), dtype=float)
            if values.size == 0:
                raise SystemExit(f"state {state_id} lacks system {system}")
            spread = float(np.max(values) - np.min(values))
            scale = max(float(np.max(np.abs(values))), 1.0e-12)
            relative = spread / scale
            if relative > float(args.relative_repeat_tolerance):
                raise SystemExit(
                    f"repeat tensor probes vary by {relative:.4g} for state {state_id}, "
                    f"system {system}"
                )
            factors.append(float(np.mean(values)))
            repeat_checks.append(
                {
                    "state_id": state_id,
                    "system": system,
                    "n_repeats": int(values.size),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                    "relative_spread": relative,
                }
            )
        state = {
            "state_id": state_id,
            **{name: float(value) for name, value in zip(STATE_AXES, coord)},
            "signed_tau_over_sigma_by_system": factors,
            **metadata.get(state_id, {}),
        }
        states.append(state)

    kernel_authorized = bool(
        kernel.metadata.get("production_parameterization_allowed", False)
    )
    authorized = bool(args.authorize_production_parameterization and kernel_authorized)
    payload = {
        "schema": SCHEMA,
        "candidate_independent": True,
        "derived_from_2d_tensor_probe": True,
        "signed_resolved_shear": True,
        "normalized_by_local_opening_stress": True,
        "fitted_to_toughness_or_fatigue": False,
        "state_axes": list(STATE_AXES),
        "kernel_family": str(args.kernel_family.resolve()),
        "kernel_family_production_authorized": kernel_authorized,
        "production_parameterization_allowed": authorized,
        "authorization_requires_independent_review": True,
        "interpolation": {
            "method": "inverse_distance",
            "neighbors": min(8, len(states)),
            "power": 2.0,
            "envelope_relative_tolerance": 1.0e-10,
        },
        "repeat_relative_tolerance": float(args.relative_repeat_tolerance),
        "repeat_checks": repeat_checks,
        "states": states,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_states": len(states),
                "n_systems": kernel.n_systems,
                "production_parameterization_allowed": authorized,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
