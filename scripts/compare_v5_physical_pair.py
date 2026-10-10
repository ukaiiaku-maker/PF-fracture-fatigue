#!/usr/bin/env python3
"""Compare physical enabled/suppressed V5 checkpoints without conflating metadata."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.voiding_production_v5 import observables


FIELDS = (
    ("displacement", "m"), ("ep_gp", "strain"), ("rho_gp", "m^-2"),
    ("damage", "1"), ("vacancy_fraction_gp", "1"),
)


def _load_pair(path: Path):
    payload = json.loads(path.read_text())
    result = {}
    for arm, raw in payload["arms"].items():
        source = Path(raw)
        if not source.is_absolute():
            source = path.parent / source
        result[arm] = restore_checkpoint(source)
    return payload, result


def _field(left, right, name, unit):
    a = np.asarray(getattr(left, name), dtype=float)
    b = np.asarray(getattr(right, name), dtype=float)
    delta = a - b
    scale = max(float(np.linalg.norm(a)), float(np.linalg.norm(b)))
    return {
        "unit": unit, "shape": list(a.shape), "exact": bool(np.array_equal(a, b)),
        "max_abs_difference": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "l2_difference": float(np.linalg.norm(delta)),
        "relative_l2_difference": None if scale == 0.0 else float(np.linalg.norm(delta) / scale),
    }


def _numeric_observables(left, right):
    a = observables(left, "paired_comparison")
    b = observables(right, "paired_comparison")
    rows = {}
    for key in sorted(set(a).intersection(b)):
        if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
            x, y = float(a[key]), float(b[key])
            if math.isfinite(x) and math.isfinite(y):
                scale = max(abs(x), abs(y))
                rows[key] = {
                    "disabled": x, "enabled": y, "absolute_difference": abs(y - x),
                    "relative_difference": None if scale == 0.0 else abs(y - x) / scale,
                }
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_checkpoint", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--physical-time-s", type=float, required=True)
    args = parser.parse_args(argv)
    manifest, states = _load_pair(args.pair_checkpoint.resolve())
    disabled, enabled = states["disabled"], states["enabled"]
    payload = {
        "schema": "v5-physical-paired-state-comparison/1",
        "physical_time_s": args.physical_time_s,
        "pair_checkpoint": str(args.pair_checkpoint.resolve()),
        "pair_checkpoint_sha256": hashlib.sha256(args.pair_checkpoint.read_bytes()).hexdigest(),
        "structural_differences": {
            "cavity_enabled": {"disabled": False, "enabled": True},
            "void_owner_present": {"disabled": disabled.void_state is not None, "enabled": enabled.void_state is not None},
            "cavity_rng_independent": enabled.void_state is not None and enabled.void_state.rng_state is not None,
        },
        "shared_mesh_exact": bool(
            np.array_equal(disabled.mesh.nodes, enabled.mesh.nodes)
            and np.array_equal(disabled.mesh.elems, enabled.mesh.elems)
        ),
        "shared_fields": {name: _field(disabled, enabled, name, unit) for name, unit in FIELDS},
        "fracture_rng_exact": dict(disabled.rng_state) == dict(enabled.rng_state),
        "crack_network_exact": disabled.crack_network.to_json() == enabled.crack_network.to_json(),
        "observables": _numeric_observables(disabled, enabled),
        "terminal_fingerprints": {
            arm: complete_accepted_state_fingerprint(state) for arm, state in states.items()
        },
        "manifest_fingerprints_verified": all(
            complete_accepted_state_fingerprint(states[arm]) == expected
            for arm, expected in manifest["fingerprints"].items()
        ),
    }
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
