#!/usr/bin/env python3
"""Adaptive four-class production points toward da/dN = 1e-3 m/cycle."""
from __future__ import annotations

import importlib

try:
    from scripts import v10230_driving_force_ladder_supervisor as ladder
    from scripts import v10230_qualification_supervisor as qualification
except ModuleNotFoundError:
    import v10230_driving_force_ladder_supervisor as ladder
    import v10230_qualification_supervisor as qualification

ladder = importlib.reload(ladder)

TARGET_POINTS = {
    "peak": (1.180, 1.190),
    "dbtt": (1.130, 1.140),
    "weakT": (1.190, 1.195),
    "ceramic": (1.250, 1.300),
}
ladder.MANIFEST_NAME = "four_class_1e3_rate_matrix.json"


def matrix() -> list[dict]:
    rows = []
    for label, fractions in TARGET_POINTS.items():
        option, seed, reference = qualification.OPTIONS[label]
        for fraction in fractions:
            delta_k = reference * fraction
            kmax = delta_k / (1.0 - ladder.R)
            rows.append({
                "case": ladder.case_name(label, fraction, seed),
                "label": label,
                "parameter_option": option,
                "seed": seed,
                "fraction": fraction,
                "reference_deltaK_MPa_sqrt_m": reference,
                "deltaK_MPa_sqrt_m": delta_k,
                "Kmax_MPa_sqrt_m": kmax,
                "Kmin_MPa_sqrt_m": ladder.R * kmax,
                "R": ladder.R,
                "frequency_Hz": ladder.FREQUENCY_HZ,
                "temperature_K": ladder.TEMPERATURE_K,
                "target_extension_um": ladder.TARGET_EXTENSION_UM,
                "cycle_horizon": ladder.CYCLES_MAX,
            })
    return rows


ladder.matrix = matrix


if __name__ == "__main__":
    raise SystemExit(ladder.main())
