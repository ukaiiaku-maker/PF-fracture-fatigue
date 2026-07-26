#!/usr/bin/env python3
"""Write fixed-extension anchors for accepted production-state kernel capture."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.kernel_configuration_v10227 import load_configuration


def _round_up(value: float, quantum: float) -> float:
    return math.ceil((value - 1.0e-12 * quantum) / quantum) * quantum


def _maximum_event_um(da_um: float, minimum_factor: float, maximum_factor: float) -> float:
    lower = max(float(minimum_factor), 0.0)
    upper = max(float(maximum_factor), lower)
    clipped_mean = max(lower + math.exp(-lower) - math.exp(-upper), 1.0e-300)
    return float(da_um) * upper / clipped_mean


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mechanical-config", type=Path, required=True)
    parser.add_argument("--required-max-extension-um", type=float, required=True)
    parser.add_argument("--temperature-K", type=float, required=True)
    parser.add_argument("--event-minimum-factor", type=float, default=0.5)
    parser.add_argument("--event-maximum-factor", type=float, default=4.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    configuration = load_configuration(args.mechanical_config)
    required_um = float(args.required_max_extension_um)
    temperature = float(args.temperature_K)
    if not math.isfinite(required_um) or required_um <= 0.0:
        raise SystemExit("required maximum extension must be positive and finite")
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise SystemExit("capture temperature must be positive and finite")

    da_um = 1.0e6 * configuration.da_phys_m
    spacing_um = max(
        _round_up(1.0e6 * configuration.atlas_anchor_spacing_m, da_um),
        da_um,
    )
    maximum_um = _round_up(required_um, spacing_um)
    count = max(int(round(maximum_um / spacing_um)), 1)
    maximum_event_um = _maximum_event_um(
        da_um,
        args.event_minimum_factor,
        args.event_maximum_factor,
    )
    # The production trajectory uses variable stochastic event lengths.  Capture
    # the nearest accepted equilibrium within one maximum admissible event of the
    # nominal anchor instead of requiring an artificial fixed 5 um landing.
    tolerance_um = max(1.05 * maximum_event_um, 0.51 * da_um, 1.0e-3)
    tolerance_m = tolerance_um * 1.0e-6

    rows = []
    for index in range(count + 1):
        extension_um = index * spacing_um
        rows.append(
            {
                "state_id": f"E{int(round(extension_um)):07d}",
                "temperature_K": temperature,
                "cumulative_crack_path_extension_m": extension_um * 1.0e-6,
                "extension_tolerance_m": tolerance_m,
                "interaction_ell_m": configuration.interaction_length_m,
            }
        )

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"state_table={output}")
    print(f"state_count={len(rows)}")
    print(f"anchor_spacing_um={spacing_um:.17g}")
    print(f"maximum_extension_um={maximum_um:.17g}")
    print(f"maximum_event_um={maximum_event_um:.17g}")
    print(f"extension_tolerance_um={tolerance_um:.17g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
