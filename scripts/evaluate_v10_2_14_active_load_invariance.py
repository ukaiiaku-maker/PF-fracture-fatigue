#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import arrhenius_fracture

_EXPECTED_PROJECT = "PF-fracture-fatigue"
_PACKAGE_ROOT = Path(arrhenius_fracture.__file__).resolve().parent
if getattr(arrhenius_fracture, "PROJECT_ID", None) != _EXPECTED_PROJECT:
    raise RuntimeError(
        "wrong arrhenius_fracture distribution imported: expected "
        f"{_EXPECTED_PROJECT!r}, got "
        f"{getattr(arrhenius_fracture, 'PROJECT_ID', None)!r} from {_PACKAGE_ROOT}"
    )
if _PACKAGE_ROOT != REPOSITORY_ROOT / "arrhenius_fracture":
    raise RuntimeError(
        "stale editable arrhenius_fracture import detected: expected "
        f"{REPOSITORY_ROOT / 'arrhenius_fracture'}, got {_PACKAGE_ROOT}"
    )

from arrhenius_fracture import physical_fem_station_responses_v10212 as _station_responses

_STATION_RESPONSE_SCHEMA = (
    "v10.2.14_exact_endpoint_active_signed_spatial_station_responses"
)


def _exact_first_last_station_indices(
    coordinates: tuple[float, ...],
    minimum_spacing_m: float,
    minimum_distance_m: float = 0.0,
) -> list[int]:
    """Restore the frozen v10.2.14 active-station selection contract.

    The first and last exact MPZ bins are always retained.  Intermediate bins are
    selected only by the established spacing rule.  ``minimum_distance_m`` is
    accepted for call compatibility but may not omit, move, or snap physical bins.
    If bin zero is not resolvable, exact endpoint construction fails closed later.
    """
    values = [float(value) for value in coordinates]
    if not values:
        return []
    if any(not math.isfinite(value) for value in values):
        raise ValueError("MPZ coordinates must be finite")
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError("MPZ coordinates must be nondecreasing")
    spacing = float(minimum_spacing_m)
    if not math.isfinite(spacing) or spacing <= 0.0:
        raise ValueError("minimum station spacing must be positive and finite")
    minimum_distance = float(minimum_distance_m)
    if not math.isfinite(minimum_distance) or minimum_distance < 0.0:
        raise ValueError("minimum station distance must be finite and nonnegative")

    selected = [0]
    for index in range(1, len(values) - 1):
        if values[index] - values[selected[-1]] >= spacing:
            selected.append(index)
    if len(values) > 1 and selected[-1] != len(values) - 1:
        selected.append(len(values) - 1)
    return selected


_station_responses.MODEL_ID = _STATION_RESPONSE_SCHEMA
_station_responses._station_indices = _exact_first_last_station_indices

from arrhenius_fracture.frozen_geometry_load_invariance_v10213 import (
    evaluate_frozen_geometry_load_invariance,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--load-scales", type=float, nargs="+", required=True)
    parser.add_argument("--magnitudes", type=float, nargs="+", required=True)
    parser.add_argument("--linearity-tolerance", type=float, default=0.03)
    parser.add_argument("--load-invariance-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--minimum-residual-stiffness-fraction", type=float, default=1.0e-3
    )
    parser.add_argument("--ribbon-width-m", type=float)
    parser.add_argument("--minimum-station-spacing-m", type=float)
    args = parser.parse_args()

    payload = evaluate_frozen_geometry_load_invariance(
        args.snapshot,
        outroot=args.outroot,
        load_scales=args.load_scales,
        perturbation_magnitudes=args.magnitudes,
        ribbon_width_m=args.ribbon_width_m,
        minimum_station_spacing_m=args.minimum_station_spacing_m,
        linearity_tolerance=args.linearity_tolerance,
        load_invariance_tolerance=args.load_invariance_tolerance,
        minimum_residual_stiffness_fraction=(
            args.minimum_residual_stiffness_fraction
        ),
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "station_response_schema": _STATION_RESPONSE_SCHEMA,
                "active_station_policy": "exact_first_last_no_omission",
                "parent_state_id": payload["parent_state_id"],
                "load_invariance_passed": payload["load_invariance_passed"],
                "active_kernel_mechanically_measured": payload[
                    "active_kernel_mechanically_measured"
                ],
                "wake_shielding_supported": payload[
                    "wake_shielding_supported"
                ],
                "maximum_within_load_relative_spread": payload["checks"][
                    "maximum_within_load_relative_spread"
                ],
                "maximum_relative_load_variation": payload["checks"][
                    "maximum_relative_load_variation"
                ],
                "project_id": arrhenius_fracture.PROJECT_ID,
                "package_root": str(_PACKAGE_ROOT),
                "report": str(
                    (
                        args.outroot
                        / "frozen_geometry_load_invariance.json"
                    ).resolve()
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
