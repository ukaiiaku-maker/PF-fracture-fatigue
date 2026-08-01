#!/usr/bin/env python3
"""Audit completed v10.4.0 cases for selective reuse in v10.4.1."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.reuse_v1040_v1041 import audit_campaign


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--source-commit",
        default="37549595157b67d3c3444cdddcd948aada02a420",
    )
    parser.add_argument(
        "--target-commit",
        required=True,
        help="Exact v10.4.1 commit that will consume the reuse manifest.",
    )
    parser.add_argument("--target-extension-um", type=float, default=1000.0)
    parser.add_argument("--theta-deg", type=float, default=0.0)
    parser.add_argument("--loading-rate-factor", type=float, default=1.0)
    parser.add_argument(
        "--max-cumulative-equivalent-strain-difference",
        type=float,
        default=1.0e-6,
        help=(
            "Maximum conservative accumulated |delta ep_eq| permitted for reuse. "
            "Cases exceeding this value are marked for rerun."
        ),
    )
    parser.add_argument("--max-stress-GPa", type=float, default=50.0)
    parser.add_argument("--stress-points", type=int, default=401)
    parser.add_argument("--rho-points", type=int, default=65)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = audit_campaign(
        args.source_root,
        output_json=args.output,
        source_commit=args.source_commit,
        target_commit=args.target_commit,
        target_extension_um=args.target_extension_um,
        theta_deg=args.theta_deg,
        loading_rate_factor=args.loading_rate_factor,
        max_cumulative_strain_difference=(
            args.max_cumulative_equivalent_strain_difference
        ),
        max_stress_GPa=args.max_stress_GPa,
        stress_points=args.stress_points,
        rho_points=args.rho_points,
    )
    print(json.dumps({
        "audit": str(args.output.resolve()),
        "audited_existing_case_count": payload["audited_existing_case_count"],
        "approved_reuse_case_count": payload["approved_reuse_case_count"],
        "rerun_case_count_among_existing": payload["rerun_case_count_among_existing"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
