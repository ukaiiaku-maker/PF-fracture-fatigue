"""Loading-rate specification shared by v10.2.28 orientation campaigns.

The monotonic 2-D driver applies a nominal displacement increment ``dU`` over a
nominal physical time increment ``dt``.  Adaptive event stepping multiplies both
by the same accepted fraction, so ``dU/dt`` remains the imposed opening rate.
This module resolves rate factors without touching constitutive parameters.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
import json
import math


BASE_DU_M = 2.0e-7
BASE_DT_S = 8.4


@dataclass(frozen=True)
class LoadingRateSpec:
    loading_rate_factor: float
    nominal_dU_m: float
    base_dt_s: float
    nominal_dt_s: float
    nominal_opening_rate_m_per_s: float
    rate_tag: str


def _positive_finite(name: str, value: float) -> float:
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return resolved


def rate_tag(factor: float | str) -> str:
    """Return a filesystem-safe canonical rate-factor tag."""
    try:
        value = Decimal(str(factor)).normalize()
    except InvalidOperation as exc:
        raise ValueError("loading-rate factor must be numeric") from exc
    if not value.is_finite() or value <= 0:
        raise ValueError("loading-rate factor must be positive and finite")
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "rate" + text.replace("-", "m").replace(".", "p") + "x"


def resolve_loading_rate(
    factor: float,
    nominal_dU_m: float = BASE_DU_M,
    base_dt_s: float = BASE_DT_S,
) -> LoadingRateSpec:
    factor_value = _positive_finite("loading-rate factor", factor)
    dU = _positive_finite("nominal dU", nominal_dU_m)
    base_dt = _positive_finite("base dt", base_dt_s)
    dt = base_dt / factor_value
    rate = dU / dt
    return LoadingRateSpec(
        loading_rate_factor=factor_value,
        nominal_dU_m=dU,
        base_dt_s=base_dt,
        nominal_dt_s=dt,
        nominal_opening_rate_m_per_s=rate,
        rate_tag=rate_tag(factor),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor", type=float, required=True)
    parser.add_argument("--dU-m", type=float, default=BASE_DU_M, dest="dU_m")
    parser.add_argument("--base-dt-s", type=float, default=BASE_DT_S)
    parser.add_argument("--format", choices=("json", "tsv", "tag"), default="json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        spec = resolve_loading_rate(args.factor, args.dU_m, args.base_dt_s)
    except ValueError as exc:
        raise SystemExit(f"ERROR: {exc}") from exc

    if args.format == "tag":
        print(spec.rate_tag)
    elif args.format == "tsv":
        print(
            "\t".join(
                (
                    f"{spec.loading_rate_factor:.17g}",
                    f"{spec.nominal_dU_m:.17g}",
                    f"{spec.base_dt_s:.17g}",
                    f"{spec.nominal_dt_s:.17g}",
                    f"{spec.nominal_opening_rate_m_per_s:.17g}",
                    spec.rate_tag,
                )
            )
        )
    else:
        print(json.dumps(asdict(spec), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BASE_DT_S",
    "BASE_DU_M",
    "LoadingRateSpec",
    "rate_tag",
    "resolve_loading_rate",
]
