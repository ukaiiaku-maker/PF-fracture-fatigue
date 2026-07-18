"""v10.2.2 entry point: fixed-DeltaK fatigue with uncapped physical shielding."""
from __future__ import annotations

from pathlib import Path
import sys

from .physical_shielding_v1022 import (
    install_uncapped_physical_shielding,
    reset_physical_shielding_audit,
    write_physical_shielding_audit,
)
from . import sharp_front_v10_2_1 as _fixed_deltaK


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    out = _option_value(args, "--out")
    if not out:
        raise SystemExit("v10.2.2 requires --out so shielding diagnostics are auditable")

    reset_physical_shielding_audit()
    print(
        "  v10.2.2 physical shielding: constitutive_K_cap=off "
        "saturation=finite_source+backstress+transport+recovery+moving_frame"
    )
    with install_uncapped_physical_shielding():
        result = _fixed_deltaK.main(args)

    audit = write_physical_shielding_audit(Path(out))
    print(
        "  v10.2.2 shielding audit: "
        f"samples={audit['n_shielding_samples']} "
        f"max|Kshield|={audit['maximum_abs_raw_K_shield_Pa_sqrt_m']/1.0e6:.6g} MPa*sqrt(m) "
        f"legacy-cap exceedances={audit['n_samples_above_legacy_cap_reference']} "
        f"raw==effective={int(audit['raw_equals_effective_within_relative_1e_12'])}"
    )
    return result


if __name__ == "__main__":
    main()
