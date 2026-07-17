"""v10.1.6 entry point for emergent temperature-response validation.

This wrapper intentionally changes no constitutive physics.  Source capacity,
source refresh, back-stress scaling, shielding limits, and blunting remain
independent of temperature at runtime.  Temperature dependence enters only
through the promoted cleavage, emission, Peierls, Taylor, and recovery kinetics
used by v10.1.5.
"""
from __future__ import annotations

import sys

from . import sharp_front_v10_1_5 as _campaign


def main(argv=None):
    print(
        "  v10.1.6 temperature emergence: runtime source parameters are "
        "temperature-independent; T enters through Arrhenius kinetics only"
    )
    return _campaign.main(sys.argv[1:] if argv is None else argv)


if __name__ == "__main__":
    main()
