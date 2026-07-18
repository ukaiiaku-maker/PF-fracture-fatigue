"""v10.2.3 shared-core uncapped shielding entry point.

This entry point preserves the current monotonic fracture solver chain while
making the versioned shared-core shielding semantics explicit in console output.
The constitutive change itself lives in CampaignCalibratedTipEngine and therefore
also applies to the fatigue entry point.
"""
from __future__ import annotations

import sys

from . import sharp_front_v10_1_7_5 as _protected


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.2.3 shared shielding core: constitutive_K_cap=off "
        "legacy_manifest_cap=diagnostic_only loading_paths=monotonic+fatigue"
    )
    return _protected.main(args)


if __name__ == "__main__":
    main()
