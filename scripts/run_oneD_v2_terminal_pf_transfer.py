#!/usr/bin/env python3
"""Analysis-only entry for bounded V2 finalist transfer to production PF."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

from arrhenius_fracture import sharp_front_v10_2_27 as paper
from arrhenius_fracture import sharp_front_v10_2_30_energy_gated_fatigue as entry


def main() -> int:
    registry = Path(os.environ["ONED_V2_TRANSFER_REGISTRY"]).resolve()
    selection = Path(os.environ["ONED_V2_TRANSFER_SELECTION"]).resolve()
    with registry.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    options = {row["option_key"]: row["candidate_id"] for row in rows}
    if not rows or len(options) != len(rows):
        raise SystemExit("terminal PF transfer registry must contain unique nonempty options")
    paper.DEFAULT_REGISTRY = registry
    paper.SELECTION_RECORD = selection
    paper.VALID_OPTIONS = options
    entry.main(sys.argv[1:])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
