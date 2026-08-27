"""Qualified monotonic v10 entry with an immutable external candidate registry."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

from . import sharp_front_v10_2_27 as _paper
from . import sharp_front_v10_2_29_fatigue_audited as _monotonic


def main(argv=None):
    registry = Path(os.environ.get("V10230_CANDIDATE_REGISTRY", "")).resolve()
    selection = Path(os.environ.get("V10230_CANDIDATE_SELECTION", "")).resolve()
    if not registry.is_file() or not selection.is_file():
        raise SystemExit("set V10230_CANDIDATE_REGISTRY and V10230_CANDIDATE_SELECTION")
    with registry.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    valid = {row["option_key"]: row["candidate_id"] for row in rows}
    if not rows or len(valid) != len(rows):
        raise SystemExit("candidate registry is empty or contains duplicate option keys")
    old = (_paper.DEFAULT_REGISTRY, _paper.SELECTION_RECORD, _paper.VALID_OPTIONS)
    _paper.DEFAULT_REGISTRY, _paper.SELECTION_RECORD, _paper.VALID_OPTIONS = registry, selection, valid
    try:
        return _monotonic.main(sys.argv[1:] if argv is None else argv)
    finally:
        _paper.DEFAULT_REGISTRY, _paper.SELECTION_RECORD, _paper.VALID_OPTIONS = old


if __name__ == "__main__":
    main()
