"""Explicit custom-registry entry for refined-v10 candidate measurements."""
from __future__ import annotations

import csv
import os
from pathlib import Path
import sys

from . import sharp_front_v10_2_27 as _paper
from . import sharp_front_v10_2_30_fixed_deltaK as _fixed


MODEL_ID = "v10.2.30_refined_candidate_fixed_deltaK"


def main(argv=None):
    registry = Path(os.environ.get("V10230_CANDIDATE_REGISTRY", "")).resolve()
    selection = Path(os.environ.get("V10230_CANDIDATE_SELECTION", "")).resolve()
    if not registry.is_file() or not selection.is_file():
        raise SystemExit(
            "set V10230_CANDIDATE_REGISTRY and V10230_CANDIDATE_SELECTION "
            "to immutable generated candidate artifacts"
        )
    with registry.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit("candidate registry is empty")
    valid = {row["option_key"]: row["candidate_id"] for row in rows}
    if len(valid) != len(rows):
        raise SystemExit("candidate registry contains duplicate option keys")

    old_registry, old_selection, old_valid = (
        _paper.DEFAULT_REGISTRY, _paper.SELECTION_RECORD, _paper.VALID_OPTIONS
    )
    _paper.DEFAULT_REGISTRY = registry
    _paper.SELECTION_RECORD = selection
    _paper.VALID_OPTIONS = valid
    try:
        return _fixed.main(sys.argv[1:] if argv is None else argv)
    finally:
        _paper.DEFAULT_REGISTRY = old_registry
        _paper.SELECTION_RECORD = old_selection
        _paper.VALID_OPTIONS = old_valid


if __name__ == "__main__":
    main()
