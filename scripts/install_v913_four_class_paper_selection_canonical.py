#!/usr/bin/env python3
"""Install the v10.2.27 registry with four explicit material-class labels.

The original v10.2.27 installer intentionally preserves the accepted v10.2.25
peak row verbatim.  In v10.2.25 that row carries ``material_class=DBTT`` even
though its option, role, and paper class are peak.  This wrapper retains every
hash, fingerprint, fixed-closure, and candidate validation from the original
installer, but normalizes the installed metadata to the four canonical labels:
``peak``, ``DBTT``, ``weakT``, and ``ceramic``.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


_BASE_PATH = Path(__file__).with_name("install_v913_four_class_paper_selection.py")
_SPEC = importlib.util.spec_from_file_location(
    "install_v913_four_class_paper_selection_base", _BASE_PATH
)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"could not load base installer: {_BASE_PATH}")
_base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_base)

_ORIGINAL_INSTALL_ROWS = _base.install_rows
_EXPECTED_LABEL_BY_OPTION = {
    "v913_paper_peak01_0242980_persistent_sites": "peak",
    "v913_paper_dbtt01_0202500_persistent_sites": "DBTT",
}
_EXPECTED_FINAL_LABELS = ["peak", "DBTT", "weakT", "ceramic"]


def install_rows(
    base_rows: list[dict[str, str]],
    handoff_rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]]]:
    """Run the audited installer logic and normalize only class metadata."""
    fields, installed, selected_metadata = _ORIGINAL_INSTALL_ROWS(
        base_rows, handoff_rows, manifest
    )

    for row in installed:
        option = str(row["option_key"])
        if option in _EXPECTED_LABEL_BY_OPTION:
            row["material_class"] = _EXPECTED_LABEL_BY_OPTION[option]

    labels = [str(row["material_class"]) for row in installed]
    if labels != _EXPECTED_FINAL_LABELS:
        raise RuntimeError(
            "canonical four-class labels mismatch: "
            f"expected={_EXPECTED_FINAL_LABELS}, observed={labels}"
        )

    installed_by_candidate = {
        str(row["candidate_id"]): str(row["material_class"]) for row in installed
    }
    for metadata in selected_metadata:
        candidate_id = str(metadata["candidate_id"])
        metadata["material_class_2d"] = installed_by_candidate[candidate_id]

    return fields, installed, selected_metadata


def main() -> int:
    original = _base.install_rows
    _base.install_rows = install_rows
    try:
        return _base.main()
    finally:
        _base.install_rows = original


if __name__ == "__main__":
    raise SystemExit(main())
