#!/usr/bin/env python3
"""Add audited-reuse short-circuiting to the v10.4.2 production launcher.

Materialized v10.4.1 cases already carry a v10.4.2 reuse audit that verifies
source hashes, detailed-balance provenance, target completion, and the corrected
positive directional-J history.  They must not then be rejected for lacking the
native v10.4.2 command-line terminal options that were intentionally never used
to generate those inherited results.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_positive_j_builder():
    path = Path(__file__).with_name("build_v10_4_2_positive_J_launcher.py")
    spec = importlib.util.spec_from_file_location("v1042_positive_j_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v10.4.2 positive-J builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    text = _load_positive_j_builder().transform(source)

    marker = "'''bulk_audit = json.loads(\n"
    replacement = "'''v1042_reuse_path = root / \"v10_4_2_reuse_audit.json\"\n" \
        "if v1042_reuse_path.is_file():\n" \
        "    from arrhenius_fracture.reuse_v1041_v1042 import (\n" \
        "        verify_materialized_case,\n" \
        "        verify_source_case,\n" \
        "    )\n\n" \
        "    verify_materialized_case(root)\n" \
        "    verify_source_case(root)\n" \
        "    raise SystemExit(0)\n\n" \
        "bulk_audit = json.loads(\n"

    return _replace_once(
        text,
        marker,
        replacement,
        "v10.4.2 audited-reuse scheduler short-circuit",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
