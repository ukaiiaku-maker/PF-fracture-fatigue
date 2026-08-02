#!/usr/bin/env python3
"""Add audited-reuse short-circuiting to the v10.4.2 production launcher.

Materialized v10.4.1 cases already carry a v10.4.2 reuse audit.  The generated
scheduler must verify that audit before applying native v10.4.2 contract checks.
This builder edits the nested scheduler generator, not merely the intermediate
wrapper text, so the ordering is enforced in the executable final scheduler.
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

    # The final scheduler's verifier reads the case contract before building the
    # native expected dictionary.  Inject the audited-reuse branch at that exact
    # point.  The previous anchor (bulk_audit = json.loads) was about 100 lines
    # too late and could never be reached by a v10.4.1-era inherited contract.
    guard = (
        'replace_scheduler_exact(\n'
        '    \'contract = json.loads((root / "v10_2_27_case_contract.json").read_text())\',\n'
        '    \'\'\'v1042_reuse_path = root / "v10_4_2_reuse_audit.json"\n'
        'if v1042_reuse_path.is_file():\n'
        '    from arrhenius_fracture.reuse_v1041_v1042 import (\n'
        '        verify_materialized_case,\n'
        '        verify_source_case,\n'
        '    )\n'
        '\n'
        '    try:\n'
        '        reuse_audit = verify_materialized_case(root)\n'
        '        verify_source_case(Path(reuse_audit["source_case"]))\n'
        '    except Exception as exc:\n'
        '        print(f"FAILED_REUSE_VERIFICATION {root}: {exc}")\n'
        '        raise\n'
        '    print(f"SKIP_REUSED_VERIFIED {root}")\n'
        '    raise SystemExit(0)\n'
        '\n'
        'contract = json.loads((root / "v10_2_27_case_contract.json").read_text())\'\'\',\n'
        '    label="v10.4.2 audited-reuse short-circuit before native contract checks",\n'
        ')\n\n'
    )

    tail_marker = "plotter = source_plotter.read_text()"
    text = _replace_once(
        text,
        tail_marker,
        guard + tail_marker,
        "v10.4.2 audited-reuse scheduler short-circuit",
    )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
