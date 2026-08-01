#!/usr/bin/env python3
"""Add corrected positive directional-J provenance to the v10.4.2 launcher."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_base_builder():
    path = Path(__file__).with_name("build_v10_4_2_plastic_terminal_launcher.py")
    spec = importlib.util.spec_from_file_location("v1042_terminal_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v10.4.2 terminal builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    text = _load_base_builder().transform(source)
    text = _replace_once(
        text,
        '"full_field_bulk_peierls_taylor_detailed_balance_with_plastic_flow_terminal"',
        '"full_field_bulk_peierls_taylor_detailed_balance_positive_directional_J_with_plastic_flow_terminal"',
        "v10.4.2 positive directional-J production provenance",
    )
    text = _replace_once(
        text,
        '    "v10_4_1_completed_fracture_cases_physics_compatible": True,\n',
        '    "directional_J_sign_convention": "positive_raw_signed_J_is_forward_configurational_work",\n'
        '    "directional_J_effective_definition": "max(J_signed,0)",\n'
        '    "directional_J_first_nonzero_sign_latch_used": False,\n'
        '    "directional_J_absolute_value_used": False,\n'
        '    "v10_4_1_completed_fracture_cases_physics_compatible": "only_after_positive_directional_J_history_audit",\n',
        "v10.4.2 conditional inherited-case compatibility",
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
