#!/usr/bin/env python3
"""Build the v10.4.4 full-field bulk-plasticity orientation campaign.

The v10.2.30 launcher remains the geometry, barrier, seed, loading-rate, and
hazard-energy-gate source. This builder changes the public model entry and
arranges for the generated scheduler to be patched by the dedicated v10.4.4
scheduler transformer.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

OLD_ENTRY = (
    "arrhenius_fracture."
    "sharp_front_v10_2_30_hazard_energy_gated_audited"
)
MODEL_ENTRY = (
    "arrhenius_fracture."
    "sharp_front_v10_4_4_plasticity_dominated_audited"
)
LOCK_SCHEMA = "v10.4.4_full_field_bulk_plasticity_orientation_rate_lock_v1"


def _load_gate_builder():
    path = Path(__file__).with_name(
        "build_v10_2_30_rate_enabled_orientation_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("v10230_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load gate builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_exact(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label} changed: expected one occurrence, found {count}"
        )
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    text = _load_gate_builder().transform(source)
    entry_count = text.count(OLD_ENTRY)
    if entry_count < 2:
        raise RuntimeError(
            f"outer launcher model entry changed: expected at least 2, found {entry_count}"
        )
    text = text.replace(OLD_ENTRY, MODEL_ENTRY)

    text = _replace_exact(
        text,
        '"schema": "v10.2.30_hazard_energy_gated_orientation_rate_lock_v1",',
        f'"schema": "{LOCK_SCHEMA}",',
        label="v10.4.4 campaign-lock schema",
    )

    lock_marker = (
        f'    "schema": "{LOCK_SCHEMA}",\n'
        f'    "model_entry": "{MODEL_ENTRY}",\n'
        '    "hazard_energy_gate": True,\n'
    )
    lock_replacement = lock_marker + (
        '    "bulk_plasticity_mode": "full_field",\n'
        '    "plasticity_dominated_campaign_terminal": True,\n'
        '    "plasticity_terminal_allows_partial_fracture": True,\n'
        '    "plasticity_terminal_projected_hazard_role": "diagnostic_only",\n'
    )
    text = _replace_exact(
        text,
        lock_marker,
        lock_replacement,
        label="v10.4.4 outer campaign-lock fields",
    )

    marker = "plotter = source_plotter.read_text()"
    patch_scheduler = '''patcher_path = source_scheduler.parent / "patch_v10_4_4_generated_scheduler.py"
patcher_namespace = {}
exec(
    compile(patcher_path.read_text(), str(patcher_path), "exec"),
    patcher_namespace,
)
scheduler = patcher_namespace["transform"](scheduler)

'''
    text = _replace_exact(
        text,
        marker,
        patch_scheduler + marker,
        label="v10.4.4 generated-scheduler patch hook",
    )
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
