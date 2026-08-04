#!/usr/bin/env python3
"""Build the v10.4.9 full-field bulk-plasticity orientation campaign.

The v10.2.30 launcher remains the geometry, barrier, seed, loading-rate, and
hazard-energy-gate source. This builder keeps the qualified v10.4.8 numerical
failure model entry and fixes the generated-scheduler patcher execution
contract by supplying an explicit ``__file__`` and ``__name__`` namespace.
The fracture and constitutive physics remain unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

OLD_ENTRY = (
    "arrhenius_fracture."
    "sharp_front_v10_2_30_hazard_energy_gated_audited"
)
V1044_ENTRY = (
    "arrhenius_fracture."
    "sharp_front_v10_4_4_plasticity_dominated_audited"
)
MODEL_ENTRY = (
    "arrhenius_fracture."
    "sharp_front_v10_4_8_numerical_failure_audited"
)
LOCK_SCHEMA = "v10.4.9_full_field_bulk_plasticity_orientation_rate_lock_v1"
LAUNCHER_REVISION = "v10.4.9_exec_namespace_contract"


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
        label="v10.4.9 campaign-lock schema",
    )

    lock_marker = (
        f'    "schema": "{LOCK_SCHEMA}",\n'
        f'    "model_entry": "{MODEL_ENTRY}",\n'
        '    "hazard_energy_gate": True,\n'
    )
    lock_replacement = lock_marker + (
        f'    "launcher_revision": "{LAUNCHER_REVISION}",\n'
        '    "bulk_plasticity_mode": "full_field",\n'
        '    "plasticity_dominated_campaign_terminal": True,\n'
        '    "plasticity_terminal_allows_partial_fracture": True,\n'
        '    "plasticity_terminal_projected_hazard_role": "diagnostic_only",\n'
        '    "plasticity_terminal_severe_substep_energy_ratios_role": "diagnostic_only",\n'
        '    "plasticity_terminal_severe_substep_positive_Wp_sufficient": False,\n'
        '    "plasticity_terminal_severe_substep_cumulative_fraction_threshold": 0.90,\n'
        '    "numerical_stagnation_fail_fast": True,\n'
        '    "numerical_stagnation_is_successful_terminal": False,\n'
        '    "numerical_stagnation_exit_code": 4,\n'
        '    "numerical_fixed_point_failure_fail_fast": True,\n'
        '    "numerical_fixed_point_failure_is_successful_terminal": False,\n'
        '    "numerical_fixed_point_failure_exit_code": 5,\n'
        '    "nonzero_solver_exit_bookkeeping_fail_closed": True,\n'
        '    "generated_patcher_file_context_supplied": True,\n'
    )
    text = _replace_exact(
        text,
        lock_marker,
        lock_replacement,
        label="v10.4.9 outer campaign-lock fields",
    )

    marker = "plotter = source_plotter.read_text()"
    patch_scheduler = f'''patcher_path = source_scheduler.parent / "patch_v10_4_8_generated_scheduler.py"
patcher_namespace = {{
    "__file__": str(patcher_path),
    "__name__": "v10_4_8_generated_scheduler_patcher",
}}
exec(
    compile(patcher_path.read_text(), str(patcher_path), "exec"),
    patcher_namespace,
)
# The outer v10.4.9 transform has already replaced the legacy model entry in
# the scheduler-builder source. Normalize the generated scheduler back to the
# v10.2.30 entry expected by the established scheduler patcher, apply the
# v10.4.8 fail-closed scheduler patch, and then promote every case command to
# the qualified v10.4.8 model entry.
scheduler = scheduler.replace(
    "{MODEL_ENTRY}",
    "{OLD_ENTRY}",
)
scheduler = patcher_namespace["transform"](scheduler)
scheduler = scheduler.replace(
    "{V1044_ENTRY}",
    "{MODEL_ENTRY}",
)
if scheduler.count("{MODEL_ENTRY}") < 4:
    raise RuntimeError("v10.4.9 generated scheduler model-entry contract is incomplete")

'''
    text = _replace_exact(
        text,
        marker,
        patch_scheduler + marker,
        label="v10.4.9 generated-scheduler patch hook",
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
