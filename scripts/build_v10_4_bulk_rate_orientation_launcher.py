#!/usr/bin/env python3
"""Build a v10.4.1 full-field bulk Peierls--Taylor orientation/rate launcher."""
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
    "sharp_front_v10_4_bulk_peierls_taylor_audited"
)


def _load_v10230_builder():
    path = Path(__file__).with_name(
        "build_v10_2_30_rate_enabled_orientation_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("v10230_rate_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v10.2.30 builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def _scheduler_adapter() -> str:
    return r"""
replace_scheduler_exact(
    '    --bulk-plasticity-mode tip_only',
    '''    --bulk-plasticity-mode full_field
    --bulk-mult-frac 1
    --tip-source-rho-per-emit 0
    --rho-transport-c 0''',
    label="v10.4.1 full-field bulk command",
)
replace_scheduler_exact(
    '"v10.2.30_hazard_energy_gated_orientation_rate_campaign_v1"',
    '"v10.4_bulk_peierls_taylor_orientation_rate_campaign_v1"',
    label="v10.4.1 campaign manifest schema",
)
replace_scheduler_exact(
    '"v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1"',
    '"v10.4_bulk_peierls_taylor_orientation_rate_case_contract_v1"',
    expected_count=1,
    label="v10.4.1 case contract schema",
)

bulk_contract_marker = '    "hazard_energy_gate": True,'
bulk_contract_count = scheduler.count(bulk_contract_marker)
if bulk_contract_count < 3:
    raise SystemExit(
        "ERROR: v10.4.1 bulk contract insertion expected at least 3 gate fields; "
        f"found {bulk_contract_count}"
    )
scheduler = scheduler.replace(
    bulk_contract_marker,
    '''    "hazard_energy_gate": True,
    "bulk_plasticity_mode": "full_field",
    "bulk_kinetics_model": "emission_derived_peierls_taylor_multihit",
    "bulk_initial_density_from_selected_row": True,
    "tip_and_bulk_source_populations_distinct": True,
    "direct_tip_to_bulk_density_transfer": False,
    "bulk_net_slip_model": "detailed_balance_forward_minus_reverse",
    "zero_stress_net_plastic_rate_exactly_zero": True,
    "v10_4_0_outputs_physics_compatible": False,
    "selective_reuse_permitted_with_case_audit": True,''',
)

replace_scheduler_exact(
    '    root / "v10_2_30_hazard_energy_gate_audit.json",',
    '''    root / "v10_2_30_hazard_energy_gate_audit.json",
    root / "v10_4_bulk_peierls_taylor_coupling_audit.json",
    root / "v10_4_bulk_coupled_model_audit.json",''',
    label="v10.4.1 completed-case common audit files",
)
replace_scheduler_exact(
    'command = (root / "command.sh").read_text()',
    '''bulk_audit = json.loads(
    (root / "v10_4_bulk_peierls_taylor_coupling_audit.json").read_text()
)
if bulk_audit.get("bulk_kinetics_model") != "emission_derived_peierls_taylor_multihit":
    raise SystemExit(1)
if bulk_audit.get("tip_and_bulk_populations_distinct") is not True:
    raise SystemExit(1)
if bulk_audit.get("direct_tip_to_bulk_density_transfer") is not False:
    raise SystemExit(1)
if float(bulk_audit.get("bulk_multiplication_fraction", 0.0)) != 1.0:
    raise SystemExit(1)
runtime_bulk = bulk_audit.get("runtime_diagnostics", {})
if runtime_bulk.get("local_plastic_work_nonnegative") is not True:
    raise SystemExit(1)

bulk_model_audit = json.loads(
    (root / "v10_4_bulk_coupled_model_audit.json").read_text()
)
if bulk_model_audit.get("bulk_plasticity_mode") != "full_field":
    raise SystemExit(1)
if bulk_model_audit.get("v10_2_30_code_path_modified") is not False:
    raise SystemExit(1)

reuse_path = root / "v10_4_1_reuse_audit.json"
detailed_balance_path = root / "v10_4_1_bulk_detailed_balance_audit.json"
if reuse_path.is_file():
    from arrhenius_fracture.reuse_v1040_v1041 import verify_materialized_reuse

    reuse = verify_materialized_reuse(root)
    if bulk_model_audit.get("execution_mode") != "audited_v10_4_0_reuse":
        raise SystemExit(1)
    if bulk_model_audit.get("source_one_way_arrhenius_rate_used_as_net_slip") is not True:
        raise SystemExit(1)
    if reuse.get("target_model") != "v10.4.1_detailed_balance_forward_minus_reverse":
        raise SystemExit(1)
    if detailed_balance_path.exists():
        raise SystemExit(1)
else:
    if not detailed_balance_path.is_file():
        raise SystemExit(1)
    if bulk_model_audit.get("zero_stress_net_plastic_rate_exactly_zero") is not True:
        raise SystemExit(1)
    if bulk_model_audit.get("v10_4_0_outputs_physics_compatible") is not False:
        raise SystemExit(1)
    detailed_balance_audit = json.loads(detailed_balance_path.read_text())
    if detailed_balance_audit.get("one_way_arrhenius_rate_used_as_net_slip") is not False:
        raise SystemExit(1)
    if detailed_balance_audit.get("zero_stress_net_plastic_rate_exactly_zero") is not True:
        raise SystemExit(1)
    if detailed_balance_audit.get("new_fitted_parameters") != 0:
        raise SystemExit(1)

command = (root / "command.sh").read_text()''',
    label="v10.4.1 completed-case native/reuse audit verification",
)
replace_scheduler_exact(
    '    f"--parameter-option {expected[\'option\']}",',
    '''    "--bulk-plasticity-mode full_field",
    "--bulk-mult-frac 1",
    "--tip-source-rho-per-emit 0",
    "--rho-transport-c 0",
    f"--parameter-option {expected['option']}",''',
    label="v10.4.1 completed-case command verification",
)
"""


def transform(source: str) -> str:
    text = _load_v10230_builder().transform(source)
    if OLD_ENTRY not in text:
        raise RuntimeError("v10.2.30 model entry is missing from generated launcher")
    text = text.replace(OLD_ENTRY, MODEL_ENTRY)

    replacements = {
        "v10.2.30_hazard_energy_gated_orientation_rate_lock_v1":
            "v10.4_bulk_peierls_taylor_orientation_rate_lock_v1",
        "v10_2_30_hazard_energy_gate_campaign_lock.json":
            "v10_4_bulk_peierls_taylor_campaign_lock.json",
        "v10_2_30_hazard_energy_gate_generated_scheduler.sh":
            "v10_4_bulk_peierls_taylor_generated_scheduler.sh",
    }
    for old, new in replacements.items():
        if old not in text:
            raise RuntimeError(f"generated v10.2.30 token is missing: {old}")
        text = text.replace(old, new)

    text = _replace_once(
        text,
        '    "direct_prescribed_geometry": True,\n'
        '    "production_physics_modified": False,\n'
        '    "persistent_sites": True,\n',
        '    "direct_prescribed_geometry": True,\n'
        '    "production_physics_modified": True,\n'
        '    "production_physics_change": '
        '"full_field_bulk_peierls_taylor_coupling",\n'
        '    "kernel_family_production_physics_modified": False,\n'
        '    "persistent_sites": True,\n',
        "v10.4.1 campaign physics provenance",
    )

    outer = (
        f'    "model_entry": "{MODEL_ENTRY}",\n'
        '    "hazard_energy_gate": True,\n'
    )
    outer_bulk = outer + (
        '    "bulk_plasticity_mode": "full_field",\n'
        '    "bulk_kinetics_model": "emission_derived_peierls_taylor_multihit",\n'
        '    "bulk_initial_density_from_selected_row": True,\n'
        '    "tip_and_bulk_source_populations_distinct": True,\n'
        '    "direct_tip_to_bulk_density_transfer": False,\n'
        '    "bulk_net_slip_model": "detailed_balance_forward_minus_reverse",\n'
        '    "zero_stress_net_plastic_rate_exactly_zero": True,\n'
        '    "v10_4_0_outputs_physics_compatible": False,\n'
        '    "selective_reuse_permitted_with_case_audit": True,\n'
    )
    if outer not in text:
        raise RuntimeError("outer v10.4.1 model/gate lock block is missing")
    text = text.replace(outer, outer_bulk, 1)

    marker = "plotter = source_plotter.read_text()"
    text = _replace_once(
        text,
        marker,
        _scheduler_adapter() + "\n" + marker,
        "embedded v10.4.1 scheduler adapter",
    )
    return text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
