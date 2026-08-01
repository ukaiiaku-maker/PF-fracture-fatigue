#!/usr/bin/env python3
"""Build the isolated v10.4.2 plastic-flow terminal orientation/rate launcher."""
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
    "sharp_front_v10_4_2_plastic_flow_audited"
)


def _load_v10230_builder():
    path = Path(__file__).with_name(
        "build_v10_2_30_rate_enabled_orientation_launcher.py"
    )
    spec = importlib.util.spec_from_file_location("v10230_rate_builder_v1042", path)
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
    --rho-transport-c 0
    --plastic-flow-terminal
    --plastic-flow-window-steps 2000
    --plastic-flow-min-step 2000
    --plastic-flow-contour-multipliers "1 2 4 8"''',
    label="v10.4.2 full-field bulk and terminal command",
)
replace_scheduler_exact(
    '"v10.2.30_hazard_energy_gated_orientation_rate_campaign_v1"',
    '"v10.4.2_bulk_plastic_flow_orientation_rate_campaign_v1"',
    label="v10.4.2 campaign manifest schema",
)
replace_scheduler_exact(
    '"v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1"',
    '"v10.4.2_bulk_plastic_flow_orientation_rate_case_contract_v1"',
    expected_count=1,
    label="v10.4.2 case contract schema",
)

bulk_contract_marker = '    "hazard_energy_gate": True,'
bulk_contract_count = scheduler.count(bulk_contract_marker)
if bulk_contract_count < 3:
    raise SystemExit(
        "ERROR: v10.4.2 bulk contract insertion expected at least 3 gate fields; "
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
    "plastic_flow_terminal_enabled": True,
    "plastic_flow_terminal_status": "plastic_flow_no_sharp_fracture",
    "plastic_flow_window_steps": 2000,
    "J_pl_diss_is_diagnostic_only": True,
    "contour_shielding_is_diagnostic_only": True,
    "plastic_work_enters_fracture_measure": False,
    "plastic_work_enters_cleavage_hazard": False,''',
)

replace_scheduler_exact(
    '''required = [
    root / "COMPLETE",
    root / "stage3_case_status.json",''',
    '''terminal_markers = [root / "COMPLETE", root / "PLASTIC_FLOW"]
if sum(path.is_file() for path in terminal_markers) != 1:
    raise SystemExit(1)
required = [
    root / "stage3_case_status.json",''',
    label="v10.4.2 terminal marker verification",
)
replace_scheduler_exact(
    '''if status.get("complete") is not True:
    raise SystemExit(1)''',
    '''is_plastic_terminal = (root / "PLASTIC_FLOW").is_file()
if is_plastic_terminal:
    if status.get("campaign_terminal") is not True:
        raise SystemExit(1)
    if status.get("status") != "plastic_flow_no_sharp_fracture":
        raise SystemExit(1)
    terminal_audit_path = root / "plastic_flow_terminal_audit.json"
    if not terminal_audit_path.is_file():
        raise SystemExit(1)
    terminal_audit = json.loads(terminal_audit_path.read_text())
    if terminal_audit.get("terminal") is not True:
        raise SystemExit(1)
    if terminal_audit.get("plastic_work_enters_fracture_measure") is not False:
        raise SystemExit(1)
    if terminal_audit.get("plastic_work_enters_cleavage_hazard") is not False:
        raise SystemExit(1)
    if terminal_audit.get("contour_shielding_enters_fracture_hazard") is not False:
        raise SystemExit(1)
else:
    if status.get("complete") is not True:
        raise SystemExit(1)''',
    label="v10.4.2 successful terminal status verification",
)
replace_scheduler_exact(
    '    root / "v10_2_30_hazard_energy_gate_audit.json",',
    '''    root / "v10_2_30_hazard_energy_gate_audit.json",
    root / "v10_4_bulk_peierls_taylor_coupling_audit.json",
    root / "v10_4_bulk_coupled_model_audit.json",''',
    label="v10.4.2 completed-case common audit files",
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

v1042_reuse_path = root / "v10_4_2_reuse_audit.json"
if v1042_reuse_path.is_file():
    from arrhenius_fracture.reuse_v1041_v1042 import verify_materialized_case

    verify_materialized_case(root)
elif bulk_model_audit.get("schema") != "v10.4.2_bulk_detailed_balance_plastic_flow_terminal":
    raise SystemExit(1)

command = (root / "command.sh").read_text()
if is_plastic_terminal and "--plastic-flow-terminal" not in command:
    raise SystemExit(1)
if bulk_model_audit.get("schema") == "v10.4.2_bulk_detailed_balance_plastic_flow_terminal":
    if "--plastic-flow-terminal" not in command:
        raise SystemExit(1)''',
    label="v10.4.2 completed-case native/reuse/terminal audit verification",
)
replace_scheduler_exact(
    '    f"--parameter-option {expected[\'option\']}",',
    '''    "--bulk-plasticity-mode full_field",
    "--bulk-mult-frac 1",
    "--tip-source-rho-per-emit 0",
    "--rho-transport-c 0",
    f"--parameter-option {expected['option']}",''',
    label="v10.4.2 completed-case command verification",
)
replace_scheduler_exact(
    'scripts/classify_v10_2_15_stage3_case.py',
    'scripts/classify_v10_4_2_case.py',
    label="v10.4.2 postrun classifier",
)
replace_scheduler_exact(
    '''    if [[ -f "$case_root/COMPLETE" ]]; then
      echo "ERROR: complete-looking case failed contract verification: $case_root" >&2
      return 3
    fi''',
    '''    if [[ -f "$case_root/COMPLETE" || -f "$case_root/PLASTIC_FLOW" ]]; then
      echo "ERROR: terminal-looking case failed contract verification: $case_root" >&2
      return 3
    fi''',
    label="v10.4.2 terminal-looking case rejection",
)
replace_scheduler_exact(
    '''            "complete": (
                status.get("complete") is True
                and (case_root / "COMPLETE").is_file()
                and not (case_root / "RUN_FAILED").exists()
            ),''',
    '''            "complete": (
                (
                    status.get("complete") is True
                    and (case_root / "COMPLETE").is_file()
                )
                or (
                    status.get("campaign_terminal") is True
                    and status.get("status") == "plastic_flow_no_sharp_fracture"
                    and (case_root / "PLASTIC_FLOW").is_file()
                    and (case_root / "plastic_flow_terminal_audit.json").is_file()
                )
            ) and not (case_root / "RUN_FAILED").exists(),''',
    label="v10.4.2 campaign terminal acceptance",
)
replace_scheduler_exact(
    '''"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \\
  --outroot "$OUTROOT" \\
  --target-extension-um "$TARGET_EXT_UM" || {
    echo "ERROR: four-class R-curve postprocessing failed" >&2
    exit 1
  }''',
    '''if find "$OUTROOT" -type f -name COMPLETE -print -quit | grep -q .; then
  "$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \\
    --outroot "$OUTROOT" \\
    --target-extension-um "$TARGET_EXT_UM" || {
      echo "ERROR: four-class R-curve postprocessing failed" >&2
      exit 1
    }
else
  echo "No sharp-fracture COMPLETE cases; skipping fracture-only R-curve postprocessing"
fi''',
    label="v10.4.2 terminal-aware fracture postprocessing",
)
"""


def transform(source: str) -> str:
    text = _load_v10230_builder().transform(source)
    if OLD_ENTRY not in text:
        raise RuntimeError("v10.2.30 model entry is missing from generated launcher")
    text = text.replace(OLD_ENTRY, MODEL_ENTRY)

    replacements = {
        "v10.2.30_hazard_energy_gated_orientation_rate_lock_v1":
            "v10.4.2_bulk_plastic_flow_orientation_rate_lock_v1",
        "v10_2_30_hazard_energy_gate_campaign_lock.json":
            "v10_4_2_bulk_plastic_flow_campaign_lock.json",
        "v10_2_30_hazard_energy_gate_generated_scheduler.sh":
            "v10_4_2_bulk_plastic_flow_generated_scheduler.sh",
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
        '"full_field_bulk_peierls_taylor_detailed_balance_with_plastic_flow_terminal",\n'
        '    "kernel_family_production_physics_modified": False,\n'
        '    "persistent_sites": True,\n',
        "v10.4.2 campaign physics provenance",
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
        '    "v10_4_1_completed_fracture_cases_physics_compatible": True,\n'
        '    "selective_reuse_permitted_with_case_audit": True,\n'
        '    "plastic_flow_terminal_enabled": True,\n'
        '    "plastic_flow_terminal_status": "plastic_flow_no_sharp_fracture",\n'
        '    "plastic_flow_window_steps": 2000,\n'
        '    "J_pl_diss_is_diagnostic_only": True,\n'
        '    "contour_shielding_is_diagnostic_only": True,\n'
        '    "plastic_work_enters_fracture_measure": False,\n'
        '    "plastic_work_enters_cleavage_hazard": False,\n'
    )
    if outer not in text:
        raise RuntimeError("outer v10.4.2 model/gate lock block is missing")
    text = text.replace(outer, outer_bulk, 1)

    marker = "plotter = source_plotter.read_text()"
    text = _replace_once(
        text,
        marker,
        _scheduler_adapter() + "\n" + marker,
        "embedded v10.4.2 scheduler adapter",
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
