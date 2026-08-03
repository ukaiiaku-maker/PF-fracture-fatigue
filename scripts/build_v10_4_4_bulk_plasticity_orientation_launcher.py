#!/usr/bin/env python3
"""Build the v10.4.4 full-field bulk-plasticity orientation campaign.

The v10.2.30 launcher remains the geometry, barrier, seed, loading-rate, and
hazard-energy-gate source. This builder changes the model entry, enables the
qualified full-field coupling and adaptive stagger controls, and accepts either
fracture-target completion or a plasticity-dominated campaign terminal.
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


def _scheduler_adapter() -> str:
    return r"""

# v10.4.4 full-field bulk-plasticity campaign adapter.
def replace_v1044_scheduler_exact(old, new, expected_count=1, label="v10.4.4 token"):
    global scheduler
    actual = scheduler.count(old)
    if actual != expected_count:
        raise SystemExit(
            f"ERROR: {label} changed: expected {expected_count}, found {actual}"
        )
    scheduler = scheduler.replace(old, new)


old_entry = "arrhenius_fracture.sharp_front_v10_2_30_hazard_energy_gated_audited"
new_entry = "arrhenius_fracture.sharp_front_v10_4_4_plasticity_dominated_audited"
old_entry_count = scheduler.count(old_entry)
if old_entry_count:
    scheduler = scheduler.replace(old_entry, new_entry)
entry_count = scheduler.count(new_entry)
if entry_count < 4:
    raise SystemExit(
        f"ERROR: v10.4.4 model-entry contract changed: found {entry_count}"
    )

replace_v1044_scheduler_exact(
    '    --dU "$DU_M" --dt "$DT_S" --n-stagger 2',
    '''    --dU "$DU_M" --dt "$DT_S" --n-stagger 80
    --stagger-relaxation 0.25
    --stagger-rtol 1e-6
    --stagger-ep-atol 1e-12
    --stagger-rho-atol-m2 1000
    --stagger-dt-shrink 0.5
    --stagger-min-dt-fraction 1e-8
    --stagger-max-dt-retries 20
    --adaptive-grow 1''',
    label="full-field stagger command",
)
replace_v1044_scheduler_exact(
    '    --bulk-plasticity-mode tip_only',
    '''    --bulk-plasticity-mode full_field
    --bulk-mult-frac 1
    --tip-source-rho-per-emit 0
    --rho-transport-c 0
    --plastic-flow-terminal
    --plastic-flow-window-steps "${PLASTIC_FLOW_WINDOW_STEPS:-32}"
    --plastic-flow-min-step "${PLASTIC_FLOW_MIN_STEP:-32}"
    --plastic-flow-max-da-fraction "${PLASTIC_FLOW_MAX_DA_FRACTION:-0.1}"
    --plastic-flow-min-plastic-fraction "${PLASTIC_FLOW_MIN_PLASTIC_FRACTION:-0.90}"
    --plastic-flow-min-cumulative-plastic-fraction "${PLASTIC_FLOW_MIN_CUMULATIVE_FRACTION:-0.90}"
    --plastic-flow-max-elastic-fraction "${PLASTIC_FLOW_MAX_ELASTIC_FRACTION:-0.05}"
    --plastic-flow-max-tangent-fraction "${PLASTIC_FLOW_MAX_TANGENT_FRACTION:-0.05}"
    --plastic-flow-contour-multipliers "1 2 4 8"''',
    label="full-field bulk-plasticity command",
)

replace_v1044_scheduler_exact(
    '''events = json.loads(
    (root / "stochastic_avalanche_geometry_events.json").read_text()
)
if not events:
    raise SystemExit(1)
for event in events:
    gate = event.get("hazard_energy_gate")
    if event.get("hazard_energy_gate_active") is not True or not isinstance(gate, dict):
        raise SystemExit(1)
    proposed = float(gate.get("proposed_event_advance_m", float("nan")))
    accepted = float(gate.get("accepted_event_advance_m", float("nan")))
    available = float(gate.get("energy_available_integrated_J_per_m", float("nan")))
    dissipated = float(gate.get("energy_dissipated_integrated_J_per_m", float("nan")))
    if not all(math.isfinite(value) for value in (proposed, accepted, available, dissipated)):
        raise SystemExit(1)
    if proposed <= 0.0 or accepted <= 0.0 or accepted > proposed * (1.0 + 1.0e-10):
        raise SystemExit(1)
    tolerance = 1.0e-10 * max(abs(available), abs(dissipated), 1.0)
    if dissipated > available + tolerance:
        raise SystemExit(1)
    if float(gate.get("gamma_rel", 0.0)) <= 0.0:
        raise SystemExit(1)
    if float(gate.get("DeltaG_cleave_eff_eV", 0.0)) <= 0.0:
        raise SystemExit(1)

command = (root / "command.sh").read_text()''',
    '''events = json.loads(
    (root / "stochastic_avalanche_geometry_events.json").read_text()
)
plasticity_dominated = (
    (root / "PLASTICITY_DOMINATED").is_file()
    or (root / "PLASTIC_FLOW").is_file()
)
if not plasticity_dominated and not events:
    raise SystemExit(1)
for event in events:
    gate = event.get("hazard_energy_gate")
    if event.get("hazard_energy_gate_active") is not True or not isinstance(gate, dict):
        raise SystemExit(1)
    proposed = float(gate.get("proposed_event_advance_m", float("nan")))
    accepted = float(gate.get("accepted_event_advance_m", float("nan")))
    available = float(gate.get("energy_available_integrated_J_per_m", float("nan")))
    dissipated = float(gate.get("energy_dissipated_integrated_J_per_m", float("nan")))
    if not all(math.isfinite(value) for value in (proposed, accepted, available, dissipated)):
        raise SystemExit(1)
    if proposed <= 0.0 or accepted <= 0.0 or accepted > proposed * (1.0 + 1.0e-10):
        raise SystemExit(1)
    tolerance = 1.0e-10 * max(abs(available), abs(dissipated), 1.0)
    if dissipated > available + tolerance:
        raise SystemExit(1)
    if float(gate.get("gamma_rel", 0.0)) <= 0.0:
        raise SystemExit(1)
    if float(gate.get("DeltaG_cleave_eff_eV", 0.0)) <= 0.0:
        raise SystemExit(1)

command = (root / "command.sh").read_text()''',
    label="dual-outcome energy-gate verification",
)

replace_v1044_scheduler_exact(
    '''tokens = [
    "-m arrhenius_fracture.sharp_front_v10_4_4_plasticity_dominated_audited",''',
    '''tokens = [
    "-m arrhenius_fracture.sharp_front_v10_4_4_plasticity_dominated_audited",
    "--bulk-plasticity-mode full_field",
    "--plastic-flow-terminal",
    "--n-stagger 80",''',
    label="completed-case full-field command verification",
)

replace_v1044_scheduler_exact(
    '''  "$PYTHON_BIN" scripts/classify_v10_2_15_stage3_case.py \\
    --case-root "$case_root" \\
    --target-extension-um "$TARGET_EXT_UM" >> "$log" 2>&1 || {
      echo "classification_failed" > "$case_root/RUN_FAILED"
      tail -n 100 "$log" >&2 || true
      return 1
    }

  if ! verified_complete "$case_root" "$option" "$candidate" "$T" "$case_seed"; then''',
    '''  if [[ -f "$case_root/PLASTICITY_DOMINATED" || -f "$case_root/PLASTIC_FLOW" ]]; then
    CASE_ROOT="$case_root" TARGET_EXT_UM="$TARGET_EXT_UM" T="$T" \\
    "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["CASE_ROOT"])
terminal = json.loads((root / "plastic_flow_terminal_audit.json").read_text())
summary_path = root / "summary.json"
summary_payload = json.loads(summary_path.read_text()) if summary_path.is_file() else []
summary = summary_payload[0] if isinstance(summary_payload, list) and summary_payload else {}
projected_m = float(summary.get("geometry_projected_extension_m", 0.0) or 0.0)
classification = str(
    terminal.get(
        "campaign_classification",
        terminal.get("classification", "plasticity_dominated"),
    )
)
payload = {
    "schema": "v10.4.4_plasticity_dominated_case_status_v1",
    "complete": True,
    "status": "plasticity_dominated",
    "terminal_status": classification,
    "temperature_K": float(os.environ["T"]),
    "target_extension_um": float(os.environ["TARGET_EXT_UM"]),
    "projected_extension_um": projected_m * 1.0e6,
    "fracture_target_reached": False,
    "plasticity_dominated": True,
    "J_elastic_positive_J_per_m2": terminal.get("J_elastic_positive_J_per_m2"),
    "J_plastic_dissipation_J_per_m2": terminal.get("J_plastic_dissipation_J_per_m2"),
    "J_apparent_total_J_per_m2": terminal.get("J_apparent_total_J_per_m2"),
    "K_apparent_plasticity_limited_MPa_sqrt_m": terminal.get(
        "K_apparent_plasticity_limited_MPa_sqrt_m"
    ),
}
(root / "stage3_case_status.json").write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\\n"
)
(root / "COMPLETE").write_text(classification + "\\n")
PY
  else
    "$PYTHON_BIN" scripts/classify_v10_2_15_stage3_case.py \\
      --case-root "$case_root" \\
      --target-extension-um "$TARGET_EXT_UM" >> "$log" 2>&1 || {
        echo "classification_failed" > "$case_root/RUN_FAILED"
        tail -n 100 "$log" >&2 || true
        return 1
      }
  fi

  if ! verified_complete "$case_root" "$option" "$candidate" "$T" "$case_seed"; then''',
    label="dual-outcome postrun classification",
)

replace_v1044_scheduler_exact(
    '''"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \\
  --outroot "$OUTROOT" \\
  --target-extension-um "$TARGET_EXT_UM" || {
    echo "ERROR: four-class R-curve postprocessing failed" >&2
    exit 1
  }''',
    '''"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \\
  --outroot "$OUTROOT" \\
  --target-extension-um "$TARGET_EXT_UM" || {
    echo "WARNING: fracture-only R-curve plotting skipped or incomplete for mixed terminal outcomes" >&2
  }

"$PYTHON_BIN" scripts/summarize_v10_4_4_bulk_plasticity_campaign.py \\
  --outroot "$OUTROOT"''',
    label="mixed-outcome campaign postprocessing",
)
"""


def transform(source: str) -> str:
    text = _load_gate_builder().transform(source)
    count = text.count(OLD_ENTRY)
    if count < 2:
        raise RuntimeError(
            f"outer launcher model entry changed: expected at least 2, found {count}"
        )
    text = text.replace(OLD_ENTRY, MODEL_ENTRY)

    marker = "plotter = source_plotter.read_text()"
    text = _replace_exact(
        text,
        marker,
        _scheduler_adapter() + "\n" + marker,
        label="v10.4.4 embedded scheduler adapter",
    )
    text = _replace_exact(
        text,
        '"schema": "v10.2.30_hazard_energy_gated_orientation_rate_lock_v1",',
        '"schema": "v10.4.4_full_field_bulk_plasticity_orientation_rate_lock_v1",',
        label="v10.4.4 campaign-lock schema",
    )
    text = _replace_exact(
        text,
        '    "hazard_energy_gate": True,\n',
        '    "hazard_energy_gate": True,\n'
        '    "bulk_plasticity_mode": "full_field",\n'
        '    "plasticity_dominated_campaign_terminal": True,\n'
        '    "plasticity_terminal_allows_partial_fracture": True,\n'
        '    "plasticity_terminal_projected_hazard_role": "diagnostic_only",\n',
        label="v10.4.4 campaign-lock fields",
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
