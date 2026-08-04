from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script(name: str):
    path = Path(__file__).parents[1] / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_builder_preserves_campaign_source_and_installs_scheduler_patch_hook():
    root = Path(__file__).parents[1]
    source = (
        root / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    builder = _load_script(
        "build_v10_4_4_bulk_plasticity_orientation_launcher.py"
    )
    generated = builder.transform(source)

    assert builder.MODEL_ENTRY in generated
    assert builder.LOCK_SCHEMA in generated
    assert '"bulk_plasticity_mode": "full_field"' in generated
    assert '"plasticity_dominated_campaign_terminal": True' in generated
    assert '"plasticity_terminal_severe_substep_positive_Wp_sufficient": False' in generated
    assert '"plasticity_terminal_severe_substep_cumulative_fraction_threshold": 0.90' in generated
    assert "patch_v10_4_4_generated_scheduler.py" in generated

    normalization = f'''scheduler = scheduler.replace(
    "{builder.MODEL_ENTRY}",
    "{builder.OLD_ENTRY}",
)'''
    patch_call = 'scheduler = patcher_namespace["transform"](scheduler)'
    promotion = f'''scheduler = scheduler.replace(
    "{builder.V1044_ENTRY}",
    "{builder.MODEL_ENTRY}",
)'''

    assert normalization in generated
    assert patch_call in generated
    assert promotion in generated
    assert generated.index(normalization) < generated.index(patch_call)
    assert generated.index(patch_call) < generated.index(promotion)
    assert "v10.4.6 generated scheduler model-entry contract is incomplete" in generated

    assert "v913_paper_peak01_0242980_persistent_sites" in generated
    assert "v913_paper_dbtt01_0202500_persistent_sites" in generated
    assert "v913_paper_weakT01_0129902_persistent_sites" in generated
    assert "v913_paper_ceramic01_0077080_persistent_sites" in generated


def test_scheduler_patcher_inserts_full_field_dual_terminal_contract():
    patcher = _load_script("patch_v10_4_4_generated_scheduler.py")
    entry = patcher.MODEL_ENTRY
    source = f'''{entry}\n{entry}\n{entry}\n{entry}\n
    --dU "$DU_M" --dt "$DT_S" --n-stagger 2
    --bulk-plasticity-mode tip_only

events = json.loads(
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

command = (root / "command.sh").read_text()
tokens = [
    "-m {entry}",

  "$PYTHON_BIN" scripts/classify_v10_2_15_stage3_case.py \\
    --case-root "$case_root" \\
    --target-extension-um "$TARGET_EXT_UM" >> "$log" 2>&1 || {{
      echo "classification_failed" > "$case_root/RUN_FAILED"
      tail -n 100 "$log" >&2 || true
      return 1
    }}

  if ! verified_complete "$case_root" "$option" "$candidate" "$T" "$case_seed"; then

"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \\
  --outroot "$OUTROOT" \\
  --target-extension-um "$TARGET_EXT_UM" || {{
    echo "ERROR: four-class R-curve postprocessing failed" >&2
    exit 1
  }}
'''
    generated = patcher.transform(source)

    assert "--bulk-plasticity-mode full_field" in generated
    assert "--n-stagger 80" in generated
    assert "--plastic-flow-terminal" in generated
    assert "PLASTICITY_DOMINATED" in generated
    assert "plasticity_dominated = (" in generated
    assert "summarize_v10_4_4_bulk_plasticity_campaign.py" in generated
    assert "--plastic-flow-contour-multipliers \"1 2 4 8\"" in generated
