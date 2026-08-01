from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_builder():
    path = ROOT / "scripts" / "build_v10_4_2_plastic_terminal_launcher.py"
    spec = importlib.util.spec_from_file_location("v1042_adapter_builder", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_v1042_scheduler_adapter_executes_full_terminal_contract():
    builder = _load_builder()
    scheduler = "\n".join(
        [
            "    --bulk-plasticity-mode tip_only",
            '"v10.2.30_hazard_energy_gated_orientation_rate_campaign_v1"',
            '"v10.2.30_hazard_energy_gated_orientation_rate_case_contract_v1"',
            '    "hazard_energy_gate": True,',
            '    "hazard_energy_gate": True,',
            '    "hazard_energy_gate": True,',
            'required = [',
            '    root / "COMPLETE",',
            '    root / "stage3_case_status.json",',
            'if status.get("complete") is not True:',
            '    raise SystemExit(1)',
            '    root / "v10_2_30_hazard_energy_gate_audit.json",',
            'command = (root / "command.sh").read_text()',
            '    f"--parameter-option {expected[\'option\']}",',
            'scripts/classify_v10_2_15_stage3_case.py',
            '    if [[ -f "$case_root/COMPLETE" ]]; then',
            '      echo "ERROR: complete-looking case failed contract verification: $case_root" >&2',
            '      return 3',
            '    fi',
            '            "complete": (',
            '                status.get("complete") is True',
            '                and (case_root / "COMPLETE").is_file()',
            '                and not (case_root / "RUN_FAILED").exists()',
            '            ),',
            '"$PYTHON_BIN" scripts/plot_v10_2_27_paper_four_class_rcurves.py \\',
            '  --outroot "$OUTROOT" \\',
            '  --target-extension-um "$TARGET_EXT_UM" || {',
            '    echo "ERROR: four-class R-curve postprocessing failed" >&2',
            '    exit 1',
            '  }',
        ]
    )
    namespace = {"scheduler": scheduler}

    def replace_scheduler_exact(
        old: str,
        new: str,
        expected_count: int = 1,
        label: str = "scheduler token",
    ) -> None:
        actual = namespace["scheduler"].count(old)
        if actual != expected_count:
            raise RuntimeError(
                f"{label} changed: expected {expected_count}, found {actual}"
            )
        namespace["scheduler"] = namespace["scheduler"].replace(old, new)

    namespace["replace_scheduler_exact"] = replace_scheduler_exact
    exec(builder._scheduler_adapter(), namespace)
    transformed = namespace["scheduler"]

    assert "--plastic-flow-terminal" in transformed
    assert "v10.4.2_bulk_plastic_flow_orientation_rate_campaign_v1" in transformed
    assert "v10.4.2_bulk_plastic_flow_orientation_rate_case_contract_v1" in transformed
    assert "terminal_markers = [root / \"COMPLETE\", root / \"PLASTIC_FLOW\"]" in transformed
    assert "scripts/classify_v10_4_2_case.py" in transformed
    assert "verify_materialized_case" in transformed
    assert "plastic_flow_no_sharp_fracture" in transformed
    assert "No sharp-fracture COMPLETE cases" in transformed
    assert "contour_shielding_is_diagnostic_only" in transformed
    assert "plastic_work_enters_fracture_measure" in transformed
