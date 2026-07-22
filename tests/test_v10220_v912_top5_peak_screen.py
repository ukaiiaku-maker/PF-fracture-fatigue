from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from arrhenius_fracture.parameter_registry_v9111 import read_registry, select_option
from arrhenius_fracture import sharp_front_v10_2_20 as entry


def test_v912_registry_contains_exact_five_rows_and_spatial_contract():
    rows = read_registry(entry.default_v912_registry_path())
    assert tuple(row["option_key"] for row in rows) == entry.V912_OPTIONS
    assert len(rows) == 5
    for row in rows:
        assert row["material_class"] == "DBTT"
        assert float(row["Tref_K"]) == 481.33
        assert int(float(row["n_slip_channels"])) == 2
        assert float(row["rho_forest_floor_m2"]) == 5.0e12
        assert float(row["mobile_shield_fraction"]) == 0.0
        assert float(row["source_recovery_rate_s"]) == 0.0
        assert float(row["L_pz_um_recommended"]) == 50.0
        assert int(float(row["n_bins_recommended"])) == 80
        assert row["v912_parameterization_semantics"].startswith(
            "peak cleavage resistance"
        )


def test_v912_candidate_fingerprints_and_control_role():
    expected = {
        "v912_peak_0368": "v912_targeted_local_peak_013476_0368",
        "v912_peak_0314": "v912_targeted_local_peak_013476_0314",
        "v912_peak_0162": "v912_targeted_local_peak_013476_0162",
        "v912_late_0118": "v912_targeted_local_peak_005518_0118",
        "v912_plateau_0403": "v912_targeted_local_plateau_010759_0403",
    }
    for option, candidate in expected.items():
        selected = select_option(
            option,
            entry.default_v912_registry_path(),
            canonical_stage3_only=False,
        )
        assert selected.candidate_id == candidate
    control = select_option(
        "v912_plateau_0403",
        entry.default_v912_registry_path(),
        canonical_stage3_only=False,
    )
    assert control.row["v912_source_exhausted_control"] == "true"


def test_prepare_writes_visible_active_and_inactive_parameter_audit(tmp_path: Path):
    case = tmp_path / "case"
    args = [
        "--parameter-option", "v912_peak_0368",
        "--parameter-registry", str(entry.default_v912_registry_path()),
        "--out", str(case),
        "--mode", "2d",
    ]
    selected, manifest, audit = entry._prepare_v912_option(args)
    payload = json.loads(audit.read_text())
    assert selected.candidate_id == "v912_targeted_local_peak_013476_0368"
    assert manifest.is_file()
    assert payload["mechanics_changed"] is False
    assert payload["source_model_changed"] is False
    assert payload["inactive_field_policy"] == "fail_visible_not_silent"
    assert payload["active_source_and_recovery_fields"]["source_sites_per_system"] > 0.0
    inactive = payload["campaign_fields_retained_for_audit_but_inactive_in_frozen_solver"]
    assert inactive["rho_source0_m2"] > 0.0
    assert inactive["recovery_H0_eV"] > 0.0
    assert payload["J_reporting_policy"]["J_init_stable_tearing"] == (
        "not represented by cleavage-only geometry law"
    )
    assert args[args.index("--mpz-length-um") + 1] == "50.0"
    assert args[args.index("--mpz-n-bins") + 1] == "80"


def test_runner_and_analyzer_parse_and_preserve_exact_controls():
    root = Path(__file__).resolve().parents[1]
    runner = root / "scripts" / "run_v10_2_20_v912_top5_peak_screen.sh"
    analyzer = root / "scripts" / "plot_v10_2_20_v912_top5_peak_screen.py"
    syntax = subprocess.run(
        ["bash", "-n", str(runner)], capture_output=True, text=True, check=False
    )
    assert syntax.returncode == 0, syntax.stderr
    text = runner.read_text()
    for token in (
        "sharp_front_v10_2_20",
        "--bulk-plasticity-mode tip_only",
        "--tip-plasticity",
        "--active-shielding",
        "--signed-active-shielding",
        "--mobile-shield-fraction 0",
        "--no-wake-shielding",
        "--crystal-aniso --crystal-compete",
        "--max-fronts 1",
        "--crack-backend sharp_wake",
        "seed = base + int(round(temperature))",
        "DEFAULT_TARGET=50",
    ):
        assert token in text
    compiled = subprocess.run(
        [sys.executable, "-m", "py_compile", str(analyzer), str(root / "arrhenius_fracture" / "sharp_front_v10_2_20.py")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert compiled.returncode == 0, compiled.stderr
    help_result = subprocess.run(
        [sys.executable, str(analyzer), "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--outroot" in help_result.stdout
    assert "--require-cases" in help_result.stdout


def test_analyzer_reports_J_and_refuses_to_invent_tearing_or_ctod():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "plot_v10_2_20_v912_top5_peak_screen.py").read_text()
    assert "J_c_from_KJ_J_per_m2" in text
    assert "J_from_KJ_kJ_per_m2" in text
    assert "absorbed_work_at_cleavage" in text
    assert '"J_init_stable_tearing": "not represented"' in text
    assert '"CTOD": "not available in frozen solver output"' in text
    assert "not presumed valid K_IC" in text
