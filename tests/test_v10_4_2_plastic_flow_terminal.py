from __future__ import annotations

from collections import deque
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

# Tests must exercise the same outermost transform used by production.  Importing
# the accepted-work overlay directly bypasses both the positive-J repair and the
# v10.4.3 stagger-consistency repair.
from arrhenius_fracture.plastic_flow_stagger_consistent_v1043 import (
    load_transformed_sharp_front,
    transform_source,
)

ROOT = Path(__file__).resolve().parents[1]


def test_transformed_sharp_front_compiles_and_preserves_fracture_measure():
    source = (ROOT / "arrhenius_fracture" / "sharp_front.py").read_text()
    transformed = transform_source(source)
    compile(transformed, "sharp_front.py[v10.4.3-test]", "exec")
    assert "plastic_flow_no_sharp_fracture" in transformed
    assert "J_pl_diss_J_per_m2" in transformed
    assert "contour_shielding_enters_fracture_hazard': False" in transformed
    assert "plastic_work_enters_fracture_measure': False" in transformed
    assert "plastic_work_enters_cleavage_hazard': False" in transformed
    assert "eng.step(KJ, T, dt_cur)" in transformed
    assert "eng.step(J_pl" not in transformed
    assert "predict_clock_increment(J_pl" not in transformed
    assert "return_info=True" in transformed
    assert "dWp_accepted_gp" in transformed
    assert "W_bulk_plastic_primary_is_constitutive_accepted_work" in transformed
    assert "ep_gp_step0_v1043 = ep_gp.copy()" in transformed
    assert "rho_gp_step0_v1043 = rho_gp.copy()" in transformed
    assert "plastic_work_accepted_gp_v1042 +=" not in transformed
    assert "J_positive = max(sign_ref * J_signed, 0.0)" not in transformed
    assert "J_positive = max(J_signed, 0.0)" in transformed


def test_synthetic_persistent_plastic_window_is_terminal():
    module = load_transformed_sharp_front()
    window = deque(maxlen=2000)
    for step in range(1, 2001):
        window.append({
            "step": step,
            "Uapp": step * 2.0e-7,
            "Ftop": 0.0,
            "J_positive": 0.0,
            "J_signed": 0.0,
            "sigma_tip": 0.0,
            "B": 0.0,
            "lambda_c": 0.0,
            "n_fire": 0,
            "a_tip": 5.0e-4,
            "W_ext": step * 1.0e-6,
            "U_el": 0.0,
            "W_p": step * 1.0e-6,
            "W_emit": 0.0,
        })

    class Args:
        plastic_flow_window_steps = 2000
        plastic_flow_min_step = 2000
        plastic_flow_max_da_fraction = 0.1
        plastic_flow_J_abs_tol_J_per_m2 = 1e-6
        plastic_flow_J_rel_tol = 1e-6
        plastic_flow_sigma_rel_tol = 1e-6
        plastic_flow_min_plastic_fraction = 0.9
        plastic_flow_min_cumulative_plastic_fraction = 0.9
        plastic_flow_max_elastic_fraction = 0.05
        plastic_flow_max_force_fraction = 0.1
        plastic_flow_max_tangent_fraction = 0.05
        plastic_flow_max_dB_window = 1e-6
        plastic_flow_min_cleavage_horizon_ratio = 100.0

    result = module._v1042_terminal_metrics(
        window,
        Args(),
        Eprime=4.0e11,
        da_phys=5.0e-6,
        sigma_reference=3.0e10,
        peak_J_positive=1.0e5,
        peak_force=1.0,
        stiffness_reference=1.0e8,
        remaining_steps=100000,
        nominal_dt_s=8.4,
        cumulative_Wp=2.0e-3,
        cumulative_Uel=0.0,
        cumulative_Wemit=0.0,
    )
    assert result is not None
    assert result["criteria_pass"] is True
    assert all(result["criteria"].values())
    assert result["predicted_remaining_cleavage_time_s"] is None
    assert result["predicted_remaining_cleavage_time_infinite"] is True


def test_nonzero_tip_drive_blocks_terminal():
    module = load_transformed_sharp_front()
    window = deque(maxlen=4)
    for step in range(1, 5):
        window.append({
            "step": step,
            "Uapp": float(step),
            "Ftop": 0.0,
            "J_positive": 10.0,
            "J_signed": 10.0,
            "sigma_tip": 0.0,
            "B": 0.0,
            "lambda_c": 0.0,
            "n_fire": 0,
            "a_tip": 5.0e-4,
            "W_ext": float(step),
            "U_el": 0.0,
            "W_p": float(step),
            "W_emit": 0.0,
        })

    class Args:
        plastic_flow_window_steps = 4
        plastic_flow_min_step = 4
        plastic_flow_max_da_fraction = 0.1
        plastic_flow_J_abs_tol_J_per_m2 = 1e-6
        plastic_flow_J_rel_tol = 1e-6
        plastic_flow_sigma_rel_tol = 1e-6
        plastic_flow_min_plastic_fraction = 0.9
        plastic_flow_min_cumulative_plastic_fraction = 0.9
        plastic_flow_max_elastic_fraction = 0.05
        plastic_flow_max_force_fraction = 0.1
        plastic_flow_max_tangent_fraction = 0.05
        plastic_flow_max_dB_window = 1e-6
        plastic_flow_min_cleavage_horizon_ratio = 100.0

    result = module._v1042_terminal_metrics(
        window,
        Args(),
        Eprime=4.0e11,
        da_phys=5.0e-6,
        sigma_reference=3.0e10,
        peak_J_positive=10.0,
        peak_force=1.0,
        stiffness_reference=1.0,
        remaining_steps=100,
        nominal_dt_s=1.0,
        cumulative_Wp=4.0,
        cumulative_Uel=0.0,
        cumulative_Wemit=0.0,
    )
    assert result is not None
    assert result["criteria_pass"] is False
    assert result["criteria"]["negligible_positive_tip_J"] is False


def test_contour_scan_keeps_shielding_diagnostic_and_raw_positive():
    module = load_transformed_sharp_front()

    def fake_compute(mesh, u, sigma, psi, damage, tip, direction, mat, ell,
                     crack_segments=None, exclude_radius=0.0):
        J = 2.0 * ell
        return J, np.sqrt(J * 4.0), {
            "J_signed": J,
            "r_inner": ell,
            "r_outer": 8.0 * ell,
            "n_active_elements": 20,
        }

    records = module._v1042_contour_scan(
        fake_compute,
        mesh=None,
        u=None,
        sigma_gp=None,
        psi_gp=None,
        damage=None,
        tip_xy=[0.0, 0.0],
        direction=[1.0, 0.0],
        mat=None,
        base_ell_m=1.0,
        multipliers="1 2 4",
        crack_segments=None,
        exclude_radius_m=0.0,
        sign_reference=1.0,
    )
    assert [row["contour_multiplier"] for row in records] == [1.0, 2.0, 4.0]
    assert records[-1]["J_positive_root_convention_J_per_m2"] == 8.0


def test_contour_scan_rejects_legacy_negative_sign_reference():
    module = load_transformed_sharp_front()

    def fake_compute(*args, **kwargs):
        return 1.0, 1.0, {"J_signed": 1.0, "r_outer": 1.0}

    import pytest

    with pytest.raises(RuntimeError, match="non-production directional-J sign reference"):
        module._v1042_contour_scan(
            fake_compute,
            mesh=None,
            u=None,
            sigma_gp=None,
            psi_gp=None,
            damage=None,
            tip_xy=[0.0, 0.0],
            direction=[1.0, 0.0],
            mat=None,
            base_ell_m=1.0,
            multipliers="1",
            crack_segments=None,
            exclude_radius_m=0.0,
            sign_reference=-1.0,
        )


def test_launcher_generation_compiles_and_contains_terminal_contract(tmp_path: Path):
    builder_path = ROOT / "scripts" / "build_v10_4_2_plastic_terminal_launcher.py"
    spec = importlib.util.spec_from_file_location("v1042_builder", builder_path)
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    source = (
        ROOT / "scripts" / "run_v10_2_28_paper_four_class_theta30_1000um.sh"
    ).read_text()
    generated = builder.transform(source)
    output = tmp_path / "generated.sh"
    output.write_text(generated)
    subprocess.run(["bash", "-n", str(output)], check=True)
    assert "sharp_front_v10_4_2_plastic_flow_audited" in generated
    assert "--plastic-flow-terminal" in generated
    assert "scripts/classify_v10_4_2_case.py" in generated
    assert "PLASTIC_FLOW" in generated
    assert "plastic_work_enters_fracture_measure" in generated
    assert "contour_shielding_is_diagnostic_only" in generated
    assert "verify_materialized_case" in generated
    assert "v10_4_2_reuse_audit.json" in generated


def test_plastic_classifier_status_contract(tmp_path: Path):
    scripts = ROOT / "scripts"
    sys.path.insert(0, str(scripts))
    try:
        import classify_v10_4_2_case as classifier
    finally:
        sys.path.pop(0)

    case = tmp_path / "case"
    case.mkdir()
    (case / "PLASTIC_FLOW").write_text("plastic_flow_no_sharp_fracture\n")
    (case / "plastic_flow_terminal_audit.json").write_text(json.dumps({
        "schema": "v10.4.2_plastic_flow_terminal_audit_v1",
        "classification": "plastic_flow_no_sharp_fracture",
        "terminal": True,
        "temperature_K": 1000.0,
        "sharp_fracture_occurred": False,
        "plastic_work_enters_fracture_measure": False,
        "plastic_work_enters_cleavage_hazard": False,
        "contour_shielding_enters_fracture_hazard": False,
        "J_pl_diss_J_per_m2": 123.0,
        "K_pl_equivalent_MPa_sqrt_m": 7.0,
        "J_contour_shielding_J_per_m2": 10.0,
    }))
    payload = classifier.classify(case, 1000.0)
    assert payload["campaign_terminal"] is True
    assert payload["complete"] is False
    assert payload["status"] == "plastic_flow_no_sharp_fracture"
