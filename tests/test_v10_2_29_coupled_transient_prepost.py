import json

from arrhenius_fracture.persistent_site_cyclic_coupled_audited_v10229 import (
    _coupled_fields,
)
from scripts.analyze_v10_2_29_coupled_transient_screen import summarize_case


def test_coupled_audit_preserves_pre_block_state():
    result = {
        "state_mobile_count_pre": 2.0,
        "state_retained_count_pre": 3.0,
        "state_emitted_total_pre": 4.0,
        "state_sigma_back_Pa_pre": 5.0,
        "coupled_hazard_lambda_start_s": 6.0,
    }
    fields = _coupled_fields(result)
    assert fields == result


def test_transient_screen_uses_true_single_block_pre_state(tmp_path):
    root = tmp_path / "case"
    root.mkdir()
    control = root / "v10_2_29_fixed_deltaK_control.json"
    control.write_text(
        json.dumps(
            {
                "parameter_option": "v913_paper_dbtt01_0202500_persistent_sites",
                "target_deltaK_MPa_sqrt_m": 12.0,
                "target_Kmax_MPa_sqrt_m": 13.0,
                "R": 0.1,
                "cycles_max": 1.0e8,
            }
        )
    )
    record = {
        "loading_mode": "cyclic",
        "temperature_K": 900.0,
        "cycles_consumed": 1.0e8,
        "fired": False,
        "B_pre": 0.02,
        "B": 0.07,
        "dB_block": 0.05,
        "state_sigma_back_Pa_pre": 10.0,
        "persistent_sigma_back_Pa": 30.0,
        "state_mobile_count_pre": 1.0,
        "state_mobile_count": 4.0,
        "state_retained_count_pre": 2.0,
        "state_retained_count": 5.0,
        "state_emitted_total_pre": 3.0,
        "state_emitted_total": 9.0,
        "state_active_K_shield_signed_Pa_sqrt_m_pre": -2.0,
        "state_active_K_shield_signed_Pa_sqrt_m": -7.0,
        "coupled_hazard_log_lambda_span_decades": 1.0,
        "coupled_hazard_transient_cycles": 1.0e6,
        "coupled_hazard_stationary_tail_cycles": 9.9e7,
        "coupled_hazard_accepted_segments": 3,
        "coupled_hazard_rejected_splits": 2,
        "coupled_hazard_lambda_min_s": 1.0e-9,
        "coupled_hazard_lambda_max_s": 1.0e-8,
        "coupled_hazard_segments": [{"state_target_ratio": 0.5}],
    }
    (root / "kinetic_tip_cell_audit_v101.json").write_text(
        json.dumps({"records": [record]})
    )

    summary = summarize_case(control)
    assert summary is not None
    assert summary["explicit_pre_block_state_available"] is True
    assert summary["B_initial"] == 0.02
    assert summary["B_final"] == 0.07
    assert summary["B_change"] == 0.05
    assert summary["sigma_back_initial_Pa"] == 10.0
    assert summary["sigma_back_final_Pa"] == 30.0
    assert summary["mobile_initial"] == 1.0
    assert summary["mobile_final"] == 4.0
    assert summary["retained_initial"] == 2.0
    assert summary["retained_final"] == 5.0
    assert summary["emitted_initial"] == 3.0
    assert summary["emitted_final"] == 9.0
    assert summary["active_K_shield_initial_Pa_sqrt_m"] == -2.0
    assert summary["active_K_shield_final_Pa_sqrt_m"] == -7.0
