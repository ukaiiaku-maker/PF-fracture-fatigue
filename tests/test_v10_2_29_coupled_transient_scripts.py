import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANALYZER = ROOT / "scripts" / "analyze_v10_2_29_coupled_transient_screen.py"
RUNNER = ROOT / "scripts" / "run_v10_2_29_high_cycle_fatigue.sh"
WRAPPER = ROOT / "scripts" / "run_v10_2_29_dbtt_peak_coupled_transient_screen.sh"


def write_case(root: Path, *, fired: bool) -> None:
    root.mkdir(parents=True)
    (root / "v10_2_29_fixed_deltaK_control.json").write_text(
        json.dumps(
            {
                "parameter_option": "v913_paper_dbtt01_0202500_persistent_sites",
                "target_deltaK_MPa_sqrt_m": 4.0,
                "target_Kmax_MPa_sqrt_m": 4.444444444444445,
                "R": 0.1,
                "cycles_max": 1.0e9,
            }
        )
    )
    records = [
        {
            "loading_mode": "cyclic",
            "temperature_K": 700.0,
            "cycles_consumed": 1.0e5,
            "fired": False,
            "B": 0.01,
            "persistent_sigma_back_Pa": 1.0e8,
            "state_mobile_count": 2.0,
            "state_retained_count": 0.2,
            "state_emitted_total": 2.2,
            "state_active_K_shield_signed_Pa_sqrt_m": 1.0e5,
            "coupled_hazard_lambda_min_s": 1.0e-10,
            "coupled_hazard_lambda_max_s": 1.0e-8,
            "coupled_hazard_log_lambda_span_decades": 2.0,
            "coupled_hazard_transient_cycles": 1.0e5,
            "coupled_hazard_stationary_tail_cycles": 0.0,
            "coupled_hazard_accepted_segments": 3,
            "coupled_hazard_rejected_splits": 2,
            "coupled_hazard_segments": [{"state_target_ratio": 0.20}],
        },
        {
            "loading_mode": "cyclic",
            "temperature_K": 700.0,
            "cycles_consumed": 2.0e5,
            "fired": fired,
            "B": 0.0 if fired else 0.02,
            "persistent_sigma_back_Pa": 2.0e8,
            "state_mobile_count": 3.0,
            "state_retained_count": 0.6,
            "state_emitted_total": 3.6,
            "state_active_K_shield_signed_Pa_sqrt_m": 3.0e5,
            "coupled_hazard_lambda_min_s": 1.0e-9,
            "coupled_hazard_lambda_max_s": 1.0e-7,
            "coupled_hazard_log_lambda_span_decades": 2.0,
            "coupled_hazard_transient_cycles": 2.0e5,
            "coupled_hazard_stationary_tail_cycles": 0.0,
            "coupled_hazard_accepted_segments": 4,
            "coupled_hazard_rejected_splits": 1,
            "coupled_hazard_segments": [{"state_target_ratio": 0.15}],
        },
    ]
    (root / "kinetic_tip_cell_audit_v101.json").write_text(
        json.dumps({"records": records})
    )


def test_coupled_transient_analyzer_emits_strict_json_and_classifies(tmp_path):
    write_case(tmp_path / "censored", fired=False)
    write_case(tmp_path / "event", fired=True)
    completed = subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            str(tmp_path),
            "--minimum-log-lambda-span-decades",
            "0.3",
            "--minimum-state-target-ratio",
            "0.05",
            "--require-candidate",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "NaN" not in completed.stdout
    payload = json.loads((tmp_path / "coupled_transient_screen.json").read_text())
    assert payload["candidate_count"] == 2
    assert payload["delayed_event_candidate_count"] == 1
    assert payload["censored_transient_candidate_count"] == 1
    censored = next(row for row in payload["cases"] if row["right_censored"])
    event = next(row for row in payload["cases"] if not row["right_censored"])
    assert censored["first_event_cycle"] is None
    assert event["first_event_cycle"] == 3.0e5


def test_coupled_transient_shell_scripts_are_valid_and_explicit():
    subprocess.run(["bash", "-n", str(RUNNER)], check=True)
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    runner = RUNNER.read_text()
    wrapper = WRAPPER.read_text()
    assert "MODE must be horizon, transient, or growth" in runner
    assert "analyze_v10_2_29_coupled_transient_screen.py" in runner
    assert "v913_paper_dbtt01_0202500_persistent_sites" in wrapper
    assert "v913_paper_peak01_0242980_persistent_sites" in wrapper
    assert 'mkdir -p "$case_root"' in wrapper
    assert '"$PYTHON_BIN" scripts/analyze_v10_2_29_coupled_transient_screen.py' in wrapper
