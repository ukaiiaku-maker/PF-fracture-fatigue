import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.persistent_site_cyclic_coupled_audited_v10229 import (
    _active_state_snapshot,
)


class SnapshotEngine:
    def __init__(self):
        self.mpz = SimpleNamespace(
            mobile_positive=np.asarray([[1.0, 2.0, 3.0]]),
            mobile_negative=np.asarray([[0.5, 0.25, 0.0]]),
            retained_positive=np.asarray([[0.1, 0.2, 0.3]]),
            retained_negative=np.asarray([[0.0, 0.1, 0.0]]),
            accumulated_slip_positive=np.asarray([[2.0, 4.0, 6.0]]),
            accumulated_slip_negative=np.asarray([[1.0, 0.0, 1.0]]),
            wake_mobile_positive=np.zeros((1, 0)),
            wake_mobile_negative=np.zeros((1, 0)),
            wake_retained_positive=np.zeros((1, 0)),
            wake_retained_negative=np.zeros((1, 0)),
            wake_slip_positive=np.zeros((1, 0)),
            wake_slip_negative=np.zeros((1, 0)),
            mobile_count=7.0,
            retained_count=0.7,
            continuum_source_last_sigma_back_Pa=2.0e8,
            continuum_source_last_aggregate_hazard_s=3.0,
            advance_total_m=0.0,
            n_bins=3,
            wake_n_bins=0,
        )
        self.W_emit = 1.0
        self.K_prev = 2.0
        self.n_adv = 0
        self.a_adv = 0.0
        self.micro_advance_total_m = 0.0
        self.checkpoint_advance_total_m = 0.0

    def r_eff(self):
        return 1.0e-6

    def K_shield(self):
        return 0.0


def test_active_state_snapshot_is_opt_in(monkeypatch):
    engine = SnapshotEngine()
    monkeypatch.delenv("V10230_SAVE_ACTIVE_STATE_SNAPSHOT", raising=False)
    assert _active_state_snapshot(engine) == {}
    monkeypatch.setenv("V10230_SAVE_ACTIVE_STATE_SNAPSHOT", "1")
    payload = _active_state_snapshot(engine)
    snapshot = payload["coupled_hazard_active_state_snapshot"]
    assert len(snapshot["vector"]) > 0
    names = {field["name"] for field in snapshot["fields"]}
    assert "mobile_positive" in names
    assert "retained_positive" in names
    assert "accumulated_slip_positive" in names


def test_generic_launcher_requires_and_records_target_delta_k():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts" / "run_v10_2_30_weakt_high_cycle_1e12.sh").read_text()
    assert "TARGET_DELTAK" in text
    assert "set TARGET_DELTAK" in text
    assert 'export DELTA_K_MPA_SQRT_M="$TARGET_DELTAK"' in text
    assert "target_deltaK_MPa_sqrt_m" in text
    assert "V10230_SAVE_ACTIVE_STATE_SNAPSHOT" in text
    assert "analyze_v10_2_30_high_cycle_visuals.py" in text


def test_visual_analyzer_writes_mechanical_and_mpz_diagnostics(tmp_path):
    root = Path(__file__).resolve().parents[1]
    snapshot = {
        "model_id": "test",
        "vector": [
            1.0, 2.0, 3.0,
            0.5, 0.25, 0.0,
            0.1, 0.2, 0.3,
            0.0, 0.1, 0.0,
            2.0, 4.0, 6.0,
            1.0, 0.0, 1.0,
        ],
        "fields": [],
        "diagnostics": {},
        "geometry_signature": [],
    }
    cursor = 0
    for name in (
        "mobile_positive",
        "mobile_negative",
        "retained_positive",
        "retained_negative",
        "accumulated_slip_positive",
        "accumulated_slip_negative",
    ):
        snapshot["fields"].append(
            {
                "owner": "mpz",
                "name": name,
                "shape": [1, 3],
                "start": cursor,
                "stop": cursor + 3,
                "floor": 1.0,
            }
        )
        cursor += 3
    audit = {
        "records": [
            {
                "cycles_consumed": 1.0e12,
                "B": 1.0e-4,
                "physical_hazard_action_block": 2.0e-4,
                "persistent_sigma_back_Pa": 5.0e8,
                "state_mobile_count": 6.0,
                "state_retained_count": 0.7,
                "state_emitted_total": 7.0,
                "state_escaped_total": 0.0,
                "coupled_hazard_active_state_snapshot": snapshot,
                "coupled_hazard_modes": [
                    {"mode": "exact_cycle_burst", "cycles": 8.0},
                    {
                        "mode": "slow_projective",
                        "cycles": 1.0e12 - 8.0,
                        "hazard_error": 5.0e-4,
                        "drift_error": 1.0e-6,
                    },
                ],
            }
        ]
    }
    (tmp_path / "kinetic_tip_cell_audit_v101.json").write_text(json.dumps(audit))
    (tmp_path / "high_cycle_summary.json").write_text(
        json.dumps({"cycles_consumed": 1.0e12, "fired_records": 0})
    )
    (tmp_path / "v10_2_30_fixed_deltaK_control.json").write_text(
        json.dumps({"fatigue_censor_status": "right_censored_no_event"})
    )
    (tmp_path / "exit_code.txt").write_text("0\n")
    (tmp_path / "wall_seconds.txt").write_text("10\n")
    (tmp_path / "steps_0300K.csv").write_text(
        "sigma_tip_Pa,sigma_back_Pa,mpz_mobile_count,mpz_retained_count,"
        "fatigue_DeltaK_target_Pa_sqrtm,fatigue_Kmax_target_Pa_sqrtm\n"
        "3e9,5e8,6,0.7,9.5e6,1.05e7\n"
    )
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts" / "analyze_v10_2_30_high_cycle_visuals.py"),
            str(tmp_path),
        ],
        check=True,
        cwd=root,
    )
    expected = {
        "high_cycle_mode_timeline.png",
        "high_cycle_validation_history.png",
        "final_mechanical_response.png",
        "final_mpz_state_profiles.png",
        "mpz_activity_proxy.png",
        "high_cycle_summary_panel.png",
        "high_cycle_visual_manifest.json",
    }
    assert expected.issubset({path.name for path in tmp_path.iterdir()})
    manifest = json.loads((tmp_path / "high_cycle_visual_manifest.json").read_text())
    assert manifest["active_state_snapshot_available"] is True
    assert manifest["damage_field_claimed"] is False
