import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.analyze_pf_current_source_branching_corrected_v5_2_result import (
    normalize,
    signed_and_population_audit,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "analysis_outputs/pf_current_source_branching_corrected_v5_2_execution_20260831T190053Z"


def test_prebranch_normalization_removes_only_read_only_identity_fields():
    physical = {"step": 12, "directional_K_Pa_sqrt_m": 3.0, "candidate_id": "c"}
    left = {**physical, "accepted_state_id": "left", "stress_field_state_id": "left-stress"}
    right = {**physical, "accepted_state_id": "right", "stress_field_state_id": "right-stress"}
    assert normalize([left], 20) == normalize([right], 20)
    right["directional_K_Pa_sqrt_m"] = 4.0
    assert normalize([left], 20) != normalize([right], 20)


def test_population_and_signed_ledger_equations():
    one = np.array([[1.0]])
    zero = np.array([[0.0]])
    fields = {
        "mobile": one,
        "mobile_positive": one,
        "mobile_negative": zero,
        "retained": one,
        "retained_positive": one,
        "retained_negative": zero,
        "accumulated_slip": np.array([[2.0]]),
        "accumulated_slip_positive": np.array([[2.0]]),
        "accumulated_slip_negative": zero,
        "wake_mobile": one,
        "wake_mobile_positive": one,
        "wake_mobile_negative": zero,
        "wake_retained": one,
        "wake_retained_positive": one,
        "wake_retained_negative": zero,
        "wake_slip": np.array([[2.0]]),
        "wake_slip_positive": np.array([[2.0]]),
        "wake_slip_negative": zero,
        "wake_discarded_mobile_total": 0.5,
        "wake_discarded_retained_total": 0.5,
        "wake_discarded_slip_total": 2.0,
        "escaped_total": 1.0,
        "recovered_total": 0.0,
        "emitted_total": 6.0,
    }
    audit = signed_and_population_audit(fields)
    assert audit["population_closure_residual"] == 0.0
    assert audit["slip_closure_residual"] == 0.0
    assert audit["maximum_unsigned_vs_signed_population_residual"] == 0.0


def test_committed_v5_2_result_packet_hashes_and_fail_closed_decision():
    decision = json.loads((RESULT / "final_two_axis_decision.json").read_text())
    assert decision["pair_terminal_result"] == "CORRECTED_THETA40_REPLAY_STOPPED_FAIL_CLOSED_SIGNED_KERNEL_ENVELOPE"
    assert decision["predictive_branching_physics_validated"] is False
    assert decision["raw_partial_evidence"]["prebranch_identity_pass"] is True
    assert decision["raw_partial_evidence"]["same_tip_ownership_pass"] is True
    assert decision["raw_partial_evidence"]["closure_gates_pass_through_stop"] is True

    manifest = json.loads((RESULT / "compact_packet_manifest.json").read_text())
    for item in manifest["files"]:
        path = RESULT / item["path"]
        assert path.stat().st_size == item["bytes"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == item["sha256"]
