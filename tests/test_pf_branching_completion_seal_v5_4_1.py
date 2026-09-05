import csv
import inspect
import json

from arrhenius_fracture.sharp_front_v11_branching import _restore_shared_engine
from scripts.plot_pf_branching_archived_fields_v5_4_1 import event_curve
from scripts import qualify_pf_branching_exact_prefix_weights_v5_4_1 as weights


class DummyMPZ:
    def __init__(self):
        self.recorded = "initializer"
        self.unrecorded_default = 7


class DummyEngine:
    def __init__(self):
        self.mpz = DummyMPZ()
        self.recorded = "initializer"
        self.unrecorded_default = 11


def test_exact_restore_drops_initializer_defaults_and_preserves_payload_alias():
    engine = DummyEngine()
    shared = {"family": "one-object"}
    payload = {
        "schema": "v11.shared-production-engine-state/1",
        "engine_type": "DummyEngine",
        "engine_fields": {"recorded": 3, "family": shared},
        "mpz_type": "DummyMPZ",
        "mpz_fields": {"recorded": 5, "family": shared},
    }
    restored = _restore_shared_engine(engine, payload)
    assert restored.recorded == 3
    assert restored.mpz.recorded == 5
    assert not hasattr(restored, "unrecorded_default")
    assert not hasattr(restored.mpz, "unrecorded_default")
    assert restored.family is restored.mpz.family


def test_restore_sentinel_is_default_off_and_precedes_production_loop():
    source = inspect.getsource(__import__(
        "arrhenius_fracture.sharp_front_v11_branching", fromlist=["run_2d"]
    ).run_2d)
    marker = 'os.environ.get(\n            "PF_V5_4_1_RESTORE_ONLY_SENTINEL_OUT"'
    assert marker in source
    assert source.index(marker) < source.index("for step in range")
    assert "return None" in source[source.index(marker):source.index("for step in range")]


def test_weight_audit_exposes_required_fail_closed_fields():
    source = inspect.getsource(weights.main)
    for field in (
        "resolver_mode", "legacy_prefix_delegation_flag", "resolved_state_ids",
        "weights_by_state_id", "legacy_weight_sum", "appended_weight_sum",
        "boundary_action", "operator_digest_sha256",
    ):
        assert field in source
    assert "BEYOND_QUALIFIED_ENVELOPE_FAIL_CLOSED" in source
    assert "envelope_relative_tolerance" in source
    assert "interpolation_error_metadata" in source


def test_event_curve_uses_selected_event_owner_not_participant_order(tmp_path):
    case = tmp_path / "case"
    case.mkdir()
    with (case / "fronts.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "step", "front_id", "parent_front_id", "status", "termination_reason",
            "tip_x_m", "tip_y_m", "arclength_m",
        ))
        writer.writeheader()
        writer.writerow({"step": 373, "front_id": "b0fa_short", "parent_front_id": "root",
                         "status": "active", "termination_reason": "", "tip_x_m": 0.00059,
                         "tip_y_m": -0.00004, "arclength_m": 3e-5})
        writer.writerow({"step": 373, "front_id": "b7d_long", "parent_front_id": "root",
                         "status": "active", "termination_reason": "", "tip_x_m": 0.00061,
                         "tip_y_m": 0.00001, "arclength_m": 4e-5})
    (case / "directional_rates.jsonl").write_text(json.dumps({
        "step": 373, "tip_id": "b0fa_short", "candidate_id": "cleavage:(010)",
        "selected_event_tip_id": "b7d_long", "selected_event_candidate_id": "cleavage:(100)",
    }) + "\n")
    (case / "branch_action_trials.jsonl").write_text(json.dumps({
        "step": 373, "accepted": True, "participating_front_ids": ["b0fa_short", "b7d_long"],
        "candidate_ids": ["cleavage:(100)"], "local_J_valid": [True],
        "J_local_signed_J_per_m2": [12.0], "G_marginal_J_per_m2": [None],
        "J_kin_used_J_per_m2": [12.0], "local_J_invalid_reason": [None],
        "directional_K_Pa_sqrt_m": [2.0e6],
    }) + "\n")
    rows, _, _ = event_curve(case, "fixture")
    assert rows[0]["selected_event_tip_id"] == "b7d_long"
    assert rows[0]["candidate_owner_front_id"] == "b7d_long"


def test_corrected_figure_publisher_does_not_use_first_participant_as_owner():
    source = inspect.getsource(event_curve)
    assert 'selected_by_step.get(step)' in source
    assert 'participating_front_ids"][0]' not in source
