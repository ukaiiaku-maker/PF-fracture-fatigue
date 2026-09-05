"""Regression tests for the developed-response confirmation campaign's
window/decomposition fixes (decomposition-repair pass, post-review).

Covers, with synthetic events (no engine needed):
  - developed_interval_event_indices() excludes the transient events
    (Section A's development_extension_um=20.0 exclusion);
  - stable_growth_gate()'s exposed developed_event_indices agrees with
    developed_interval_event_indices() computed independently;
  - true_final_half_indices/true_final_six_indices give the ACTUAL tail of
    an n-event trajectory, not the old fixed [9..17]/[12..17] windows;
  - inter_event_raw_action_weighted_K_rebond() returns an explicit
    NOT_ARCHIVED status rather than a surrogate weight (and its
    compatibility alias, action_weighted_K_rebond_true_inter_event_
    weighted, still resolves to the same function);
  - apply_tail_sensitivity_gate() only ever downgrades
    DEVELOPED_REBONDING_STEEPENING_CONFIRMED, never other classifications.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture.crack_rebonding_developed_confirmation_v10230 import (  # noqa: E402
    action_weighted_K_rebond_true_inter_event_weighted,
    apply_tail_sensitivity_gate,
    developed_interval_event_indices,
    inter_event_raw_action_weighted_K_rebond,
    stable_growth_gate,
    true_final_half_indices,
    true_final_six_indices,
)


def _synthetic_events(n: int, da_um: float = 5.0, dt_s: float = 100.0) -> list[dict]:
    events = []
    cum = 0.0
    for i in range(n):
        da = da_um * 1.0e-6
        cum += da
        events.append({
            "event_index": i,
            "accepted_length_m": da,
            "cumulative_extension_m": cum,
            "waiting_time_s_this_event": dt_s,
            "action_weighted_K_rebond_Pa_sqrt_m": 9.0e5 if i >= 4 else 0.0,
            "cleavage_action": 1.0e-6 * (i + 1),
            "all_bulk_action_qualified": True,
        })
    return events


def test_developed_interval_excludes_transient_events():
    events = _synthetic_events(30)
    idx = developed_interval_event_indices(events, development_extension_um=20.0)
    # 20um / 5um-per-event = 4 events excluded (indices 0-3); developed
    # starts at the first event whose POST extension exceeds 20um, i.e.
    # event_index 4 (post=25um).
    assert idx[0] == 4
    assert idx == list(range(4, 30))
    assert 0 not in idx and 1 not in idx and 2 not in idx and 3 not in idx


def test_stable_growth_gate_exposes_matching_developed_event_indices():
    events = _synthetic_events(30)
    gate = stable_growth_gate(events, frequency_Hz=1000.0)
    independent = developed_interval_event_indices(events, development_extension_um=20.0)
    assert gate["developed_event_indices"] == independent
    # The developed window must never equal the full all-event set for a
    # 30-event trajectory with a 20um transient exclusion (the aliasing
    # bug this test guards against).
    all_event_indices = [e["event_index"] for e in events]
    assert gate["developed_event_indices"] != all_event_indices


def test_true_final_windows_are_actual_tail_not_fixed_18_event_window():
    n = 30
    half = true_final_half_indices(n)
    six = true_final_six_indices(n)
    assert half == list(range(15, 30))
    assert six == list(range(24, 30))
    # Must NOT equal the old fixed [9..17]/[12..17] windows for n=30.
    assert half != list(range(9, 18))
    assert six != list(range(12, 18))


def test_inter_event_action_weighted_mean_is_not_archived():
    events = _synthetic_events(30)
    result = inter_event_raw_action_weighted_K_rebond(events)
    assert result["status"] == "NOT_ARCHIVED"
    assert result["limitation_tag"] == "INTER_EVENT_RAW_ACTION_WEIGHTING_NOT_ARCHIVED"
    assert result["unconditional_action_weighted_mean_Pa_sqrt_m"] is None
    assert result["conditional_action_weighted_mean_nonzero_only_Pa_sqrt_m"] is None
    # The raw sum is still surfaced for audit, but explicitly marked unsafe.
    assert "unvalidated_raw_cleavage_action_sum_do_not_use_for_physics" in result
    # The compatibility alias must resolve to the exact same function.
    assert action_weighted_K_rebond_true_inter_event_weighted is inter_event_raw_action_weighted_K_rebond


def test_tail_sensitivity_gate_only_downgrades_steepening_confirmed():
    other = {"classification": "DEVELOPED_REBONDING_SEED_SENSITIVE"}
    result = apply_tail_sensitivity_gate(
        other, true_terminal_delta_m_by_seed={1720: 0.01, 1001723: -0.02},
    )
    assert result["classification"] == "DEVELOPED_REBONDING_SEED_SENSITIVE"
    assert result["tail_sensitivity_check"]["evaluated"] is False

    confirmed_pass = {"classification": "DEVELOPED_REBONDING_STEEPENING_CONFIRMED"}
    result_pass = apply_tail_sensitivity_gate(
        confirmed_pass, true_terminal_delta_m_by_seed={1720: 0.6, 1001723: 0.65},
    )
    assert result_pass["classification"] == "DEVELOPED_REBONDING_STEEPENING_CONFIRMED"

    confirmed_fail = {"classification": "DEVELOPED_REBONDING_STEEPENING_CONFIRMED"}
    result_fail = apply_tail_sensitivity_gate(
        confirmed_fail, true_terminal_delta_m_by_seed={1720: 0.6, 1001723: 0.1},
    )
    assert result_fail["classification"] == "DEVELOPED_REBONDING_TAIL_SENSITIVE"
