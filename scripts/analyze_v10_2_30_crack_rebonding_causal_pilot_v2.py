"""Censor-aware causal analysis for the corrected crack-rebonding pilot v2.

Reads the portable, tracked event ledger (artifacts/
crack_rebonding_causal_pilot_v2/event_ledger.json, built by
build_v2_event_ledger.py from a run's raw trajectories.json) -- NOT the
gitignored runs/.../trajectories.json directly -- evaluates the mission's
seven hard gates plus the zero/finite-cohesion causal comparison
(Delta t = t_finite - t_zero at matched event indices, for intervals that
actually contain a complete compressive excursion), and writes
interval_causal_analysis.csv and causal_decision.json.

Terminology note: gate 1 checks EVENT-TIME-AND-LENGTH parity specifically
(accepted_length_m and waiting_time_s_this_event at rel_tol=1e-9) -- not an
unqualified claim that every other state variable (MPZ populations,
emission ledgers, etc) is bit-identical between C0 and C1, which was never
separately checked.

The four "compression-containing" rows gate 4 counts are two UNIQUE
physical event intervals (creation event 0->1 and 2->3, from the shared
C0/C1/C2*/C3* threshold stream before any cohesive divergence), each
evaluated once under the reversible kinetics preset and once under the
persistent kinetics preset -- not four statistically independent samples.

Usage:
    <pinned interpreter> scripts/analyze_v10_2_30_crack_rebonding_causal_pilot_v2.py \\
        [--ledger artifacts/crack_rebonding_causal_pilot_v2/event_ledger.json] \\
        [--out artifacts/crack_rebonding_causal_pilot_v2]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from arrhenius_fracture.crack_rebonding_v10230 import CONTACT_SEMANTICS_LABEL  # noqa: E402

ARTIFACTS_DIR = REPO_ROOT / "artifacts/crack_rebonding_causal_pilot_v2"
DEFAULT_LEDGER_PATH = ARTIFACTS_DIR / "event_ledger.json"
EXPANSION_THRESHOLD_LOG10_DECADE = 0.05

_STRICT_PARITY_REL_TOL = 1.0e-9


def _event_pairs(a: dict, b: dict) -> list[tuple[dict, dict]]:
    return list(zip(a["events"], b["events"]))


def _strict_event_time_and_length_parity(a: dict, b: dict) -> dict[str, Any]:
    """Checks ONLY accepted_length_m and waiting_time_s_this_event at
    rel_tol=1e-9 -- an event-time-and-length parity claim, not a full-state
    one (MPZ populations, emission ledgers, etc are not compared here)."""
    pairs = _event_pairs(a, b)
    mismatches = []
    for ea, eb in pairs:
        for key in ("accepted_length_m", "waiting_time_s_this_event"):
            va, vb = float(ea[key]), float(eb[key])
            denom = max(abs(va), abs(vb), 1.0e-300)
            rel = abs(va - vb) / denom
            if rel > _STRICT_PARITY_REL_TOL:
                mismatches.append({
                    "event_index": ea["event_index"], "key": key,
                    "a": va, "b": vb, "rel_diff": rel, "rel_tol": _STRICT_PARITY_REL_TOL,
                })
    identical = len(mismatches) == 0 and a["n_accepted_events"] == b["n_accepted_events"] and len(pairs) > 0
    return {
        "identical": identical, "n_pairs": len(pairs),
        "n_accepted_events_a": a["n_accepted_events"], "n_accepted_events_b": b["n_accepted_events"],
        "mismatches": mismatches,
        "fields_compared": ["accepted_length_m", "waiting_time_s_this_event"],
        "note": "event-time-and-length parity only, not an unqualified full-state parity claim",
    }


def _zero_bonding(res: dict) -> bool:
    return all(
        e["max_pB_post_commit"] == 0.0 and e["max_K_rebond_post_commit_Pa_sqrt_m"] == 0.0
        for e in res["events"]
    )


def _dynamic_nonzero_bonding(res: dict) -> bool:
    return any(
        e["max_pB_post_commit"] > 0.0 and e["max_K_rebond_post_commit_Pa_sqrt_m"] > 0.0
        for e in res["events"]
    )


def _all_bulk_action_qualified(res: dict) -> bool:
    """Reads the per-event aggregate already computed at run time -- the
    portable ledger drops the raw per-bisection-iteration bulk_action_
    records list (reproducibility detail, not a gate input)."""
    return all(bool(e["all_bulk_action_qualified"]) for e in res["events"])


def _matched_delay_rows(zero_res: dict, finite_res: dict, label: str) -> list[dict[str, Any]]:
    """For each post-first-event interval common to both trajectories
    (matched by event index -- both start from the identical event-0
    threshold stream and diverge only through the cohesive effect itself),
    compute DeltaN_wait = t_finite - t_zero and its decade ratio, tagging
    whether THAT interval (in the zero-cohesion twin, which by construction
    has baseline-identical timing to RB0/RB1) actually contains a complete
    compressive excursion. Also carries the mission's requested
    cleavage-action-weighted K_rebond and finite-minus-zero cleavage-action
    diagnostics for the matched event on each side.
    """
    zero_intervals = {iv["next_event_index"]: iv for iv in zero_res["post_first_event_intervals"]}
    finite_intervals = {iv["next_event_index"]: iv for iv in finite_res["post_first_event_intervals"]}
    zero_events = {e["event_index"]: e for e in zero_res["events"]}
    finite_events = {e["event_index"]: e for e in finite_res["events"]}
    rows = []
    for idx in sorted(set(zero_intervals) & set(finite_intervals)):
        iv_zero = zero_intervals[idx]
        iv_finite = finite_intervals[idx]
        t_zero = iv_zero["waiting_time_s_this_event"]
        t_finite = iv_finite["waiting_time_s_this_event"]
        delta_t = t_finite - t_zero
        log10_ratio = (
            abs(math.log10(t_finite / t_zero)) if t_zero > 0.0 and t_finite > 0.0 else float("inf")
        )
        e_zero = zero_events.get(idx, {})
        e_finite = finite_events.get(idx, {})
        action_zero = e_zero.get("cleavage_action")
        action_finite = e_finite.get("cleavage_action")
        finite_minus_zero_action = (
            action_finite - action_zero
            if action_zero is not None and action_finite is not None else None
        )
        rows.append({
            "config_pair": label,
            # These 4 rows (2 config_pairs x 2 unique intervals) are TWO
            # unique physical event intervals under two kinetics presets,
            # not four independent samples -- see interval_group_id.
            "interval_group_id": f"event_{iv_zero['creation_event_index']}_to_{idx}",
            "creation_event_index": iv_zero["creation_event_index"],
            "next_event_index": idx,
            "contains_complete_negative_excursion": iv_zero["complete_negative_excursion"],
            "negative_contact_duration_s_zero": iv_zero["negative_contact_duration_s"],
            "elapsed_cycles_zero": iv_zero["elapsed_cycles"],
            "t_zero_cohesion_s": t_zero,
            "t_finite_cohesion_s": t_finite,
            "delta_t_s": delta_t,
            "log10_ratio_abs_decade": log10_ratio,
            "expansion_threshold_exceeded": log10_ratio >= EXPANSION_THRESHOLD_LOG10_DECADE,
            "pre_event_p_B_zero": e_zero.get("pre_event_max_pB"),
            "pre_event_p_B_finite": e_finite.get("pre_event_max_pB"),
            "pre_event_K_rebond_finite_Pa_sqrt_m": e_finite.get("pre_event_K_rebond_Pa_sqrt_m"),
            "max_phase_resolved_K_rebond_finite_Pa_sqrt_m": e_finite.get(
                "max_phase_resolved_K_rebond_Pa_sqrt_m"
            ),
            "action_weighted_K_rebond_finite_Pa_sqrt_m": e_finite.get(
                "action_weighted_K_rebond_Pa_sqrt_m"
            ),
            "cleavage_action_zero": action_zero,
            "cleavage_action_finite": action_finite,
            "finite_minus_zero_cleavage_action": finite_minus_zero_action,
            "accepted_length_m_zero": e_zero.get("accepted_length_m"),
            "accepted_length_m_finite": e_finite.get("accepted_length_m"),
        })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", default=str(DEFAULT_LEDGER_PATH))
    parser.add_argument("--out", default=str(ARTIFACTS_DIR))
    args = parser.parse_args(argv)

    ledger_path = Path(args.ledger)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ledger = json.loads(ledger_path.read_text())
    trajectories: dict[str, dict] = ledger["trajectories"]
    c0, c1 = trajectories["C0"], trajectories["C1"]
    c2r, c3r = trajectories["C2R"], trajectories["C3R"]
    c2p, c3p = trajectories["C2P"], trajectories["C3P"]
    c4, c5 = trajectories["C4"], trajectories["C5"]

    gates: dict[str, Any] = {}

    gates["gate_1_c0_c1_event_time_and_length_parity"] = _strict_event_time_and_length_parity(c0, c1)
    gates["gate_1_c0_c1_event_time_and_length_parity"]["pass"] = gates["gate_1_c0_c1_event_time_and_length_parity"]["identical"]

    gate2_c0_c4_bonding = _zero_bonding(c4)
    gate2_c5_zero = _zero_bonding(c5)
    gates["gate_2_c5_exactly_zero_bond_formation_at_R_positive"] = {
        "pass": gate2_c5_zero, "c4_control_also_zero_bonding": gate2_c0_c4_bonding,
    }

    p2r_dynamic = _dynamic_nonzero_bonding(c2r)
    p3r_dynamic = _dynamic_nonzero_bonding(c3r)
    p2p_dynamic = _dynamic_nonzero_bonding(c2p)
    p3p_dynamic = _dynamic_nonzero_bonding(c3p)
    gates["gate_3_dynamic_nonzero_bonding_from_event_created_patch"] = {
        "c2r_dynamic": p2r_dynamic, "c3r_dynamic": p3r_dynamic,
        "c2p_dynamic": p2p_dynamic, "c3p_dynamic": p3p_dynamic,
        "pass": p3r_dynamic or p3p_dynamic,
    }

    rows_reversible = _matched_delay_rows(c2r, c3r, "reversible")
    rows_persistent = _matched_delay_rows(c2p, c3p, "persistent")
    all_rows = rows_reversible + rows_persistent

    compression_rows = [r for r in all_rows if r["contains_complete_negative_excursion"]]
    unique_compression_intervals = sorted({r["interval_group_id"] for r in compression_rows})
    gates["gate_4_delay_in_at_least_two_compression_containing_intervals"] = {
        "n_compression_containing_rows": len(compression_rows),
        "n_unique_compression_containing_intervals": len(unique_compression_intervals),
        "unique_interval_ids": unique_compression_intervals,
        "note": (
            "each unique interval is evaluated once under the reversible "
            "preset and once under the persistent preset -- rows are not "
            "independent samples of distinct physical intervals"
        ),
        "pass": len(unique_compression_intervals) >= 2,
        "rows": compression_rows,
    }

    gates["gate_5_periodic_orbit_or_explicit_actions_only"] = {
        "c0": _all_bulk_action_qualified(c0), "c1": _all_bulk_action_qualified(c1),
        "c2r": _all_bulk_action_qualified(c2r), "c3r": _all_bulk_action_qualified(c3r),
        "c2p": _all_bulk_action_qualified(c2p), "c3p": _all_bulk_action_qualified(c3p),
        "c4": _all_bulk_action_qualified(c4), "c5": _all_bulk_action_qualified(c5),
    }
    gates["gate_5_periodic_orbit_or_explicit_actions_only"]["pass"] = all(
        gates["gate_5_periodic_orbit_or_explicit_actions_only"][k]
        for k in ("c0", "c1", "c2r", "c3r", "c2p", "c3p", "c4", "c5")
    )

    gates["gate_6_contact_semantics_label"] = {
        "pass": True, "contact_semantics_label": CONTACT_SEMANTICS_LABEL,
    }
    gates["gate_7_no_paris_slope_inference"] = {
        "pass": True,
        "note": "single Kmax (18 MPa*sqrt(m)) pilot only; multi-K matrix not launched",
    }

    max_ratio_compression = max(
        (r["log10_ratio_abs_decade"] for r in compression_rows), default=0.0
    )
    expansion_exceeded = max_ratio_compression >= EXPANSION_THRESHOLD_LOG10_DECADE

    overall_gate_pass = all(
        gates[k]["pass"]
        for k in (
            "gate_1_c0_c1_event_time_and_length_parity",
            "gate_2_c5_exactly_zero_bond_formation_at_R_positive",
            "gate_3_dynamic_nonzero_bonding_from_event_created_patch",
            "gate_4_delay_in_at_least_two_compression_containing_intervals",
            "gate_5_periodic_orbit_or_explicit_actions_only",
            "gate_6_contact_semantics_label",
            "gate_7_no_paris_slope_inference",
        )
    )

    if not overall_gate_pass:
        classification = "HARD_GATE_FAILURE_SEE_GATES"
    elif expansion_exceeded:
        classification = "CONTACT_GATED_REBONDING_CAUSAL_EFFECT_DEMONSTRATED"
    else:
        classification = "REBONDING_KINETICALLY_ACTIVE_BUT_MACROSCOPICALLY_SMALL"

    decision = {
        "schema": "v10.2.30_crack_rebonding_causal_pilot_v2_causal_decision_v1",
        "source_ledger_sha256_note": "see file_hashes.json for event_ledger.json's hash",
        "gates": gates,
        "overall_gate_pass": overall_gate_pass,
        "expansion_threshold_log10_decade": EXPANSION_THRESHOLD_LOG10_DECADE,
        "max_log10_ratio_abs_decade_in_compression_containing_intervals": max_ratio_compression,
        "expansion_threshold_exceeded": expansion_exceeded,
        "classification": classification,
        "contact_semantics_label": CONTACT_SEMANTICS_LABEL,
        "multi_K_paris_slope_campaign_authorized": False,
    }

    decision_path = out_dir / "causal_decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    print(f"wrote {decision_path}")
    print(f"classification={classification}")

    csv_path = out_dir / "interval_causal_analysis.csv"
    fieldnames = [
        "config_pair", "interval_group_id", "creation_event_index", "next_event_index",
        "contains_complete_negative_excursion", "negative_contact_duration_s_zero",
        "elapsed_cycles_zero", "t_zero_cohesion_s", "t_finite_cohesion_s", "delta_t_s",
        "log10_ratio_abs_decade", "expansion_threshold_exceeded",
        "pre_event_p_B_zero", "pre_event_p_B_finite", "pre_event_K_rebond_finite_Pa_sqrt_m",
        "max_phase_resolved_K_rebond_finite_Pa_sqrt_m", "action_weighted_K_rebond_finite_Pa_sqrt_m",
        "cleavage_action_zero", "cleavage_action_finite", "finite_minus_zero_cleavage_action",
        "accepted_length_m_zero", "accepted_length_m_finite",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"wrote {csv_path}")

    return 0 if overall_gate_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
