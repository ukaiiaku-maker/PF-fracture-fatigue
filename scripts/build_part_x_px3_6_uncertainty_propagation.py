"""PX3.6 section 5: propagate the numerical/action uncertainty bound the
mission's own frozen gate requires (|S_h| > max(0.005, 5*epsilon_S_h)) for
D3, D5, and D6, using this system's ALREADY-CERTIFIED, enforced per-event
bulk-action relative-error tolerance (CrackRebondingControls.
bulk_action_error_rel_tol = 1e-3) rather than an unvalidated new estimate.

bulk_action_error_rel_tol is not a documentation constant -- it is an
actively enforced gate: persistent_site_cyclic_energy_gated_v10230.py
refuses to accept an event whose accumulated bulk-action tail bound exceeds
this relative tolerance (crack_rebonding_v10230.py's
`tail_bound <= cfg.bulk_action_error_rel_tol * reference_scale` checks).
Every accepted event in every PX3/PX3.5/PX3.6 trajectory is therefore
already certified to have its accumulated cleavage action correct to this
relative tolerance.

Propagation (first-order, conservative):
  g = accepted_length_m / cumulative_cycles. accepted_length_m is a fixed
  physical quantity (not subject to this action-integration error).
  cumulative_cycles is inferred from the accumulated-action threshold
  crossing, so its relative error is taken equal to the certified action
  relative tolerance (a standard, conservative first-order carry-over,
  assuming no cancellation across the ~12 events a trajectory aggregates).

  epsilon_log10_rate = bulk_action_error_rel_tol / ln(10)  (relative
  error in g propagated through log10, holding accepted_length_m exact)

  epsilon_S_h = 2 * epsilon_log10_rate  (linear, worst-case combination of
  the finite and zero trajectories' independent errors -- no assumed
  beneficial cancellation)

This is a conservative, first-order, order-of-magnitude bound built from a
real certified system constant -- not a claim of exact error propagation
through the full adaptive bisection search. Documented as such.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls

MEASURABLE_GATE = 0.01
FLOOR_GATE = 0.005


def main() -> None:
    bulk_action_error_rel_tol = CrackRebondingControls().bulk_action_error_rel_tol
    epsilon_log10_rate = bulk_action_error_rel_tol / math.log(10.0)
    epsilon_S_h = 2.0 * epsilon_log10_rate
    operative_floor = max(FLOOR_GATE, 5.0 * epsilon_S_h)

    freq_data = json.loads((ARTIFACTS_DIR / "px3_5_frequency_bisection.json").read_text())
    dwell_data = json.loads((ARTIFACTS_DIR / "px3_5_dwell_audit_classification.json").read_text())
    persistent_path = REPO_ROOT / "runs" / "crack_rebonding_part_x_v1" / "px3_5_persistent_at_transition_summary.json"
    persistent_data = json.loads(persistent_path.read_text()) if persistent_path.exists() else None

    decisions = {}

    # D3: frequency-transition S_h at the localized condition.
    if freq_data["localized"] is not None:
        S_h_D3 = freq_data["localized"]["S_h"]
        decisions["D3"] = {
            "S_h": S_h_D3, "epsilon_log10_rate_zero": epsilon_log10_rate, "epsilon_log10_rate_finite": epsilon_log10_rate,
            "epsilon_S_h": epsilon_S_h, "operative_gate": operative_floor,
            "clears_gate": abs(S_h_D3) > operative_floor,
            "margin_factor": abs(S_h_D3) / operative_floor if operative_floor > 0 else float("inf"),
        }

    # D5: passivation chemistry=1.0's live effect (from the original screen).
    pairs = json.loads((ARTIFACTS_DIR / "px3_screen_pair_analysis.json").read_text())["pairs"]
    d5_pair = next(p for p in pairs if p["protocol"] == "7.5_passivation_chemistry" and p["chemistry_factor"] == 1.0)
    S_h_D5 = d5_pair["S_h_all"]
    decisions["D5"] = {
        "S_h": S_h_D5, "epsilon_log10_rate_zero": epsilon_log10_rate, "epsilon_log10_rate_finite": epsilon_log10_rate,
        "epsilon_S_h": epsilon_S_h, "operative_gate": operative_floor,
        "clears_gate": abs(S_h_D5) > operative_floor,
        "margin_factor": abs(S_h_D5) / operative_floor if operative_floor > 0 else float("inf"),
    }

    # D6: persistent-vs-reversible distinction at the transition frequency.
    if persistent_data is not None and freq_data["localized"] is not None:
        delta_S_h_D6 = abs(persistent_data["S_h"] - freq_data["localized"]["S_h"])
        # Distinction gate compares TWO independent S_h values, so the
        # combined error is the linear sum of each S_h's own epsilon_S_h.
        epsilon_delta_D6 = 2.0 * epsilon_S_h
        operative_floor_D6 = max(FLOOR_GATE, 5.0 * epsilon_delta_D6)
        decisions["D6"] = {
            "delta_S_h": delta_S_h_D6, "epsilon_S_h_each": epsilon_S_h, "epsilon_delta_S_h": epsilon_delta_D6,
            "operative_gate": operative_floor_D6, "clears_gate": delta_S_h_D6 > operative_floor_D6,
            "margin_factor": delta_S_h_D6 / operative_floor_D6 if operative_floor_D6 > 0 else float("inf"),
        }

    result = {
        "schema": "v10230_part_x_px3_6_uncertainty_propagation_v1",
        "certified_source_constant": {
            "name": "CrackRebondingControls.bulk_action_error_rel_tol", "value": bulk_action_error_rel_tol,
            "enforcement": (
                "Actively enforced, not a documentation constant: persistent_site_cyclic_"
                "energy_gated_v10230.py refuses to accept an event whose accumulated bulk-"
                "action tail bound exceeds this relative tolerance (see crack_rebonding_v10230.py's "
                "tail_bound <= cfg.bulk_action_error_rel_tol * reference_scale checks)."
            ),
        },
        "propagation_method": (
            "First-order, conservative: cumulative_cycles' relative error is taken equal to the "
            "certified per-event bulk-action relative tolerance (no assumed error cancellation "
            "across a trajectory's ~12 events); accepted_length_m is treated as exact (a fixed "
            "physical quantity, not subject to action-integration error); epsilon_log10_rate = "
            "bulk_action_error_rel_tol / ln(10); epsilon_S_h = 2 * epsilon_log10_rate (linear "
            "worst-case combination of the independent finite/zero trajectory errors). This is an "
            "order-of-magnitude bound from a real certified system constant, not an exact "
            "propagation through the full adaptive bisection search."
        ),
        "epsilon_log10_rate": epsilon_log10_rate, "epsilon_S_h": epsilon_S_h,
        "gate_measurable_decade": MEASURABLE_GATE, "gate_floor_decade": FLOOR_GATE,
        "operative_floor_after_uncertainty": operative_floor,
        "decisions": decisions,
    }
    out_path = ARTIFACTS_DIR / "px3_6_uncertainty_propagation.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"wrote {out_path}")
    print(f"epsilon_S_h = {epsilon_S_h:.6f}, operative floor = {operative_floor:.6f}")
    for name, d in decisions.items():
        print(f"  {name}: clears_gate={d['clears_gate']} margin_factor={d['margin_factor']:.2f}x")


if __name__ == "__main__":
    main()
