"""PX5 (mission section 8, review-scoped): the scientific PX5 decision.

Reduces px5_static_shield_analysis.csv into exact, unrounded fold-
slowdown and percent-rate-reduction values for both prescribed static
controls (ceiling, analytical periodic-orbit-matched) at every (D2/D5,
Kmax) point, using the same wording convention px4_scientific_decision.py
established: "gives approximately a {fold:.1f}-fold slowdown, or a
{pct:.1f}% rate reduction" -- never "{fold:.1f}x rate reduction".

All 10 points classified MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY, but
the underlying pattern is much more specific than that categorical label
alone conveys: the ceiling control reproduces 100.0-101.9% of the dynamic
slowdown at EVERY point (the static absolute shielding LEVEL alone
explains almost the entire effect), while the periodic-orbit-matched
control reproduces only 22-40% (a naive duration-weighted-average K_b
substantially UNDERESTIMATES the true suppression) -- this asymmetry,
not "kinetic history in general," is what keeps the classification out
of FIXED_ABSOLUTE_COHESIVE_SHIELDING_DOMINANT territory under the
review's own both-controls-must-agree gate.

PROVISIONAL_PENDING_PX6_PX7, carrying the same scope labels as PX4's
decision forward.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"


def fold_and_pct(S_h: float) -> tuple[float, float]:
    fold = 10.0 ** (-S_h)
    pct = (1.0 - 10.0 ** S_h) * 100.0
    return fold, pct


def wording(fold: float, pct: float) -> str:
    return f"gives approximately a {fold:.1f}-fold slowdown, or a {pct:.1f}% rate reduction"


def main() -> None:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px5_static_shield_analysis.csv")))

    per_point = []
    for r in rows:
        S_h_dyn = float(r["S_h_dynamic"])
        fold_dyn, pct_dyn = fold_and_pct(S_h_dyn)
        entry = {
            "protocol": r["protocol"], "Kmax_MPa_sqrt_m": round(float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, 1),
            "dynamic_rebonding": {"S_h": S_h_dyn, "fold_slowdown": fold_dyn, "pct_rate_reduction": pct_dyn, "wording": wording(fold_dyn, pct_dyn)},
        }
        for control in ("ceiling_static", "orbit_matched_static"):
            S_h_c = float(r[f"{control}_S_h"])
            fold_c, pct_c = fold_and_pct(S_h_c)
            entry[control] = {
                "S_h": S_h_c, "fold_slowdown": fold_c, "pct_rate_reduction": pct_c, "wording": wording(fold_c, pct_c),
                "reproduction_fraction_of_dynamic": float(r[f"{control}_reproduction_fraction"]),
            }
        entry["classification"] = r["classification"]
        per_point.append(entry)

    ceiling_fracs = [e["ceiling_static"]["reproduction_fraction_of_dynamic"] for e in per_point]
    orbit_fracs = [e["orbit_matched_static"]["reproduction_fraction_of_dynamic"] for e in per_point]

    decision = {
        "schema": "v10230_part_x_px5_scientific_decision_v1",
        "status": "PROVISIONAL_PENDING_PX6_PX7",
        "correction_history": [
            {
                "field": "interpretation",
                "reason": (
                    "The original text asserted as fact that 'the developed-regime hazard is dominated by "
                    "whichever portions of the loading cycle see the crack tip fully shielded.' The two-"
                    "control PX5 comparison (ceiling vs cycle-mean-occupancy) is CONSISTENT WITH that "
                    "mechanism but does not uniquely identify or prove it -- it only shows a cycle-averaged "
                    "proxy underestimates the true suppression. Reworded to state the narrower, defensible "
                    "claim."
                ),
                "corrected_in": "part_x_final_decision closure pass",
            },
        ],
        "wording_convention": (
            "fold_slowdown and pct_rate_reduction are DIFFERENT numbers computed from the same S_h; "
            "report both -- NEVER '{fold}x rate reduction'."
        ),
        "per_point": per_point,
        "summary_statistics": {
            "ceiling_reproduction_fraction_min": min(ceiling_fracs), "ceiling_reproduction_fraction_max": max(ceiling_fracs),
            "orbit_matched_reproduction_fraction_min": min(orbit_fracs), "orbit_matched_reproduction_fraction_max": max(orbit_fracs),
        },
        "interpretation": (
            f"At all 10 tested (D2/D5, Kmax) points, the ceiling static control (K_b fixed at the full "
            f"900000 Pa*sqrt(m) cohesive-strength constant from the first accepted event onward) reproduces "
            f"{min(ceiling_fracs)*100:.1f}-{max(ceiling_fracs)*100:.1f}% of the dynamic-rebonding developed-regime "
            f"slowdown -- essentially the ENTIRE effect, including at Kmax=12 MPa*sqrt(m) where the dynamic "
            f"slowdown itself is largest (approximately 15.5-fold). The analytical periodic-orbit-matched "
            f"control (K_b = K_rebond_max * cycle-mean bonded occupancy, computed with zero fitting to any "
            f"physical trajectory) reproduces only {min(orbit_fracs)*100:.1f}-{max(orbit_fracs)*100:.1f}% -- a "
            f"naive duration-weighted-average shielding level substantially UNDERESTIMATES the true "
            f"suppression. This is not evidence that dynamic P/C/B kinetics timing per se is required to "
            f"explain the developed-regime rate (the ceiling control has NO kinetics, no timing, no "
            f"formation/rupture dynamics at all, yet reproduces the effect almost exactly) -- rather, the "
            f"result is CONSISTENT WITH the developed-regime hazard being dominated by whichever portions "
            f"of the loading cycle see the crack tip near-fully shielded, and shows that a simple "
            f"cycle-averaged proxy for that shielding underestimates the true suppression, because the "
            f"hazard integral is NOT linear in the instantaneous K_b. This two-control comparison does NOT "
            f"uniquely identify or prove which specific cycle phase controls the hazard -- near-ceiling "
            f"phase dominance is one interpretation consistent with the data, not the only one the data "
            f"rules out alternatives against."
        ),
        "classification_note": (
            "Classified MIXED_STATIC_SHIELDING_AND_KINETIC_HISTORY at every point under the review's "
            "both-controls-must-agree gate (dominant requires BOTH controls >=80% reproduction) -- not "
            "because dynamic kinetic history is confirmed necessary, but because the review's classification "
            "scheme correctly refuses to credit 'static shielding in general' when one of the two prescribed "
            "static proxies fails this badly. The narrower, better-supported finding is in the interpretation "
            "text above: the CEILING level specifically (not a cycle-averaged level) is what matters."
        ),
        "scope_reminder": [
            "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT", "HAZARD_ONLY_COHESIVE_FEEDBACK",
            "TOPOLOGICAL_HEALING_NOT_MODELED", "PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED",
        ],
    }
    out_path = ARTIFACTS_DIR / "px5_scientific_decision.json"
    out_path.write_text(json.dumps(decision, indent=2, default=str))
    print(f"wrote {out_path}")
    print(decision["interpretation"])


if __name__ == "__main__":
    main()
