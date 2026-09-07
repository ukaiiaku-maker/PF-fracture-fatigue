"""PX4.1 section 7: the scientific PX4 decision. Reduces px4_stage1_rate_
table.csv (itself independently reproducible from developed_event_ledger
via verify_part_x_px4_stage1.py) into exact, unrounded fold-slowdown and
percent-rate-reduction values, and the four review-mandated pairwise
comparisons (D1-vs-D2 R sensitivity, D2-vs-D5 passivation, D2-vs-D3
frequency, D3-vs-D6 persistence).

Wording convention (per review instruction -- a fold-slowdown is NOT the
same number as a percent rate reduction, and conflating them as "Nx rate
reduction" is wrong): "gives approximately a {fold:.1f}-fold slowdown, or
a {pct:.1f}% rate reduction", never "{fold:.1f}x rate reduction".
fold_slowdown = da/dN_zero / da/dN_finite = 10**(-S_h_developed).
pct_rate_reduction = (1 - da/dN_finite/da/dN_zero) * 100 = (1 - 10**S_h_developed) * 100.

All findings here are PROVISIONAL until PX5 (the scoped D2/D5 static-vs-
dynamic shielding follow-up) and PX6/PX7 (synthesis and terminal
verification) complete -- this script records the Stage-1 developed-regime
finding, not a terminal Part X conclusion.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"

KMAX_GRID_MPa = [12.0, 15.0, 18.0, 21.0, 24.3]
PROTOCOLS = ["D1", "D2", "D3", "D5", "D6_conditional_persistent"]
PROTOCOL_MEANING = {
    "D1": "R=-0.95 (near-fully-reversed), f=1000Hz, dynamic rebonding",
    "D2": "R=-0.5, f=1000Hz, dynamic rebonding -- COMPETING_REVERSIBLE",
    "D3": "R=-0.5, f=316.228Hz, dynamic rebonding",
    "D5": "R=-0.5, f=1000Hz, dynamic rebonding -- PASSIVATION_LIMITED chemistry",
    "D6_conditional_persistent": "R=-0.5, f=316.228Hz, dynamic rebonding, conditional persistence branch",
}


def fold_and_pct(S_h: float) -> tuple[float, float]:
    fold = 10.0 ** (-S_h)
    pct = (1.0 - 10.0 ** S_h) * 100.0
    return fold, pct


def wording(fold: float, pct: float) -> str:
    return f"gives approximately a {fold:.1f}-fold slowdown, or a {pct:.1f}% rate reduction"


def main() -> None:
    rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_rate_table.csv")))
    by_seed_protocol_kmax = {
        (r["seed"], r["protocol"], round(float(r["Kmax_Pa_sqrt_m"]) / 1.0e6, 1)): r for r in rows
    }

    per_protocol = {}
    for proto in PROTOCOLS:
        entries = []
        for Kmax in KMAX_GRID_MPa:
            r = by_seed_protocol_kmax.get(("1720", proto, Kmax))
            if r is None:
                continue
            S_h = float(r["S_h_developed"])
            fold, pct = fold_and_pct(S_h)
            entries.append({
                "Kmax_MPa_sqrt_m": Kmax, "S_h_developed_exact": S_h,
                "fold_slowdown_exact": fold, "pct_rate_reduction_exact": pct,
                "wording": wording(fold, pct),
            })
        per_protocol[proto] = {"meaning": PROTOCOL_MEANING[proto], "points": entries}

    def s_h_at(proto: str, Kmax: float) -> float | None:
        r = by_seed_protocol_kmax.get(("1720", proto, Kmax))
        return float(r["S_h_developed"]) if r is not None else None

    def compare(protocol_a: str, protocol_b: str, axis: str) -> dict:
        deltas = []
        for Kmax in KMAX_GRID_MPa:
            sa, sb = s_h_at(protocol_a, Kmax), s_h_at(protocol_b, Kmax)
            if sa is None or sb is None:
                continue
            deltas.append({"Kmax_MPa_sqrt_m": Kmax, "S_h_a": sa, "S_h_b": sb, "delta_S_h_b_minus_a": sb - sa})
        max_abs_delta = max((abs(d["delta_S_h_b_minus_a"]) for d in deltas), default=None)
        return {
            "axis_tested": axis, "protocol_a": protocol_a, "protocol_b": protocol_b,
            "per_Kmax": deltas, "max_abs_delta_S_h": max_abs_delta,
            "practically_indistinguishable": max_abs_delta is not None and max_abs_delta < 0.02,
        }

    comparisons = {
        "D1_vs_D2_R_sensitivity": compare("D1", "D2", "load ratio R (-0.95 vs -0.5) at fixed f=1000Hz"),
        "D2_vs_D5_passivation": compare("D2", "D5", "passivation-limited chemistry_factor vs COMPETING_REVERSIBLE at fixed R=-0.5, f=1000Hz"),
        "D2_vs_D3_frequency": compare("D2", "D3", "frequency (1000Hz vs 316.228Hz) at fixed R=-0.5"),
        "D3_vs_D6_persistence": compare("D3", "D6_conditional_persistent", "conditional persistence branch vs COMPETING_REVERSIBLE at fixed R=-0.5, f=316.228Hz"),
    }

    decision = {
        "schema": "v10230_part_x_px4_scientific_decision_v1",
        "status": "PROVISIONAL_PENDING_PX5_PX6_PX7",
        "correction_history": [
            {
                "field": "interpretation.D3_vs_D6_persistence",
                "reason": (
                    "The original text described the D3-vs-D6 separation as 'an order-of-magnitude "
                    "difference' without qualifying it as a low-K-endpoint-only statement. The exact "
                    "recomputed separation is ~6.18-fold at Kmax=12 (NOT a full decade) and decays to "
                    "~1.02-fold by Kmax=24.3 -- an explicitly K-dependent result, not a single order-of-"
                    "magnitude finding."
                ),
                "old_text": (
                    "an order-of-magnitude difference driven entirely by the persistence assumption, not "
                    "by frequency, since D3 and D6 share the same frequency and R."
                ),
                "corrected_in": "part_x_final_decision closure pass",
            },
        ],
        "wording_convention": (
            "fold_slowdown and pct_rate_reduction are DIFFERENT numbers computed from the same "
            "S_h_developed; report both, e.g. 'D1 Kmax=12 gives approximately a 15.5-fold slowdown, "
            "or a 93.5% rate reduction' -- NEVER '15.5x rate reduction' (that conflates the two)."
        ),
        "per_protocol_seed_1720": per_protocol,
        "pairwise_comparisons": comparisons,
        "interpretation": {
            "D1_vs_D2_R_sensitivity": (
                f"max|delta S_h|={comparisons['D1_vs_D2_R_sensitivity']['max_abs_delta_S_h']:.4f}: "
                "R has a negligible effect on the developed-regime rebonding slowdown across this Kmax "
                "range -- D1 (R=-0.95) and D2 (R=-0.5) are practically indistinguishable."
            ),
            "D2_vs_D5_passivation": (
                f"max|delta S_h|={comparisons['D2_vs_D5_passivation']['max_abs_delta_S_h']:.4f}: "
                "the passivation-limited chemistry factor produces an almost identical developed-regime "
                "slowdown to the COMPETING_REVERSIBLE baseline -- passivation state does not materially "
                "change the STABILIZED rate once the developed window is reached, though it may still "
                "govern the transient approach to that window (not assessed by S_h_developed, which is "
                "computed only over the stable/developed interval)."
            ),
            "D2_vs_D3_frequency": (
                f"max|delta S_h|={comparisons['D2_vs_D3_frequency']['max_abs_delta_S_h']:.4f}: "
                "frequency has a LARGE effect -- at Kmax=12 MPa*sqrt(m), D2 (1000Hz) shows a "
                f"{per_protocol['D2']['points'][0]['fold_slowdown_exact']:.2f}-fold slowdown "
                f"while D3 (316.228Hz) shows only a {per_protocol['D3']['points'][0]['fold_slowdown_exact']:.2f}-fold "
                "slowdown; the effect shrinks with Kmax but the frequency-dependence itself does not "
                "vanish across the grid."
            ),
            "D3_vs_D6_persistence": (
                f"max|delta S_h|={comparisons['D3_vs_D6_persistence']['max_abs_delta_S_h']:.4f}: "
                "at the SAME frequency (316.228Hz), the conditional-persistence branch (D6) shows a "
                f"{per_protocol['D6_conditional_persistent']['points'][0]['fold_slowdown_exact']:.2f}-fold "
                f"slowdown at Kmax=12 versus D3's {per_protocol['D3']['points'][0]['fold_slowdown_exact']:.2f}-fold -- "
                f"a {10.0 ** comparisons['D3_vs_D6_persistence']['max_abs_delta_S_h']:.2f}-fold separation at the "
                "low-K endpoint (NOT a full order of magnitude), decaying strongly as Kmax increases "
                f"(per-Kmax fold separation: {', '.join(f'{d['Kmax_MPa_sqrt_m']:.1f}MPa={10.0 ** abs(d['delta_S_h_b_minus_a']):.2f}x' for d in comparisons['D3_vs_D6_persistence']['per_Kmax'])}). "
                "Persistent bonding materially amplifies the reduced-frequency low-K retardation, with the "
                "separation decaying strongly as Kmax increases -- driven entirely by the persistence "
                "assumption, not by frequency, since D3 and D6 share the same frequency and R. CORRECTED "
                "(see correction_history): an earlier draft of this text described this as 'an "
                "order-of-magnitude difference,' which the exact recomputed separation (6.18-fold at "
                "Kmax=12, decaying to ~1.02-fold by Kmax=24.3) does not support -- a full decade would "
                "require a 10.0-fold separation at the cited point."
            ),
        },
        "second_seed_confirmation": "see px4_stage1_slope_table.csv: all 5 protocol axes classify REBONDING_SEED_ROBUST",
        "scope_reminder": [
            "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT", "HAZARD_ONLY_COHESIVE_FEEDBACK",
            "TOPOLOGICAL_HEALING_NOT_MODELED", "PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED",
        ],
    }
    out_path = ARTIFACTS_DIR / "px4_scientific_decision.json"
    out_path.write_text(json.dumps(decision, indent=2, default=str))
    print(f"wrote {out_path}")
    for name, text in decision["interpretation"].items():
        print(f"\n{name}:\n  {text}")


if __name__ == "__main__":
    main()
