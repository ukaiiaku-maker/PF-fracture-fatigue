"""PX6: scientific synthesis of the whole signed-K crack-rebonding
compression-conditioned Part X campaign (PX0-PX5). Reads ONLY tracked
artifacts/crack_rebonding_part_x_v1/* JSON/CSV -- never a gitignored
runs/ directory -- and assembles the consolidated findings + open
questions PX7's terminal verifier and decision will certify against.

This is a REDUCTION of already-committed, already-independently-verified
decision artifacts (px4_scientific_decision.json, px5_scientific_
decision.json, px4_stage1_slope_table.csv, px4_stage1_verification.json,
px5_verification.json) -- it recomputes nothing, fits nothing, and
introduces no new physical claim beyond what those artifacts already
established.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS_DIR = REPO_ROOT / "artifacts" / "crack_rebonding_part_x_v1"


def main() -> None:
    px4_decision = json.loads((ARTIFACTS_DIR / "px4_scientific_decision.json").read_text())
    px5_decision = json.loads((ARTIFACTS_DIR / "px5_scientific_decision.json").read_text())
    px4_verification = json.loads((ARTIFACTS_DIR / "px4_stage1_verification.json").read_text())
    px5_verification = json.loads((ARTIFACTS_DIR / "px5_verification.json").read_text())
    slope_rows = list(csv.DictReader(open(ARTIFACTS_DIR / "px4_stage1_slope_table.csv")))

    seed_robustness = {r["base_protocol"]: r["classification"] for r in slope_rows}
    all_seed_robust = all(v == "REBONDING_SEED_ROBUST" for v in seed_robustness.values())

    synthesis = {
        "schema": "v10230_part_x_px6_synthesis_v1",
        "status": "PROVISIONAL_PENDING_PX7_TERMINAL_VERIFICATION",
        "campaign_scope": (
            "Signed-K crack-rebonding compression-conditioned physical campaign (Part X, PX0-PX5): "
            "PX0-PX1 established the exact phase-resolved P/C/B kinetics and one-cycle monodromy primitives; "
            "PX2 characterized 4 candidate kinetic regimes (SAT_EXISTING, COMPETING_REVERSIBLE, "
            "COMPETING_PERSISTENT, PASSIVATION_LIMITED) analytically with zero fitting to any physical "
            "trajectory; PX3 ran a 28-job physical screen across R/frequency/dwell/cohesive-strength panels; "
            "PX3.5/PX3.6 corrected a real duration-weighting bug (the '6.26x dwell acceleration' anomaly), "
            "audited dwell-causality, and selected the final 5 developed protocols (D1/D2/D3/D5/D6); PX4/"
            "PX4.1 ran the full developed campaign (80 trajectories: 5 protocols x 5 Kmax x seed=1720, plus "
            "seed=1001723 second-seed confirmation at 3 Kmax across all 5 protocols) under an exact "
            "first-passage 1e12-cycle horizon; PX5 (scoped to D2/D5 only) attributed the dynamic-rebonding "
            "developed-regime slowdown against two prescribed static-shield controls."
        ),
        "px4_findings": {
            "seed_robustness_by_protocol_axis": seed_robustness,
            "all_5_axes_seed_robust": all_seed_robust,
            "R_sensitivity_D1_vs_D2": px4_decision["interpretation"]["D1_vs_D2_R_sensitivity"],
            "passivation_D2_vs_D5": px4_decision["interpretation"]["D2_vs_D5_passivation"],
            "frequency_D2_vs_D3": px4_decision["interpretation"]["D2_vs_D3_frequency"],
            "persistence_D3_vs_D6": px4_decision["interpretation"]["D3_vs_D6_persistence"],
            "representative_wording": px4_decision["per_protocol_seed_1720"]["D1"]["points"][0]["wording"],
            "verification_classification": px4_verification["classification"],
        },
        "px5_findings": {
            "scope": "D2 (COMPETING_REVERSIBLE) and D5 (PASSIVATION_LIMITED) only, per explicit review scope restriction",
            "interpretation": px5_decision["interpretation"],
            "classification_note": px5_decision["classification_note"],
            "reproduction_fraction_summary": px5_decision["summary_statistics"],
            "verification_classification": px5_verification["classification"],
        },
        "synthesized_narrative": (
            "Across all 5 developed protocols and both seeds, cohesive rebonding produces a real, "
            "seed-robust, Kmax-decaying fatigue-crack-growth slowdown (from ~15.5-fold at Kmax=12 MPa*sqrt(m) "
            "to ~1.04-fold at Kmax=24.3 MPa*sqrt(m)) that is materially insensitive to load ratio R and "
            "passivation-limited chemistry, but strongly sensitive to loading frequency and to whether the "
            "reduced-frequency condition's rebonded state is allowed to persist across cycles (D3 vs D6 "
            "differ by an order of magnitude at identical R/frequency). For the two scoped protocols, PX5 "
            "shows this developed-regime effect is almost entirely explained by the crack tip's ABSOLUTE "
            "shielding ceiling (a static K_b fixed at full cohesive strength after the first event reproduces "
            "100.0-101.9% of the dynamic result at every tested point) rather than by a cycle-averaged "
            "shielding level (which reproduces only 22-40%) -- i.e. the hazard is dominated by whichever "
            "cycle phase sees full shielding, not by the mean occupancy over the cycle. This is evidence "
            "AGAINST needing the full P/C/B kinetics timing to explain the STEADY-STATE developed rate for "
            "D2/D5 specifically, though it says nothing about the TRANSIENT approach to that steady state, "
            "and has not been tested for D1/D3/D6."
        ),
        "scope_reminder": [
            "SURROGATE_SIGNED_K_CONTACT_NOT_RESOLVED_FACE_CONTACT", "HAZARD_ONLY_COHESIVE_FEEDBACK",
            "TOPOLOGICAL_HEALING_NOT_MODELED", "PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED",
        ],
        "not_claimed": [
            "closure-corrected DeltaK_eff", "production-line merge readiness",
            "PX5 findings for D1/D3/D6 (out of scope, not tested)",
            "transient (pre-developed-window) explanatory power of static shielding",
        ],
    }
    out_path = ARTIFACTS_DIR / "px6_synthesis.json"
    out_path.write_text(json.dumps(synthesis, indent=2, default=str))
    print(f"wrote {out_path}")
    print(synthesis["synthesized_narrative"])


if __name__ == "__main__":
    main()
