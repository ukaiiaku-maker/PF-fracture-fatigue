"""Paper-evidence completion package (mission Section K): the remaining
cross-repository summary/manifest files, built from the same audit
recorded in paper_claim_evidence_matrix.* and paper_completion_registries
outputs. Run build_paper_claim_evidence_matrix.py and build_paper_
completion_registries.py first.
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"

TEMPERATURE_FATIGUE_SUMMARY = [
    dict(material_class="Ceramic", n_physical_targets_at_class=None, temperatures_tested_K="300, 700-900, 1200",
         status="Propagates at middle/high anchors at ALL three temperatures; middle-anchor rate decreases "
                "3.98e-11 (300K) -> 5.72e-12 m/cycle (1200K); no numerical exclusions; 3 ordinary log-slopes "
                "survive (74.73-78.46), all Ceramic.",
         manuscript_relevant_conclusion="Consistent with a monotonically decreasing, well-resolved fatigue-rate response."),
    dict(material_class="DBTT", n_physical_targets_at_class=None, temperatures_tested_K="300, 700, 1200",
         status="Propagates at 300/700K middle+high anchors; ALL THREE 1200K points are genuine "
                "COMPLETE_PHYSICAL_CYCLE_CENSORS (not numerical exclusions) -- suppression survives censor-"
                "aware analysis and is intrinsic-opening controlled.",
         manuscript_relevant_conclusion="High-temperature suppression is a real physical censor, not imputed as da/dN=0."),
    dict(material_class="Peak", n_physical_targets_at_class=None, temperatures_tested_K="600 (fast middle point); "
         "3 rows numerically excluded",
         status="A genuine intermediate-temperature maximum is NOT demonstrated: 3 Peak rows are numerical "
                "nonterminations, and the remaining high-K results are cooperative-renewal-ceiling influenced.",
         manuscript_relevant_conclusion="Matches the manuscript's own hedged Sec 3.2.1/4.4 language: peak "
                "behavior is architecture-capable, not proven invariant/robust under every representation."),
    dict(material_class="Weak-T", n_physical_targets_at_class=None, temperatures_tested_K="matrix incomplete",
         status="High-anchor rates span a 42.2x factor across temperature; weak fatigue-temperature dependence "
                "is explicitly NOT supported by the incomplete matrix. Developed states show measurable "
                "mobile/retained population, backstress, blunting (r_eff/r0 up to 1.478), and shielding up to "
                "0.280 MPa*sqrt(m).",
         manuscript_relevant_conclusion="No overly strong 'weakly temperature-dependent' fatigue claim should "
                "be made for this class without qualification -- consistent with the manuscript not making one."),
]

REBONDING_SUMMARY = [
    dict(topic="Crack rebonding (Part X)", status="COMPLETE, PUBLISHED, NOT REFERENCED BY THE CURRENT MANUSCRIPT",
         primary_classification="SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE",
         branch="codex/v10.2.30-crack-rebonding-part-x", commit="30db009ff7172728a6bdc885886f21b94cbec225",
         action="None required for this paper. See excluded_future_scope.csv."),
]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    claim_matrix = json.loads((OUT_DIR / "paper_claim_evidence_matrix.json").read_text())["rows"]

    # --- paper_figure_manifest / paper_table_manifest ---
    figure_rows = []
    table_rows = []
    for row in claim_matrix:
        entry = {
            "section": row["section"], "canonical_source": row["existing_figure_table"],
            "status": row["completeness"], "provenance": row["terminal_classification"],
            "repository": row["repository"], "branch": row["branch"], "commit": row["commit"],
            "verification_depth": row["verification_depth"],
        }
        if "Fig." in row["existing_figure_table"] or "figure" in row["existing_figure_table"].lower():
            figure_rows.append(entry)
        else:
            table_rows.append(entry)
    if figure_rows:
        with (OUT_DIR / "paper_figure_manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(figure_rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(figure_rows)
    if table_rows:
        with (OUT_DIR / "paper_table_manifest.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table_rows[0].keys()), lineterminator="\n")
            w.writeheader(); w.writerows(table_rows)

    # --- paper_authoritative_result_registry / attempt / censor-exclusion / cross-fidelity ---
    campaigns = list(csv.DictReader(open(OUT_DIR / "authoritative_campaign_inventory.csv")))
    with (OUT_DIR / "paper_authoritative_result_registry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(campaigns[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(campaigns)

    attempt_rows = [
        dict(campaign="Crack rebonding Part X (+ PX5 transient child)", n_admitted_attempts=128,
             n_interrupted_or_superseded=7, note="126 admitted campaign-wide (per Part X admissibility map) + "
             "2 admitted PX5 wall-budget retry attempts counted separately in the Part X totals; 3 interrupted, "
             "2 superseded, 2 invalidated -- see Part X's own part_x_admissibility_map.csv for the full "
             "authoritative breakdown, not duplicated here."),
        dict(campaign="Canonical temperature-fatigue (joint fracture-fatigue archetype atlas)", n_admitted_attempts=18,
             n_interrupted_or_superseded=18, note="36 total terminal rows = 18 physical targets (admitted "
             "science) + 11 qualified physical cycle censors (admitted as censors, not failures) + 7 numerical "
             "nonterminations (excluded from science, not rerun this session)."),
    ]
    with (OUT_DIR / "paper_simulation_attempt_registry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(attempt_rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(attempt_rows)

    censor_rows = [
        dict(campaign="Canonical temperature-fatigue", row_class="DBTT", condition="1200K, all 3 anchor points",
             disposition="COMPLETE_PHYSICAL_CYCLE_CENSOR",
             note="Genuine physical censor (suppressed but not zero rate); da/dN NOT imputed as zero; survives censor-aware analysis."),
        dict(campaign="Canonical temperature-fatigue", row_class="Peak", condition="3 rows (temperatures not "
             "individually re-extracted this session)", disposition="NUMERICAL_NONTERMINATION_EXCLUDED",
             note="Excluded from science per the campaign's own terminal analysis; not reinterpreted as a censor or as da/dN=0."),
        dict(campaign="Canonical temperature-fatigue", row_class="(unspecified, 4 additional rows)",
             condition="remaining 4 of the 7 total numerical nonterminations",
             disposition="NUMERICAL_NONTERMINATION_EXCLUDED",
             note="Row-level temperature/class breakdown not individually re-extracted this session beyond "
                  "the 3 Peak rows explicitly named in the campaign's own progress log; recorded as a "
                  "verification-depth limitation."),
        dict(campaign="Crack rebonding Part X", row_class="D1 (screen dwell, hold=0.0005s and 0.002s)",
             condition="finite cohesion, PX3 screen", disposition="INVALIDATED_DWELL_DURATION_WEIGHTING_BUG",
             note="A real duration-weighting bug (found and fixed in PX3.5); formally invalidated, never used in any scientific count."),
    ]
    with (OUT_DIR / "paper_censor_and_exclusion_registry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(censor_rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(censor_rows)

    cross_fidelity_rows = [
        dict(comparison="V1 vs PF/sharp-front vs FEM/CZM first-passage K_c(T), 1x rate, 4 classes",
             agreement="Broad ceramic/weakT/DBTT topology preserved across all 3 representations; peak "
                        "amplitude is architecture-sensitive (V1/PF retain a pronounced maximum, FEM/CZM shows "
                        "only a muted shoulder). RMS deviations from V1: ceramic 0.23, weakT 0.23, DBTT 1.33, peak 1.18 MPa*sqrt(m).",
             source="Manuscript Fig. 2 / Sec 3.2.1 (MANUSCRIPT_TEXT_ONLY verification depth)"),
        dict(comparison="Local 1D constitutive-kernel vs PF front-local vs FEM applied-load transfer (matched F0/F1/F2 population)",
             agreement="Matched-1D median absolute relative errors 0.066%/0.066%/0.102%; FEM F2 median remains "
                        "36.9% with no transfer map fitted or nominal-K equivalence assumed.",
             source="Canonical temperature-fatigue campaign progress log, 2026-08-28 entry (EXACT_COMMIT_AND_COUNTS_CONFIRMED)"),
        dict(comparison="Monotonic K_c vs rate-defined fatigue threshold DeltaK_th, pooled across barrier ensemble",
             agreement="Pooled log-space Pearson r; context-specific values 0.958-0.999 across 1360 matched observations.",
             source="Manuscript Fig. 6B / Sec 3.4 (MANUSCRIPT_TEXT_ONLY verification depth)"),
    ]
    with (OUT_DIR / "paper_cross_fidelity_comparison.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cross_fidelity_rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(cross_fidelity_rows)

    with (OUT_DIR / "paper_temperature_fracture_fatigue_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(TEMPERATURE_FATIGUE_SUMMARY[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(TEMPERATURE_FATIGUE_SUMMARY)

    with (OUT_DIR / "paper_rebonding_summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(REBONDING_SUMMARY[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(REBONDING_SUMMARY)

    # --- terminal decision ---
    decision = {
        "schema": "v1_paper_simulation_completion_decision",
        "status": "COMPLETE_WITH_DISCLOSED_VERIFICATION_DEPTH_LIMITATIONS",
        "primary_finding": "The common-hazard fracture/fatigue manuscript is a mature draft that already "
        "reports complete, specific, quantitative results for every technical claim examined in this closure "
        "(four-class K_c(T) comparison, R-curve, 4-decade rate sweep, fatigue crack-growth atlas, cross-"
        "phenomenon correlations). No manuscript claim or figure was found to require a new physical "
        "simulation. Crack rebonding (Part X, independently complete) is not referenced by the current draft.",
        "physical_simulations_launched_this_session": 0,
        "campaigns_reused_not_duplicated": [
            "Four-class 2D PF/CZM (peak/DBTT/weakT/ceramic) -- sobol parameter IDs byte-matched",
            "Canonical temperature-fatigue (36-row, 18/11/7) -- exact commit and counts confirmed",
            "Crack rebonding Part X -- exact commit confirmed, already published this session",
        ],
        "new_work_this_session": [
            "Pushed crack rebonding Part X to origin and opened a draft PR",
            "PX5 transient-regime analysis (resolves Part X's one NOT_ARCHIVED limitation)",
            "This paper-evidence completion package",
        ],
        "genuine_gaps_found": [
            "Optional SI sensitivity/identifiability analysis -- explicitly framed as optional in discussion_outline.docx, not built",
        ],
        "verification_depth_limitations": (
            "Five named campaigns (two-scale virtual C(T), A_NATIVE+PT panel, R-ratio/deltaK, analytical "
            "overlay, physical-slope-transfer/inverse-design) were confirmed to exist as real branches with "
            "plausible terminal commits but were not individually re-verified at the test-suite or figure-"
            "number level this session. None were pushed to origin pending that verification. The specific "
            "commit(s) that generated the manuscript's Figs. 2-5 were not individually pinpointed beyond "
            "confirming the parameter registry and campaign family."
        ),
        "not_claimed": [
            "closure-corrected DeltaK_eff", "resolved opposing-face contact", "topological crack healing",
            "calibrated physical chemistry", "production-line merge readiness",
        ],
    }
    (OUT_DIR / "paper_simulation_completion_decision.json").write_text(json.dumps(decision, indent=2, default=str))

    md = ["# Paper Simulation Completion Decision\n",
          f"**Status:** `{decision['status']}`\n",
          f"## Primary finding\n\n{decision['primary_finding']}\n",
          "## Campaigns reused (not duplicated)\n"] + [f"- {c}" for c in decision["campaigns_reused_not_duplicated"]] + [
          "\n## New work this session\n"] + [f"- {c}" for c in decision["new_work_this_session"]] + [
          "\n## Genuine gaps found\n"] + [f"- {c}" for c in decision["genuine_gaps_found"]] + [
          f"\n## Verification depth limitations\n\n{decision['verification_depth_limitations']}\n",
          "\n## Not claimed\n"] + [f"- {c}" for c in decision["not_claimed"]]
    (OUT_DIR / "paper_simulation_completion_decision.md").write_text("\n".join(md) + "\n")

    handoff = """# Paper Simulation Handoff

This branch (`codex/v10.2.30-paper-simulation-completion`) is an evidence
and analysis coordinator for the common-hazard fracture/fatigue
manuscript at `/Volumes/Data/working-papers/fracture_and_fatigue/`
(current draft: `Fatigue_and_fracture_revised_FEM_PF_Rcurve.docx`).

## What this branch contains

All files under `artifacts/paper_simulation_completion/`:
- `paper_claim_evidence_matrix.{csv,json}` -- every manuscript figure/
  section mapped to its authoritative campaign and a completeness class.
- `authoritative_campaign_inventory.csv` -- every named campaign's branch,
  commit, and verification depth.
- `unfinished_physical_simulation_registry.csv` -- explicitly empty of
  launched work; every conditional simulation program (four-class PF/CZM,
  temperature-fatigue replacements, rate panel, rebonding) was found
  NOT to be required, with the reasoning recorded per item.
- `analysis_only_gap_registry.csv` -- the PX5 transient-regime analysis
  (complete) and the optional SI sensitivity analysis (not built,
  explicitly optional).
- `excluded_future_scope.csv` -- scope boundaries that remain out of
  scope for this paper.
- `push_publication_record.json` -- exactly what was pushed where.
- `paper_completion_contract.json` -- the scoping rules this closure followed.
- `paper_simulation_completion_decision.{json,md}` -- the terminal decision.
- `paper_authoritative_result_registry.csv`, `paper_simulation_attempt_
  registry.csv`, `paper_censor_and_exclusion_registry.csv`, `paper_cross_
  fidelity_comparison.csv`, `paper_temperature_fracture_fatigue_summary.csv`,
  `paper_rebonding_summary.csv`, `paper_figure_manifest.csv`,
  `paper_table_manifest.csv` -- supporting cross-repository summaries.

## What was NOT done, and why

No new physical simulation was launched. The manuscript already reports
complete, specific, quantitative results for every technical claim
examined. Five of the seven named campaigns were confirmed to exist as
real branches with plausible terminal commits but were not individually
re-verified (test suite, strict verifier, exact figure numbers) this
session, and were therefore not pushed to `origin` -- pushing them would
require completing that verification first (see `push_publication_
record.json`'s `reason_other_campaigns_not_pushed` field).

## Key correction to remember

Crack rebonding (Part X) is NOT referenced anywhere in the current
manuscript draft. It remains an independently complete, published result
(`codex/v10.2.30-crack-rebonding-part-x`) that this paper does not need.
"""
    (REPO_ROOT / "artifacts" / "paper_simulation_completion" / "PAPER_SIMULATION_HANDOFF.md").write_text(handoff)

    # --- file hashes (must run LAST, after every other file in this dir exists) ---
    hashes = {}
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file() and path.name != "file_hashes.json":
            hashes[str(path.relative_to(OUT_DIR))] = _sha256(path)
    (OUT_DIR / "file_hashes.json").write_text(json.dumps({"schema": "v1_file_hashes", "n_files": len(hashes), "files": hashes}, indent=2, sort_keys=True))

    print(f"Wrote the full paper-evidence completion package to {OUT_DIR}")
    print(f"file_hashes.json: {len(hashes)} files hashed")


if __name__ == "__main__":
    main()
