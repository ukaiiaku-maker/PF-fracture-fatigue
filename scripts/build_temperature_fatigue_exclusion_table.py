"""Paper-evidence provenance closure: one row per excluded/censored
temperature-fatigue trajectory (mission Section 6).

IMPORTANT LIMITATION, disclosed rather than worked around: the canonical
temperature-fatigue campaign's raw runs/mechanical_transfer_temperature_
anchors_v1/ output directory (36 rows, generated at commit 4e51077 on
codex/v10.2.30-joint-fracture-fatigue-archetype-atlas) is gitignored by
that branch's own .gitignore and was NOT found on this filesystem in
this session (not present in any currently accessible worktree or
/private/tmp location). Re-deriving the exact per-row (class, T, Kmax)
identity of all 7 numerical nonterminations and all 11 physical censors
would require either locating that raw output elsewhere or re-running
the campaign -- the latter is explicitly prohibited by the mission
except to fill a demonstrated, manuscript-critical gap, which was not
found (see paper_claim_evidence_matrix_v2.csv).

This table therefore reproduces exactly what the campaign's own
committed progress log (CODEX_PROGRESS.md at commit 4e51077, read
directly via `git show`) states in prose, at the granularity that
document supports -- 3 of the 7 numerical exclusions are individually
attributed to the Peak class; the remaining 4 are not individually
named in that document and are recorded as UNRESOLVED_ROW_IDENTITY
rather than guessed.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"

SOURCE_COMMIT = "4e51077bb4d2dbe00143f5a7cb9fb4b6f473d8bb"
SOURCE_BRANCH = "codex/v10.2.30-joint-fracture-fatigue-archetype-atlas"

ROWS = [
    dict(row_group="DBTT @ 1200K (all 3 anchor points: low/middle/high K)", material_class="DBTT",
         temperature_K=1200, disposition="COMPLETE_PHYSICAL_CYCLE_CENSOR",
         manuscript_statement_potentially_affected="DBTT high-temperature suppression / rate shift (Sec 3.2.3, Fig 4)",
         adjacent_points_already_bound_conclusion="Yes -- 300K and 700K DBTT anchors both propagate normally; "
             "the 1200K suppression is a genuine physical censor (survives censor-aware analysis, intrinsic-"
             "opening controlled per the campaign's own analysis), not a numerical failure needing replacement.",
         replacement_required="NO", exact_reason="A physical cycle censor is a valid scientific outcome (the "
             "crack genuinely did not propagate within the cycle budget) -- it must not be reinterpreted as an "
             "unfinished calculation or assigned da/dN=0, and was not."),
    dict(row_group="Peak: 3 rows (temperature/Kmax not individually named in the source progress log)",
         material_class="Peak", temperature_K="not individually resolved (see limitation note)",
         disposition="NUMERICAL_NONTERMINATION_EXCLUDED",
         manuscript_statement_potentially_affected="Whether Peak has a genuine intermediate-temperature fatigue "
             "maximum (distinct from the already-separately-confirmed monotonic K_c(T) peak in Figs 2-4)",
         adjacent_points_already_bound_conclusion="Partially -- a fast 600K middle point IS available, but the "
             "campaign's own analysis explicitly states a genuine intermediate-T maximum is NOT demonstrated "
             "because these 3 rows are excluded and the remaining high-K results are cooperative-renewal-"
             "ceiling-influenced.",
         replacement_required="NO", exact_reason="No current manuscript claim asserts a resolved fatigue-rate "
             "intermediate-temperature Peak maximum (distinct from the monotonic K_c(T) peak, which IS "
             "separately and robustly confirmed via Figs 2-4). The manuscript's own hedged language (Sec 4.4: "
             "'peak-like maxima are less architecture-robust... not... guaranteeing a solver-independent peak') "
             "does not depend on resolving these 3 rows."),
    dict(row_group="Remaining 4 of 7 numerical nonterminations (class/temperature not individually named in "
             "the source progress log)", material_class="unresolved (weakT and/or other classes, per the "
             "campaign's own 'Weak-T high-anchor... incomplete matrix' statement)",
         temperature_K="not individually resolved (see limitation note)",
         disposition="NUMERICAL_NONTERMINATION_EXCLUDED",
         manuscript_statement_potentially_affected="Whether weakT fatigue rate is genuinely temperature-"
             "independent (42.2x rate span across T observed on the available points)",
         adjacent_points_already_bound_conclusion="The campaign's own analysis explicitly states weak "
             "fatigue-temperature dependence is NOT supported by the incomplete matrix -- i.e. this limitation "
             "is already disclosed rather than silently absorbed into a stronger claim.",
         replacement_required="NO", exact_reason="No current manuscript claim asserts weakT fatigue-rate "
             "temperature-independence; the manuscript's Fig 2/3 weakT claims are about the separately-"
             "confirmed monotonic K_c(T) response (RMS deviation 0.23 MPa*sqrt(m) from V1), not the cyclic "
             "da/dN(T) response this campaign's incomplete matrix concerns."),
]

LIMITATION_NOTE = (
    "The exact (class, T, Kmax, exact_numerical_failure_mode, surviving_bound) identity of all 7 numerical "
    "nonterminations and all 11 physical censors could not be individually re-derived this session: the "
    "campaign's raw 36-row output directory (runs/mechanical_transfer_temperature_anchors_v1/) is gitignored "
    "and was not found on this filesystem in any currently accessible location. Re-running the campaign to "
    "regenerate that table was judged NOT justified, because (a) the campaign's own committed progress-log "
    "prose already establishes the two manuscript-relevant conclusions this closure needed to check (Peak "
    "intermediate-T maximum not demonstrated; weakT temperature-independence not supported), (b) no current "
    "manuscript claim was found to depend on the specific identity of any one of the remaining 4 unnamed "
    "excluded rows, and (c) the mission explicitly prohibits rerunning the 36-row matrix as a whole and "
    "authorizes a virgin replacement only when a specific manuscript conclusion changes depending on a named "
    "excluded row -- a condition not met here. This is recorded as an explicit provenance limitation, not "
    "resolved by inference or by re-running physics."
)


def main() -> None:
    with (OUT_DIR / "temperature_fatigue_exclusion_review.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(ROWS)

    (OUT_DIR / "temperature_fatigue_exclusion_review.json").write_text(json.dumps({
        "schema": "v1_temperature_fatigue_exclusion_review",
        "source_branch": SOURCE_BRANCH, "source_commit": SOURCE_COMMIT,
        "total_terminal_rows": 36, "physical_targets": 18, "qualified_physical_cycle_censors": 11,
        "numerical_nonterminations_excluded": 7,
        "row_groups_reviewed": len(ROWS),
        "replacements_authorized": 0,
        "limitation": LIMITATION_NOTE,
        "rows": ROWS,
    }, indent=2, default=str))
    print(f"Wrote temperature_fatigue_exclusion_review.{{csv,json}}: {len(ROWS)} row-groups, 0 replacements authorized")


if __name__ == "__main__":
    main()
