"""Paper-evidence provenance closure: lightweight verification registry
for the 6 named campaign branches not independently deep-verified in the
earlier paper-simulation-completion pass (mission Section 5).

For each branch, this session:
  - created a disposable, detached worktree at the branch tip;
  - ran `git diff --check` (clean on all 6);
  - ran `py_compile` over every tracked .py file (clean on all 6, 478-512
    files each);
  - attempted to run the campaign's own named strict verifier script.

The verifiers could NOT complete: each requires a `--root` (or
equivalent) pointing at that campaign's raw physical result directory
(e.g. `A_native_plus_8PT_registry.csv`), and those directories are
gitignored `runs/`-style outputs that were not found anywhere on this
filesystem in this session -- the same portability limitation already
disclosed for the canonical temperature-fatigue campaign. Full data-
level re-verification (figure numbers, censor counts, exact artifact
content) was therefore NOT achieved for these 6 campaigns this session.

None of these 6 campaigns were found to be directly cited by name or by
traceable figure/table provenance in the current manuscript draft during
this session's reading and grep search (unlike the four-class FEM/CZM
campaign and the canonical temperature-fatigue campaign, both of which
WERE traced). They are therefore recorded as NOT_CONCLUSIVELY_LINKED_TO_
CURRENT_MANUSCRIPT rather than pushed to origin -- pushing a branch
"because it might be needed" is not the same as confirming it is needed,
and the mission explicitly permits marking an unused branch
NOT_REQUIRED_FOR_CURRENT_PAPER without pushing it.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"

ROWS = [
    dict(branch="codex/v10.2.30-two-scale-virtual-CT", commit="d727cbde36f240214086ac2134985bcd023742fc",
         py_compile="PASS (493 files)", git_diff_check="PASS", strict_verifier="verify_v10_2_30_two_scale_virtual_CT.py "
         "-- requires --root pointing at a raw result directory not present on this filesystem; not completed",
         manuscript_link_found="NO", disposition="NOT_CONCLUSIVELY_LINKED_TO_CURRENT_MANUSCRIPT", pushed="NO"),
    dict(branch="codex/v10.2.30-A-native-TP-panel", commit="f99693632376a5e0308e8e112e9cd1c36a0692f6",
         py_compile="PASS (478 files)", git_diff_check="PASS", strict_verifier="verify_v10_2_30_A_native_plus_8PT_study.py "
         "-- fails on missing A_native_plus_8PT_registry.csv (gitignored raw output, not present); not completed",
         manuscript_link_found="INDIRECT -- referenced BY NAME in the canonical temperature-fatigue campaign's own "
         "progress log as an 'exact-row A_NATIVE/PT03/PT08 archival control, reused without resume or retuning' "
         "-- i.e. treated as an already-qualified upstream input elsewhere in this codebase, not as unfinished work",
         disposition="NOT_CONCLUSIVELY_LINKED_TO_CURRENT_MANUSCRIPT_DIRECTLY_BUT_REUSED_UPSTREAM", pushed="NO"),
    dict(branch="codex/v10.2.30-R-ratio-nominal-deltaK", commit="4b554006d60539d7dec659f4bfe4bf298728780a",
         py_compile="PASS (485 files)", git_diff_check="PASS", strict_verifier="verify_v10_2_30_R_nominal_deltaK_study.py "
         "-- requires --root; not completed",
         manuscript_link_found="NO", disposition="NOT_CONCLUSIVELY_LINKED_TO_CURRENT_MANUSCRIPT", pushed="NO"),
    dict(branch="codex/v10.2.30-analytical-overlay-all-1d", commit="985ea82992bc22dffcfc365ea68aa9bba4022296",
         py_compile="PASS (498 files)", git_diff_check="PASS", strict_verifier="verify_v10_2_30_analytical_overlay_all_1d.py "
         "-- requires --root; not completed",
         manuscript_link_found="NO", disposition="NOT_CONCLUSIVELY_LINKED_TO_CURRENT_MANUSCRIPT", pushed="NO"),
    dict(branch="codex/v10.2.30-physical-slope-transfer", commit="43b5ec9a259b636c5a950eec1f6fffbf3ef4b65c",
         py_compile="PASS (512 files)", git_diff_check="PASS", strict_verifier="verify_v10_2_30_physical_slope_transfer.py "
         "-- requires --root; not completed",
         manuscript_link_found="INDIRECT -- CONFIRMED to be the exact starting HEAD the canonical temperature-fatigue "
         "campaign branch forked from (43b5ec9a matches that campaign's own recorded 'starting HEAD' verbatim)",
         disposition="NOT_CONCLUSIVELY_LINKED_TO_CURRENT_MANUSCRIPT_DIRECTLY_BUT_CONFIRMED_UPSTREAM_LINEAGE", pushed="NO"),
    dict(branch="codex/v10.2.30-inverse-fatigue-barrier-design", commit="5034046963418254f39fc0116c30563bd17c1635",
         py_compile="PASS (507 files)", git_diff_check="PASS", strict_verifier="verify_v10_2_30_inverse_fatigue_barrier_design.py "
         "-- requires --root; not completed",
         manuscript_link_found="POSSIBLE -- the manuscript's Sec 4.5 synthetic-identifiability/inverse-recovery "
         "study (20-55%/1-10% recovery errors) is thematically consistent with 'inverse fatigue barrier design', "
         "but this was NOT confirmed by locating the exact dataset (see paper_claim_evidence_matrix_v2.csv "
         "SI-identifiability-recovery-errors row)",
         disposition="NOT_CONCLUSIVELY_LINKED_TO_CURRENT_MANUSCRIPT_PLAUSIBLE_BUT_UNCONFIRMED", pushed="NO"),
]


def main() -> None:
    with (OUT_DIR / "campaign_branch_verification_registry.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ROWS[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(ROWS)
    (OUT_DIR / "campaign_branch_verification_registry.json").write_text(json.dumps({
        "schema": "v1_campaign_branch_verification_registry",
        "n_branches": len(ROWS),
        "all_py_compile_pass": True, "all_git_diff_check_pass": True,
        "all_strict_verifiers_blocked_by_missing_raw_results": True,
        "n_pushed_this_session": 0,
        "rows": ROWS,
    }, indent=2, default=str))
    print(f"Wrote campaign_branch_verification_registry.{{csv,json}}: {len(ROWS)} branches")


if __name__ == "__main__":
    main()
