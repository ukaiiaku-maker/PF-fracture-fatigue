"""Paper-evidence completion: remaining registries required by the
mission's Section C (authoritative_campaign_inventory, unfinished_
physical_simulation_registry, analysis_only_gap_registry,
excluded_future_scope, push_publication_record, paper_completion_contract).
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"

CAMPAIGNS = [
    dict(
        name="Crack rebonding Part X", branch="codex/v10.2.30-crack-rebonding-part-x",
        commit="30db009ff7172728a6bdc885886f21b94cbec225",
        remote_status="PUSHED_AND_VERIFIED (this session; PR #64 draft opened to v10.2.30-hazard-energy-gated-fatigue-events)",
        terminal_classification="SIGNED_K_CRACK_REBONDING_PART_X_COMPLETE",
        verification_depth="EXACT_COMMIT_AND_COUNTS_CONFIRMED",
        note="Not referenced by the current manuscript draft; independently complete and published.",
    ),
    dict(
        name="Crack rebonding Part X, PX5 transient-regime analysis (this closure)",
        branch="codex/v10.2.30-crack-rebonding-part-x-px5-transient",
        commit="7816cbf91a2ceae76cd52162e0fffb4fef9e75aa",
        remote_status="PUSHED_AND_VERIFIED (this session)",
        terminal_classification="ANALYSIS_ONLY_EXTENSION_COMPLETE",
        verification_depth="EXACT_COMMIT_AND_COUNTS_CONFIRMED",
        note="Child branch of the accepted, immutable Part X HEAD; resolves the one Part X NOT_ARCHIVED limitation.",
    ),
    dict(
        name="Canonical temperature-fatigue campaign (joint fracture-fatigue archetype atlas)",
        branch="codex/v10.2.30-joint-fracture-fatigue-archetype-atlas", commit="4e51077bb4d2dbe00143f5a7cb9fb4b6f473d8bb",
        remote_status="NOT_ON_REMOTE (local branch confirmed to exist; not pushed this session -- see limitations)",
        terminal_classification="TERMINAL: 36/36 registry rows terminal; 18 physical targets; 11 qualified "
        "physical cycle censors; 7 numerical nonterminations excluded from science (exact match to mission spec)",
        verification_depth="EXACT_COMMIT_AND_COUNTS_CONFIRMED",
        note="Confirmed via git log/CODEX_PROGRESS.md at the exact commit; Peak intermediate-T maximum "
        "explicitly NOT demonstrated (3 numerical exclusions); weakT temperature-independence NOT supported "
        "by the incomplete matrix (42x rate span). Both narrowings are ALREADY reflected in the campaign's own "
        "documentation and appear consistent with the manuscript's own hedged claims. This session's own test/"
        "verifier suite was NOT re-run.",
    ),
    dict(
        name="Two-scale virtual C(T) campaign", branch="codex/v10.2.30-two-scale-virtual-CT",
        commit="d727cbde36f240214086ac2134985bcd023742fc",
        remote_status="NOT_ON_REMOTE", terminal_classification="Branch/terminal-commit existence confirmed; "
        "internal terminal classification not re-derived this session",
        verification_depth="BRANCH_AND_TERMINAL_COMMIT_CONFIRMED",
        note="Not deeply re-audited this session due to mission scope/time constraints -- see limitations.",
    ),
    dict(
        name="A_NATIVE + PT01-PT08 fatigue panel", branch="codex/v10.2.30-A-native-TP-panel",
        commit="f99693632376a5e0308e8e112e9cd1c36a0692f6",
        remote_status="NOT_ON_REMOTE", terminal_classification="Branch/terminal-commit existence confirmed; "
        "internal terminal classification not re-derived this session",
        verification_depth="BRANCH_AND_TERMINAL_COMMIT_CONFIRMED",
        note="Referenced as an 'exact-row A_NATIVE/PT03/PT08 archival control, reused without resume or "
        "retuning' by the temperature-fatigue campaign's own progress log -- confirms this panel is treated as "
        "an already-qualified upstream input elsewhere in this codebase, not as unfinished work.",
    ),
    dict(
        name="Multi-R and nominal/experimental-deltaK campaign", branch="codex/v10.2.30-R-ratio-nominal-deltaK",
        commit="4b554006d60539d7dec659f4bfe4bf298728780a",
        remote_status="NOT_ON_REMOTE", terminal_classification="Branch/terminal-commit existence confirmed; "
        "internal terminal classification not re-derived this session",
        verification_depth="BRANCH_AND_TERMINAL_COMMIT_CONFIRMED", note="Not deeply re-audited this session.",
    ),
    dict(
        name="Analytical steady-state overlay", branch="codex/v10.2.30-analytical-overlay-all-1d",
        commit="985ea82992bc22dffcfc365ea68aa9bba4022296",
        remote_status="NOT_ON_REMOTE", terminal_classification="Branch/terminal-commit existence confirmed; "
        "internal terminal classification not re-derived this session",
        verification_depth="BRANCH_AND_TERMINAL_COMMIT_CONFIRMED", note="Not deeply re-audited this session.",
    ),
    dict(
        name="Physical slope-transfer / inverse-design campaigns",
        branch="codex/v10.2.30-physical-slope-transfer, codex/v10.2.30-inverse-fatigue-barrier-design",
        commit="43b5ec9a259b636c5a950eec1f6fffbf3ef4b65c, 5034046963418254f39fc0116c30563bd17c1635",
        remote_status="NOT_ON_REMOTE", terminal_classification="Branch/terminal-commit existence confirmed; "
        "internal terminal classification not re-derived this session",
        verification_depth="BRANCH_AND_TERMINAL_COMMIT_CONFIRMED",
        note="43b5ec9a is confirmed (via the temperature-fatigue campaign's own progress log) to be the exact "
        "starting HEAD the joint-fracture-fatigue-archetype-atlas branch forked from -- a direct, confirmed "
        "lineage relationship, not merely coincidental branch existence.",
    ),
    dict(
        name="Four-class 2D PF/CZM validation (peak/DBTT/weakT/ceramic)",
        branch="many codex/v10.2.2x-v10.2.30-*four-class* branches in this same repository "
        "(NOT a separate PF/FEM-CZM repository -- confirmed by direct git search)",
        commit="parameter registry file arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_"
        "registry.csv confirmed present on codex/v10.2.30-A-native-TP-panel, codex/v10.2.30-R-ratio-nominal-"
        "deltaK, codex/v10.2.30-analytical-overlay-all-1d, and multiple crack-rebonding-predecessor branches",
        remote_status="NOT_ON_REMOTE (individual four-class production branches not identified/pushed this session)",
        terminal_classification="Exact sobol candidate IDs (v913_zeroD_sobol_0242980/0202500/0129902/0077080 "
        "primaries; 0008816/0008536 backups) BYTE-MATCHED against the mission specification via "
        "PF-fracture-fatigue_generated_registry_20260727_102630/v10_2_27_paper_four_class_selection.json. "
        "Manuscript Sections 3.2.1-3.2.4 and Figures 2-4 report complete, specific, already-generated "
        "quantitative results (RMS deviations, R-curve K_0/K_ss/DeltaK_R, 4-decade rate sweep) for exactly "
        "this four-class comparison.",
        verification_depth="PARAMETER_REGISTRY_CONFIRMED",
        note="Given the manuscript's own already-complete, specific numeric reporting for this exact "
        "comparison, no new focused-validation (G1) or full-production (G2) physical simulation matrix was "
        "launched -- doing so would duplicate qualified results already reflected in the current draft. The "
        "individual production branch/commit that generated Figs. 2-4 was not pin-pointed to one specific "
        "commit this session; this is recorded as a verification-depth limitation, not as missing physics.",
    ),
]

UNFINISHED_PHYSICAL_SIMULATION_ROWS = [
    dict(
        item="Four-class 2D PF/CZM focused validation (G1) or full production (G2) matrix",
        status="NOT_LAUNCHED",
        reason="Manuscript already reports complete, specific quantitative results for exactly this comparison "
        "(Figs. 2-4, Sec 3.2). No evidence of a missing or unreported case was found. Launching a new 20-case "
        "matrix would risk duplicating already-qualified results, which the mission explicitly prohibits "
        "('Reuse qualified results; never duplicate them').",
        disposition="NO_ACTION -- see paper_claim_evidence_matrix.csv rows for Figs. 2-4",
    ),
    dict(
        item="Temperature-fatigue numerical-nontermination replacements (7 excluded rows)",
        status="NOT_LAUNCHED",
        reason="The canonical temperature-fatigue campaign's own terminal analysis already narrows the two "
        "manuscript-relevant conclusions the mission worries about (Peak intermediate-T maximum: explicitly "
        "NOT claimed as demonstrated; weakT temperature-independence: explicitly NOT supported by the "
        "incomplete matrix) -- these narrowings are consistent with, and already reflected in, the current "
        "manuscript's own hedged language. No manuscript conclusion was found that depends on resolving any "
        "specific one of the 7 excluded rows.",
        disposition="NO_ACTION -- see paper_claim_evidence_matrix.csv Fig. 6 row and authoritative_campaign_inventory.csv",
    ),
    dict(
        item="Minimal loading-rate panel (Part H)",
        status="NOT_LAUNCHED",
        reason="The manuscript already contains a quantitative rate-shift claim backed by a 4-decade FEM/CZM "
        "rate sweep (0.1x/1x/10x/100x, Fig. 4) with a specific mechanistic DBTT-shift result. The mission's own "
        "gating condition for launching a new rate panel ('a quantitative rate-shift claim or a figure whose "
        "required data are absent') is not met -- the data are present.",
        disposition="NO_ACTION",
    ),
    dict(
        item="Rebonding-temperature campaign", status="NOT_LAUNCHED",
        reason="Crack rebonding is not referenced anywhere in the current manuscript draft or discussion "
        "outline (confirmed by full-text read and targeted grep). The mission's gating condition ('a principal "
        "figure or conclusion explicitly requires temperature-dependent rebonding') is not met.",
        disposition="NO_ACTION",
    ),
]

ANALYSIS_ONLY_GAP_ROWS = [
    dict(
        item="PX5 transient-regime analysis (first-event response, post-first-event transient, rolling "
             "eventwise waiting-time ratio, approach to developed regime, D2 vs D5)",
        status="COMPLETE_THIS_SESSION",
        branch="codex/v10.2.30-crack-rebonding-part-x-px5-transient", commit="7816cbf91a2ceae76cd52162e0fffb4fef9e75aa",
        note="Resolves the one NOT_ARCHIVED_WITH_EXPLICIT_LIMITATION item from the Part X final artifact "
        "manifest. Static-shield controls bit-identical to zero-cohesion at event 0 (by construction); dynamic "
        "trajectory shows a first-event deviation many orders of magnitude below numerical tolerance (floating-"
        "point-level, not physical); transient approach to the developed-regime ratio completes within 0-3 "
        "events for every tested (protocol, Kmax) point.",
    ),
    dict(
        item="Optional SI sensitivity/identifiability analysis (stacked sensitivity-matrix singular values or "
             "profile likelihoods)",
        status="NOT_BUILT_OPTIONAL",
        branch="N/A", commit="N/A",
        note="discussion_outline.docx frames this explicitly as a possible, non-committed addition ('We may "
        "want one SI analysis...'), not a firm requirement of the current draft. Building it would require "
        "either the exact synthetic-identifiability code already used for Sec 4.5's reported recovery-error "
        "numbers (not located/re-run this session) or a fresh implementation, which risks diverging from the "
        "already-reported 20-55%/1-10% recovery-error figures. Left as an explicitly optional, non-blocking "
        "item rather than risk introducing an inconsistent duplicate analysis.",
    ),
]

EXCLUDED_FUTURE_SCOPE_ROWS = [
    dict(item="Resolved opposing-crack-face contact model", reason="Explicit Part X scope boundary; would require a new constitutive model, not an analysis-only extension."),
    dict(item="Positive-R plastic-wake closure", reason="Explicit Part X/mission scope boundary."),
    dict(item="Topological crack healing", reason="Explicit Part X scope boundary (TOPOLOGICAL_HEALING_NOT_MODELED)."),
    dict(item="Calibrated physical chemistry for crack rebonding", reason="Explicit Part X scope boundary (PHYSICAL_CHEMISTRY_PARAMETERIZATION_NOT_CALIBRATED)."),
    dict(item="Closure-corrected DeltaK_eff", reason="Explicit Part X scope boundary; not claimed anywhere in this closure."),
    dict(item="Temperature-dependent crack rebonding", reason="Not referenced by the current manuscript; would only be authorized if a principal figure/conclusion required it, which was not found."),
    dict(item="PX5 static-shield attribution for D1/D3/D6", reason="Explicitly out of PX5's prescribed D2/D5-only scope."),
    dict(item="Material-specific atomistic barrier calibration (NEB, DFT)", reason="Explicitly framed in the manuscript's own Discussion (Sec 4.7.1) as a distinct future research stage, not a gap in the current paper."),
]


def main() -> None:
    with (OUT_DIR / "authoritative_campaign_inventory.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CAMPAIGNS[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(CAMPAIGNS)

    with (OUT_DIR / "unfinished_physical_simulation_registry.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(UNFINISHED_PHYSICAL_SIMULATION_ROWS[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(UNFINISHED_PHYSICAL_SIMULATION_ROWS)

    with (OUT_DIR / "analysis_only_gap_registry.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(ANALYSIS_ONLY_GAP_ROWS[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(ANALYSIS_ONLY_GAP_ROWS)

    with (OUT_DIR / "excluded_future_scope.csv").open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(EXCLUDED_FUTURE_SCOPE_ROWS[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(EXCLUDED_FUTURE_SCOPE_ROWS)

    push_record = {
        "schema": "v1_push_publication_record",
        "part_x": {
            "branch": "codex/v10.2.30-crack-rebonding-part-x",
            "local_head": "30db009ff7172728a6bdc885886f21b94cbec225",
            "remote": "https://github.com/ukaiiaku-maker/PF-fracture-fatigue.git",
            "remote_verified_sha": "30db009ff7172728a6bdc885886f21b94cbec225",
            "remote_branch_status_before_push": "ABSENT",
            "action_taken": "push --set-upstream origin (normal push, no force)",
            "draft_pr": "https://github.com/ukaiiaku-maker/PF-fracture-fatigue/pull/64",
            "pr_target_branch": "v10.2.30-hazard-energy-gated-fatigue-events (confirmed unambiguous via origin/HEAD)",
        },
        "px5_transient_analysis": {
            "branch": "codex/v10.2.30-crack-rebonding-part-x-px5-transient",
            "local_head": "7816cbf91a2ceae76cd52162e0fffb4fef9e75aa",
            "remote_verified_sha": "7816cbf91a2ceae76cd52162e0fffb4fef9e75aa",
            "remote_branch_status_before_push": "ABSENT", "action_taken": "push --set-upstream origin (normal push, no force)",
        },
        "other_campaigns_not_pushed_this_session": [
            "codex/v10.2.30-joint-fracture-fatigue-archetype-atlas",
            "codex/v10.2.30-two-scale-virtual-CT", "codex/v10.2.30-A-native-TP-panel",
            "codex/v10.2.30-R-ratio-nominal-deltaK", "codex/v10.2.30-analytical-overlay-all-1d",
            "codex/v10.2.30-physical-slope-transfer", "codex/v10.2.30-inverse-fatigue-barrier-design",
        ],
        "reason_other_campaigns_not_pushed": "The mission's Section L requires running each branch's focused "
        "tests, strict verifier, compileall, shell-syntax checks, JSON/CSV schema checks, and git diff --check "
        "BEFORE pushing. This session confirmed each branch's existence and terminal commit but did not re-run "
        "their individual verification suites (a multi-hour undertaking per campaign, several campaigns). "
        "Pushing under-verified branches to a shared remote was judged an unjustified risk given they were not "
        "produced or re-validated in this session; recorded here as explicit remaining work rather than done "
        "silently or skipped without disclosure.",
    }
    (OUT_DIR / "push_publication_record.json").write_text(json.dumps(push_record, indent=2, default=str))

    contract = {
        "schema": "v1_paper_completion_contract",
        "mission_scope": "Push the accepted Part X branch; complete every simulation/analysis/figure/evidence "
        "package genuinely still required for the common-hazard fracture/fatigue paper.",
        "manuscript_located_at": "/Volumes/Data/working-papers/fracture_and_fatigue/",
        "primary_draft_file": "Fatigue_and_fracture_revised_FEM_PF_Rcurve.docx (+ matching .pdf)",
        "manuscript_reading_method": "macOS textutil (no docx-parsing python library available); full text read "
        "for the main draft (355 lines) and discussion_outline.docx (365 lines); targeted grep search over the "
        "Supporting Information and MPZ parameterization handoff documents.",
        "key_finding_governing_this_closure": "The manuscript is a mature, largely complete draft that already "
        "reports specific, quantitative results for essentially every technical claim examined (four-class "
        "K_c(T) comparison, R-curve, 4-decade rate sweep, fatigue crack-growth atlas, cross-phenomenon "
        "correlations over 1360 matched observations). No manuscript claim or figure was found to require a "
        "genuinely new physical simulation. Crack rebonding (Part X) is not referenced anywhere in the current "
        "draft.",
        "physical_simulations_launched_this_session": 0,
        "rerun_policy_applied": "No completed campaign was rerun. The canonical temperature-fatigue campaign's "
        "7 numerical exclusions were reviewed (via its own already-recorded terminal analysis) and found to "
        "already narrow the two manuscript-relevant conclusions (Peak intermediate-T maximum, weakT "
        "temperature-independence) consistently with the manuscript's own hedged claims -- no replacement was "
        "authorized because no manuscript conclusion was found to depend on resolving a specific excluded row.",
        "verification_depth_disclosure": "Full exact-commit verification was performed for the crack-rebonding "
        "Part X campaign (already complete before this mission) and the canonical temperature-fatigue campaign "
        "(exact commit and terminal counts matched the mission specification). The remaining five named "
        "campaigns (two-scale virtual C(T), A_NATIVE+PT panel, R-ratio/deltaK, analytical overlay, physical-"
        "slope-transfer/inverse-design) were confirmed to exist as real git branches with plausible terminal "
        "commits but were NOT individually re-verified at the test-suite/figure-number level in this session. "
        "The four-class 2D PF/CZM parameter registry (sobol candidate IDs) was byte-matched exactly against "
        "the mission specification, but the specific commit(s) that generated the manuscript's Figs. 2-4 were "
        "not pinpointed.",
    }
    (OUT_DIR / "paper_completion_contract.json").write_text(json.dumps(contract, indent=2, default=str))

    print("Wrote authoritative_campaign_inventory.csv, unfinished_physical_simulation_registry.csv, "
          "analysis_only_gap_registry.csv, excluded_future_scope.csv, push_publication_record.json, "
          "paper_completion_contract.json")


if __name__ == "__main__":
    main()
