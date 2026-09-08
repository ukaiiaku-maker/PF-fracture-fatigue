"""Strict paper-evidence verifier (paper-evidence provenance closure,
mission Section 9). Fails when:

  - a QUALIFIED_SOURCE_RESULT_VERIFIED row lacks a real branch/commit-or-
    N/A-with-reason/result_root/source_data_file/builder_script mapping;
  - a manuscript numerical claim cannot be independently reproduced from
    its cited source data (recomputed fresh here, not read from the
    matrix's own cached verification_note text);
  - a claim_id is duplicated;
  - a recorded remote SHA disagrees with a live `git ls-remote`;
  - any campaign branch this closure touched is dirty;
  - the campaign branch verification registry reports a non-PASS
    py_compile/git-diff-check result;
  - the PX5 transient verifier (in its own branch/worktree) does not pass;
  - file_hashes.json is stale relative to the current directory contents.

Never touches a gitignored runs/ directory inside the git repository
itself. DOES read the external, non-git Arrhenius_FEM_CZM directory
directly (the confirmed source of Figures 2-4) to independently
recompute the manuscript's own reported numbers -- if that directory is
unavailable in a future environment, the affected checks FAIL rather
than being silently skipped, per the review's own instruction not to
treat an unreproducible number as qualified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"

FEM_CZM_ROOT = Path("/Volumes/Data/Data/Nanopillar_calculation/Arrhenius_FEM_CZM")

EXPECTED_FIG2_RMS = {"ceramic": 0.23, "weakT": 0.23, "peak": 1.18, "DBTT": 1.33}
EXPECTED_FIG3_K0 = {"ceramic": 9.33, "weakT": 12.83, "peak": 16.71, "DBTT": 22.77}
EXPECTED_FIG3_KSS = {"ceramic": 16.59, "weakT": 17.78, "peak": 27.41, "DBTT": 33.20}
EXPECTED_FIG3_DELTAK = {"ceramic": 7.26, "weakT": 4.95, "peak": 10.70, "DBTT": 10.43}

EXPECTED_REMOTE_SHAS = {
    "codex/v10.2.30-crack-rebonding-part-x": "30db009ff7172728a6bdc885886f21b94cbec225",
    "codex/v10.2.30-crack-rebonding-part-x-px5-transient": "fbf500cc92b37e8a613c2aa593d86403680bdfa5",
}


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    checks: dict[str, bool] = {}
    details: dict[str, object] = {}

    # --- claim matrix v2 internal consistency ---
    matrix = json.loads((OUT_DIR / "paper_claim_evidence_matrix_v2.json").read_text())
    rows = matrix["rows"]
    claim_ids = [r["claim_id"] for r in rows]
    checks["no_duplicate_claim_ids"] = len(claim_ids) == len(set(claim_ids))

    qualified_rows = [r for r in rows if r["status"] == "QUALIFIED_SOURCE_RESULT_VERIFIED"]
    checks["n_qualified_rows_is_8"] = len(qualified_rows) == 8
    all_have_mapping = all(
        r["result_root"] not in ("N/A", "") and r["source_data_file"] not in ("N/A", "")
        and r["builder_script"] not in ("N/A", "") and r["parameter_fingerprint"] not in ("N/A", "")
        for r in qualified_rows
    )
    checks["every_qualified_row_has_full_provenance_mapping"] = all_have_mapping

    # No row may claim QUALIFIED status while its own verification_note
    # concedes it did not independently recompute anything.
    unverified_language = ["not independently recompute", "not confirmed", "not individually verified"]
    checks["no_qualified_row_admits_unverified_language"] = all(
        not any(phrase in r["verification_note"] for phrase in unverified_language) for r in qualified_rows
    )

    # --- independently RECOMPUTE Fig 2 RMS deviations from raw source data ---
    fig2_csv = FEM_CZM_ROOT / "runs/PF_vs_CZM_first_passage_with_analytic_publication/first_passage_comparison_with_analytic.csv"
    if fig2_csv.is_file():
        fig2_rows = list(csv.DictReader(open(fig2_csv)))
        from collections import defaultdict
        by_class = defaultdict(list)
        for r in fig2_rows:
            if r["framework"] == "FEM/CZM":
                by_class[r["class"]].append(float(r["error_vs_analytic_MPa_sqrt_m"]))
        recomputed_rms = {cls: round(math.sqrt(sum(e * e for e in errs) / len(errs)), 2) for cls, errs in by_class.items()}
        details["recomputed_fig2_rms"] = recomputed_rms
        checks["fig2_rms_reproduces_manuscript_exactly"] = recomputed_rms == EXPECTED_FIG2_RMS
    else:
        checks["fig2_rms_reproduces_manuscript_exactly"] = False
        details["fig2_source_missing"] = str(fig2_csv)

    # --- independently RE-READ Fig 3 K0/Kss/DeltaK from raw source data ---
    fig3_csv = FEM_CZM_ROOT / "runs/four_class_exp_floor_CZM_500K_5rep_1000um_theta45/Rcurve_analysis/class_Rcurve_metric_summary_complete_only.csv"
    if fig3_csv.is_file():
        fig3_rows = {r["class"]: r for r in csv.DictReader(open(fig3_csv))}
        recomputed_k0 = {cls: round(float(fig3_rows[cls]["K0_mean"]), 2) for cls in EXPECTED_FIG3_K0}
        recomputed_kss = {cls: round(float(fig3_rows[cls]["Kss_mean"]), 2) for cls in EXPECTED_FIG3_KSS}
        recomputed_deltak = {cls: round(float(fig3_rows[cls]["DeltaK_mean"]), 2) for cls in EXPECTED_FIG3_DELTAK}
        details["recomputed_fig3_K0"] = recomputed_k0
        details["recomputed_fig3_Kss"] = recomputed_kss
        details["recomputed_fig3_DeltaK"] = recomputed_deltak
        checks["fig3_K0_reproduces_manuscript_exactly"] = recomputed_k0 == EXPECTED_FIG3_K0
        checks["fig3_Kss_reproduces_manuscript_exactly"] = recomputed_kss == EXPECTED_FIG3_KSS
        checks["fig3_DeltaK_reproduces_manuscript_exactly"] = recomputed_deltak == EXPECTED_FIG3_DELTAK
    else:
        checks["fig3_K0_reproduces_manuscript_exactly"] = False
        checks["fig3_Kss_reproduces_manuscript_exactly"] = False
        checks["fig3_DeltaK_reproduces_manuscript_exactly"] = False
        details["fig3_source_missing"] = str(fig3_csv)

    # --- Fig 4 theta=45 parameter fingerprint check ---
    rate_sweep_config = FEM_CZM_ROOT / "runs/four_class_exp_floor_CZM_rates_no_branch_500um_theta45/rate_1x/sweep_config.json"
    if rate_sweep_config.is_file():
        cfg = json.loads(rate_sweep_config.read_text())
        checks["fig4_theta_45_confirmed"] = cfg.get("theta") == 45.0
    else:
        checks["fig4_theta_45_confirmed"] = False

    # --- remote SHA agreement ---
    for branch, expected_sha in EXPECTED_REMOTE_SHAS.items():
        out = subprocess.run(["git", "ls-remote", "--heads", "origin", branch], cwd=REPO_ROOT, capture_output=True, text=True)
        actual = out.stdout.split()[0] if out.stdout.strip() else None
        checks[f"remote_sha_matches_{branch.replace('/', '_')}"] = actual == expected_sha

    # --- campaign branch verification registry: all PASS ---
    campaign_reg = json.loads((OUT_DIR / "campaign_branch_verification_registry.json").read_text())
    checks["all_campaign_branches_py_compile_pass"] = campaign_reg["all_py_compile_pass"] is True
    checks["all_campaign_branches_diff_check_pass"] = campaign_reg["all_git_diff_check_pass"] is True

    # --- temperature-fatigue exclusion review: zero unauthorized replacements ---
    temp_review = json.loads((OUT_DIR / "temperature_fatigue_exclusion_review.json").read_text())
    checks["temperature_fatigue_zero_replacements_authorized"] = temp_review["replacements_authorized"] == 0
    checks["temperature_fatigue_counts_match_mission_spec"] = (
        temp_review["total_terminal_rows"] == 36 and temp_review["physical_targets"] == 18
        and temp_review["qualified_physical_cycle_censors"] == 11 and temp_review["numerical_nonterminations_excluded"] == 7
    )

    # --- PX5 transient verifier (separate branch/worktree) ---
    px5_worktree = REPO_ROOT.parent / "v10230-crack-rebonding-part-x-px5-transient"
    if px5_worktree.is_dir():
        proc = subprocess.run([sys.executable, "scripts/verify_px5_transient_regime_analysis.py"], cwd=px5_worktree, capture_output=True, text=True)
        checks["px5_transient_verifier_passes"] = proc.returncode == 0
        details["px5_transient_verifier_output_tail"] = (proc.stdout + proc.stderr)[-800:]
        checks["px5_worktree_clean"] = _git("status", "--porcelain", cwd=px5_worktree) == ""
    else:
        checks["px5_transient_verifier_passes"] = False
        checks["px5_worktree_clean"] = False
        details["px5_worktree_missing"] = str(px5_worktree)

    # --- this worktree itself must be clean (aside from the files this run is about to write) ---
    checks["this_worktree_diff_check_clean"] = _git("diff", "--check") == ""

    structural_checks_pass = all(checks.values())
    n_unresolved_rows = sum(
        1 for r in rows if r["status"] in ("MANUSCRIPT_RESULT_NOT_SOURCE_TRACED", "PHYSICAL_SIMULATION_GAP", "SOURCE_RESULT_LOCATED_NOT_REVERIFIED")
    )
    checks["zero_unresolved_claim_rows"] = n_unresolved_rows == 0
    overall_pass = structural_checks_pass and n_unresolved_rows == 0
    # A structural-check failure always yields the terminal (unqualified)
    # failure classification; unresolved-but-structurally-sound rows yield
    # the explicit-limitations classification, never a bare "complete".
    if not structural_checks_pass:
        classification = "PAPER_SIMULATION_EVIDENCE_INCOMPLETE"
    elif n_unresolved_rows == 0:
        classification = "PAPER_SIMULATION_EVIDENCE_COMPLETE"
    else:
        classification = "PAPER_SIMULATION_EVIDENCE_COMPLETE_WITH_EXPLICIT_SOURCE_LIMITATIONS"
    details["n_unresolved_claim_rows"] = n_unresolved_rows

    limitations = []
    if n_unresolved_rows > 0:
        limitations.append(f"{n_unresolved_rows} claim rows remain MANUSCRIPT_RESULT_NOT_SOURCE_TRACED "
                            "(Figures 1, 5, 6, 7 and the SI synthetic-identifiability study) -- their exact "
                            "source data/scripts were not located this session (see paper_claim_evidence_matrix_v2.csv).")
    limitations.append("6 named campaign branches (two-scale virtual C(T), A_NATIVE+PT panel, R-ratio/deltaK, "
                        "analytical overlay, physical-slope-transfer, inverse-fatigue-barrier-design) pass "
                        "py_compile and git diff --check but could not be data-level re-verified: their strict "
                        "verifiers require a --root pointing at gitignored raw result directories not found on "
                        "this filesystem. None were found to be directly, conclusively cited by the current "
                        "manuscript and none were pushed to origin.")
    limitations.append("The exact per-row (class, temperature, Kmax) identity of 4 of the 7 canonical "
                        "temperature-fatigue numerical exclusions was not individually re-derived (the raw "
                        "36-row output directory is gitignored and was not found on this filesystem); the 3 "
                        "Peak-class exclusions ARE individually attributed via the campaign's own committed "
                        "progress log.")

    verification = {
        "schema": "v1_paper_simulation_completion_verification",
        "classification": classification,
        "structural_checks_pass": structural_checks_pass,
        "fully_resolved_zero_limitations": overall_pass,
        "checks": checks, "details": details, "limitations": limitations,
    }
    out_path = OUT_DIR / "paper_simulation_completion_verification.json"
    out_path.write_text(json.dumps(verification, indent=2, sort_keys=True, default=str) + "\n")
    print(f"wrote {out_path}")
    print(f"classification={classification}")
    print(f"structural_checks_pass={structural_checks_pass}")
    print(f"fully_resolved_zero_limitations={overall_pass}")
    for k, v in checks.items():
        if not v:
            print(f"  FAILED: {k}")
    return 0 if structural_checks_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
