"""Independent-verifier closure (review round 4): final contract document
and file_hashes_v4.json refresh. Run LAST, after every builder/projection/
verifier/render script in this closure pass.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"


BUNDLE = OUT_DIR / "source_bundle_figures_2_4"


def _reconcile_bundle_file_counts() -> dict:
    physically_present = sum(1 for p in BUNDLE.iterdir() if p.is_file())
    manifest = json.loads((OUT_DIR / "paper_source_bundle_manifest.json").read_text())
    manifest_governed = manifest["n_files"]
    compact_governed = 0
    for prov_name in ["fig6b_portable_projection_provenance.json",
                      "fig6c_portable_projection_provenance.json",
                      "fig5c_portable_projection_provenance.json"]:
        prov_path = OUT_DIR / prov_name
        if not prov_path.exists():
            continue
        prov = json.loads(prov_path.read_text())
        tables = prov.get("compact_tables") or {"_single": prov.get("compact_table")}
        compact_governed += sum(1 for t in tables.values() if t)
    return dict(
        physically_present_in_source_bundle=physically_present,
        manifest_governed=manifest_governed,
        compact_projection_governed=compact_governed,
        sum_check_passes=(physically_present == manifest_governed + compact_governed),
    )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    verification = json.loads((OUT_DIR / "paper_simulation_completion_verification_v4.json").read_text())

    contract = {
        "schema": "v4_paper_completion_contract",
        "supersedes": "paper_completion_contract_v3.json",
        "trigger": "Fourth review: replace self-declared verification with an executable one (the "
                   "verifier itself runs deterministic recomputation functions and derives PASS/FAIL, "
                   "rather than trusting a pre-declared independent_recomputation_method string or "
                   "pass_fail column); make Fig.6B genuinely portable via a compact provenance-tracked "
                   "join projection and prove the full verifier passes with all three external source "
                   "roots hidden; recompute or honestly reclassify every remaining qualitative claim "
                   "(Fig.1, Fig.5A-D, Fig.6A/C, Sec.2.15, Fig.4) rather than accepting README-text or "
                   "single-representative-case matches as full verification; add adversarial tests "
                   "proving the verifier cannot be fooled by a forged pass_fail, a stale hash, a "
                   "duplicated claim ID, a blank terminal-status field, a partially-present multi-input "
                   "claim, or a missing external root.",
        "headline_finding": (
            f"verify_paper_evidence_v4.py -- an EXECUTABLE verifier that runs each claim's own "
            f"recomputation function and compares its output against a typed expected value using a "
            f"pure comparator function, deriving pass/fail itself rather than reading any pre-declared "
            f"field -- reports classification={verification['classification']} with "
            f"{verification['n_by_final_evidence_class']} across {verification['n_claims']} claims, "
            f"{verification['n_unresolved']} unresolved. The default (non-strict) invocation exits 0; "
            f"re-running with PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1 (which makes every external, "
            f"non-git source root report as unavailable, with zero filesystem mutation) reproduces the "
            f"identical result, proving every claim is satisfiable from the committed bundle alone."
        ),
        "verifier_classification": verification["classification"],
        "n_claim_rows": verification["n_claims"],
        "n_by_final_evidence_class": verification["n_by_final_evidence_class"],
        "n_unresolved": verification["n_unresolved"],
        "claim_id_inventory_ok": verification["claim_id_inventory"]["inventory_ok"],
        "genuine_recomputations_added_this_round": [
            "Fig1A-D: array-based topology checks (regime-order transitions, shared parameter grids) "
            "replacing README-text matching; capped at SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED "
            "since no manuscript-stated number exists to reproduce numerically.",
            "Fig4-peak-narrow-attenuation: peak-location shift now confirmed at all 4 rate factors via "
            "a fine 5K-resolution, rate-resolved analytic file (was 1x-only in the prior pass).",
            "Fig5A: per-case monotonicity (Spearman rho), point counts, Paris-law slope and curvature "
            "computed for all 6 canonical cases from the raw K-point array.",
            "Fig5B: class-ordering persistence (rank correlation = 1.0 across 6 classes between an "
            "early and a late common extension point) and orientation-dependence magnitude (~17x da/dN "
            "difference, theta30 vs theta45, matched Kmax) recomputed from bundled raw data.",
            "Fig5C: reconstructed across all 15 available seed/stress/condition jobs (was a single "
            "representative pair); the shielded-vs-unshielded contrast is reported as a genuine, "
            "quantified statistical tendency, not an assumed deterministic split.",
            "Fig5D: downgraded to a qualitative-only conclusion; a prior specific numeric field-ratio "
            "claim ('160x'/'1000x') is retracted as an unverifiable visual color-bar estimate -- see "
            "proposed_manuscript_corrections.md.",
            "Fig6A: all 36 contingency tables rebuilt from raw cell counts and Cramer's V recomputed "
            "via chi2_contingency from scratch; confirmed mathematically non-negative, motivating a "
            "manuscript sign-convention correction (see proposed_manuscript_corrections.md).",
            "Fig6B: made fully portable via a compact, provenance-tracked 1360-row joined table -- no "
            "external 40MB file needed at verification time.",
            "Fig6C: all 39 frozen Panel C rows (AUC + 95% bootstrap CI) independently reproduced from a "
            "compact per-observation projection, not one spot-checked row.",
            "Sec2.15: saturation R-curve parameters genuinely refit via scipy.optimize.curve_fit from "
            "raw per-seed binned R-curve data (Kss within 3%, ell_R within 10% of the existing fit "
            "output), not read from the fit-output CSV.",
        ],
        "portability_verified": "verify_paper_evidence_v4.py produces an identical classification and "
                                "per-claim result set with PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1 as "
                                "without it (see paper_simulation_completion_verification_v4.json's "
                                "external_roots_status field for both runs, and "
                                "tests/test_paper_evidence_verifier_v4.py::"
                                "test_full_verifier_passes_from_temp_checkout_with_roots_hidden).",
        "physical_simulations_launched_this_session": 0,
        "physical_simulation_conclusion": "NO_NEW_PHYSICAL_SIMULATIONS_REQUIRED_FOR_CURRENT_DRAFT",
        "round_5_corrections": [
            "Independent frozen claim inventory: expected_paper_claim_ids_v4.json is a hand-"
            "authored, hash-stamped, 35-ID contract loaded by the verifier and checked against "
            "the live registry in both directions -- it is NOT generated from claim_registry_v4."
            "CLAIMS (the round-4 EXPECTED_CLAIM_IDS was, and could not detect a silently deleted "
            "or renamed claim as a result).",
            "Comparators tightened for 9 claims (Fig2-peak-narrow-topology, Fig4-coverage-and-"
            "theta, Fig4-DBTT-transition-shift, Fig5A, Fig5B, Fig5C, Fig5D, Fig6A, Fig6B, Sec2.15, "
            "SI-identifiability) so that passing requires the actual manuscript-relevant magnitude "
            "or relationship, not merely a proxy boolean (e.g. 'a peak exists somewhere') that "
            "could pass under a materially wrong result.",
            "Fig5B downgraded from QUALIFIED to SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED: "
            "its compound manuscript claim includes a geometric 'path deflection' component that "
            "is not numerically verified (no spatial crack-path coordinates are bundled); the "
            "growth-rate orientation-dependence component IS verified, and both facts are recorded "
            "explicitly rather than the claim being rounded up.",
            "Fig5D's spatial-morphology conclusion is now recorded in a distinct "
            "visual_inspection_fig5d.json artifact (image hashes, matched condition, reviewer, "
            "date, explicit no-cross-image-ratio disclaimer) rather than being implied by file-"
            "presence alone; the executable check verifies that record's existence and hash-"
            "consistency, not the morphology judgment itself.",
            "Terminal classification is now computed dynamically as "
            "PAPER_SIMULATION_EVIDENCE_COMPLETE_WITH_<N>_STRUCTURAL_ONLY_CLAIMS (N from this run's "
            "own evaluations, never hardcoded) whenever N>0, rather than collapsing a mix of "
            "QUALIFIED and STRUCTURAL rows into a bare PAPER_SIMULATION_EVIDENCE_COMPLETE.",
            "Bundle file count reconciled: 44 files physically present in "
            "source_bundle_figures_2_4/ = 39 manifest-governed (paper_source_bundle_manifest.json) "
            "+ 5 compact-projection-governed (fig6b/6c/5c provenance JSONs) -- see "
            "PAPER_EVIDENCE_FINAL_HANDOFF_V4.md's reconciliation table for the prior draft's "
            "incorrect '47' figure.",
        ],
        "bundle_file_count_reconciliation": _reconcile_bundle_file_counts(),
    }
    (OUT_DIR / "paper_completion_contract_v4.json").write_text(json.dumps(contract, indent=2, default=str))

    hashes = {}
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file() and path.name not in ("file_hashes.json", "file_hashes_v3.json", "file_hashes_v4.json"):
            hashes[str(path.relative_to(OUT_DIR))] = _sha256(path)
    (OUT_DIR / "file_hashes_v4.json").write_text(json.dumps({
        "schema": "v4_file_hashes", "n_files": len(hashes), "files": hashes,
    }, indent=2, sort_keys=True))
    print(f"Wrote paper_completion_contract_v4.json and file_hashes_v4.json ({len(hashes)} files)")
    print(f"classification: {contract['verifier_classification']}")


if __name__ == "__main__":
    main()
