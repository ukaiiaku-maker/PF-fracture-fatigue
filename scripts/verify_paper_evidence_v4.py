"""Independent-verifier closure (review round 4, corrected in round 5): the
executable verifier.

This is the one and only place PASS/FAIL is decided. It does NOT trust any
pre-computed `pass_fail` column or `independent_recomputation_method`
free-text field the way verify_paper_simulation_completion_v3.py did. For
every claim in claim_registry_v4.CLAIMS it:

  1. Checks the claim_id is unique across the registry and that the set of
     claim_ids exactly matches the INDEPENDENT frozen inventory in
     expected_paper_claim_ids_v4.json -- a hand-authored, hash-stamped
     contract that is NOT generated from claim_registry_v4.CLAIMS. (Round 4
     used `EXPECTED_CLAIM_IDS = tuple(sorted(c.claim_id for c in CLAIMS))`,
     which the fifth review correctly identified as unable to detect a
     claim silently deleted or renamed from CLAIMS, since the "expected"
     list was derived from the very thing being checked. This is fixed.)
     Silently dropping, renaming, or adding a claim is now a hard failure.
  2. Checks every required_inputs file exists in the bundle AND its SHA-256
     matches a recorded provenance hash (either the original-copy manifest
     from build_source_bundle_figures_2_4.py, or one of the three portable-
     projection provenance JSONs for the compact derived tables) -- this is
     `source_bundle_present`, computed here, not read from a CSV column.
  3. Calls the claim's `recompute(bundle_dir)` function inside a try/except.
     Any exception is caught and recorded as a FAIL with the exception text;
     it is never silently downgraded to "skipped".
  4. Calls the claim's `compare(actual, expected)` function to get a
     boolean pass/fail -- this boolean is what "PASS"/"FAIL" means from here
     on, and nothing upstream of this call can override it.
  5. Requires `terminal_or_censor_status` in the recompute result to be a
     non-blank string.
  6. Assigns final_evidence_class = max_evidence_class if all checks pass,
     else fail_evidence_class.

Supports PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS=1 (see external_roots.py) to
prove every claim is satisfiable from the committed bundle alone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import claim_registry_v4 as registry  # noqa: E402
import external_roots  # noqa: E402

REPO_ROOT = SCRIPT_DIR.parent
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"
BUNDLE = OUT_DIR / "source_bundle_figures_2_4"

TERMINAL_OK_LABEL = "PAPER_SIMULATION_EVIDENCE_COMPLETE"
INCOMPLETE_LABEL = "PAPER_SIMULATION_EVIDENCE_INCOMPLETE"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_provenance_hashes() -> dict[str, str]:
    """Map bundle filename -> the sha256 it is supposed to have, drawn from
    every provenance source this branch maintains."""
    hashes: dict[str, str] = {}
    manifest_path = OUT_DIR / "paper_source_bundle_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for f in manifest["files"]:
            if f.get("copy_status") == "OK":
                hashes[f["bundle"]] = f["original_sha256"]
    for prov_name in ["fig6b_portable_projection_provenance.json",
                      "fig6c_portable_projection_provenance.json",
                      "fig5c_portable_projection_provenance.json"]:
        prov_path = OUT_DIR / prov_name
        if not prov_path.exists():
            continue
        prov = json.loads(prov_path.read_text())
        tables = prov.get("compact_tables") or {"_single": prov.get("compact_table")}
        for entry in tables.values():
            if entry is None:
                continue
            fname = Path(entry["path"]).name
            hashes[fname] = entry["sha256"]
    return hashes


def load_expected_claim_ids(path: Path = None) -> tuple[str, ...]:
    """Load the INDEPENDENT, hand-authored frozen claim-id inventory. This
    file is never written by any code that also builds claim_registry_v4.CLAIMS
    -- it is a separate, literal contract checked into the same commit."""
    path = path or (OUT_DIR / "expected_paper_claim_ids_v4.json")
    doc = json.loads(path.read_text())
    ids = tuple(doc["claim_ids"])
    payload = "\n".join(sorted(ids)).encode()
    actual_hash = hashlib.sha256(payload).hexdigest()
    recorded_hash = doc["sha256_of_sorted_newline_joined_ids"]
    if actual_hash != recorded_hash:
        raise ValueError(
            f"{path} is internally inconsistent: its own sha256_of_sorted_newline_joined_ids "
            f"({recorded_hash}) does not match a hash of its own claim_ids list ({actual_hash}). "
            f"This file must be regenerated deliberately, not hand-edited."
        )
    return ids


def check_claim_id_inventory(claims: list, expected_ids: tuple[str, ...]) -> dict:
    ids = [c.claim_id for c in claims]
    duplicates = sorted({cid for cid in ids if ids.count(cid) > 1})
    actual_set = set(ids)
    expected_set = set(expected_ids)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    return dict(
        n_claims=len(claims), n_unique=len(actual_set), n_expected=len(expected_set),
        duplicates=duplicates,
        missing_from_registry=missing, unexpected_in_registry=unexpected,
        inventory_ok=(not duplicates and not missing and not unexpected),
    )


def check_source_bundle_present(claim, provenance_hashes: dict[str, str], bundle_dir: Path = BUNDLE) -> dict:
    missing_files = []
    hash_mismatches = []
    multi_input_single_citation = False
    for fname in claim.required_inputs:
        fpath = bundle_dir / fname
        if not fpath.is_file():
            missing_files.append(fname)
            continue
        expected_hash = provenance_hashes.get(fname)
        if expected_hash is None:
            hash_mismatches.append((fname, "NO_PROVENANCE_RECORD"))
            continue
        actual_hash = _sha256_file(fpath)
        if actual_hash != expected_hash:
            hash_mismatches.append((fname, "HASH_MISMATCH"))
    if len(claim.required_inputs) > 1 and len(claim.required_inputs) - len(missing_files) == 1 and missing_files:
        multi_input_single_citation = True
    return dict(
        n_required=len(claim.required_inputs), n_missing=len(missing_files),
        missing_files=missing_files, hash_mismatches=hash_mismatches,
        present=(not missing_files and not hash_mismatches),
        multi_input_but_only_one_present=multi_input_single_citation,
    )


def evaluate_claim(claim, provenance_hashes: dict[str, str], bundle_dir: Path = BUNDLE) -> dict:
    bundle_check = check_source_bundle_present(claim, provenance_hashes, bundle_dir=bundle_dir)

    result = dict(
        claim_id=claim.claim_id, figure_panel=claim.figure_panel,
        manuscript_claim_text=claim.manuscript_claim_text,
        source_data_level=claim.source_data_level,
        source_repository_or_archive=claim.source_repository_or_archive,
        producer_script=claim.producer_script,
        parameter_config_fingerprint=claim.parameter_config_fingerprint,
        required_inputs=list(claim.required_inputs),
        tolerance=claim.tolerance, notes=claim.notes,
        source_bundle_present=bundle_check["present"],
        source_bundle_check=bundle_check,
    )

    if not bundle_check["present"]:
        result.update(
            recompute_succeeded=False, recompute_error="source_bundle_present is False; recompute not attempted",
            actual_value=None, terminal_or_censor_status="", terminal_status_present=False,
            numerical_comparison_passed=False, independent_recomputation_performed=False,
            final_evidence_class=claim.fail_evidence_class, pass_fail="FAIL",
        )
        return result

    try:
        recompute_result = claim.recompute(bundle_dir)
        actual = recompute_result["actual"]
        terminal_status = recompute_result.get("terminal_or_censor_status", "")
        diagnostics = recompute_result.get("diagnostics", {})
        recompute_succeeded = True
        recompute_error = None
    except Exception:  # noqa: BLE001 -- intentionally broad: any failure is a FAIL, not a skip
        actual = None
        terminal_status = ""
        diagnostics = {}
        recompute_succeeded = False
        recompute_error = traceback.format_exc(limit=5)

    terminal_status_present = bool(str(terminal_status).strip())
    numerical_comparison_passed = False
    if recompute_succeeded:
        try:
            numerical_comparison_passed = bool(claim.compare(actual, claim.expected))
        except Exception:  # noqa: BLE001
            numerical_comparison_passed = False
            if recompute_error is None:
                recompute_error = "compare() raised: " + traceback.format_exc(limit=5)

    overall_pass = recompute_succeeded and numerical_comparison_passed and terminal_status_present
    final_evidence_class = claim.max_evidence_class if overall_pass else claim.fail_evidence_class

    result.update(
        recompute_succeeded=recompute_succeeded, recompute_error=recompute_error,
        actual_value=actual, expected_value=claim.expected,
        terminal_or_censor_status=terminal_status, terminal_status_present=terminal_status_present,
        numerical_comparison_passed=numerical_comparison_passed,
        independent_recomputation_performed=recompute_succeeded,
        diagnostics=diagnostics,
        final_evidence_class=final_evidence_class,
        pass_fail="PASS" if overall_pass else "FAIL",
    )
    return result


UNRESOLVED_CLASSES = {registry.NOT_REVERIFIED}
STRUCTURAL_OK_CLASSES = {registry.QUALIFIED, registry.STRUCTURAL}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-all-qualified", action="store_true",
                     help="Stricter-than-default mode: also require zero "
                          "SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED rows for "
                          "PAPER_SIMULATION_EVIDENCE_COMPLETE. The default mode (matching the review's "
                          "own final-classification instruction) treats a passing STRUCTURAL row as an "
                          "honest ceiling for a topological claim with no manuscript-stated number to "
                          "reproduce, not a failure -- only SOURCE_RESULT_LOCATED_NOT_REVERIFIED "
                          "(a claim whose comparator actually returned False) blocks completion by "
                          "default.")
    args = ap.parse_args()

    expected_ids = load_expected_claim_ids()
    inventory = check_claim_id_inventory(registry.CLAIMS, expected_ids)
    provenance_hashes = _load_provenance_hashes()

    evaluations = [evaluate_claim(c, provenance_hashes) for c in registry.CLAIMS]

    n_by_class: dict[str, int] = {}
    for e in evaluations:
        n_by_class[e["final_evidence_class"]] = n_by_class.get(e["final_evidence_class"], 0) + 1

    unresolved = [e for e in evaluations if e["final_evidence_class"] in UNRESOLVED_CLASSES]
    structural_only = [e for e in evaluations if e["final_evidence_class"] == registry.STRUCTURAL]

    if not inventory["inventory_ok"]:
        classification = "VERIFIER_INTEGRITY_FAILURE_CLAIM_INVENTORY_MISMATCH"
    elif unresolved:
        classification = INCOMPLETE_LABEL
    elif structural_only and args.require_all_qualified:
        classification = INCOMPLETE_LABEL
    elif structural_only:
        # Round-5 correction: do NOT collapse "30 QUALIFIED + 5(+) STRUCTURAL" into a bare
        # PAPER_SIMULATION_EVIDENCE_COMPLETE that reads as if every claim were fully
        # quantitatively verified. The label itself now names the exact count of claims held at
        # the honest SOURCE_RESULT_LOCATED_AND_STRUCTURALLY_MATCHED ceiling (topology-only claims
        # with no manuscript-stated number to reproduce numerically), computed from this run's
        # own evaluations -- never hardcoded. This is still a terminal, passing classification
        # (exit 0): zero claims are unresolved/FAILed, and the physical-simulation conclusion
        # (NO_NEW_PHYSICAL_SIMULATIONS_REQUIRED_FOR_CURRENT_DRAFT) is unaffected by a claim being
        # honestly qualitative rather than numeric.
        classification = f"PAPER_SIMULATION_EVIDENCE_COMPLETE_WITH_{len(structural_only)}_STRUCTURAL_ONLY_CLAIMS"
    else:
        # Every retained claim is QUALIFIED_SOURCE_RESULT_VERIFIED; no honest-ceiling exceptions.
        classification = TERMINAL_OK_LABEL

    external_status = external_roots.all_roots_status()

    output = dict(
        schema="v4_executable_verifier",
        claim_id_inventory=inventory,
        n_claims=len(evaluations),
        n_by_final_evidence_class=n_by_class,
        n_unresolved=len(unresolved),
        unresolved_claim_ids=[e["claim_id"] for e in unresolved],
        n_structural_only=len(structural_only),
        structural_only_claim_ids=[e["claim_id"] for e in structural_only],
        require_all_qualified_flag=args.require_all_qualified,
        external_roots_status=external_status,
        classification=classification,
        physical_simulation_conclusion="NO_NEW_PHYSICAL_SIMULATIONS_REQUIRED_FOR_CURRENT_DRAFT",
        physical_simulations_launched_this_run=0,
        evaluations=evaluations,
    )
    (OUT_DIR / "paper_simulation_completion_verification_v4.json").write_text(
        json.dumps(output, indent=2, default=str))

    print(f"n_claims={len(evaluations)}  by_class={n_by_class}")
    print(f"inventory_ok={inventory['inventory_ok']}  duplicates={inventory['duplicates']} "
          f"missing={inventory['missing_from_registry']} unexpected={inventory['unexpected_in_registry']}")
    print(f"external_roots_hidden={external_status['FEM_CZM_ROOT']['hiding_enabled']}")
    print(f"classification={classification}")
    if unresolved:
        print("UNRESOLVED:")
        for e in unresolved:
            print(f"  {e['claim_id']}: pass_fail={e.get('pass_fail')} error={e.get('recompute_error')}")
    if structural_only:
        print(f"STRUCTURAL-ONLY (not QUALIFIED): {[e['claim_id'] for e in structural_only]}")

    passing = classification == TERMINAL_OK_LABEL or classification.startswith(
        "PAPER_SIMULATION_EVIDENCE_COMPLETE_WITH_"
    )
    return 0 if passing else 1


if __name__ == "__main__":
    sys.exit(main())
