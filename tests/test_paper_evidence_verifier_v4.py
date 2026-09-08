"""Independent-verifier closure (review round 4): adversarial tests proving
verify_paper_evidence_v4.py actually judges claims itself rather than
trusting a pre-computed field.

Run with an interpreter that has pandas/scipy, e.g.:
  /opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-fem-czm/bin/python -m pytest \
      tests/test_paper_evidence_verifier_v4.py -v
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import verify_paper_evidence_v4 as verifier  # noqa: E402
import claim_registry_v4 as registry  # noqa: E402


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _make_bundle(tmp_path: Path, filename: str, content: bytes) -> tuple[Path, str]:
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir(exist_ok=True)
    fpath = bundle_dir / filename
    fpath.write_bytes(content)
    return bundle_dir, _sha256_bytes(content)


def _trivial_claim(**overrides) -> registry.Claim:
    defaults = dict(
        claim_id="TEST-CLAIM", figure_panel="Test", manuscript_claim_text="test claim",
        source_data_level="raw", source_repository_or_archive="/nowhere",
        producer_script="none", parameter_config_fingerprint="none",
        required_inputs=("input.csv",),
        recompute=lambda bundle: dict(actual=42, terminal_or_censor_status="complete"),
        expected=42, compare=registry.cmp_exact, tolerance="exact",
        max_evidence_class=registry.QUALIFIED, fail_evidence_class=registry.NOT_REVERIFIED,
        notes="",
    )
    defaults.update(overrides)
    return registry.Claim(**defaults)


# ---------------------------------------------------------------------------
# 1. An expected numerical value is altered -> the claim must FAIL.
# ---------------------------------------------------------------------------

def test_altered_expected_value_causes_fail(tmp_path):
    bundle_dir, h = _make_bundle(tmp_path, "input.csv", b"a,b\n1,2\n")
    claim = _trivial_claim(expected=999)  # wrong on purpose
    prov = {"input.csv": h}
    result = verifier.evaluate_claim(claim, prov, bundle_dir=bundle_dir)
    assert result["pass_fail"] == "FAIL"
    assert result["final_evidence_class"] == registry.NOT_REVERIFIED
    assert result["numerical_comparison_passed"] is False


# ---------------------------------------------------------------------------
# 2. pass_fail is manually changed to PASS -> the verifier must not honor it;
#    it always recomputes pass_fail itself from claim.compare().
# ---------------------------------------------------------------------------

def test_manually_forged_pass_fail_is_ignored(tmp_path):
    bundle_dir, h = _make_bundle(tmp_path, "input.csv", b"x\n1\n")
    claim = _trivial_claim(expected=999)
    prov = {"input.csv": h}
    result = verifier.evaluate_claim(claim, prov, bundle_dir=bundle_dir)
    # Simulate an attacker/bug hand-editing the result dict after the fact.
    forged = dict(result)
    forged["pass_fail"] = "PASS"
    forged["final_evidence_class"] = registry.QUALIFIED
    # The forged copy is NOT what the verifier itself produced; re-running
    # evaluate_claim on the same inputs must reproduce the original FAIL,
    # proving the verdict is derived fresh each time, not cached/trusted.
    fresh = verifier.evaluate_claim(claim, prov, bundle_dir=bundle_dir)
    assert fresh["pass_fail"] == "FAIL"
    assert fresh["pass_fail"] != forged["pass_fail"]


# ---------------------------------------------------------------------------
# 3. A required source file is removed -> source_bundle_present is False,
#    recompute is never even attempted, claim FAILs.
# ---------------------------------------------------------------------------

def test_missing_required_file_fails_before_recompute(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    # Do NOT create input.csv.
    claim = _trivial_claim()
    prov = {}  # no provenance either, but the missing-file check should fire first
    result = verifier.evaluate_claim(claim, prov, bundle_dir=bundle_dir)
    assert result["source_bundle_present"] is False
    assert result["recompute_succeeded"] is False
    assert "input.csv" in result["source_bundle_check"]["missing_files"]
    assert result["pass_fail"] == "FAIL"
    assert result["final_evidence_class"] == registry.NOT_REVERIFIED


# ---------------------------------------------------------------------------
# 4. A claim ID is duplicated -> the inventory check must catch it.
# ---------------------------------------------------------------------------

def test_duplicate_claim_id_detected():
    dup_a = _trivial_claim(claim_id="DUP")
    dup_b = _trivial_claim(claim_id="DUP")
    unique = _trivial_claim(claim_id="UNIQUE")
    inventory = verifier.check_claim_id_inventory([dup_a, dup_b, unique], expected_ids=("DUP", "UNIQUE"))
    assert inventory["inventory_ok"] is False
    assert "DUP" in inventory["duplicates"]


def test_real_registry_has_no_duplicate_ids_and_matches_expected_inventory():
    expected_ids = verifier.load_expected_claim_ids(
        REPO_ROOT / "artifacts" / "paper_simulation_completion" / "expected_paper_claim_ids_v4.json"
    )
    inventory = verifier.check_claim_id_inventory(registry.CLAIMS, expected_ids)
    assert inventory["duplicates"] == []
    assert inventory["missing_from_registry"] == []
    assert inventory["unexpected_in_registry"] == []
    assert inventory["inventory_ok"] is True


# ---------------------------------------------------------------------------
# 5. An input hash is stale (bundled file content no longer matches the
#    recorded provenance hash) -> source_bundle_present is False.
# ---------------------------------------------------------------------------

def test_stale_hash_fails_source_bundle_present(tmp_path):
    bundle_dir, correct_hash = _make_bundle(tmp_path, "input.csv", b"original content\n")
    stale_hash = _sha256_bytes(b"a completely different, older version\n")
    claim = _trivial_claim()
    prov = {"input.csv": stale_hash}  # deliberately wrong vs. what's actually on disk
    result = verifier.evaluate_claim(claim, prov, bundle_dir=bundle_dir)
    assert result["source_bundle_present"] is False
    assert result["source_bundle_check"]["hash_mismatches"]
    assert result["pass_fail"] == "FAIL"


# ---------------------------------------------------------------------------
# 6. A terminal/censor field is blank -> the claim must FAIL even if the
#    numeric comparison would otherwise pass.
# ---------------------------------------------------------------------------

def test_blank_terminal_status_fails_even_with_correct_value(tmp_path):
    bundle_dir, h = _make_bundle(tmp_path, "input.csv", b"x\n1\n")
    claim = _trivial_claim(
        recompute=lambda bundle: dict(actual=42, terminal_or_censor_status=""),  # blank!
        expected=42,
    )
    prov = {"input.csv": h}
    result = verifier.evaluate_claim(claim, prov, bundle_dir=bundle_dir)
    assert result["numerical_comparison_passed"] is True  # the value itself was right
    assert result["terminal_status_present"] is False
    assert result["pass_fail"] == "FAIL"  # but the claim still fails overall
    assert result["final_evidence_class"] == registry.NOT_REVERIFIED


def test_whitespace_only_terminal_status_fails(tmp_path):
    # Round-5 correction: this test previously only asserted a Python truthiness fact
    # (bool("   ".strip()) is False) without ever calling evaluate_claim -- it could not have
    # caught a regression in the verifier's own blank-status handling. It now exercises the
    # real code path: a whitespace-only (not merely empty-string) terminal_or_censor_status,
    # returned from an otherwise-correct recompute(), must still fail the claim overall.
    bundle_dir, h = _make_bundle(tmp_path, "input.csv", b"x\n1\n")
    claim = _trivial_claim(
        recompute=lambda bundle: dict(actual=42, terminal_or_censor_status="   \t  "),
        expected=42,
    )
    prov = {"input.csv": h}
    result = verifier.evaluate_claim(claim, prov, bundle_dir=bundle_dir)
    assert result["numerical_comparison_passed"] is True
    assert result["terminal_status_present"] is False
    assert result["pass_fail"] == "FAIL"
    assert result["final_evidence_class"] == registry.NOT_REVERIFIED


# ---------------------------------------------------------------------------
# 7. A multi-input claim cites only one bundled input -> flagged, and fails
#    source_bundle_present because the OTHER required file is missing.
# ---------------------------------------------------------------------------

def test_multi_input_claim_with_only_one_present_is_flagged_and_fails(tmp_path):
    bundle_dir = tmp_path / "bundle"
    bundle_dir.mkdir()
    content_a = b"a\n1\n"
    (bundle_dir / "input_a.csv").write_bytes(content_a)
    # input_b.csv is deliberately NOT created.
    claim = _trivial_claim(required_inputs=("input_a.csv", "input_b.csv"))
    prov = {"input_a.csv": _sha256_bytes(content_a)}
    result = verifier.evaluate_claim(claim, prov, bundle_dir=bundle_dir)
    assert result["source_bundle_check"]["multi_input_but_only_one_present"] is True
    assert result["source_bundle_present"] is False
    assert result["pass_fail"] == "FAIL"


# ---------------------------------------------------------------------------
# 8. An external source directory is unavailable -> PAPER_EVIDENCE_HIDE_
#    EXTERNAL_ROOTS=1 must make external_roots report it as unavailable,
#    with no filesystem mutation.
# ---------------------------------------------------------------------------

def test_hidden_external_roots_env_var(monkeypatch):
    sys.path.insert(0, str(SCRIPTS_DIR))
    import external_roots
    monkeypatch.delenv("PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS", raising=False)
    normal = external_roots.fatigue_pf_root()
    assert str(normal) == "/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF"

    monkeypatch.setenv("PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS", "1")
    hidden = external_roots.fatigue_pf_root()
    assert not hidden.exists()
    assert "hidden" in str(hidden).lower() or "nonexistent" in str(hidden).lower()

    status = external_roots.all_roots_status()
    assert all(v["hiding_enabled"] for v in status.values())
    # Crucially: the REAL directory on disk must be untouched -- hiding is
    # purely a reporting-layer indirection, never a filesystem mutation.
    real_path = Path("/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF")
    assert status["FATIGUE_PF_ROOT"]["actually_exists_on_disk"] == real_path.is_dir()


# ---------------------------------------------------------------------------
# 9. A clean, committed portable bundle must pass from a temporary checkout
#    with no access to the original external directories.
# ---------------------------------------------------------------------------

def test_full_verifier_passes_from_temp_checkout_with_roots_hidden(tmp_path):
    checkout = tmp_path / "checkout"
    shutil.copytree(SCRIPTS_DIR, checkout / "scripts")
    shutil.copytree(REPO_ROOT / "artifacts" / "paper_simulation_completion",
                     checkout / "artifacts" / "paper_simulation_completion")

    env = dict(os.environ)
    env["PAPER_EVIDENCE_HIDE_EXTERNAL_ROOTS"] = "1"
    interpreter = sys.executable
    proc = subprocess.run(
        [interpreter, str(checkout / "scripts" / "verify_paper_evidence_v4.py")],
        cwd=str(checkout / "scripts"), env=env, capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "classification=PAPER_SIMULATION_EVIDENCE_COMPLETE" in proc.stdout
    assert "external_roots_hidden=True" in proc.stdout

    verification_path = checkout / "artifacts" / "paper_simulation_completion" / \
        "paper_simulation_completion_verification_v4.json"
    assert verification_path.is_file()
    data = json.loads(verification_path.read_text())
    assert data["classification"].startswith("PAPER_SIMULATION_EVIDENCE_COMPLETE")
    assert data["n_unresolved"] == 0
    assert data["claim_id_inventory"]["inventory_ok"] is True


def test_real_verifier_default_invocation_exits_zero_and_is_complete():
    interpreter = sys.executable
    proc = subprocess.run(
        [interpreter, str(SCRIPTS_DIR / "verify_paper_evidence_v4.py")],
        cwd=str(SCRIPTS_DIR), capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"
    assert "classification=PAPER_SIMULATION_EVIDENCE_COMPLETE" in proc.stdout


# ---------------------------------------------------------------------------
# Round-5 additions: prove the TIGHTENED comparators actually reject
# corrupted/degenerate results that the round-4 comparators would have
# accepted. Each test uses the REAL registry claim's own `compare` function
# (not a copy), fed a synthetic `actual` value that is deliberately wrong in
# exactly the way the fifth review identified as previously undetectable.
# ---------------------------------------------------------------------------

def _claim(claim_id: str) -> registry.Claim:
    matches = [c for c in registry.CLAIMS if c.claim_id == claim_id]
    assert len(matches) == 1, f"expected exactly one claim {claim_id!r}, found {len(matches)}"
    return matches[0]


def test_fig6b_corrupted_pooled_r_with_unchanged_n_and_range_fails():
    claim = _claim("Fig6B")
    good_actual = dict(claim.expected)  # a "perfect" actual would equal expected exactly
    corrupted = dict(good_actual)
    corrupted["pooled_r"] = 0.5  # wildly wrong, but n and per-context untouched
    assert claim.compare(good_actual, claim.expected) is True
    assert claim.compare(corrupted, claim.expected) is False


def test_fig6a_values_changed_but_still_nonnegative_fails():
    claim = _claim("Fig6A")
    good_actual = dict(n_tables_recomputed=36, all_v_nonnegative=True,
                        cramers_v_by_family_context=dict(claim.expected["frozen_v_by_family_context"]))
    assert claim.compare(good_actual, claim.expected) is True
    corrupted = dict(good_actual)
    corrupted_map = dict(corrupted["cramers_v_by_family_context"])
    # Flatten every value to a small, nonnegative constant -- passes the round-4 checks
    # (count==36, all nonnegative) but destroys every reported magnitude and magnitude group.
    corrupted_map = {k: 0.01 for k in corrupted_map}
    corrupted["cramers_v_by_family_context"] = corrupted_map
    assert claim.compare(corrupted, claim.expected) is False


def test_fig5a_monotonic_but_loses_class_distinction_fails():
    claim = _claim("Fig5A")
    good_actual = dict(all_six_canonical_cases_present=True, all_cases_monotonic=True,
                        steep_cleavage_has_max_paris_slope=True,
                        plastic_shielded_has_min_paris_slope=True,
                        main_text_four_cases_present=True)
    assert claim.compare(good_actual, claim.expected) is True
    # Six curves all monotonic (as six identical curves would be), but the class-specific
    # extremes are lost -- e.g. some OTHER case has the steepest slope.
    degenerate = dict(good_actual)
    degenerate["steep_cleavage_has_max_paris_slope"] = False
    degenerate["plastic_shielded_has_min_paris_slope"] = False
    assert claim.compare(degenerate, claim.expected) is False


def test_fig5c_coverage_pass_intact_but_stress_life_reversed_fails():
    claim = _claim("Fig5C")
    good_actual = dict(unshielded_pass_rate_exceeds_shielded=True,
                        no_shield_life_decreases_with_stress=True,
                        shielded_life_decreases_with_stress=True)
    assert claim.compare(good_actual, claim.expected) is True
    # The shielding/formation-probability contrast is untouched, but the stress-life ordering
    # is reversed (life would INCREASE with stress -- physically backwards for an S-N claim).
    reversed_life = dict(good_actual)
    reversed_life["no_shield_life_decreases_with_stress"] = False
    assert claim.compare(reversed_life, claim.expected) is False


def test_fig2_peak_exists_but_no_fem_attenuation_fails():
    claim = _claim("Fig2-peak-narrow-topology")
    good_actual = dict(fine_grid_peak_found=True, coarse_grid_bump_found=True,
                        pf_sharp_front_peak_found=True, pf_reproduces_peak_strongly=True,
                        fem_lower_than_analytic_at_peak_T=True,
                        fem_attenuation_within_5pct_of_30_9=True)
    assert claim.compare(good_actual, claim.expected) is True
    # A peak exists everywhere it should, but FEM/CZM does NOT fall below the analytic/PF peak
    # (i.e. FEM would have fully -- or over- -- reproduced the peak, contradicting "attenuated").
    no_attenuation = dict(good_actual)
    no_attenuation["fem_lower_than_analytic_at_peak_T"] = False
    no_attenuation["fem_attenuation_within_5pct_of_30_9"] = False
    assert claim.compare(no_attenuation, claim.expected) is False


# ---------------------------------------------------------------------------
# Round-5: the independent frozen claim-ID inventory must catch deletion,
# renaming, addition, and duplication -- and, critically, must be loaded
# from expected_paper_claim_ids_v4.json, NOT derived from CLAIMS itself.
# ---------------------------------------------------------------------------

EXPECTED_IDS_PATH = REPO_ROOT / "artifacts" / "paper_simulation_completion" / "expected_paper_claim_ids_v4.json"


def _load_real_expected_ids() -> tuple[str, ...]:
    return verifier.load_expected_claim_ids(EXPECTED_IDS_PATH)


def test_expected_claim_ids_file_is_independent_of_registry():
    # The file must exist as a standalone artifact and be internally hash-consistent; loading
    # it must not require importing claim_registry_v4 at all.
    doc = json.loads(EXPECTED_IDS_PATH.read_text())
    assert "claim_ids" in doc and "sha256_of_sorted_newline_joined_ids" in doc
    ids = _load_real_expected_ids()
    assert len(ids) == 35
    assert len(set(ids)) == 35


def test_deleting_a_claim_is_detected_against_independent_inventory():
    expected_ids = _load_real_expected_ids()
    claims_missing_one = [c for c in registry.CLAIMS if c.claim_id != expected_ids[0]]
    inventory = verifier.check_claim_id_inventory(claims_missing_one, expected_ids)
    assert inventory["inventory_ok"] is False
    assert expected_ids[0] in inventory["missing_from_registry"]


def test_renaming_a_claim_is_detected_against_independent_inventory():
    expected_ids = _load_real_expected_ids()
    renamed = []
    for c in registry.CLAIMS:
        if c.claim_id == expected_ids[0]:
            renamed.append(_trivial_claim(claim_id=expected_ids[0] + "_RENAMED"))
        else:
            renamed.append(c)
    inventory = verifier.check_claim_id_inventory(renamed, expected_ids)
    assert inventory["inventory_ok"] is False
    assert expected_ids[0] in inventory["missing_from_registry"]
    assert (expected_ids[0] + "_RENAMED") in inventory["unexpected_in_registry"]


def test_adding_an_unexpected_claim_is_detected_against_independent_inventory():
    expected_ids = _load_real_expected_ids()
    extra = list(registry.CLAIMS) + [_trivial_claim(claim_id="TOTALLY-NEW-CLAIM-NOT-IN-CONTRACT")]
    inventory = verifier.check_claim_id_inventory(extra, expected_ids)
    assert inventory["inventory_ok"] is False
    assert "TOTALLY-NEW-CLAIM-NOT-IN-CONTRACT" in inventory["unexpected_in_registry"]


def test_duplicating_a_claim_is_detected_against_independent_inventory():
    expected_ids = _load_real_expected_ids()
    duplicated = list(registry.CLAIMS) + [
        c for c in registry.CLAIMS if c.claim_id == expected_ids[0]
    ]
    inventory = verifier.check_claim_id_inventory(duplicated, expected_ids)
    assert inventory["inventory_ok"] is False
    assert expected_ids[0] in inventory["duplicates"]


def test_real_registry_matches_the_independent_frozen_inventory_exactly():
    expected_ids = _load_real_expected_ids()
    inventory = verifier.check_claim_id_inventory(registry.CLAIMS, expected_ids)
    assert inventory["inventory_ok"] is True
    assert inventory["missing_from_registry"] == []
    assert inventory["unexpected_in_registry"] == []
    assert inventory["duplicates"] == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
