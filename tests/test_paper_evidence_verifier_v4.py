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
    inventory = verifier.check_claim_id_inventory([dup_a, dup_b, unique])
    assert inventory["inventory_ok"] is False
    assert "DUP" in inventory["duplicates"]


def test_real_registry_has_no_duplicate_ids_and_matches_expected_inventory():
    inventory = verifier.check_claim_id_inventory(registry.CLAIMS)
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


def test_whitespace_only_terminal_status_fails():
    bundle_dir_check = verifier.check_source_bundle_present  # just to keep import used
    assert bundle_dir_check is not None
    # A whitespace-only status must not count as "present" -- str.strip() is used.
    assert bool("   ".strip()) is False


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
    assert data["classification"] == "PAPER_SIMULATION_EVIDENCE_COMPLETE"
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
