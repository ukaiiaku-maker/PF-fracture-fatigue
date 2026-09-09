"""One focused and one full merge validation; never starts physical work.

The previous publication suite and package remain immutable. New invocations
are exclusive-claimed in a separate output directory; failed gates never
silently rerun tests. Scratch uses Macintosh HD, records use the Data drive.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile

from scripts.historical_product_fixtures import BUNDLED, MANIFEST_NAME
from scripts.run_v13_heldout_materials import (
    OUT as PUBLICATION, ROOT, DEST, PHYSICS, HELPERS, verify_accepted, verify_source,
)

OUT = ROOT / "analysis_outputs/v13_merge_baseline_review"
FINAL = PUBLICATION / "final_four_material_record"
PRIOR = FINAL / "full_suite_review.json"
PRIOR_SHA = "12bdb50e17896e877322fd6f4012b2ea2c3447f4db1b7190d6d9d8bc8d56a4c7"
ACCEPTED = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_1_1/pf_general_multifront_validation_v6_1_1.json"
PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(name, data):
    OUT.mkdir(exist_ok=True)
    path = OUT / name
    with path.open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def node(row):
    return row["test"]["classname"].replace(".", "/") + ".py::" + row["test"]["name"]


def signature(message):
    # Only volatile pytest temporary paths, repr escapes and whitespace change.
    message = re.sub(r"\S+/S00\.audit\.json", "<PYTEST_TMP>/S00.audit.json", message)
    return re.sub(r"\s+", " ", message.replace("\\", "")).strip()


def baseline():
    assert sha(PRIOR) == PRIOR_SHA, "previous one-time suite changed"
    old = json.loads(PRIOR.read_text())
    missing = {node(r): r for r in old["failures"] if r["message"].startswith("FileNotFoundError:")}
    inherited = {node(r): r for r in old["failures"] if node(r) not in missing}
    accepted = json.loads(ACCEPTED.read_text())["full_suite"]["failure_node_ids"]
    assert len(missing) == 24 and len(inherited) == 7 and set(inherited) == set(accepted)
    catalog = json.loads((BUNDLED / MANIFEST_NAME).read_text())
    assert set(catalog["tests"]) == set(missing)
    return old, missing, inherited


def verify_immutable():
    plan = verify_source()
    frozen = verify_accepted()
    pre = json.loads((PUBLICATION / "publication_preflight.json").read_text())
    provenance = json.loads((FINAL / "provenance.json").read_text())
    for mapping in (pre["heldout_files"], provenance["files"]):
        for name, digest in mapping.items():
            assert sha(name) == digest, "accepted artifact changed: " + name
    archive = PUBLICATION / "V13_FOUR_MATERIAL_REVIEW.zip"
    assert sha(archive) == "35234f3e78afcbd6679b523aacf402ea025ad9b6ec48f982140729ddcb83e519"
    with zipfile.ZipFile(archive) as pack:
        members = json.loads(pack.read("SHA256_MANIFEST.json"))
        assert len(pack.namelist()) == len(set(pack.namelist())) == len(members) + 1
        assert set(pack.namelist()) == set(members) | {"SHA256_MANIFEST.json"}
        for name, digest in members.items():
            assert hashlib.sha256(pack.read(name)).hexdigest() == digest
        for path in FINAL.iterdir():
            if path.is_file():
                assert sha(path) == members["final/" + path.name]
    assert json.loads((DEST / "queue_status.json").read_text()) == {
        "active": {}, "pending": [], "paused": False,
    }
    subprocess.run(["git", "diff", "--exit-code", PHYSICS, "--", "arrhenius_fracture", *HELPERS],
                   cwd=ROOT, check=True, capture_output=True)
    sealed = {str(p): sha(p) for p in PUBLICATION.iterdir() if p.is_file()}
    sealed.update({str(p): sha(p) for p in FINAL.iterdir() if p.is_file()})
    return dict(publication_hashes=sealed, accepted_freezes=frozen,
                source_registry_files=len(plan["source_and_registry_files"]),
                heldout_seals=len(pre["heldout_files"]), report_input_hashes=len(provenance["files"]),
                archive_members=len(members) + 1, all_archive_member_hashes_verified=True)


def code_hashes():
    paths = [ROOT / "scripts/historical_product_fixtures.py", Path(__file__).resolve()]
    paths += list((ROOT / "tests").glob("*.py"))
    paths += [p for p in BUNDLED.rglob("*") if p.is_file()]
    return {str(p.relative_to(ROOT)): sha(p) for p in paths}


def inventory(xml):
    rows = []
    for case in ET.parse(xml).getroot().iter("testcase"):
        status = "passed"
        child = None
        for tag in ("failure", "error", "skipped"):
            found = case.find(tag)
            if found is not None:
                status, child = ("failed" if tag in ("failure", "error") else tag), found
                break
        rows.append(dict(test=dict(case.attrib), status=status,
                         kind=child.tag if child is not None else None,
                         message=child.get("message", "") if child is not None else "",
                         detail=child.text or "" if child is not None else ""))
    return rows


def compare(rows, returncode, *, focused=False):
    old, missing, inherited = baseline()
    current = {node(r): r for r in rows}
    failures = {n: r for n, r in current.items() if r["status"] == "failed"}
    comparison = []
    for n, expected in inherited.items():
        actual = failures.get(n)
        comparison.append(dict(node_id=n, baseline_message=expected["message"],
            current_message=actual["message"] if actual else None,
            baseline_signature=signature(expected["message"]),
            current_signature=signature(actual["message"]) if actual else None,
            match=actual is not None and signature(actual["message"]) == signature(expected["message"]),
            category="ENVIRONMENT_LIMITATION_SANDBOX_PS" if "stage3_status" in n else "INHERITED_BASELINE_FAILURE"))
    historical = []
    for n in missing:
        r = current.get(n)
        allowed = r is not None and (r["status"] == "passed" or
            r["status"] == "skipped" and "HISTORICAL_PRODUCT_FIXTURE_UNAVAILABLE" in r["message"])
        historical.append(dict(node_id=n, status=r["status"] if r else "not_collected",
                               reason=r["message"] if r else None, allowed=allowed))
    original_v13 = {node(r) for r in old["test_inventory"] if "test_v13_" in r["test"]["classname"]}
    v13 = [r for r in rows if "test_v13_" in r["test"]["classname"]]
    original_v13_pass = len(original_v13) == 131 and all(
        n in current and current[n]["status"] == "passed" for n in original_v13)
    new = sorted(set(failures) - set(inherited))
    # These two legacy failures depend on full-suite collection importing the
    # old zero_event_summary_v10215 monkey-patch. A focused PASS is not a fix.
    isolated_passes = {n for n in inherited if
        n.startswith("tests/test_v10_2_27_zero_event_summary.py::")
        and current.get(n, {}).get("status") == "passed"} if focused else set()
    passed = (returncode == 1 and set(failures) == set(inherited) - isolated_passes
              and all(r["match"] or r["node_id"] in isolated_passes for r in comparison)
              and all(r["allowed"] for r in historical)
              and original_v13_pass and all(r["status"] == "passed" for r in v13))
    return dict(relative_gate_pass=passed, status_counts=dict(Counter(r["status"] for r in rows)),
                legacy_comparison=comparison, historical_products=historical,
                original_131_v13_pass=original_v13_pass, all_v13_named_tests=len(v13),
                v13_passed=sum(r["status"] == "passed" for r in v13),
                new_failure_node_ids=new, inventory=rows,
                focused_collection_dependent_legacy_passes=sorted(isolated_passes))


def run(phase):
    old, missing, inherited = baseline()
    OUT.mkdir(exist_ok=True)
    if phase == "full":
        focus = json.loads((OUT / "focused_result.json").read_text())
        contextual = compare(focus["inventory"], focus["returncode"], focused=True)
        assert contextual["relative_gate_pass"], "focused relative gate has not passed"
        changed = [n for n, h in code_hashes().items() if focus["source_file_hashes"].get(n) != h]
        assert set(changed) <= {"scripts/validate_v13_merge_baseline.py"}, "test or loader source changed after focused validation"
        snapshot = OUT / "focused_runner_source.py"
        assert sha(snapshot) == focus["source_file_hashes"]["scripts/validate_v13_merge_baseline.py"]
        save("focused_context_review.json", dict(
            strict_original_result_preserved=True, focused_invocations=1,
            focused_result_sha256=sha(OUT / "focused_result.json"),
            contextual_relative_gate_pass=contextual["relative_gate_pass"],
            isolated_legacy_passes=contextual["focused_collection_dependent_legacy_passes"],
            source_changes_after_focus=changed,
            reason="Full collection imports zero_event_summary_v10215, which replaces the summary function at import time. Only validation orchestration recognizes the focused context; numerical tests and source are unchanged. Full gate still requires all seven exact signatures."))
    # Observed prior scratch was 9 MiB. No trajectory data is copied or deleted.
    assert shutil.disk_usage(ROOT).free > 64 * 1024**2, "insufficient durable record headroom"
    assert shutil.disk_usage("/private/tmp").free > 512 * 1024**2, "insufficient temporary test headroom"
    before = verify_immutable()
    if phase == "full":
        assert before == focus["immutable_after"]
    scratch = tempfile.mkdtemp(prefix="v13-merge-" + phase + "-", dir="/private/tmp")
    command = [PYTHON, "-m", "pytest", "-q", "--junitxml=" + str(OUT / (phase + ".xml"))]
    if phase == "focused":
        command += sorted(missing) + sorted(inherited)
        command += [str(p.relative_to(ROOT)) for p in sorted((ROOT / "tests").glob("test_v13_*.py"))]
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONHASHSEED="0",
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
               MPLCONFIGDIR="/tmp/pf-current-source-v13-parent-mpl", TMPDIR=scratch)
    started = datetime.now(timezone.utc).isoformat()
    claim = dict(phase=phase, command=command, start_utc=started, invocations=1,
                 parent_publication_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                 source_file_hashes=code_hashes(), pid=os.getpid(),
                 environment={k:env[k] for k in ("PYTHONPATH","PYTHONHASHSEED","OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MPLCONFIGDIR","TMPDIR")},
                 durable_free_bytes=shutil.disk_usage(ROOT).free,
                 scratch_free_bytes=shutil.disk_usage(scratch).free)
    save(phase + "_claim.json", claim)  # exclusive; never rerun for a better count
    with (OUT / (phase + ".log")).open("x") as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    rows = inventory(OUT / (phase + ".xml"))
    review = compare(rows, result.returncode)
    after = verify_immutable()
    assert after == before, "accepted publication changed during validation"
    review.update(returncode=result.returncode, invocations=1, command=command,
                  source_file_hashes=claim["source_file_hashes"], immutable_before=before,
                  immutable_after=after, start_utc=started, end_utc=datetime.now(timezone.utc).isoformat())
    save(phase + "_result.json", review)
    print(json.dumps({k:review[k] for k in ("status_counts","relative_gate_pass","new_failure_node_ids","v13_passed")}), flush=True)
    return 0 if review["relative_gate_pass"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("focused", "full"))
    sys.exit(run(parser.parse_args().phase))
