import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_4"
ARCHIVE = OUT / "Archive_PF_GENERAL_MULTIFRONT_V6_4_COMPLETE.zip"
REQUIRED = {
    "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_4.md",
    "pf_general_multifront_v6_4_source_provenance.json",
    "pf_general_multifront_integrated_interval_lifecycle_v6_4.csv",
    "pf_general_multifront_hook_interface_contract_v6_4.json",
    "pf_general_multifront_stress_identity_v6_4.json",
    "pf_general_multifront_owner_process_and_renewal_v6_4.json",
    "pf_general_multifront_event_finalization_v6_4.csv",
    "pf_general_multifront_exact_trial_delta_v6_4.json",
    "pf_general_multifront_arbitrary_region_request_v6_4.json",
    "pf_general_multifront_atomic_output_checkpoint_v6_4.json",
    "pf_general_multifront_cached_end_to_end_parity_v6_4.json",
    "pf_general_multifront_integrated_cli_dry_run_v6_4.json",
    "pf_general_multifront_archive_sufficiency_inventory_v6_4.csv",
    "pf_general_multifront_archive_sufficiency_summary_v6_4.json",
}


def verify(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        expected = set(manifest["members"]) | {"MANIFEST.json"}
        if set(archive.namelist()) != expected:
            raise ValueError("inventory mismatch")
        for name, evidence in manifest["members"].items():
            value = archive.read(name)
            if hashlib.sha256(value).hexdigest() != evidence["sha256"]:
                raise ValueError(f"hash mismatch: {name}")
            if len(value) != evidence["size_bytes"]:
                raise ValueError(f"size mismatch: {name}")


def test_required_v6_4_products_and_fail_closed_decision():
    assert REQUIRED.issubset({path.name for path in OUT.iterdir()})
    cli = json.loads((OUT / "pf_general_multifront_integrated_cli_dry_run_v6_4.json").read_text())
    cached = json.loads((OUT / "pf_general_multifront_cached_end_to_end_parity_v6_4.json").read_text())
    provenance = json.loads((OUT / "pf_general_multifront_v6_4_source_provenance.json").read_text())
    sufficiency = json.loads((OUT / "pf_general_multifront_archive_sufficiency_summary_v6_4.json").read_text())
    assert cli["integrated_interval_transaction"] == "FAIL_CLOSED"
    assert not cli["trajectory_execution_authorized"]
    assert cli["provider_solve_count"] == cli["mechanics_solve_count"] == cli["workers_started"] == 0
    assert cached["qualification"] == "FAIL_CLOSED"
    assert provenance["v6_3_archive"]["sha256"] == "b46f9b5581ecd952dc244d2f21d23eca342c456a1be5e71f5220deaae0496e5d"
    assert not provenance["executable_v12_source_or_tests_changed_after_v6_3_implementation"]
    assert sufficiency["admitted_interval_count"] == 0
    assert sufficiency["cached_N1_end_to_end_parity"] == "FAIL_CLOSED_NO_USABLE_ARCHIVED_STRESS_INTERVAL"
    assert sufficiency["cached_N2_end_to_end_parity"] == "FAIL_CLOSED_INCOMPLETE_FRONT_CANDIDATE_COVERAGE"


def test_v6_4_archive_positive_and_tamper_verification():
    original = ARCHIVE.read_bytes(); verify(original)
    altered = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(altered, "w") as target:
        victim = "pf_general_multifront_exact_trial_delta_v6_4.json"
        for item in source.infolist():
            value = source.read(item.filename)
            target.writestr(item, value + (b"x" if item.filename == victim else b""))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify(altered.getvalue())

    removed = io.BytesIO()
    with zipfile.ZipFile(io.BytesIO(original)) as source, zipfile.ZipFile(removed, "w") as target:
        victim = "pf_general_multifront_exact_trial_delta_v6_4.json"
        for item in source.infolist():
            if item.filename != victim:
                target.writestr(item, source.read(item.filename))
    with pytest.raises(ValueError, match="inventory mismatch"):
        verify(removed.getvalue())
