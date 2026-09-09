import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "analysis_outputs/pf_current_source_general_multifront_v6_3"
ARCHIVE = OUT / "Archive_PF_GENERAL_MULTIFRONT_V6_3_COMPLETE.zip"


REQUIRED = {
    "PF_CURRENT_SOURCE_GENERAL_MULTIFRONT_V6_3.md",
    "pf_general_multifront_v6_3_source_provenance.json",
    "pf_general_multifront_stateful_hook_context_v6_3.json",
    "pf_general_multifront_hook_lifecycle_v6_3.csv",
    "pf_general_multifront_event_finalization_v6_3.csv",
    "pf_general_multifront_daughter_rng_identity_v6_3.json",
    "pf_general_multifront_exact_trial_cache_v6_3.json",
    "pf_general_multifront_transaction_interruption_v6_3.json",
    "pf_general_multifront_native_N1_N2_restore_v6_3.json",
    "pf_general_multifront_stateful_production_dry_run_v6_3.json",
}


def verify_archive(data: bytes) -> None:
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        manifest = json.loads(archive.read("MANIFEST.json"))
        members = manifest["members"]
        if set(archive.namelist()) != set(members) | {"MANIFEST.json"}:
            raise ValueError("archive member inventory mismatch")
        for name, expected in members.items():
            value = archive.read(name)
            if hashlib.sha256(value).hexdigest() != expected["sha256"]:
                raise ValueError(f"archive member hash mismatch: {name}")
            if len(value) != expected["size_bytes"]:
                raise ValueError(f"archive member size mismatch: {name}")


def test_v6_3_required_products_and_decisions(historical_product):
    assert REQUIRED.issubset({path.name for path in OUT.iterdir()})
    context = json.loads((OUT / "pf_general_multifront_stateful_hook_context_v6_3.json").read_text())
    dry = json.loads((OUT / "pf_general_multifront_stateful_production_dry_run_v6_3.json").read_text())
    native = json.loads((OUT / "pf_general_multifront_native_N1_N2_restore_v6_3.json").read_text())
    assert context["qualification"] == "QUALIFIED"
    assert context["actual_engine_interval"]["update_count_after"] == context["actual_engine_interval"]["update_count_before"] + 1
    assert context["mechanics_solve_count"] == context["provider_solve_count"] == 0
    assert dry["qualification"] == "PASS" and not dry["trajectory_execution_authorized"]
    assert dry["provider_solve_count"] == dry["mechanics_solve_count"] == 0
    assert native["cases"]["branch_disabled_N1_control_restore"]["native_branching_mode"] == "disabled"
    assert native["cases"]["enabled_N2_terminal_restore"]["active_front_count"] == 2


def test_v6_3_archive_positive_and_negative_verification(historical_product):
    original = ARCHIVE.read_bytes()
    verify_archive(original)
    source = io.BytesIO(original)
    removed = io.BytesIO()
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(removed, "w") as zout:
        omitted = "pf_general_multifront_exact_trial_cache_v6_3.json"
        for info in zin.infolist():
            if info.filename != omitted:
                zout.writestr(info, zin.read(info.filename))
    with pytest.raises(ValueError, match="inventory"):
        verify_archive(removed.getvalue())

    source = io.BytesIO(original)
    modified = io.BytesIO()
    with zipfile.ZipFile(source) as zin, zipfile.ZipFile(modified, "w") as zout:
        target = "pf_general_multifront_transaction_interruption_v6_3.json"
        for info in zin.infolist():
            value = zin.read(info.filename)
            if info.filename == target:
                value += b"modified"
            zout.writestr(info, value)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_archive(modified.getvalue())
