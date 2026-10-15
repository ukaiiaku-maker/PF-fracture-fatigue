from pathlib import Path

import pytest

from arrhenius_fracture.parameter_dossier import build_parameter_dossier
from arrhenius_fracture.qualification_manifest import (
    assert_qualification_launch, capacity_preflight, manifest_payload,
)


ROOT = Path(__file__).resolve().parents[1]


def test_qualification_manifest_is_separate_and_fail_closed():
    payload = manifest_payload(ROOT)
    assert payload["production_execution_authorized"] is False
    assert payload["maximum_substantial_workers"] == 1
    assert {j["job_id"] for j in payload["jobs"]} >= {"Q-MECH-PATCH", "Q-PF-FEM-3LEVEL"}
    assert all(not j["production_job"] for j in payload["jobs"])
    mechanics = assert_qualification_launch("Q-MECH-PATCH")
    assert mechanics.evidence_status == "QUALIFIED_COMPACT_AFFINE_PATCH"
    assert mechanics.native_entrypoint == "scripts/qualify_mechanics_constraint_patch.py"
    with pytest.raises(PermissionError):
        assert_qualification_launch("Q-MECH-PATCH", production_execution_authorized=True)
    comparison = assert_qualification_launch("Q-PF-FEM-3LEVEL")
    assert comparison.evidence_status == "QUALIFIED_COMPACT_INDEPENDENT_SUPPORTS_REPRESENTATION_OFFSET"
    with pytest.raises(KeyError):
        assert_qualification_launch("P1_temperature_rate")


def test_capacity_preflight_accounts_for_peak_plus_reserve():
    free = capacity_preflight(ROOT, 0.01, reserve_GiB=0.0)
    assert free.admitted and free.free_GiB > 0
    blocked = capacity_preflight(ROOT, free.free_GiB + 1.0, reserve_GiB=0.0)
    assert not blocked.admitted and "DEFERRED_RESOURCE" in blocked.reason


def test_parameter_dossier_checks_every_parent_and_mechanism():
    dossier = build_parameter_dossier(ROOT, temperatures=(950.0, 1050.0), stress_fractions=(0.0, 0.5))
    assert {p["family"] for p in dossier["parents"]} == {"peak", "DBTT", "weakT", "ceramic"}
    for parent in dossier["parents"]:
        assert {m["mechanism"] for m in parent["mechanisms"]} == {
            "cleavage", "emission", "peierls_completion", "taylor_release"}
        assert all(m["time_unit"] == "s" for m in parent["mechanisms"] if "time_unit" in m)
    dbtt = next(p for p in dossier["parents"] if p["family"] == "DBTT")
    invalid_emit = [m for m in dbtt["mechanisms"] if m["mechanism"] == "emission"
                    and m["T_K"] == 1050.0 and m["domain_status"] == "UNSUPPORTED_RAW_SCALE"]
    assert invalid_emit
    assert dossier["selection_status"].startswith("parent examples")
