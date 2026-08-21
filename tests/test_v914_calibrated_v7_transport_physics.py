from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_v914_calibrated_v7_transport_physics import (  # noqa: E402
    CHANGED_FIELD,
    DERIVED_FILENAME,
    PARENT_SHA256,
    PROVENANCE_FILENAME,
    build,
    changed_leaf_paths,
)


V914 = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "Arrhenius_FEM_CZM_MPZ_v9_14_cyclic_fatigue_knee_search"
)
PARENT = (
    V914
    / "runs/v9_13_v10222_rcurve_alpha0p95_all50_v1/"
    "v9_13_R_curve_calibrated_common_physics.json"
)
LEGACY = V914 / "mpz_v9_13_persistent_sites_common_physics.json"


def test_derived_transport_physics_changes_only_velocity_scale(tmp_path):
    provenance = build(PARENT, tmp_path, LEGACY)
    parent = json.loads(PARENT.read_text())
    derived = json.loads((tmp_path / DERIVED_FILENAME).read_text())

    assert provenance["parent"]["sha256"] == PARENT_SHA256
    assert changed_leaf_paths(parent, derived) == [CHANGED_FIELD]
    assert parent["common_physics"]["mobile_transport_velocity_scale"] == 0.0
    assert derived["common_physics"]["mobile_transport_velocity_scale"] == 1.0
    assert provenance["derived"]["all_other_parent_fields_preserved"] is True
    assert (tmp_path / PROVENANCE_FILENAME).is_file()


def test_builder_output_is_byte_deterministic(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    build(PARENT, first, LEGACY)
    build(PARENT, second, LEGACY)
    assert (first / DERIVED_FILENAME).read_bytes() == (
        second / DERIVED_FILENAME
    ).read_bytes()
    assert (first / PROVENANCE_FILENAME).read_bytes() == (
        second / PROVENANCE_FILENAME
    ).read_bytes()
