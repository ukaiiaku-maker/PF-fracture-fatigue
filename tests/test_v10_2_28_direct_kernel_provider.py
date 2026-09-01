from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from arrhenius_fracture.kernel_configuration_v10227 import MechanicalKernelConfiguration
from arrhenius_fracture.kernel_normalization_contract_v10228 import (
    DEFAULT_BURGERS_M,
    DEFAULT_KINETIC_PACKET_LENGTH_M,
    KernelNormalizationContract,
)
from arrhenius_fracture.kernel_resolver_v10228 import (
    BUILD_SCHEMA,
    VALIDATION_SCHEMA,
    _validate_direct_evidence,
)
from arrhenius_fracture.prescribed_geometry_kernel_v10228 import (
    plan_explicit_prescribed_geometry_anchors,
    plan_prescribed_geometry_anchors,
    prescribed_crack_direction,
)

ROOT = Path(__file__).resolve().parents[1]


def _configuration(**overrides) -> MechanicalKernelConfiguration:
    payload = MechanicalKernelConfiguration().canonical_payload()
    payload.update(overrides)
    extra = dict(payload.get("extra", {}))
    extra.update(
        {
            "prescribed_crack_path_policy": "forward_100_cleavage_trace",
            "burgers_m": DEFAULT_BURGERS_M,
            "kinetic_packet_length_m": DEFAULT_KINETIC_PACKET_LENGTH_M,
        }
    )
    payload["extra"] = extra
    return MechanicalKernelConfiguration.from_mapping(payload)


def test_normalization_contract_preserves_current_production_defaults():
    from arrhenius_fracture.config import ElasticProperties
    from arrhenius_fracture.kinetic_tip_cell import KineticTipConfig

    contract = KernelNormalizationContract().validate()
    assert contract.burgers_m == pytest.approx(ElasticProperties().b, rel=0.0, abs=0.0)
    assert contract.kinetic_packet_length_m == pytest.approx(
        KineticTipConfig().packet_length_m, rel=0.0, abs=0.0
    )
    assert contract.activation_to_line_content == pytest.approx(
        KineticTipConfig().packet_length_m / ElasticProperties().b
    )


def test_anchor_plan_is_geometric_and_seed_free():
    configuration = _configuration(theta_deg=30.0, atlas_anchor_spacing_m=200.0e-6)
    anchors = plan_prescribed_geometry_anchors(configuration, 1000.0)
    assert anchors[0].extension_m == 0.0
    assert anchors[-1].extension_m >= 1000.0e-6
    assert all(right.extension_m > left.extension_m for left, right in zip(anchors, anchors[1:]))
    direction = np.asarray(anchors[0].crack_direction)
    assert direction[0] == pytest.approx(np.cos(np.deg2rad(30.0)))
    assert direction[1] == pytest.approx(np.sin(np.deg2rad(30.0)))
    assert all(anchor.crack_direction == anchors[0].crack_direction for anchor in anchors)


def test_explicit_anchor_plan_preserves_append_only_irregular_levels():
    configuration = _configuration(theta_deg=40.0, da_phys_m=5.0e-6)
    levels = (0.0, 200.0, 400.0, 415.0, 420.0, 425.0, 600.0, 745.0)
    anchors = plan_explicit_prescribed_geometry_anchors(configuration, levels)
    assert [anchor.state_id for anchor in anchors] == [
        "E0000000",
        "E0000200",
        "E0000400",
        "E0000415",
        "E0000420",
        "E0000425",
        "E0000600",
        "E0000745",
    ]
    assert [1.0e6 * anchor.extension_m for anchor in anchors] == pytest.approx(levels)


def test_explicit_anchor_plan_rejects_off_quantum_and_unsorted_levels():
    configuration = _configuration(da_phys_m=5.0e-6)
    with pytest.raises(ValueError, match="align"):
        plan_explicit_prescribed_geometry_anchors(configuration, (0.0, 421.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        plan_explicit_prescribed_geometry_anchors(configuration, (0.0, 420.0, 415.0))


def test_orientation_changes_direction_and_mechanical_fingerprint():
    c0 = _configuration(theta_deg=0.0)
    c18 = _configuration(theta_deg=18.0)
    c30 = _configuration(theta_deg=30.0)
    assert c0.fingerprint() != c18.fingerprint() != c30.fingerprint()
    assert not np.allclose(prescribed_crack_direction(c0), prescribed_crack_direction(c30))


def test_specimen_or_notch_change_changes_fingerprint():
    base = _configuration()
    larger = _configuration(specimen_length_x_m=base.specimen_length_x_m * 1.1)
    longer_notch = _configuration(initial_crack_length_m=base.initial_crack_length_m * 1.1)
    assert base.fingerprint() != larger.fingerprint()
    assert base.fingerprint() != longer_notch.fingerprint()


def test_direct_provider_source_has_no_production_kinetics_imports():
    source = (ROOT / "arrhenius_fracture" / "prescribed_geometry_kernel_v10228.py").read_text()
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
    forbidden_fragments = (
        "sharp_front",
        "stochastic",
        "hazard",
        "persistent_site_source",
        "fractional_moving_frame",
        "state_resolved_signed_engine",
        "moving_pz",
    )
    assert not [
        name for name in imported if any(fragment in name for fragment in forbidden_fragments)
    ]


def test_direct_promotion_evidence_replaces_fixed_point(tmp_path: Path):
    audit = {
        "file_sha256": "family-sha",
        "physics_fingerprint": "physics-sha",
    }
    configuration_fingerprint = "configuration-sha"
    build = {
        "schema": BUILD_SCHEMA,
        "configuration_fingerprint": configuration_fingerprint,
        "family_sha256": audit["file_sha256"],
        "family_physics_fingerprint": audit["physics_fingerprint"],
        "direct_provider_validated": True,
        "production_parameterization_promotion_allowed": True,
        "prior_kernel_family_required": False,
        "stochastic_trajectory_required": False,
        "production_physics_modified": False,
    }
    validation = {
        "schema": VALIDATION_SCHEMA,
        "passed": True,
        "configuration_fingerprint": configuration_fingerprint,
        "family_sha256": audit["file_sha256"],
        "family_physics_fingerprint": audit["physics_fingerprint"],
        "direct_prescribed_geometry": True,
        "material_option_independent_by_construction": True,
        "hazard_seed_independent_by_construction": True,
        "prior_family_independent_by_construction": True,
        "stochastic_event_independent_by_construction": True,
        "fracture_hazard_imported_or_invoked": False,
        "source_emission_imported_or_invoked": False,
        "moving_process_zone_imported_or_invoked": False,
        "load_invariance_passed": True,
        "positive_negative_multi_amplitude_linearity_passed": True,
    }
    (tmp_path / "kernel_build_manifest.json").write_text(json.dumps(build))
    (tmp_path / "direct_kernel_validation_manifest.json").write_text(
        json.dumps(validation)
    )
    result = _validate_direct_evidence(tmp_path, audit, configuration_fingerprint)
    assert result["direct_provider_validated"] is True
    assert not (tmp_path / "kernel_self_consistency_manifest.json").exists()
