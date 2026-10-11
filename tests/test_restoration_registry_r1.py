from types import SimpleNamespace

import numpy as np

from arrhenius_fracture.restoration_registry_r1 import (
    BINDINGS,
    RestorationModelIdentity,
    live_geometry_source_void_contract,
    registry_payload,
    resolve_restoration_identity,
)


def test_three_model_identities_are_explicit_and_reduced_is_the_only_default():
    assert set(BINDINGS) == set(RestorationModelIdentity)
    assert [item.identity for item in BINDINGS.values() if item.default] == [
        RestorationModelIdentity.RESTORED_REDUCED_REFERENCE
    ]
    seam = resolve_restoration_identity("SEAM_DIAGNOSTIC_V4_10")
    assert seam.default is False
    assert seam.mechanics_strategy == "linear_elastic_circular_seam_cut"
    assert registry_payload()["no_mechanics_mixing"] is True
    reduced_void = resolve_restoration_identity("REDUCED_VOID_V3_REFERENCE")
    assert reduced_void.default is False
    assert reduced_void.cavity_owner == "reduced_fracture_v3 physical cavity state"
    assert "not calibrated 2-D transfer" in reduced_void.physical_scope


def test_contract_reads_native_persistent_geometry_once_and_keeps_void_radius_distinct(monkeypatch):
    state = SimpleNamespace(
        _persistent_site_cfg=object(),
        mobile_positive=np.array([[2.0]]),
        mobile_negative=np.array([[1.0]]),
        retained_positive=np.array([[3.0]]),
        retained_negative=np.array([[0.5]]),
        accumulated_slip_positive=np.array([[5.0]]),
        accumulated_slip_negative=np.array([[2.0]]),
        returned_slip_positive=np.array([[1.5]]),
        returned_slip_negative=np.array([[0.25]]),
        wake_mobile=np.array([[0.75]]),
        wake_retained=np.array([[1.25]]),
        cumulative_source_slip_wake_transfer=np.array([[0.5]]),
        void_state=SimpleNamespace(cavities=(SimpleNamespace(radius_m=8.0e-6),)),
    )
    native = {
        "tip_radius_m": 1.2e-6,
        "front_width_m": 8.0e-6,
        "rho_width_m2": 8.0e12,
        "source_area_m2": 9.6e-12,
        "multiplicity_per_system": 1.2e5,
    }
    monkeypatch.setattr(
        "arrhenius_fracture.persistent_site_source_v10221._source_geometry",
        lambda owner: native,
    )
    audit = live_geometry_source_void_contract(
        state, identity=RestorationModelIdentity.RESTORED_REDUCED_REFERENCE
    )
    assert audit["source_geometry"] == native
    assert audit["effective_kinetic_tip_radius_m"] == 1.2e-6
    assert audit["cavity_radii_m"] == [8.0e-6]
    assert audit["effective_kinetic_tip_radius_m"] != audit["cavity_radii_m"][0]
    assert audit["signed_state"]["accumulated_slip_positive"] == 5.0
    assert audit["signed_state"]["returned_slip_positive"] == 1.5
    assert audit["contribution_ownership"]["mixed_seam_plus_persistent_radius"] is False


def test_contract_does_not_relabel_sharp_spatial_geometry_as_rounded_shape():
    state = SimpleNamespace(r_tip_m=2.0e-6, void_state=None)
    audit = live_geometry_source_void_contract(
        state, identity=RestorationModelIdentity.FULL_SPATIAL_REFERENCE
    )
    assert audit["effective_kinetic_tip_radius_m"] == 2.0e-6
    assert "not a rounded FEM notch" in BINDINGS[
        RestorationModelIdentity.FULL_SPATIAL_REFERENCE
    ].geometry_owner
