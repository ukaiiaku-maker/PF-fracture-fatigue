from types import SimpleNamespace

import pytest

from arrhenius_fracture.branch_scale_identity_v11 import (
    BranchScaleIdentity, resolve_branch_scale_identity,
)


def _identity(**changes):
    values = dict(
        physical_process_zone_length_m=50e-6, branch_handoff_length_m=50e-6,
        local_J_contour_radius_m=3e-6, interaction_integral_length_m=2e-6,
        tip_h_fine_m=1e-6, actual_local_hbar_m=.8e-6,
        event_length_da_phys_m=5e-6,
        physical_process_zone_source="mechanical.process_zone_length_m",
        branch_handoff_source="mechanical.process_zone_length_m",
        local_J_contour_source="live_selected", interaction_integral_source="mechanical.interaction",
        tip_h_fine_source="args.tip_h_fine", event_length_source="args.da_phys",
    )
    values.update(changes)
    return BranchScaleIdentity(**values)


@pytest.mark.parametrize("source", ["runtime_args.tip_h_fine", "live_J_contour_radius", "interaction_integral_length"])
def test_handoff_rejects_numerical_scale_identity_even_if_value_matches(source):
    with pytest.raises(ValueError, match="physical process-zone identity"):
        _identity(branch_handoff_source=source)


def test_handoff_rejects_value_different_from_physical_process_zone():
    with pytest.raises(ValueError, match="must equal"):
        _identity(branch_handoff_length_m=1e-6)


def test_runtime_fallback_uses_promoted_mpz_not_legacy_L_pz(monkeypatch):
    monkeypatch.delenv("MECHANICAL_CONFIG", raising=False)
    args = SimpleNamespace(
        mpz_length_um=50.0, L_pz=1e-6, rJ=None,
        tip_h_fine=1e-6, da_phys=5e-6,
    )
    mesh = SimpleNamespace(hbar_tip=.8e-6, hbar=2e-6)
    identity = resolve_branch_scale_identity(args, mesh)
    assert identity.physical_process_zone_length_m == pytest.approx(50e-6)
    assert identity.branch_handoff_length_m == pytest.approx(50e-6)
    assert identity.local_J_contour_radius_m == pytest.approx(1e-6)
    assert identity.interaction_integral_length_m == pytest.approx(1e-6)
