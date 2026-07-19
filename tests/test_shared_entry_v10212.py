from types import SimpleNamespace

import pytest

import arrhenius_fracture.sharp_front_v10_2_12 as shared_entry


def _family(authorized=False):
    return SimpleNamespace(
        states=[object(), object()],
        metadata={"production_parameterization_allowed": authorized},
        audit_payload=lambda: {
            "schema": "v10.2.12_real_signed_state_resolved_2d_shielding_atlas",
            "kernel_radius_axis_policy": "disabled_constant_compatibility",
        },
    )


def _install_mocks(monkeypatch, family):
    monkeypatch.setattr(
        shared_entry.RealSigned2DShieldingKernelFamily,
        "from_json",
        classmethod(lambda cls, path: family),
    )
    monkeypatch.setattr(
        shared_entry.StateResolvedSignedBurgersTipEngine,
        "configure_state_resolved_physics",
        classmethod(lambda cls, supplied, mode, **kwargs: None),
    )
    monkeypatch.setattr(
        shared_entry._transport,
        "normalize_transport_mode",
        lambda value: "validated_scalar",
    )


def test_monotonic_and_fatigue_install_the_identical_v10212_engine(monkeypatch):
    family = _family()
    _install_mocks(monkeypatch, family)
    original_anisotropic = shared_entry._anisotropic.AnisotropicStochasticAvalancheTipEngine
    original_entry = shared_entry._entry74.AnisotropicStochasticAvalancheTipEngine
    seen = []

    def fake_main(args):
        seen.append(
            (
                shared_entry._anisotropic.AnisotropicStochasticAvalancheTipEngine
                is shared_entry.StateResolvedSignedBurgersTipEngine,
                shared_entry._entry74.AnisotropicStochasticAvalancheTipEngine
                is shared_entry.StateResolvedSignedBurgersTipEngine,
                "--fatigue-cycles" in args,
            )
        )
        return "ok"

    monkeypatch.setattr(shared_entry._transport, "main", fake_main)
    assert shared_entry.main(["--signed-kernel-family", "family.json"]) == "ok"
    assert shared_entry.main(
        ["--signed-kernel-family", "family.json", "--fatigue-cycles"]
    ) == "ok"
    assert seen == [(True, True, False), (True, True, True)]
    assert (
        shared_entry._anisotropic.AnisotropicStochasticAvalancheTipEngine
        is original_anisotropic
    )
    assert shared_entry._entry74.AnisotropicStochasticAvalancheTipEngine is original_entry


def test_parameter_campaign_fails_closed_without_real_atlas_authorization(monkeypatch):
    family = _family(authorized=False)
    _install_mocks(monkeypatch, family)
    monkeypatch.setenv("PARAMETER_CAMPAIGN", "1")
    with pytest.raises(SystemExit, match="has not authorized"):
        shared_entry.main(["--signed-kernel-family", "family.json"])
