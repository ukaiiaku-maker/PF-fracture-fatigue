from types import SimpleNamespace

import pytest

import arrhenius_fracture.sharp_front_v10_2_6 as shared_entry


def _family(authorized=False):
    return SimpleNamespace(
        states=[object(), object()],
        interpolation={"method": "inverse_distance"},
        metadata={"production_parameterization_allowed": authorized},
        audit_payload=lambda: {
            "schema": "v10.2.6_state_resolved_signed_kernel_family"
        },
    )


def _install_mocks(monkeypatch, family):
    monkeypatch.setattr(
        shared_entry.StateResolvedSignedShieldingKernelFamily,
        "from_json",
        classmethod(lambda cls, path: family),
    )
    monkeypatch.setattr(
        shared_entry.StateResolvedSignedBurgersTipEngine,
        "configure_state_resolved_physics",
        classmethod(lambda cls, supplied, mode: None),
    )
    monkeypatch.setattr(
        shared_entry._transport,
        "normalize_transport_mode",
        lambda value: "validated_scalar",
    )


def test_monotonic_and_fatigue_install_identical_state_resolved_engine(monkeypatch):
    family = _family()
    _install_mocks(monkeypatch, family)
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


def test_parameter_campaign_fails_closed_without_family_authorization(monkeypatch):
    family = _family(authorized=False)
    _install_mocks(monkeypatch, family)
    monkeypatch.setenv("PARAMETER_CAMPAIGN", "1")
    with pytest.raises(SystemExit, match="has not authorized"):
        shared_entry.main(["--signed-kernel-family", "family.json"])
