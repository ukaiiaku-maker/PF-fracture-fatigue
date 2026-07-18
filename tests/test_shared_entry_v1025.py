import numpy as np

import arrhenius_fracture.sharp_front_v10_2_5 as shared_entry
from arrhenius_fracture.signed_burgers_shared_v1025 import SignedShieldingKernel


def _kernel():
    return SignedShieldingKernel(
        active_kernel=np.zeros((2, 1)),
        wake_kernel=np.zeros((2, 0)),
        active_x_m=np.array([0.5]),
        wake_x_m=np.zeros(0),
        activation_to_line_content=np.ones(2),
        source_capacity_bounds=np.array([[0.0, 10.0], [0.0, 10.0]]),
        metadata={},
        source_path="unit-test.json",
    )


def test_monotonic_and_fatigue_install_identical_engine(monkeypatch):
    fake_kernel = _kernel()
    monkeypatch.setattr(
        shared_entry.SignedShieldingKernel,
        "from_json",
        classmethod(lambda cls, path: fake_kernel),
    )
    monkeypatch.setattr(
        shared_entry.SignedBurgersAnisotropicTipEngine,
        "configure_signed_physics",
        classmethod(lambda cls, supplied, mode: None),
    )
    monkeypatch.setattr(
        shared_entry._transport,
        "normalize_transport_mode",
        lambda value: "validated_scalar",
    )
    seen = []

    def fake_main(args):
        seen.append(
            (
                shared_entry._anisotropic.AnisotropicStochasticAvalancheTipEngine
                is shared_entry.SignedBurgersAnisotropicTipEngine,
                shared_entry._entry74.AnisotropicStochasticAvalancheTipEngine
                is shared_entry.SignedBurgersAnisotropicTipEngine,
                "--fatigue-cycles" in args,
            )
        )
        return "ok"

    monkeypatch.setattr(shared_entry._transport, "main", fake_main)
    assert shared_entry.main(["--signed-shielding-kernel", "kernel.json"]) == "ok"
    assert shared_entry.main(
        ["--signed-shielding-kernel", "kernel.json", "--fatigue-cycles"]
    ) == "ok"
    assert seen == [(True, True, False), (True, True, True)]
