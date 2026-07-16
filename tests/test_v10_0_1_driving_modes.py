from __future__ import annotations

import numpy as np
import pytest

from arrhenius_fracture import plasticity
from arrhenius_fracture import sharp_front_v10_1 as driver


def test_prepare_args_adds_abs_forward_only_when_requested():
    args, bulk, jmode = driver._prepare_args([
        "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "abs_forward",
        "--material-class", "ceramic",
    ])
    assert bulk == "tip_only"
    assert jmode == "abs_forward"
    assert "--allow-abs-directional-J" in args
    assert "--bulk-plasticity-mode" not in args
    assert "--directional-j-mode" not in args


def test_tip_only_update_is_transactional_noop():
    ep = np.arange(12.0).reshape(3, 4)
    rho = np.arange(4.0) + 10.0
    sigma = np.zeros((3, 4))
    ep2, rho2, dot, info = driver._tip_only_update_plasticity(
        ep, rho, sigma, None, 700.0, 8.4, None, None, return_info=True
    )
    assert np.array_equal(ep2, ep)
    assert np.array_equal(rho2, rho)
    assert np.all(dot == 0.0)
    assert np.all(info["dWp_accepted_gp"] == 0.0)
    assert ep2 is not ep
    assert rho2 is not rho


def test_main_restores_bulk_update(monkeypatch):
    seen = {}

    def fake_main(args):
        seen["args"] = list(args)
        seen["patched"] = plasticity.update_plasticity is driver._tip_only_update_plasticity
        return 7

    original = plasticity.update_plasticity
    monkeypatch.setattr(driver.sharp_front, "main", fake_main)
    result = driver.main([
        "--material-class", "weakT",
        "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed",
    ])
    assert result == 7
    assert seen["patched"] is True
    assert "--allow-abs-directional-J" not in seen["args"]
    assert plasticity.update_plasticity is original


def test_unparameterized_full_field_is_blocked():
    with pytest.raises(SystemExit, match="not yet mapped"):
        driver.main([
            "--material-class", "DBTT",
            "--bulk-plasticity-mode", "full_field",
        ])
