from __future__ import annotations

import numpy as np
import pytest

from arrhenius_fracture.crack_rebonding_kinetics_v10230 import CrackRebondingControls, RebondModelLevel
from arrhenius_fracture.crack_rebonding_v10230 import contact_diagnostics, patch_Q


def _cfg(model_level):
    return CrackRebondingControls(
        model_level=model_level,
        bond_activation_volume_m3=2.0e-29,
        bond_barrier_eV=0.3,
        rupture_activation_volume_m3=2.0e-29,
        rupture_barrier_eV=0.3,
    ).validate()


@pytest.mark.parametrize("K_s", [-2.0e7, -1.0e6, -1.0e-3])
def test_contact_proxy_only_produces_identity_generator_at_negative_R(K_s):
    cfg = _cfg(RebondModelLevel.CONTACT_PROXY_ONLY)
    Q = patch_Q(K_s, s_j_m=1.0e-7, r_contact_m=1.0e-8, cfg=cfg, T_K=300.0)
    assert np.allclose(Q, 0.0)


@pytest.mark.parametrize("K_s", [1.0e7, 5.0e6, 1.0e-3])
def test_rebond_off_produces_identity_generator(K_s):
    cfg = _cfg(RebondModelLevel.REBOND_OFF)
    Q = patch_Q(K_s, s_j_m=1.0e-7, r_contact_m=1.0e-8, cfg=cfg, T_K=300.0)
    assert np.allclose(Q, 0.0)


def test_contact_diagnostics_nonzero_at_negative_R_zero_at_positive():
    cfg = _cfg(RebondModelLevel.CONTACT_PROXY_ONLY)
    negative = contact_diagnostics(-2.0e7, s_j_m=1.0e-7, r_contact_m=1.0e-8, cfg=cfg)
    positive = contact_diagnostics(2.0e7, s_j_m=1.0e-7, r_contact_m=1.0e-8, cfg=cfg)
    assert negative["sigma_comp_Pa"] > 0.0
    assert positive["sigma_comp_Pa"] == 0.0


def test_contact_proxy_only_is_physically_identical_to_rebond_off():
    K_s_values = [-2.0e7, -1.0e6, 0.0, 1.0e6, 2.0e7]
    cfg_off = _cfg(RebondModelLevel.REBOND_OFF)
    cfg_proxy = _cfg(RebondModelLevel.CONTACT_PROXY_ONLY)
    for K_s in K_s_values:
        Q_off = patch_Q(K_s, s_j_m=2.0e-7, r_contact_m=1.0e-8, cfg=cfg_off, T_K=300.0)
        Q_proxy = patch_Q(K_s, s_j_m=2.0e-7, r_contact_m=1.0e-8, cfg=cfg_proxy, T_K=300.0)
        assert np.array_equal(Q_off, Q_proxy)
