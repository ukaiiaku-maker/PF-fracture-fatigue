from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from arrhenius_fracture.config import ElasticProperties
from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.sharp_front import FrontConfig, default_cleavage_barrier, default_emission_barrier
from arrhenius_fracture.unified_front import UnifiedMPZFrontEngine
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState


def engine(class_name: str) -> UnifiedMPZFrontEngine:
    mat = ElasticProperties()
    f = FrontConfig(); f.r0 = 1e-6; f.sigma_cap = 0; f.da = 5e-6; f.L_pz = 100e-6
    return UnifiedMPZFrontEngine(
        f, default_cleavage_barrier(), default_emission_barrier(mat.b),
        mat.G, mat.nu, mat.b,
        MaterialManifest.from_csv(default_manifest_path(class_name)),
        MPZConfig(length_m=100e-6, n_bins=200, wake_length_m=100e-6),
    )


def total_state(s: UnifiedMPZState) -> float:
    return float(
        np.sum(s.mobile) + np.sum(s.retained) + np.sum(s.wake_mobile)
        + np.sum(s.wake_retained)
    )


def test_manifests_are_exact_promoted_classes():
    expected = {
        "ceramic": "ceramic_restart02_candidate00",
        "weakT": "weakT_restart00_candidate00",
        "DBTT": "DBTT_restart01_candidate05",
    }
    for name, candidate in expected.items():
        m = MaterialManifest.from_csv(default_manifest_path(name))
        assert m.name == name
        assert m.candidate_id == candidate
        assert m.source_sites_per_system > 0
        assert m.source_refresh_length_m > 0


def test_matched_stress_emission_ordering():
    K = {"ceramic": 11.820868, "weakT": 16.949365, "DBTT": 29.197177}
    rates = {}
    for name in K:
        e = engine(name)
        sigma = e.sigma_tip(K[name] * 1e6)
        rates[name] = e.lambda_emit(sigma, 700)[0]
    assert rates["ceramic"] < rates["weakT"] < rates["DBTT"]
    assert rates["weakT"] / max(rates["ceramic"], 1e-300) > 1e10


def test_one_renewal_per_geometry_transaction():
    e = engine("ceramic")
    e.B = 3.25
    out = e.step(0.0, 700.0, 0.0)
    assert out["fired"]
    assert out["n_fire"] == 1
    assert e.n_adv == 1
    assert math.isclose(e.B, 2.25, rel_tol=0, abs_tol=1e-12)


def test_active_state_moves_to_persistent_wake_conservatively():
    e = engine("weakT")
    e.mpz.mobile[:, :4] = 1.0
    e.mpz.retained[:, :4] = 2.0
    before = total_state(e.mpz)
    audit = e.mpz.advance(1.0e-6)
    after = total_state(e.mpz)
    discarded = e.mpz.wake_discarded_mobile_total + e.mpz.wake_discarded_retained_total
    assert math.isclose(before, after + discarded, rel_tol=1e-12, abs_tol=1e-12)
    assert audit["wake_mobile_postcommit"] > 0.0
    assert audit["wake_retained_postcommit"] > 0.0


def test_branch_split_does_not_duplicate_state():
    e = engine("DBTT")
    e.mpz.mobile[:, :3] = 1.0
    e.mpz.retained[:, :3] = 2.0
    e.mpz.wake_retained[:, :3] = 3.0
    before = total_state(e.mpz)
    child = e.clone_split(0.35)
    assert math.isclose(before, total_state(e.mpz) + total_state(child.mpz), rel_tol=1e-12)
    assert e.mpz.site_capacity.sum() + child.mpz.site_capacity.sum() > 0


def test_wake_ablation_changes_only_wake_shielding():
    e = engine("weakT")
    e.mpz.wake_retained[:, 0] = 1.0
    on = e.mpz.wake_K_shielding(e.G, e.nu, e.b)
    active = e.mpz.active_K_shielding(e.G, e.nu, e.b)
    e.mpz.cfg.wake_shielding = False
    off = e.mpz.wake_K_shielding(e.G, e.nu, e.b)
    assert on > 0.0
    assert off == 0.0
    assert e.mpz.active_K_shielding(e.G, e.nu, e.b) == active
