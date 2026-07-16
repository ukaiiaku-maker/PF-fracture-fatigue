from __future__ import annotations

from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState


def test_diagnostics_expose_active_wake_and_total_fields():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    state = UnifiedMPZState(
        manifest,
        MPZConfig(length_m=100e-6, n_bins=200, wake_length_m=100e-6, wake_shielding=True),
    )
    state.mobile[:, :2] = 1.0
    state.retained[:, :2] = 2.0
    state.advance(5e-6)
    d = state.diagnostics(160e9, 0.28, 2.74e-10, 1e-6)
    for key in (
        "mpz_active_K_shield_Pa_sqrt_m",
        "mpz_wake_K_shield_Pa_sqrt_m",
        "mpz_total_K_shield_Pa_sqrt_m",
        "mpz_wake_mobile_count",
        "mpz_wake_retained_count",
    ):
        assert key in d
    assert d["mpz_total_K_shield_Pa_sqrt_m"] == (
        d["mpz_active_K_shield_Pa_sqrt_m"] + d["mpz_wake_K_shield_Pa_sqrt_m"]
    )
