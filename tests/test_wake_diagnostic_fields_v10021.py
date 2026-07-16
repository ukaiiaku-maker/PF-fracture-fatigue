from __future__ import annotations

from arrhenius_fracture.material_manifest import MaterialManifest, default_manifest_path
from arrhenius_fracture.sharp_front_v10_1 import _diagnostics_with_csv_aliases
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState


def test_persistent_wake_diagnostics_and_csv_aliases():
    manifest = MaterialManifest.from_csv(default_manifest_path("weakT"))
    state = UnifiedMPZState(
        manifest,
        MPZConfig(length_m=100e-6, n_bins=200, wake_length_m=100e-6, wake_shielding=True),
    )
    state.mobile[:, :2] = 1.0
    state.retained[:, :2] = 2.0
    state.advance(5e-6)
    d = _diagnostics_with_csv_aliases(state, 160e9, 0.28, 2.74e-10, 1e-6)
    assert d["mpz_wake_retained_count"] > 0.0
    assert d["mpz_wake_retained_total"] == d["mpz_wake_retained_count"]
    assert d["mpz_K_shield_Pa_sqrt_m"] == d["mpz_total_K_shield_Pa_sqrt_m"]
    assert d["mpz_local_slip_count"] >= 0.0
