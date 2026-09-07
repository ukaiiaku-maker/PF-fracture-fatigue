"""Real A_NATIVE production-engine construction for the v10.2.30 crack-
rebonding causal pilot v2.

Deliberately separate from ``tests/_crack_rebonding_engine_fixture.py``
(which builds a fallback DBTT candidate, ``DBTT_A0003837``, for fast unit
tests) -- the mission for this pilot explicitly requires the exact
reconstructed A_NATIVE row, not the test fixture. See
``artifacts/crack_rebonding_causal_pilot_v2/A_native_provenance.json`` (built
by ``scripts/build_v2_A_native_provenance.py``) for the full row-recovery
provenance and classification
(``A_NATIVE_REGISTRY_DETERMINISTICALLY_RECONSTRUCTED_FROM_QUALIFIED_INPUTS``).

Manifest construction path (see the provenance JSON's
``manifest_construction_pathway`` for why): builds
``parameter_registry_v9111.SelectedResponseOption`` directly from the
recovered row (bypassing ``select_option()``'s inapplicable legacy
``Tref_K==481.33`` campaign gate and Stage-3 canonical-candidate fingerprint
check, neither of which apply to A_NATIVE), then reuses
``write_compatibility_manifest``/``MaterialManifest.from_csv`` verbatim --
byte-identical to the real production conversion for every candidate that
IS routable through ``select_option``.

Everything else (kernel family, persistent-site config, generic solver
mechanics constants G_Pa/poisson/b_m, numerical substep controls) mirrors
``tests/_crack_rebonding_engine_fixture.py::build_real_engine`` exactly --
those are campaign-wide solver conventions, not candidate-specific data, and
this module has no reason to deviate from the already-qualified recipe for
them.
"""
from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from .anisotropic_emission_v10174 import (
    AnisotropicEmissionConfig,
    install_anisotropic_campaign_emission,
)
from .crack_rebonding_kinetics_v10230 import CrackRebondingControls
from .crack_rebonding_v10230 import install_crack_rebonding
from .material_manifest import MaterialManifest
from .parameter_registry_v9111 import SelectedResponseOption, write_compatibility_manifest, sha256_file
from .persistent_site_cyclic_energy_gated_corrected_v10230 import (
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine,
)
from .persistent_site_source_v10221 import PersistentSiteConfig
from .reduced_shared_state_v1023 import (
    SharedReducedConfig,
    _front_config,
    _mpz_config,
    _tip_config,
    install_campaign_calibrated_source,
)
from .signed_kernel_family_v10214 import ActiveOnlySigned2DShieldingKernelFamily, KernelState

PROVENANCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "artifacts/crack_rebonding_causal_pilot_v2/A_native_provenance.json"
)

MPZ_N_BINS = 80


def load_a_native_provenance() -> dict[str, Any]:
    return json.loads(PROVENANCE_PATH.read_text())


def _synthetic_family() -> ActiveOnlySigned2DShieldingKernelFamily:
    """Identical to tests/_crack_rebonding_engine_fixture.py::synthetic_family
    -- a generic, campaign-wide diagnostic shielding-kernel stand-in, not
    candidate-specific data."""
    source_x = (np.arange(40, dtype=float) + 0.5) * (100.0e-6 / 40.0)
    rows_I = np.vstack((1.0 + 2.0e4 * source_x, -0.5 + 1.0e4 * source_x))
    rows_II = 0.25 * rows_I
    states = []
    for state_id, extension, scale in (("E000", 0.0, 1.0), ("E200", 200.0e-6, 1.1)):
        states.append(
            KernelState(
                state_id=state_id,
                coordinates=np.asarray([1.0, 0.0, extension]),
                active_I=scale * rows_I,
                wake_I=np.zeros((2, 40)),
                active_II=scale * rows_II,
                wake_II=np.zeros((2, 40)),
                metadata={},
            )
        )
    family = ActiveOnlySigned2DShieldingKernelFamily(
        states=states,
        active_x_m=source_x,
        wake_x_m=source_x.copy(),
        activation_to_line_content=np.ones(2),
        source_capacity_bounds=np.asarray([[0.0, 1.0e5], [0.0, 1.0e5]]),
        fixed_kernel_assessment={"fixed_kernel_accepted": False},
        interpolation={"method": "inverse_distance", "neighbors": 2},
        metadata={"schema": "a_native_v2"},
        source_path="synthetic",
    )
    family._opening_boundary_policy = {"policy": "strict"}
    family._last_boundary_action = "none"
    family._last_observed_analytical_r_eff_over_r0 = 1.0
    family._last_observed_opening_strength_fraction = 0.0
    family._runtime_grid_binding = None
    return family


def build_a_native_manifest() -> tuple[MaterialManifest, dict[str, Any]]:
    """Build the real MaterialManifest for A_NATIVE from the recovered row,
    via the exact production row->manifest conversion
    (SelectedResponseOption -> write_compatibility_manifest ->
    MaterialManifest.from_csv), bypassing only select_option()'s two checks
    that are inapplicable to A_NATIVE (see module docstring). Returns
    (manifest, audit) where audit records the manifest CSV path/hash for
    the frozen-configuration record.
    """
    provenance = load_a_native_provenance()
    row = provenance["complete_active_material_row"]

    selected = SelectedResponseOption(
        option_key="A_NATIVE",
        candidate_id=row["candidate_id"],
        material_class=row["material_class"],
        role=row.get("role", ""),
        mechanism_summary=row.get("mechanism_summary", ""),
        validation_status=row.get("validation_status", ""),
        mpz_length_um=float(row["L_pz_um_recommended"]),
        mpz_n_bins=int(round(float(row["n_bins_recommended"]))),
        row=dict(row),
        registry_path="artifacts/crack_rebonding_causal_pilot_v2/A_native_provenance.json",
        registry_sha256=provenance["complete_row_sha256"],
    )
    if selected.mpz_n_bins != MPZ_N_BINS:
        raise ValueError(
            f"A_NATIVE row's own n_bins_recommended={selected.mpz_n_bins} does "
            f"not match the independently-required mpz_n_bins={MPZ_N_BINS}"
        )

    manifest_dir = Path(tempfile.gettempdir()) / "v10230_a_native_v2_manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = write_compatibility_manifest(
        selected, manifest_dir / "selected_material_manifest_A_NATIVE.csv"
    )
    manifest = MaterialManifest.from_csv(manifest_path)

    audit = {
        "option_key": "A_NATIVE",
        "candidate_id": selected.candidate_id,
        "mpz_length_um": selected.mpz_length_um,
        "mpz_n_bins": selected.mpz_n_bins,
        "compatibility_manifest_path": str(manifest_path),
        "compatibility_manifest_sha256": sha256_file(manifest_path),
        "source_row_sha256": provenance["complete_row_sha256"],
        "reconstruction_classification": provenance["reconstruction_classification"],
    }
    return manifest, audit


def build_a_native_engine(
    rebonding_cfg: CrackRebondingControls | None = None,
    *,
    retained_recovery_rate_s: float = 0.0,
) -> tuple[Any, dict[str, Any]]:
    """Construct the real production engine class via genuine __init__
    construction against the reconstructed A_NATIVE manifest (mpz_n_bins=80,
    mpz_length_um=50.0 -- the row's own L_pz_um_recommended). Mirrors
    tests/_crack_rebonding_engine_fixture.py::build_real_engine's recipe for
    every campaign-wide (non-candidate-specific) control. Returns
    (engine, manifest_audit).
    """
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.configure_state_resolved_physics(
        _synthetic_family(), transport_mode="validated_scalar"
    )
    CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine.configure_persistent_sites(
        PersistentSiteConfig(rho_site0_m2=5.0e12).validate()
    )

    manifest, manifest_audit = build_a_native_manifest()
    manifest = dataclasses.replace(manifest, retained_recovery_rate_s=retained_recovery_rate_s)

    cfg = SharedReducedConfig(
        mpz_length_um=manifest_audit["mpz_length_um"],
        mpz_n_bins=manifest_audit["mpz_n_bins"],
    ).validate()
    front_cfg = _front_config(cfg)
    front_cfg.sigma_cap = 1.0e12  # must be > 0 for this engine hierarchy
    mpz_cfg = _mpz_config(cfg)

    engine = CorrectedHazardEnergyGatedPersistentSiteCyclicTipEngine(
        front_cfg, manifest.cleavage, manifest.emission,
        float(cfg.G_Pa), float(cfg.poisson), float(cfg.b_m),
        manifest, mpz_cfg,
    )
    engine.tip_cfg = _tip_config(cfg, "full")
    engine.mpz.cfg.mobile_shield_fraction = float(cfg.mobile_shield_fraction)
    install_campaign_calibrated_source(
        engine.mpz, float(cfg.b_m), float(cfg.G_Pa), backstress_scale=1.0, refresh_scale=1.0
    )
    install_anisotropic_campaign_emission(
        engine.mpz,
        AnisotropicEmissionConfig(
            enabled=True, crystal_theta_deg=float(cfg.crystal_theta_deg),
            shared_forest_density=True, require_reliable_probe=True,
        ),
    )

    if rebonding_cfg is not None and rebonding_cfg.enabled:
        install_crack_rebonding(engine, rebonding_cfg)

    return engine, manifest_audit


__all__ = [
    "PROVENANCE_PATH",
    "MPZ_N_BINS",
    "load_a_native_provenance",
    "build_a_native_manifest",
    "build_a_native_engine",
]
