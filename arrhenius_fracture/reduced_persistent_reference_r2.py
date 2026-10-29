"""R2 persistent/reversible reduced reference and resolved runtime receipt.

The historical v10.2.3 finite-source replay remains unchanged in
``reduced_shared_state_v1023``.  This module constructs the intended current
persistent source and reversible signed transport on the same production MPZ
state, while replacing only the external mechanics provider with prescribed K.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
from types import MethodType
from typing import Any

import numpy as np

from .material_manifest import KB_EV_PER_K, MaterialManifest
from .kinetic_tip_cell import KineticMovingTipFrontEngine
from .persistent_site_monotonic_reversible_v10230 import (
    MonotonicReversiblePersistentSiteTipEngine,
)
from .persistent_site_reversible_transport_v10230 import (
    install_reversible_transport, reversible_binding_receipt,
)
from .persistent_site_source_v10221 import PersistentSiteConfig, _source_geometry
from .persistent_site_bracket_fix_v10221 import (
    install_backstress_complementarity_fix,
)
from .reduced_shared_state_v1023 import (
    SharedReducedConfig, _front_config, _mpz_config, _tip_config,
)


MODEL_ID = "r2.restored_reduced_persistent_reversible/1"
HISTORICAL_REPLAY_ID = "v10.2.3_historical_finite_source_replay"
TAYLOR_LAW_ID = "MEAN_FORWARD_MINUS_ZERO_RATE_OVER_DENSITY_ORDER"
DEFAULT_MATERIAL = (
    Path(__file__).resolve().parents[1]
    / "integration_evidence" / "option_completion_v2"
    / "mechanics_base_current" / "event" / "current"
    / "selected_material_manifest_v10_2_21.csv"
)
DEFAULT_SELECTION = DEFAULT_MATERIAL.with_name("v10_2_21_parameter_selection.json")
DEFAULT_KERNEL_FAMILY = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "v10_2_28_kernel_cache/"
    "4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/"
    "family.json"
)
DEFAULT_KERNEL_SHA256 = "5aa74370e6419104684c52bbcf93323f905d2752d1bba59252f9b0b35c77e07c"


def _restore_protected_continuum_alias() -> None:
    """Undo a legacy entry-module import side effect after class resolution.

    Importing the v10.2.14 engine also imports the historical v10.1.7.3 entry,
    which intentionally rebinds the protected continuum-engine module global.
    The persistent engine class has already been resolved above and does not
    require that process-global alias.  Restoring the package's protected
    default keeps this library adapter import-order neutral for other studies.
    """
    from . import continuum_source_tip
    from .separated_source_tip import SeparatedSourceKineticTipEngine

    continuum_source_tip.ContinuumSourceKineticTipEngine = (
        SeparatedSourceKineticTipEngine
    )


_restore_protected_continuum_alias()


class ReducedPersistentReversibleTipEngine(
    MonotonicReversiblePersistentSiteTipEngine
):
    """Production constitutive engine with an explicit reduced drive provider."""

    def _plastic_half_step(self, dt, temperature_K, opening_stress_Pa):
        # The reduced K-only closure has no resolved tensor and therefore owns
        # this explicit positive-drive placeholder.  Spatial subclasses and
        # complete-input replay set the policy false and preserve the signed
        # tensor supplied by their mechanics operator.
        if getattr(self, "_r2_synthetic_drive_setup", True):
            factors = np.asarray(self.mpz._anisotropic_drive_factors, dtype=float)
            self.mpz._anisotropic_tau_signed_Pa = np.where(
                factors > 0.0, 1.0, 0.0
            )
            self.mpz._anisotropic_drive_reliable = True
        if getattr(self, "_qste_reduced_internal_stress_enabled", False):
            from .qste340tm_reduced_internal_stress import update_reduced_internal_stress
            update_reduced_internal_stress(self.mpz)
        return KineticMovingTipFrontEngine._plastic_half_step(
            self, dt, temperature_K, opening_stress_Pa
        )


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _object_sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def _method_identity(value: Any) -> dict[str, Any]:
    function = getattr(value, "__func__", value)
    module = inspect.getmodule(function)
    path = None if module is None or not getattr(module, "__file__", None) else Path(module.__file__).resolve()
    return {
        "callable": f"{getattr(function, '__module__', type(function).__module__)}."
                    f"{getattr(function, '__qualname__', type(function).__qualname__)}",
        "file": None if path is None else str(path),
        "file_sha256": None if path is None else _sha256(path),
    }


def load_persistent_config(path: str | Path = DEFAULT_SELECTION) -> PersistentSiteConfig:
    payload = json.loads(Path(path).read_text())
    raw = payload.get("persistent_site_config")
    if not isinstance(raw, dict):
        raise ValueError("selection audit does not contain persistent_site_config")
    return PersistentSiteConfig(**raw).validate()


def _stable_forward_minus_zero(surface, stress_Pa, temperature_K: float) -> np.ndarray:
    stress = np.maximum(np.asarray(stress_Pa, dtype=float), 0.0)
    zero = np.zeros_like(stress)
    driven_barrier = np.asarray(surface.values_eV(stress, temperature_K), dtype=float)
    zero_barrier = np.asarray(surface.values_eV(zero, temperature_K), dtype=float)
    kT = max(KB_EV_PER_K * float(temperature_K), 1.0e-30)
    zero_rate = float(surface.attempt_frequency_s) * np.exp(
        np.clip(-zero_barrier / kT, -700.0, 0.0)
    )
    delta = np.maximum((zero_barrier - driven_barrier) / kT, 0.0)
    return np.maximum(zero_rate * np.expm1(np.minimum(delta, 700.0)), 0.0)


def _mean_rate_transport(self, stress_profile, rho, T_K: float, b: float):
    """Current mean-rate Taylor strategy; no implicit finite-time window."""
    p_surface = self.manifest.peierls.as_surface(self.manifest.emission)
    t_surface = self.manifest.taylor.as_surface(self.manifest.emission)
    stress = np.maximum(np.asarray(stress_profile, dtype=float), 0.0)
    density = np.maximum(np.asarray(rho, dtype=float), 1.0)
    tau_p = max(float(self.cfg.peierls_stress_fraction), 0.0) * stress
    spacing = 1.0 / (2.0 * np.sqrt(density))
    phi = spacing / max(abs(float(b)), 1.0e-30)
    tau_t = max(float(self.cfg.taylor_stress_fraction), 0.0) * stress * phi
    peierls = np.asarray(p_surface.rate(tau_p, T_K), dtype=float)
    elementary_net = _stable_forward_minus_zero(t_surface, tau_t, T_K)
    order = 1.0 + max(float(self.manifest.taylor_corr_scale), 0.0) * np.sqrt(
        density / max(float(self.manifest.taylor_corr_rho_c_m2), 1.0)
    )
    release = elementary_net / np.maximum(order, 1.0)
    jump = max(float(self.cfg.jump_fraction), 0.0) * spacing
    velocity = jump * peierls
    encounter = max(float(self.manifest.encounter_efficiency), 0.0) * velocity * np.sqrt(density)
    return {
        "peierls": peierls, "taylor": release,
        "taylor_single": elementary_net, "m": order, "jump": jump,
        "velocity": velocity, "encounter": encounter,
    }


def install_mean_rate_taylor(state: Any) -> None:
    state._transport_rates = MethodType(_mean_rate_transport, state)
    state._r2_taylor_law_id = TAYLOR_LAW_ID
    state._r2_taylor_units = "s^-1"
    state._r2_taylor_elementary_rate = "forward_minus_zero_stress"
    state._r2_taylor_order = "1+scale*sqrt(rho/rho_c)"
    state._r2_taylor_window_s = None


def build_restored_engine(
    cfg: SharedReducedConfig,
    *,
    material_path: str | Path = DEFAULT_MATERIAL,
    selection_path: str | Path = DEFAULT_SELECTION,
    kernel_family_path: str | Path = DEFAULT_KERNEL_FAMILY,
    local_strength_Pa: float | None = None,
    material_manifest: MaterialManifest | None = None,
    persistent_site_config: PersistentSiteConfig | None = None,
) -> ReducedPersistentReversibleTipEngine:
    """Build the production signed engine with a reduced mechanics provider."""
    cfg = cfg.validate()
    manifest = (MaterialManifest.from_csv(material_path)
                if material_manifest is None else material_manifest)
    kernel_path = Path(kernel_family_path).resolve()
    observed_kernel_hash = _sha256(kernel_path)
    if observed_kernel_hash != DEFAULT_KERNEL_SHA256:
        raise RuntimeError(
            "R2 default requires the exact qualified signed kernel family; "
            f"expected {DEFAULT_KERNEL_SHA256}, observed {observed_kernel_hash}"
        )
    engine_type = ReducedPersistentReversibleTipEngine
    # This is the already qualified v10.2.21 numerical interpretation used by
    # the audited spatial entry: reaching the mechanical blocking state is a
    # complementarity solution, not a failed root bracket or a source cap.
    install_backstress_complementarity_fix()
    engine_type.configure_state_resolved_physics(kernel_path, "validated_scalar")
    selected_sites = (load_persistent_config(selection_path)
                      if persistent_site_config is None else persistent_site_config.validate())
    engine_type.configure_persistent_sites(selected_sites)
    front = _front_config(cfg)
    # The state-resolved production kernel requires the serialized local
    # strength coordinate.  The R2 default is the selected material's existing
    # reference strength.  R3's exact spatial-contract replay may instead pass
    # the spatial driver's independently serialized cohesive ceiling (30 GPa in
    # the qualified 900 K run); keeping that choice explicit prevents the two
    # different strength sources from being silently conflated.
    selected_strength = (
        float(manifest.cleavage.sigc0_Pa)
        if local_strength_Pa is None else float(local_strength_Pa)
    )
    if not math.isfinite(selected_strength) or selected_strength < 0.0:
        raise ValueError("local strength must be finite and nonnegative")
    front.sigma_cap = selected_strength
    engine = engine_type(
        front, manifest.cleavage, manifest.emission,
        float(cfg.G_Pa), float(cfg.poisson), float(cfg.b_m), manifest,
        _mpz_config(cfg),
    )
    engine.tip_cfg = _tip_config(cfg, "full")
    engine.mpz.cfg.mobile_shield_fraction = float(cfg.mobile_shield_fraction)
    engine.mpz._anisotropic_drive_factors = np.asarray(cfg.drive_factors, dtype=float)
    engine.mpz._anisotropic_drive_reliable = True
    engine.mpz._anisotropic_drive_serial = 0
    install_mean_rate_taylor(engine.mpz)
    engine._r2_model_id = MODEL_ID
    engine._r2_synthetic_drive_setup = True
    engine._r2_material_path = (str(Path(material_path).resolve())
                                if material_manifest is None else None)
    engine._r2_selection_path = (str(Path(selection_path).resolve())
                                 if persistent_site_config is None else None)
    engine._r2_material_manifest_embedded = material_manifest is not None
    engine._r2_persistent_site_config_embedded = persistent_site_config is not None
    engine._r2_material_sha256 = (_object_sha256(manifest.as_dict())
                                  if material_manifest is not None
                                  else _sha256(material_path))
    engine._r2_selection_sha256 = (_object_sha256(asdict(selected_sites))
                                   if persistent_site_config is not None
                                   else _sha256(selection_path))
    engine._r2_kernel_family_path = str(kernel_path)
    engine._r2_kernel_family_sha256 = observed_kernel_hash
    engine._r2_numerical_options = asdict(cfg)
    engine._r2_local_strength_source = (
        "material_manifest.cleavage.sigc0_Pa"
        if local_strength_Pa is None
        else ("explicit_disabled_moving_mpz_contract"
              if float(local_strength_Pa) == 0.0
              else "explicit_spatial_driver_cohesive_ceiling")
    )
    _restore_protected_continuum_alias()
    return engine


def runtime_receipt(engine: Any, *, stage: str) -> dict[str, Any]:
    state = engine.mpz
    geometry = dict(_source_geometry(state))
    manifest_path = (None if engine._r2_material_path is None
                     else Path(engine._r2_material_path))
    selection_path = (None if engine._r2_selection_path is None
                      else Path(engine._r2_selection_path))
    methods = {
        name: _method_identity(getattr(state, name))
        for name in (
            "_emit", "evolve", "_transport_rates", "local_slip_count",
            "blunted_radius", "advance", "diagnostics",
        )
    }
    owners = {}
    for name in (
        "mobile_positive", "mobile_negative", "retained_positive",
        "retained_negative", "accumulated_slip_positive",
        "accumulated_slip_negative", "returned_slip_positive",
        "returned_slip_negative", "available_sites", "site_capacity",
        "wake_mobile_positive", "wake_mobile_negative", "wake_retained_positive",
        "wake_retained_negative",
    ):
        value = getattr(state, name, None)
        owners[name] = None if value is None else {
            "owner": "engine.mpz", "shape": list(np.shape(value)),
            "sum": float(np.sum(np.asarray(value, dtype=float))),
        }
    return {
        "schema": "r2-resolved-reduced-runtime-receipt/1",
        "stage": str(stage),
        "selected_identity": getattr(
            engine, "_r10_opening_execution_identity", "RESTORED_REDUCED_REFERENCE"
        ),
        "material_family": getattr(engine, "_r10_material_family", None),
        "final_bound_implementation_identity": getattr(
            engine, "_r10_final_bound_implementation_identity", None
        ),
        "model_id": MODEL_ID,
        "historical_replay_identity": HISTORICAL_REPLAY_ID,
        "source_geography": "tip_surface",
        "source_lifecycle": "persistent_reusable",
        "kinetics": "reversible_signed",
        "mechanics_provider": "prescribed_K_reduced_provider",
        "local_strength": {
            "value_Pa": float(engine.f.sigma_cap),
            "source": engine._r2_local_strength_source,
        },
        "engine_class": f"{type(engine).__module__}.{type(engine).__qualname__}",
        "state_class": f"{type(state).__module__}.{type(state).__qualname__}",
        "material": {
            "path": None if manifest_path is None else str(manifest_path),
            "sha256": engine._r2_material_sha256,
            "embedded": bool(engine._r2_material_manifest_embedded),
            "definition": engine.manifest.as_dict(),
        },
        "selection": {
            "path": None if selection_path is None else str(selection_path),
            "sha256": engine._r2_selection_sha256,
            "embedded": bool(engine._r2_persistent_site_config_embedded),
            "definition": asdict(state._persistent_site_cfg),
        },
        "signed_kernel_family": {
            "path": engine._r2_kernel_family_path,
            "sha256": engine._r2_kernel_family_sha256,
        },
        "methods": methods,
        "reversible_binding": reversible_binding_receipt(state),
        "taylor_contract": {
            "law_id": TAYLOR_LAW_ID, "units": "s^-1",
            "elementary_rate": "stable forward-minus-zero-stress",
            "order": "1+scale*sqrt(rho/rho_c)", "finite_window_s": None,
            "final_consumer": methods["_transport_rates"],
        },
        "source_geometry": geometry,
        "persistent_invariants": {
            "inventory_active": False, "depletion_on_emission": False,
            "advance_refresh": False,
            "available_equals_capacity": bool(np.array_equal(state.available_sites, state.site_capacity)),
        },
        "owning_arrays": owners,
        "numerical_options": dict(engine._r2_numerical_options),
    }


def advance_signed(engine: Any, *, K_Pa_sqrt_m: float, temperature_K: float, dt_s: float):
    engine.mpz._reversible_transport_K_signed_Pa_sqrt_m = float(K_Pa_sqrt_m)
    engine.mpz._reversible_tip_radius_m = float(engine.r_eff())
    return engine.step(float(K_Pa_sqrt_m), float(temperature_K), float(dt_s))


__all__ = [
    "DEFAULT_KERNEL_FAMILY", "DEFAULT_MATERIAL", "DEFAULT_SELECTION",
    "HISTORICAL_REPLAY_ID", "MODEL_ID",
    "ReducedPersistentReversibleTipEngine",
    "TAYLOR_LAW_ID", "advance_signed", "build_restored_engine",
    "install_mean_rate_taylor", "load_persistent_config", "runtime_receipt",
]
