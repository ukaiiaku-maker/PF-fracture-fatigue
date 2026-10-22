"""QSTE340TM material binding for the accepted restored R2/R9 runtime.

This module changes material coordinates only.  Engine construction remains
owned by ``build_restored_engine`` through the archived R9 binding, including
the persistent source, signed transport, moving frame, and installed mean-rate
Taylor consumer.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

from .archived_fatigue_material_r9 import (
    ArchivedMaterialBinding,
    build_archived_current_engine,
    load_archived_material_binding,
)
from .canonical_v2_registry_v10230 import ThermodynamicBarrierAdapter
from .material_manifest import ExpFloorBarrier, KB_EV_PER_K, MaterialManifest, TransportBarrier
from .reduced_persistent_reference_r2 import TAYLOR_LAW_ID
from scripts.fatigue_informed_joint_barrier_v3_v10230 import KineticSurface, SmoothBroadBasis


TEMPERATURE_K = 298.15
TREF_K = 481.33
EXP_GT_EV_PER_K = 0.003934
EXP_ST_PA_PER_K = -6.564e5
EXP_ALPHA = 0.0845685
EXP_EXPONENT = 1.0
EXP_FLOOR_FRACTION = 0.02
STEEL_G_PA = 80.0e9
STEEL_POISSON = 0.30
STEEL_B_M = 2.48e-10
FAMILY = "QSTE340TM_RESTORED_ALTERNATIVE_GSTAR_EXP_FLOOR"
MODEL_ID = "qste340tm.restored-r2-r9-alternative-gstar-exp-floor/1"


@dataclass(frozen=True)
class PlasticCoordinates:
    emission_G0_298_eV: float
    plastic_sigc_298_Pa: float
    peierls_G0_298_eV: float
    taylor_G0_298_eV: float


@dataclass(frozen=True)
class OpeningCoordinates:
    floor_eV: float
    broad_amplitude_eV: float
    broad_sigma_Pa: float
    broad_exponent: float
    guard_amplitude_eV: float


def _exp_floor(g298_eV: float, sigc298_Pa: float, attempt_frequency_s: float) -> ExpFloorBarrier:
    return ExpFloorBarrier(
        G00_eV=float(g298_eV) - EXP_GT_EV_PER_K * (TEMPERATURE_K - TREF_K),
        gT_eV_per_K=EXP_GT_EV_PER_K,
        sigc0_Pa=float(sigc298_Pa) - EXP_ST_PA_PER_K * (TEMPERATURE_K - TREF_K),
        sT_Pa_per_K=EXP_ST_PA_PER_K,
        alpha=EXP_ALPHA,
        exponent=EXP_EXPONENT,
        floor_fraction=EXP_FLOOR_FRACTION,
        Tref_K=TREF_K,
        attempt_frequency_s=float(attempt_frequency_s),
    )


def _transport(g298_eV: float, attempt_frequency_s: float) -> TransportBarrier:
    entropy = -EXP_GT_EV_PER_K / KB_EV_PER_K
    return TransportBarrier(
        H0_eV=float(g298_eV) - EXP_GT_EV_PER_K * (TEMPERATURE_K - TREF_K),
        activation_entropy_kB=entropy,
        alpha=EXP_ALPHA,
        exponent=EXP_EXPONENT,
        attempt_frequency_s=float(attempt_frequency_s),
        stress_ratio=1.0,
    )


def opening_surface(base: KineticSurface, coordinates: OpeningCoordinates) -> KineticSurface:
    """Vary the authorized alternative shape while retaining its thermal coupling."""
    if len(base.bases) < 2:
        raise ValueError("alternative G* parent lacks its guard basis")
    broad = SmoothBroadBasis(
        sigma_b_Pa=float(coordinates.broad_sigma_Pa),
        exponent=float(coordinates.broad_exponent),
        smoothing_ratio=float(getattr(base.bases[0], "smoothing_ratio", 1.0e-4)),
    )
    amplitudes = list(base.reference_amplitudes_eV)
    amplitudes[0] = float(coordinates.broad_amplitude_eV)
    amplitudes[1] = float(coordinates.guard_amplitude_eV)
    return replace(
        base,
        reference_floor_eV=float(coordinates.floor_eV),
        reference_amplitudes_eV=tuple(amplitudes),
        bases=(broad, *base.bases[1:]),
    )


def load_parent_binding(
    alternative_registry: str | Path,
    parent_registry: str | Path,
    *,
    candidate_id: str = "R10_ALT_GSTAR_02506",
) -> ArchivedMaterialBinding:
    return load_archived_material_binding(
        alternative_registry, parent_registry, candidate_id=candidate_id,
    )


def material_binding(
    parent: ArchivedMaterialBinding,
    plastic: PlasticCoordinates,
    opening: OpeningCoordinates | None = None,
) -> ArchivedMaterialBinding:
    base_adapter = parent.manifest.cleavage
    if not isinstance(base_adapter, ThermodynamicBarrierAdapter):
        raise TypeError("QSTE binding requires the authorized alternative-G* adapter")
    surface = base_adapter.surface if opening is None else opening_surface(base_adapter.surface, opening)
    cleavage = ThermodynamicBarrierAdapter(surface=surface, parent=dict(base_adapter.parent))
    manifest = MaterialManifest(
        name=FAMILY,
        candidate_id=FAMILY,
        cleavage=cleavage,
        emission=_exp_floor(plastic.emission_G0_298_eV, plastic.plastic_sigc_298_Pa, 1.0e11),
        peierls=_transport(plastic.peierls_G0_298_eV, 1.0e12),
        taylor=_transport(plastic.taylor_G0_298_eV, 1.0e11),
        # These are retained accepted material couplings, not fixture values.
        taylor_corr_rho_c_m2=float(parent.manifest.taylor_corr_rho_c_m2),
        taylor_corr_scale=float(parent.manifest.taylor_corr_scale),
        source_sites_per_system=float(parent.manifest.source_sites_per_system),
        encounter_efficiency=float(parent.manifest.encounter_efficiency),
        # The accepted persistent-source transfer deliberately disables
        # explicit recovery.  Recovery is therefore a fixed runtime contract,
        # not a QSTE fitting coordinate in this binding.
        retained_recovery_rate_s=0.0,
        source_refresh_length_m=float(parent.manifest.source_refresh_length_m),
        c_blunt=float(parent.manifest.c_blunt),
        max_K_shield_MPa_sqrt_m=float(parent.manifest.max_K_shield_MPa_sqrt_m),
    )
    config = replace(
        parent.reduced_config,
        G_Pa=STEEL_G_PA,
        poisson=STEEL_POISSON,
        b_m=STEEL_B_M,
        cleavage_hits=float(surface.hits),
        cleavage_tau_s=float(surface.tau_s),
    )
    provenance = dict(parent.provenance)
    provenance.update({
        "qste_model_id": MODEL_ID,
        "family": FAMILY,
        "temperature_K": TEMPERATURE_K,
        "steel_G_Pa": STEEL_G_PA,
        "steel_poisson": STEEL_POISSON,
        "steel_b_m": STEEL_B_M,
        "taylor_law_required": TAYLOR_LAW_ID,
        "opening_family": "C_ALTERNATIVE_GSTAR_JOINT",
        "opening_thermal_coordinates_inherited_not_fitted": True,
    })
    return ArchivedMaterialBinding(
        parent.optimized_row,
        parent.parent_row,
        manifest,
        config,
        parent.persistent_sites,
        provenance,
    )


def build_engine(binding: ArchivedMaterialBinding, *, kernel_family_path: str | Path) -> Any:
    engine = build_archived_current_engine(
        binding,
        kernel_family_path=kernel_family_path,
        local_strength_Pa=30.0e9,
    )
    if getattr(engine.mpz, "_r2_taylor_law_id", None) != TAYLOR_LAW_ID:
        raise RuntimeError("restored mean-rate Taylor consumer is not the final bound method")
    engine._r10_opening_execution_identity = "ALTERNATIVE_GSTAR_PRODUCTION_OBSERVER"
    engine._r10_material_family = FAMILY
    engine._r10_final_bound_implementation_identity = MODEL_ID
    return engine


def install_opening(engine: Any, surface: KineticSurface) -> None:
    adapter = ThermodynamicBarrierAdapter(
        surface=surface,
        parent=dict(engine.manifest.cleavage.parent),
    )
    engine.manifest = replace(engine.manifest, cleavage=adapter)
    engine.f.m_hits = float(surface.hits)
    engine.f.tau_c = float(surface.tau_s)


def plastic_vector(value: PlasticCoordinates) -> list[float]:
    return [
        value.emission_G0_298_eV,
        value.plastic_sigc_298_Pa,
        value.peierls_G0_298_eV,
        value.taylor_G0_298_eV,
    ]


__all__ = [
    "FAMILY", "MODEL_ID", "OpeningCoordinates", "PlasticCoordinates",
    "TEMPERATURE_K", "build_engine", "install_opening", "load_parent_binding",
    "material_binding", "opening_surface", "plastic_vector",
]
