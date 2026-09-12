"""Immutable material ownership for the unified single/multi/voiding model line."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .config import ElasticProperties
from .material_manifest import ExpFloorBarrier, MaterialManifest, TransportBarrier
from .sharp_front import FrontConfig
from .unified_front import UnifiedMPZFrontEngine
from .unified_mpz import MPZConfig

SCHEMA = "v5.unified-fracture-material-bundle/1"
CORE_MODEL_ID = "v13.current-source-qualified-multitip/ca4abfe47765fcdaf0d266bfc0558ecd82d0c64e"
FRACTURE_ROWS = MappingProxyType({
    "Peak": "v913_zeroD_sobol_0242980",
    "DBTT": "v913_zeroD_sobol_0202500",
    "weak-T": "oneD_v2_focused_weak_T_0016",
    "ceramic-like": "oneD_v2_focused_ceramic_like_0018",
})
COMMON_VOID_KINETICS_ROW_ID = "v5.voiding-config.reference-one-void/1"
COMMON_ELASTIC_ROW_ID = "v5.tungsten-plane-strain-E210GPa-nu0p3/1"
BENCHMARK_ELASTIC_ROW_ID = "benchmark.retained-v5-E410GPa-nu0p28/1"
COMMON_SITE_POPULATION_ROW_ID = "v5.single-centerline-site-seed3621/1"
COMMON_SPECIMEN_LOADING_ROW_ID = "v5.square-1mm-fixed-opening-G0-companion/1"
DATA_PATH = Path(__file__).resolve().parent / "data/materials/unified_v5/fracture_rows.csv"
_METADATA = frozenset({
    "material_class", "candidate_id", "source_or_search_provenance",
    "parameter_decision", "reduced_selection_status", "same_material_row_both_providers",
})
_ZERO_DISABLED = frozenset({
    "mobile_shield_fraction", "source_recovery_rate_s", "retained_recovery_rate_s",
    "source_refresh_length_um", "recovery_nu0_s", "recovery_H0_eV",
    "recovery_activation_entropy_kB",
})


def _canonical(value: Any) -> bytes:
    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(val) for key, val in sorted(item.items(), key=lambda pair: str(pair[0]))}
        if isinstance(item, (tuple, list)):
            return [normalize(val) for val in item]
        if isinstance(item, float) and not math.isfinite(item):
            return {"nonfinite": str(item)}
        return item
    return json.dumps(normalize(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _number(row: Mapping[str, str], key: str) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"active fracture field {key!r} is absent or nonnumeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"active fracture field {key!r} is nonfinite")
    return value


@dataclass(frozen=True)
class UnifiedFractureMaterialBundle:
    fracture_material_row_id: str
    void_kinetics_row_id: str
    elastic_row_id: str
    site_population_row_id: str
    specimen_loading_row_id: str
    material_class: str
    fracture_row_items: tuple[tuple[str, str], ...]
    front_config_schema_id: str = "v13.sharp-front.FrontConfig/1"
    cleavage_barrier_schema_id: str = "v13.material-manifest.ExpFloorBarrier/1"
    emission_barrier_schema_id: str = "v13.material-manifest.ExpFloorBarrier/1"
    process_zone_schema_id: str = "v13.unified-active-persistent-wake-mpz/1"
    r_tip_law_id: str = "r_tip=r0+c_blunt*b*local_weighted_accumulated_slip/1"
    hazard_algorithm_id: str = "v13.dual-hazard-logmean-multihit/1"
    renewal_policy_id: str = "v13.one-renewal-per-accepted-transaction/1"
    energy_gate_id: str = "v10.2.30.whole-topology-hazard-energy-gate/1"
    rng_policy_id: str = "v13.front-lineage-threshold-rng-ownership/1"
    checkpoint_policy_id: str = "v13.complete-engine-mpz-owner-checkpoint/1"
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        identifiers = (
            self.fracture_material_row_id, self.void_kinetics_row_id,
            self.elastic_row_id, self.site_population_row_id,
            self.specimen_loading_row_id, self.front_config_schema_id,
            self.cleavage_barrier_schema_id, self.emission_barrier_schema_id,
            self.process_zone_schema_id, self.r_tip_law_id,
            self.hazard_algorithm_id, self.renewal_policy_id,
            self.energy_gate_id, self.rng_policy_id, self.checkpoint_policy_id,
        )
        if any(not isinstance(value, str) or not value for value in identifiers):
            raise ValueError("every material-bundle row identity must be nonempty")
        if self.fracture_material_row_id == self.void_kinetics_row_id:
            raise ValueError("fracture and void kinetics must have distinct ownership")
        row = dict(self.fracture_row_items)
        if row.get("candidate_id") != self.fracture_material_row_id:
            raise ValueError("fracture row content and identity disagree")
        if row.get("material_class") != self.material_class:
            raise ValueError("fracture row family and bundle family disagree")

    @property
    def fracture_row(self) -> Mapping[str, str]:
        return MappingProxyType(dict(self.fracture_row_items))

    @property
    def bundle_hash(self) -> str:
        return _sha(asdict(self))

    @property
    def bundle_id(self) -> str:
        return f"{self.schema}:{self.material_class}:{self.bundle_hash[:16]}"


def _registry() -> dict[str, dict[str, str]]:
    with DATA_PATH.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    by_id = {row["candidate_id"]: row for row in rows}
    expected = set(FRACTURE_ROWS.values())
    if set(by_id) != expected:
        raise ValueError(f"unified registry must contain exactly the four pinned rows: {sorted(expected)}")
    return by_id


def material_bundle(material_class: str) -> UnifiedFractureMaterialBundle:
    try:
        row_id = FRACTURE_ROWS[material_class]
    except KeyError as exc:
        raise KeyError(f"unknown unified material family {material_class!r}") from exc
    row = _registry()[row_id]
    return UnifiedFractureMaterialBundle(
        fracture_material_row_id=row_id,
        void_kinetics_row_id=COMMON_VOID_KINETICS_ROW_ID,
        elastic_row_id=COMMON_ELASTIC_ROW_ID,
        site_population_row_id=COMMON_SITE_POPULATION_ROW_ID,
        specimen_loading_row_id=COMMON_SPECIMEN_LOADING_ROW_ID,
        material_class=material_class,
        fracture_row_items=tuple((key, str(value)) for key, value in row.items()),
    )


def all_material_bundles() -> Mapping[str, UnifiedFractureMaterialBundle]:
    return MappingProxyType({family: material_bundle(family) for family in FRACTURE_ROWS})


def benchmark_material_bundle(material_class: str) -> UnifiedFractureMaterialBundle:
    """Explicitly label retained 410 GPa V5 evidence as benchmark-only."""
    return replace(material_bundle(material_class), elastic_row_id=BENCHMARK_ELASTIC_ROW_ID)


def elastic_material(bundle: UnifiedFractureMaterialBundle) -> ElasticProperties:
    if bundle.elastic_row_id == COMMON_ELASTIC_ROW_ID:
        return ElasticProperties(E=210.0e9, nu=0.3, b=2.74e-10, Tm=3695.0)
    if bundle.elastic_row_id == BENCHMARK_ELASTIC_ROW_ID:
        return ElasticProperties(E=410.0e9, nu=0.28, b=2.74e-10, Tm=3695.0)
    raise ValueError("NO_DEFAULT_FALLBACK: unsupported elastic row identity")


def _require_elastic_material(bundle: UnifiedFractureMaterialBundle, material: Any) -> None:
    expected = elastic_material(bundle)
    fields = ("E", "nu", "b", "Tm")
    if any(float(getattr(material, key)) != float(getattr(expected, key)) for key in fields):
        raise ValueError("elastic material values disagree with the bundle elastic row")


def _surface(row: Mapping[str, str], prefix: str, attempt_frequency_s: float) -> ExpFloorBarrier:
    return ExpFloorBarrier(
        G00_eV=_number(row, f"{prefix}_G00_eV"),
        gT_eV_per_K=_number(row, f"{prefix}_gT_eV_per_K"),
        sigc0_Pa=_number(row, f"{prefix}_sigc0_GPa") * 1.0e9,
        sT_Pa_per_K=_number(row, f"{prefix}_sT_GPa_per_K") * 1.0e9,
        alpha=_number(row, f"{prefix}_exp_a"),
        exponent=_number(row, f"{prefix}_exp_n"),
        floor_fraction=_number(row, f"{prefix}_floor_frac"),
        Tref_K=_number(row, "Tref_K"),
        attempt_frequency_s=float(attempt_frequency_s),
    )


def material_manifest(bundle: UnifiedFractureMaterialBundle) -> MaterialManifest:
    row = bundle.fracture_row
    return MaterialManifest(
        name=bundle.material_class,
        candidate_id=bundle.fracture_material_row_id,
        cleavage=_surface(row, "cleave", 1.0e12),
        emission=_surface(row, "emit", 1.0e11),
        peierls=TransportBarrier(
            H0_eV=_number(row, "peierls_H0_eV"),
            activation_entropy_kB=_number(row, "peierls_activation_entropy_kB"),
            alpha=_number(row, "peierls_exp_a"), exponent=_number(row, "peierls_exp_n"),
            attempt_frequency_s=_number(row, "peierls_nu0_s"),
            stress_ratio=_number(row, "peierls_stress_fraction"),
        ),
        taylor=TransportBarrier(
            H0_eV=_number(row, "taylor_H0_eV"),
            activation_entropy_kB=_number(row, "taylor_activation_entropy_kB"),
            alpha=_number(row, "taylor_exp_a"), exponent=_number(row, "taylor_exp_n"),
            attempt_frequency_s=_number(row, "taylor_nu0_s"),
            stress_ratio=_number(row, "taylor_stress_fraction"),
        ),
        taylor_corr_rho_c_m2=_number(row, "taylor_corr_rho_c_m2"),
        taylor_corr_scale=_number(row, "taylor_corr_scale"),
        source_sites_per_system=_number(row, "source_sites_per_system"),
        encounter_efficiency=_number(row, "encounter_efficiency"),
        retained_recovery_rate_s=_number(row, "retained_recovery_rate_s"),
        source_refresh_length_m=_number(row, "source_refresh_length_um") * 1.0e-6,
        c_blunt=_number(row, "c_blunt"),
        # The pinned rows explicitly disable mobile shielding. No independent
        # fitted K cap is introduced by the dimensional-transfer wrapper.
        max_K_shield_MPa_sqrt_m=0.0,
    )


def front_config(bundle: UnifiedFractureMaterialBundle) -> FrontConfig:
    row = bundle.fracture_row
    config = FrontConfig()
    config.L_pz = _number(row, "L_pz_um_recommended") * 1.0e-6
    config.rho0 = _number(row, "rho_forest_floor_m2")
    config.c_blunt = _number(row, "c_blunt")
    config.nu0_c = 1.0e12
    config.nu0_e = 1.0e11
    config.max_advances_per_step = 1
    # r0 is the retained sharp-front r_tip law parameter. It is never tied to
    # a cavity radius and therefore preserves r_tip != R_void.
    return config


def mpz_config(bundle: UnifiedFractureMaterialBundle) -> MPZConfig:
    row = bundle.fracture_row
    return MPZConfig(
        length_m=_number(row, "L_pz_um_recommended") * 1.0e-6,
        forest_density_floor_m2=_number(row, "rho_forest_floor_m2"),
        peierls_stress_fraction=_number(row, "peierls_stress_fraction"),
        taylor_stress_fraction=_number(row, "taylor_stress_fraction"),
        mobile_shield_fraction=_number(row, "mobile_shield_fraction"),
        mobile_recovery_rate_s=_number(row, "source_recovery_rate_s"),
    )


def canonical_front_engine(
    bundle: UnifiedFractureMaterialBundle, elastic_material: Any,
) -> UnifiedMPZFrontEngine:
    """The only bundle-aware front-engine constructor for all 2-D drivers."""
    _require_elastic_material(bundle, elastic_material)
    manifest = material_manifest(bundle)
    engine = UnifiedMPZFrontEngine(
        front_config(bundle), manifest.cleavage, manifest.emission,
        elastic_material.G, elastic_material.nu, elastic_material.b,
        manifest, mpz_config(bundle),
    )
    engine._unified_bundle_id = bundle.bundle_id
    engine._unified_bundle_hash = bundle.bundle_hash
    engine._unified_core_model_id = CORE_MODEL_ID
    return engine


def identity_record(bundle: UnifiedFractureMaterialBundle, elastic_material: Any) -> Mapping[str, str]:
    _require_elastic_material(bundle, elastic_material)
    fcfg = front_config(bundle)
    manifest = material_manifest(bundle)
    return MappingProxyType({
        "core_model_id": CORE_MODEL_ID,
        "material_bundle_id": bundle.bundle_id,
        "material_bundle_sha256": bundle.bundle_hash,
        "elasticity_fingerprint": _sha({
            "elastic_row_id": bundle.elastic_row_id,
            "E": float(elastic_material.E), "nu": float(elastic_material.nu),
            "G": float(elastic_material.G), "b": float(elastic_material.b),
        }),
        "plasticity_fingerprint": _sha({key: bundle.fracture_row[key] for key in (
            "rho_forest_floor_m2", "peierls_stress_fraction", "taylor_stress_fraction",
            "mobile_shield_fraction", "source_recovery_rate_s", "peierls_H0_eV",
            "peierls_activation_entropy_kB", "peierls_exp_a", "peierls_exp_n",
            "taylor_H0_eV", "taylor_activation_entropy_kB", "taylor_exp_a",
            "taylor_exp_n", "taylor_corr_rho_c_m2", "taylor_corr_scale",
            "rho_source0_m2", "recovery_nu0_s", "recovery_H0_eV",
            "recovery_activation_entropy_kB",
        )}),
        "FrontConfig_fingerprint": _sha(vars(fcfg)),
        "cleavage_barrier_fingerprint": _sha(asdict(manifest.cleavage)),
        "emission_barrier_fingerprint": _sha(asdict(manifest.emission)),
        "process_zone_fingerprint": _sha({key: bundle.fracture_row[key] for key in (
            "L_pz_um_recommended", "source_sites_per_system", "encounter_efficiency",
            "source_refresh_length_um", "reference_source_area_um2",
            "reference_front_width_um", "source_zone_length_um",
        )}),
    })


def bind_identity(state: Any, bundle: UnifiedFractureMaterialBundle) -> Any:
    junction = dict(state.junction_process_state)
    identity = dict(identity_record(bundle, state.material))
    previous = junction.get("unified_model_identity")
    if previous is not None and dict(previous) != identity:
        raise ValueError("accepted state already belongs to a different material bundle")
    junction["unified_model_identity"] = identity
    junction["unified_material_bundle"] = asdict(bundle)
    return replace(state, junction_process_state=junction)


def require_bound_identity(state: Any) -> Mapping[str, str]:
    try:
        identity = state.junction_process_state["unified_model_identity"]
        payload = state.junction_process_state["unified_material_bundle"]
    except (AttributeError, KeyError) as exc:
        raise RuntimeError("accepted state has no unified material identity") from exc
    restored = UnifiedFractureMaterialBundle(**payload)
    expected = identity_record(restored, state.material)
    if dict(identity) != dict(expected):
        raise RuntimeError("accepted state material identity is corrupt or stale")
    return identity


def active_field_mapping(bundle: UnifiedFractureMaterialBundle) -> Mapping[str, str]:
    """Return an exact fail-closed runtime target for every active row field."""
    row = bundle.fracture_row
    targets = {
        "Tref_K": "cleavage/emission ExpFloorBarrier.Tref_K",
        "rho_forest_floor_m2": "MPZConfig.forest_density_floor_m2 and FEM initial rho_gp",
        "peierls_stress_fraction": "MPZConfig.peierls_stress_fraction",
        "taylor_stress_fraction": "MPZConfig.taylor_stress_fraction",
        "L_pz_um_recommended": "FrontConfig.L_pz and MPZConfig.length_m",
        "peierls_H0_eV": "MaterialManifest.peierls.H0_eV",
        "peierls_activation_entropy_kB": "MaterialManifest.peierls.activation_entropy_kB",
        "peierls_exp_a": "MaterialManifest.peierls.alpha",
        "peierls_exp_n": "MaterialManifest.peierls.exponent",
        "taylor_H0_eV": "MaterialManifest.taylor.H0_eV",
        "taylor_activation_entropy_kB": "MaterialManifest.taylor.activation_entropy_kB",
        "taylor_exp_a": "MaterialManifest.taylor.alpha",
        "taylor_exp_n": "MaterialManifest.taylor.exponent",
        "taylor_corr_rho_c_m2": "MaterialManifest.taylor_corr_rho_c_m2",
        "taylor_corr_scale": "MaterialManifest.taylor_corr_scale",
        "source_sites_per_system": "MaterialManifest.source_sites_per_system",
        "encounter_efficiency": "MaterialManifest.encounter_efficiency",
        "c_blunt": "MaterialManifest.c_blunt and FrontConfig.c_blunt",
        "peierls_nu0_s": "MaterialManifest.peierls.attempt_frequency_s",
        "taylor_nu0_s": "MaterialManifest.taylor.attempt_frequency_s",
        "rho_source0_m2": "FEM initial rho_gp",
        "reference_source_area_um2": "process_zone_fingerprint.reference_source_area_um2",
        "reference_front_width_um": "process_zone_fingerprint.reference_front_width_um",
        "source_zone_length_um": "process_geometry_fingerprint.source_zone_length_um",
    }
    for prefix in ("cleave", "emit"):
        for suffix in ("G00_eV", "gT_eV_per_K", "sigc0_GPa", "sT_GPa_per_K", "exp_a", "exp_n", "floor_frac"):
            targets[f"{prefix}_{suffix}"] = f"MaterialManifest.{prefix}.{suffix}"
    active = {
        key for key, value in row.items()
        if key not in _METADATA and not (key in _ZERO_DISABLED and _number(row, key) == 0.0)
    }
    missing = active - set(targets)
    if missing:
        raise RuntimeError("NO_DEFAULT_FALLBACK: unmapped active fields: " + ", ".join(sorted(missing)))
    return MappingProxyType({key: targets[key] for key in sorted(active)})


__all__ = [
    "BENCHMARK_ELASTIC_ROW_ID", "CORE_MODEL_ID", "FRACTURE_ROWS", "UnifiedFractureMaterialBundle",
    "active_field_mapping", "all_material_bundles", "benchmark_material_bundle", "bind_identity",
    "canonical_front_engine", "elastic_material", "front_config", "identity_record", "material_bundle",
    "material_manifest", "mpz_config", "require_bound_identity",
]
