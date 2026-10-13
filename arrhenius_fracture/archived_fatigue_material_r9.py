"""Exact archived V3/V3.1 material binding for the restored current backend.

The archive remains authoritative.  This adapter validates and translates its
complete serialized row into current production interfaces; it does not load
the historical seam engine or recreate parameters from candidate labels.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .canonical_v2_registry_v10230 import ThermodynamicBarrierAdapter
from .material_manifest import MaterialManifest, TransportBarrier
from .persistent_site_source_v10221 import PersistentSiteConfig
from .reduced_persistent_reference_r2 import build_restored_engine
from .reduced_shared_state_v1023 import SharedReducedConfig


OPTIMIZED_ID = "V3_ceramic_V2_P25_007__V3_1_TRUST1"
PARENT_ID = "V3_ceramic_V2_P25_007"
EXPECTED_OPTIMIZED_SHA256 = (
    "0976e5d4625a925ef02e64cf7f8383e879695b9946f3d17b4ec187c1e75ed798"
)
EXPECTED_PARENT_SHA256 = (
    "660a5750ca950464b71083fa0f301dabdaf98e2378df20236c0c7b2ed980eccc"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transport(raw: Mapping[str, Any]) -> TransportBarrier:
    return TransportBarrier(
        H0_eV=float(raw["H0_eV"]),
        activation_entropy_kB=float(raw["activation_entropy_kB"]),
        alpha=float(raw["alpha"]), exponent=float(raw["exponent"]),
        attempt_frequency_s=float(raw["attempt_frequency_s"]),
        stress_ratio=float(raw.get("stress_ratio", 1.0)),
    )


@dataclass(frozen=True)
class ArchivedMaterialBinding:
    optimized_row: Mapping[str, Any]
    parent_row: Mapping[str, Any]
    manifest: MaterialManifest
    reduced_config: SharedReducedConfig
    persistent_sites: PersistentSiteConfig
    provenance: Mapping[str, Any]


def load_archived_material_binding(
    optimized_registry: str | Path,
    parent_registry: str | Path,
    *, candidate_id: str = OPTIMIZED_ID,
) -> ArchivedMaterialBinding:
    optimized_path = Path(optimized_registry).resolve()
    parent_path = Path(parent_registry).resolve()
    optimized_payload = json.loads(optimized_path.read_text())
    parent_payload = json.loads(parent_path.read_text())
    optimized_matches = [
        row for row in optimized_payload["rows"]
        if row["candidate_id"] == candidate_id
    ]
    if optimized_matches:
        optimized = optimized_matches[0]
    else:
        archived_parent = next(
            row for row in parent_payload["rows"]
            if row["candidate_id"] == candidate_id
        )
        optimized = {
            "candidate_id": candidate_id,
            "candidate_sha256": archived_parent["complete_v3_row_sha256"],
            "parent_candidate_id": candidate_id,
            "opening_surface": archived_parent["opening_surface"],
            "emission_surface": archived_parent["emission_surface"],
            "archived_parent_control": True,
        }
    if candidate_id == OPTIMIZED_ID and optimized.get("candidate_sha256") != EXPECTED_OPTIMIZED_SHA256:
        raise ValueError("archived optimized ceramic row hash differs from V3.1")
    parent_id = str(optimized["parent_candidate_id"])
    parent = next(
        row for row in parent_payload["rows"]
        if row["candidate_id"] == parent_id
    )
    if parent_id == PARENT_ID and parent.get("complete_v3_row_sha256") != EXPECTED_PARENT_SHA256:
        raise ValueError("archived parent ceramic row hash differs from V3")

    source = parent["source_complete_row"]
    raw_manifest = json.loads(source["parent_material_manifest_json"])
    from scripts.fatigue_informed_joint_barrier_v3_v10230 import deserialize_surface
    from scripts.run_v3_1_coupled_sensitivity_v10230 import deserialize_thermodynamic
    from scripts.corrected_thermodynamic_joint_search_v10230 import (
        deserialize_surface as deserialize_legacy_thermodynamic,
    )
    opening_surface = deserialize_surface(optimized["opening_surface"])
    emission_data = optimized["emission_surface"]
    emission_surface = (deserialize_thermodynamic(emission_data)
                        if any("kind" in basis for basis in emission_data["bases"])
                        else deserialize_legacy_thermodynamic(emission_data))
    opening = ThermodynamicBarrierAdapter(opening_surface, raw_manifest["cleavage"])
    emission = ThermodynamicBarrierAdapter(emission_surface, raw_manifest["emission"])
    manifest = MaterialManifest(
        name=str(raw_manifest["name"]), candidate_id=str(optimized["candidate_id"]),
        cleavage=opening, emission=emission,
        peierls=_transport(raw_manifest["peierls"]),
        taylor=_transport(raw_manifest["taylor"]),
        taylor_corr_rho_c_m2=float(raw_manifest["taylor_corr_rho_c_m2"]),
        taylor_corr_scale=float(raw_manifest["taylor_corr_scale"]),
        source_sites_per_system=float(raw_manifest["source_sites_per_system"]),
        encounter_efficiency=float(raw_manifest["encounter_efficiency"]),
        retained_recovery_rate_s=float(raw_manifest["retained_recovery_rate_s"]),
        source_refresh_length_m=float(raw_manifest["source_refresh_length_m"]),
        c_blunt=float(raw_manifest["c_blunt"]),
        max_K_shield_MPa_sqrt_m=float(raw_manifest["max_K_shield_MPa_sqrt_m"]),
    )
    row = json.loads(source["parent_complete_registry_row_json"])
    cfg = SharedReducedConfig(
        target_extension_um=100.0, checkpoint_da_um=5.0,
        mpz_length_um=float(row["L_pz_um_recommended"]),
        mpz_n_bins=int(row["n_bins_recommended"]),
        wake_length_um=float(row["L_pz_um_recommended"]),
        blunting_length_um=float(row["physics__blunting_length_m"]) * 1.0e6,
        forest_density_floor_m2=float(row["rho_forest_floor_m2"]),
        mobile_shield_fraction=float(row["mobile_shield_fraction"]),
        cleavage_hits=float(optimized["opening_surface"]["hits"]),
        cleavage_tau_s=float(optimized["opening_surface"]["tau_s"]),
        max_internal_steps=100000,
    )
    sites = PersistentSiteConfig(
        rho_site0_m2=float(row["rho_source0_m2"]),
        reference_source_area_m2=float(row["reference_source_area_um2"]) * 1.0e-12,
        reference_front_width_m=float(row["reference_front_width_um"]) * 1.0e-6,
        reference_density_m2=float(row["rho_forest_floor_m2"]),
        source_zone_length_m=float(row["source_zone_length_um"]) * 1.0e-6,
    ).validate()
    provenance = {
        "optimized_registry": {"path": str(optimized_path), "sha256": _sha256(optimized_path)},
        "parent_registry": {"path": str(parent_path), "sha256": _sha256(parent_path)},
        "candidate_id": optimized["candidate_id"],
        "candidate_sha256": optimized["candidate_sha256"],
        "archived_parent_control": bool(optimized.get("archived_parent_control", False)),
        "parent_candidate_id": parent["candidate_id"],
        "complete_v3_row_sha256": parent["complete_v3_row_sha256"],
        "source_candidate_id": parent["source_candidate_id"],
        "source_complete_bound_row_sha256": parent["source_complete_bound_row_sha256"],
        "translation": "ARCHIVED_COMPLETE_ROW_TO_CURRENT_PERSISTENT_REVERSIBLE_INTERFACES",
        "historical_engine_loaded": False,
    }
    return ArchivedMaterialBinding(optimized, parent, manifest, cfg, sites, provenance)


def build_archived_current_engine(
    binding: ArchivedMaterialBinding, *, kernel_family_path,
    local_strength_Pa: float = 30.0e9,
):
    return build_restored_engine(
        binding.reduced_config,
        material_manifest=binding.manifest,
        persistent_site_config=binding.persistent_sites,
        kernel_family_path=kernel_family_path,
        # The current effective-opening kernel requires a physical strength
        # coordinate.  Use the same independently serialized 30 GPa spatial
        # contract as R3/R4, never the barrier's characteristic stress scale.
        local_strength_Pa=float(local_strength_Pa),
    )


__all__ = [
    "ArchivedMaterialBinding", "EXPECTED_OPTIMIZED_SHA256", "EXPECTED_PARENT_SHA256",
    "OPTIMIZED_ID", "PARENT_ID", "build_archived_current_engine",
    "load_archived_material_binding",
]
