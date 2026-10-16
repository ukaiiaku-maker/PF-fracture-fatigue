"""Exact retained-row bindings for the R10 joint-search continuation.

This module translates already-sealed complete rows into the restored current
backend.  It does not fit, clip, or alter a barrier surface.  The mechanical
closure is the qualified R8/R9 prescribed-K provider.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .archived_fatigue_material_r9 import ArchivedMaterialBinding, _transport
from .canonical_v2_registry_v10230 import load_row, surface_adapters
from .material_manifest import ExpFloorBarrier, MaterialManifest, TransportBarrier
from .matched_persistent_execution_r3 import matched_config
from .persistent_site_source_v10221 import PersistentSiteConfig


STRICT_FAMILY = "A_ORIGINAL_SINGLE_EXP_FLOOR"
V2_FAMILY = "B_INHERITED_V2_GUARD_ENTROPY_EXTENSION"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=str,
    ).encode()).hexdigest()


def _f(row: Mapping[str, Any], name: str) -> float:
    value = float(row[name])
    if not value == value or abs(value) == float("inf"):
        raise ValueError(f"nonfinite retained field {name!r}")
    return value


def _exp(row: Mapping[str, Any], prefix: str, attempt: float) -> ExpFloorBarrier:
    return ExpFloorBarrier(
        G00_eV=_f(row, f"{prefix}_G00_eV"),
        gT_eV_per_K=_f(row, f"{prefix}_gT_eV_per_K"),
        sigc0_Pa=1.0e9 * _f(row, f"{prefix}_sigc0_GPa"),
        sT_Pa_per_K=1.0e9 * _f(row, f"{prefix}_sT_GPa_per_K"),
        alpha=_f(row, f"{prefix}_exp_a"),
        exponent=_f(row, f"{prefix}_exp_n"),
        floor_fraction=_f(row, f"{prefix}_floor_frac"),
        Tref_K=_f(row, "Tref_K"), attempt_frequency_s=float(attempt),
    )


def _simple_manifest(row: Mapping[str, Any]) -> MaterialManifest:
    return MaterialManifest(
        name=str(row.get("material_class", row["candidate_id"])),
        candidate_id=str(row["candidate_id"]),
        cleavage=_exp(row, "cleave", 1.0e12),
        emission=_exp(row, "emit", 1.0e11),
        peierls=TransportBarrier(
            H0_eV=_f(row, "peierls_H0_eV"),
            activation_entropy_kB=_f(row, "peierls_activation_entropy_kB"),
            alpha=_f(row, "peierls_exp_a"), exponent=_f(row, "peierls_exp_n"),
            attempt_frequency_s=_f(row, "peierls_nu0_s"),
        ),
        taylor=TransportBarrier(
            H0_eV=_f(row, "taylor_H0_eV"),
            activation_entropy_kB=_f(row, "taylor_activation_entropy_kB"),
            alpha=_f(row, "taylor_exp_a"), exponent=_f(row, "taylor_exp_n"),
            attempt_frequency_s=_f(row, "taylor_nu0_s"),
        ),
        taylor_corr_rho_c_m2=_f(row, "taylor_corr_rho_c_m2"),
        taylor_corr_scale=_f(row, "taylor_corr_scale"),
        source_sites_per_system=_f(row, "source_sites_per_system"),
        encounter_efficiency=_f(row, "encounter_efficiency"),
        retained_recovery_rate_s=_f(row, "retained_recovery_rate_s"),
        source_refresh_length_m=1.0e-6 * _f(row, "source_refresh_length_um"),
        c_blunt=_f(row, "c_blunt"),
        # The canonical paper-four registry predates serialization of this
        # diagnostic-only legacy cap.  The restored backend is explicitly
        # uncapped; complete later rows serialize the same value as zero.
        max_K_shield_MPa_sqrt_m=(
            _f(row, "max_K_shield_MPa_sqrt_m")
            if row.get("max_K_shield_MPa_sqrt_m", "") != "" else 0.0
        ),
    )


def _configuration(row: Mapping[str, Any], *, hits: float, tau_s: float):
    cfg = matched_config()
    cfg.target_extension_um = 100.0
    cfg.mpz_length_um = _f(row, "L_pz_um_recommended")
    cfg.mpz_n_bins = int(_f(row, "n_bins_recommended"))
    cfg.forest_density_floor_m2 = _f(row, "rho_forest_floor_m2")
    cfg.mobile_shield_fraction = _f(row, "mobile_shield_fraction")
    cfg.cleavage_hits = float(hits)
    cfg.cleavage_tau_s = float(tau_s)
    if "physics__blunting_length_m" in row:
        cfg.blunting_length_um = 1.0e6 * _f(row, "physics__blunting_length_m")
    return cfg.validate()


def _sites(row: Mapping[str, Any]) -> PersistentSiteConfig:
    return PersistentSiteConfig(
        rho_site0_m2=_f(row, "rho_source0_m2"),
        reference_source_area_m2=1.0e-12 * _f(row, "reference_source_area_um2"),
        reference_front_width_m=1.0e-6 * _f(row, "reference_front_width_um"),
        reference_density_m2=_f(row, "rho_forest_floor_m2"),
        source_zone_length_m=1.0e-6 * _f(row, "source_zone_length_um"),
    ).validate()


def _binding(*, row: Mapping[str, Any], manifest: MaterialManifest,
             family: str, source: Path, source_hash: str,
             row_hash: str, hits: float, tau_s: float,
             candidate_id: str | None = None) -> ArchivedMaterialBinding:
    candidate = str(row["candidate_id"] if candidate_id is None else candidate_id)
    provenance = {
        "candidate_id": candidate, "family": family,
        "source": {"path": str(source.resolve()), "sha256": source_hash},
        "complete_row_sha256": row_hash,
        "translation": "EXACT_RETAINED_ROW_TO_CURRENT_PERSISTENT_REVERSIBLE_INTERFACES",
        "historical_engine_loaded": False,
        "barrier_changed": False, "mechanics_changed": False,
        "event_law_changed": False,
    }
    return ArchivedMaterialBinding(
        optimized_row={"candidate_id": candidate, "candidate_sha256": row_hash},
        parent_row={}, manifest=manifest,
        reduced_config=_configuration(row, hits=hits, tau_s=tau_s),
        persistent_sites=_sites(row), provenance=provenance,
    )


def load_strict_original_binding(registry: str | Path, option_key: str):
    source = Path(registry).resolve()
    with source.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["option_key"] == option_key]
    if len(matches) != 1:
        raise KeyError(f"expected one strict-family option {option_key!r}")
    row = matches[0]
    return _binding(
        row=row, manifest=_simple_manifest(row), family=STRICT_FAMILY,
        source=source, source_hash=_sha256(source), row_hash=_canonical_hash(row),
        hits=3.0, tau_s=1.0e-6,
    )


def load_parent_binding(registry: str | Path, candidate_id: str):
    source = Path(registry).resolve()
    with source.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise KeyError(f"expected one retained parent {candidate_id!r}")
    record = matches[0]
    row = json.loads(record["complete_parameter_vector"])
    return _binding(
        row=row, manifest=_simple_manifest(row), family=STRICT_FAMILY,
        source=source, source_hash=_sha256(source),
        row_hash=str(record["complete_row_sha256"]),
        hits=_f(row, "physics__cleavage_hits"),
        tau_s=_f(row, "physics__cleavage_correlation_time_s"),
    )


def load_offspring_binding(registry: str | Path, candidate_id: str):
    """Load an R10 complete strict-family hypothesis without label inference."""
    source = Path(registry).resolve()
    with source.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise KeyError(f"expected one R10 offspring {candidate_id!r}")
    record = matches[0]
    if record.get("family") != STRICT_FAMILY:
        raise ValueError("R10 offspring loader accepts only the strict family")
    row = json.loads(record["complete_parameter_vector"])
    return _binding(
        row=row, manifest=_simple_manifest(row), family=STRICT_FAMILY,
        source=source, source_hash=_sha256(source),
        row_hash=str(record["complete_row_sha256"]),
        hits=float(record["cleavage_hits"]),
        tau_s=float(record["cleavage_tau_s"]),
    )


def load_v2_binding(rows_path: str | Path, manifest_path: str | Path,
                    candidate_id: str):
    source = Path(rows_path).resolve(); sealed = Path(manifest_path).resolve()
    row = load_row(candidate_id, source, sealed)
    parent = json.loads(row["parent_complete_registry_row_json"])
    raw_manifest = json.loads(row["parent_material_manifest_json"])
    opening, emission = surface_adapters(row)
    manifest = MaterialManifest(
        name=str(candidate_id), candidate_id=str(candidate_id),
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
    binding = _binding(
        row=parent, manifest=manifest, family=V2_FAMILY,
        source=source, source_hash=_sha256(source),
        row_hash=row["complete_bound_row_sha256"],
        hits=_f(row, "physics__cleavage_hits"),
        tau_s=_f(row, "physics__cleavage_correlation_time_s"),
        candidate_id=candidate_id,
    )
    provenance = dict(binding.provenance)
    provenance["sealed_manifest"] = {
        "path": str(sealed), "sha256": _sha256(sealed),
    }
    provenance["opening_surface_sha256"] = row["opening_surface_sha256"]
    provenance["emission_surface_sha256"] = row["emission_surface_sha256"]
    return ArchivedMaterialBinding(
        binding.optimized_row, binding.parent_row, binding.manifest,
        binding.reduced_config, binding.persistent_sites, provenance,
    )


__all__ = [
    "STRICT_FAMILY", "V2_FAMILY", "load_offspring_binding", "load_parent_binding",
    "load_strict_original_binding", "load_v2_binding",
]
