"""Opt-in names for immutable corrected-search V2 parameter rows.

This module is deliberately separate from every established parameter registry.
It resolves only corrected V2 aliases and exact corrected candidate IDs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .canonical_v2_registry_v10230 import load_row, load_rows


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "v2_named_parameterization_registry.json"
SUPPORTED_LOADERS = (
    "analytical",
    "monotonic_reduced_1D",
    "fatigue_reduced_1D",
    "PF_sharp_front",
    "FEM_CZM",
    "checkpoint_serialization",
)
NAMED_ALIASES = (
    "DBTT_V2",
    "DBTT_V2_P40",
    "weakT_V2_P25",
    "weakT_V2_P40",
    "ceramic_V2_P25",
    "ceramic_V2_P40",
)
EXACT_ONLY_CONTROLS = (
    "P25_TJBSV2_S_064036",
    "P40_TJBSV2_S_043821",
)
RESERVED_ABSENT_ALIASES = (
    "Peak_V2",
    "weakT_V2",
    "ceramic_V2",
)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class V2Parameterization:
    registry_alias: str | None
    display_name: str
    source_candidate_id: str
    parent_background: str
    status: str
    complete_bound_row_sha256: str
    thermodynamic_surface_fingerprint: str
    renewal_fingerprint: str
    entropy_convention: str
    temperature_response_classification: str
    rising_resistance_status: str
    fatigue_status: str
    developed_fatigue_status: str
    spatial_transfer_status: str
    limitations: tuple[str, ...]
    full_precision_row: Mapping[str, str]
    evidence: Mapping[str, Any]

    def checkpoint_identity(self) -> dict[str, str]:
        return {
            "source_candidate_id": self.source_candidate_id,
            "complete_bound_row_sha256": self.complete_bound_row_sha256,
            "parameterization_generation": "CORRECTED_JOINT_THERMODYNAMIC_V2",
            **({"v2_parameterization_alias": self.registry_alias} if self.registry_alias else {}),
        }


def _load_catalog() -> dict[str, dict[str, Any]]:
    payload = json.loads(CATALOG_PATH.read_text())
    if payload.get("schema") != "v10.2.30_opt_in_named_v2_parameterizations_v1":
        raise ValueError("named V2 catalog schema mismatch")
    if tuple(payload.get("named_aliases", ())) != NAMED_ALIASES:
        raise ValueError("named V2 alias order mismatch")
    if tuple(payload.get("reserved_absent_aliases", ())) != RESERVED_ABSENT_ALIASES:
        raise ValueError("reserved V2 alias policy mismatch")
    entries = payload.get("entries", [])
    if len(entries) != 6:
        raise ValueError("named V2 catalog must contain exactly six entries")
    result: dict[str, dict[str, Any]] = {}
    for entry in entries:
        required = {
            "registry_alias", "display_name", "source_candidate_id", "parent_background",
            "status", "complete_bound_row_sha256", "thermodynamic_surface_fingerprint",
            "renewal_fingerprint", "entropy_convention",
            "temperature_response_classification", "rising_resistance_status",
            "fatigue_status", "developed_fatigue_status", "spatial_transfer_status",
            "limitations", "response_evidence", "cross_code_loaders",
        }
        missing = sorted(required - set(entry))
        if missing:
            raise ValueError(f"named V2 entry lacks required fields: {missing}")
        alias = entry["registry_alias"]
        if alias in result:
            raise ValueError(f"duplicate named V2 alias {alias!r}")
        if tuple(entry["cross_code_loaders"]) != SUPPORTED_LOADERS:
            raise ValueError(f"{alias} does not declare every supported loader")
        result[alias] = entry
    if tuple(result) != NAMED_ALIASES:
        raise ValueError("named V2 entries differ from the frozen alias set")
    return result


@lru_cache(maxsize=1)
def _entries() -> Mapping[str, Mapping[str, Any]]:
    return MappingProxyType({key: _freeze(value) for key, value in _load_catalog().items()})


@lru_cache(maxsize=None)
def _load_exact(candidate_id: str) -> V2Parameterization:
    source_rows = load_rows()
    if candidate_id not in source_rows:
        raise KeyError(f"unknown exact corrected V2 candidate {candidate_id!r}")
    row = load_row(candidate_id)
    named = next(
        (entry for entry in _entries().values() if entry["source_candidate_id"] == candidate_id),
        None,
    )
    if named is None:
        if candidate_id not in EXACT_ONLY_CONTROLS:
            raise KeyError(f"corrected V2 candidate is not exposed by this overlay: {candidate_id!r}")
        parent = "P25" if candidate_id.startswith("P25_") else "P40"
        return V2Parameterization(
            registry_alias=None,
            display_name=candidate_id,
            source_candidate_id=candidate_id,
            parent_background=parent,
            status="ARCHIVAL_RESEARCH_CANDIDATE",
            complete_bound_row_sha256=row["complete_bound_row_sha256"],
            thermodynamic_surface_fingerprint="SOURCE_ROW_BOUND",
            renewal_fingerprint="SOURCE_ROW_BOUND",
            entropy_convention=row["entropy_representation"],
            temperature_response_classification="ACCESSIBLE_CONTROL",
            rising_resistance_status="1D_EFFECTIVE_RESISTANCE_FLAT",
            fatigue_status="NOT_NAMED_FAMILY_ROW",
            developed_fatigue_status="UNRESOLVED",
            spatial_transfer_status="NOT_TESTED",
            limitations=("EXACT_ID_ONLY", "NOT_EXPERIMENTALLY_CALIBRATED"),
            full_precision_row=_freeze(row),
            evidence=_freeze({"role": "accessible corrected-search control"}),
        )
    if named["complete_bound_row_sha256"] != row["complete_bound_row_sha256"]:
        raise ValueError(f"named V2 row hash mismatch for {named['registry_alias']}")
    required_source = (
        "opening_surface_json", "emission_surface_json",
        "parent_complete_registry_row_json", "parent_material_manifest_json",
        "physics__cleavage_hits", "physics__cleavage_correlation_time_s",
        "entropy_representation", "complete_bound_row_sha256",
    )
    missing = [field for field in required_source if not row.get(field)]
    if missing:
        raise ValueError(f"{candidate_id} lacks complete corrected V2 fields: {missing}")
    return V2Parameterization(
        registry_alias=named["registry_alias"],
        display_name=named["display_name"],
        source_candidate_id=candidate_id,
        parent_background=named["parent_background"],
        status=named["status"],
        complete_bound_row_sha256=row["complete_bound_row_sha256"],
        thermodynamic_surface_fingerprint=named["thermodynamic_surface_fingerprint"],
        renewal_fingerprint=named["renewal_fingerprint"],
        entropy_convention=named["entropy_convention"],
        temperature_response_classification=named["temperature_response_classification"],
        rising_resistance_status=named["rising_resistance_status"],
        fatigue_status=named["fatigue_status"],
        developed_fatigue_status=named["developed_fatigue_status"],
        spatial_transfer_status=named["spatial_transfer_status"],
        limitations=tuple(named["limitations"]),
        full_precision_row=_freeze(row),
        evidence=_freeze(dict(named["response_evidence"])),
    )


def load(name: str) -> V2Parameterization:
    """Load one explicit V2 alias or exact corrected candidate ID."""
    key = str(name).strip()
    if key in _entries():
        return _load_exact(str(_entries()[key]["source_candidate_id"]))
    if key in load_rows():
        return _load_exact(key)
    raise KeyError(f"unknown opt-in V2 parameterization {key!r}")


def load_exact_candidate(candidate_id: str) -> V2Parameterization:
    return _load_exact(str(candidate_id).strip())


def load_for(name: str, loader: str) -> V2Parameterization:
    if loader not in SUPPORTED_LOADERS:
        raise ValueError(f"unsupported V2 parameterization loader {loader!r}")
    return load(name)


def load_checkpoint_reference(payload: Mapping[str, Any]) -> V2Parameterization:
    alias = payload.get("v2_parameterization_alias")
    candidate = payload.get("source_candidate_id")
    record = load(str(alias)) if alias else load_exact_candidate(str(candidate))
    expected = payload.get("complete_bound_row_sha256")
    if expected is not None and expected != record.complete_bound_row_sha256:
        raise ValueError("V2 checkpoint parameter-row hash mismatch")
    return record


def clear_cache_for_testing() -> None:
    _entries.cache_clear()
    _load_exact.cache_clear()


__all__ = [
    "CATALOG_PATH", "SUPPORTED_LOADERS", "NAMED_ALIASES", "EXACT_ONLY_CONTROLS",
    "RESERVED_ABSENT_ALIASES", "V2Parameterization", "load", "load_exact_candidate",
    "load_for", "load_checkpoint_reference", "clear_cache_for_testing",
]
