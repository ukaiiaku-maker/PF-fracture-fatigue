"""Fail-closed public registry for named Legacy and corrected V2 response rows.

The catalog is an additive naming and provenance layer.  Full-precision rows
remain owned by their sealed source registries; this module verifies the
catalog copy against those sources before returning a cached immutable record.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

from .canonical_v2_registry_v10230 import load_rows as load_v2_source_rows


ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "named_parameterization_registry.json"
LEGACY_SOURCE = (
    ROOT / "arrhenius_fracture/data/materials/"
    "v10_2_27_v913_four_class_paper_registry.csv"
)
DEFAULT_FOUR_ALIASES = ("Peak", "DBTT", "weakT", "ceramic")
SUPPORTED_LOADERS = (
    "analytical",
    "monotonic_reduced_1D",
    "fatigue_reduced_1D",
    "PF_sharp_front",
    "FEM_CZM",
    "checkpoint_serialization",
)
ALLOWED_STATUSES = (
    "LEGACY_CANONICAL_PARAMETERIZATION",
    "NAMED_V2_PROSPECTIVE_PARAMETERIZATION",
    "NAMED_V2_ALTERNATE_PARAMETERIZATION",
    "ARCHIVAL_RESEARCH_CANDIDATE",
)
REQUIRED_ENTRY_FIELDS = (
    "display_name", "registry_alias", "aliases", "source_candidate_id",
    "generation", "parent_background", "status", "callable",
    "canonical_default", "source_commit", "transfer_source_commit",
    "complete_bound_row_sha256",
    "thermodynamic_surface_fingerprint", "renewal_fingerprint",
    "entropy_convention", "temperature_response_classification",
    "rising_resistance_status", "fatigue_status", "developed_fatigue_status",
    "spatial_transfer_status",
    "limitations", "cross_code_loaders", "full_precision_row",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _legacy_rows() -> dict[str, dict[str, str]]:
    with LEGACY_SOURCE.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {row["candidate_id"]: row for row in rows}


def validate_entry(entry: Mapping[str, Any], source_row: Mapping[str, str]) -> None:
    """Validate one complete catalog entry without filling missing fields."""
    missing = [key for key in REQUIRED_ENTRY_FIELDS if key not in entry]
    if missing:
        raise ValueError(f"catalog entry lacks required fields: {missing}")
    if not entry["registry_alias"] or not entry["source_candidate_id"]:
        raise ValueError("catalog aliases and source identities must be nonempty")
    if entry["callable"] is not True:
        raise ValueError("active named catalog entries must be explicitly callable")
    if entry["status"] not in ALLOWED_STATUSES:
        raise ValueError(f"unknown catalog status {entry['status']!r}")
    if entry["source_candidate_id"] != source_row.get("candidate_id"):
        raise ValueError("catalog source candidate does not match its source row")
    if dict(entry["full_precision_row"]) != dict(source_row):
        raise ValueError("catalog full-precision row differs from its sealed source")
    if entry["generation"] == "LEGACY":
        expected_hash = canonical_hash(dict(source_row))
    elif entry["generation"] == "CORRECTED_JOINT_THERMODYNAMIC_V2":
        expected_hash = source_row.get("complete_bound_row_sha256")
        required_v2 = (
            "opening_surface_json", "emission_surface_json",
            "parent_complete_registry_row_json", "parent_material_manifest_json",
            "physics__cleavage_hits", "physics__cleavage_correlation_time_s",
            "entropy_representation", "complete_bound_row_sha256",
        )
        absent = [name for name in required_v2 if not source_row.get(name)]
        if absent:
            raise ValueError(f"V2 row lacks complete source fields: {absent}")
    else:
        raise ValueError(f"unknown parameterization generation {entry['generation']!r}")
    if entry["complete_bound_row_sha256"] != expected_hash:
        raise ValueError("catalog complete-bound-row hash mismatch")
    if tuple(entry["cross_code_loaders"]) != SUPPORTED_LOADERS:
        raise ValueError("catalog entry does not declare every supported loader")


@dataclass(frozen=True)
class NamedParameterization:
    display_name: str
    registry_alias: str
    aliases: tuple[str, ...]
    source_candidate_id: str
    generation: str
    parent_background: str
    status: str
    canonical_default: bool
    source_commit: str
    transfer_source_commit: str
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
            "parameterization_alias": self.registry_alias,
            "source_candidate_id": self.source_candidate_id,
            "complete_bound_row_sha256": self.complete_bound_row_sha256,
            "generation": self.generation,
        }


@dataclass(frozen=True)
class _Catalog:
    records: Mapping[str, NamedParameterization]
    aliases: Mapping[str, NamedParameterization]
    exact_candidates: Mapping[str, NamedParameterization]
    legacy_options: Mapping[str, NamedParameterization]


@lru_cache(maxsize=1)
def _catalog() -> _Catalog:
    payload = json.loads(CATALOG_PATH.read_text())
    if payload.get("schema") != "v10.2.30_named_parameterization_registry_v1":
        raise ValueError("named parameterization catalog schema mismatch")
    legacy = _legacy_rows()
    v2 = load_v2_source_rows()
    records: dict[str, NamedParameterization] = {}
    aliases: dict[str, NamedParameterization] = {}
    exact: dict[str, NamedParameterization] = {}
    options: dict[str, NamedParameterization] = {}
    for entry in payload.get("entries", []):
        sources = legacy if entry.get("generation") == "LEGACY" else v2
        candidate = entry.get("source_candidate_id", "")
        if candidate not in sources:
            raise ValueError(f"catalog source candidate is absent: {candidate!r}")
        validate_entry(entry, sources[candidate])
        record = NamedParameterization(
            display_name=entry["display_name"],
            registry_alias=entry["registry_alias"],
            aliases=tuple(entry["aliases"]),
            source_candidate_id=candidate,
            generation=entry["generation"],
            parent_background=entry["parent_background"],
            status=entry["status"],
            canonical_default=bool(entry["canonical_default"]),
            source_commit=entry["source_commit"],
            transfer_source_commit=entry["transfer_source_commit"],
            complete_bound_row_sha256=entry["complete_bound_row_sha256"],
            thermodynamic_surface_fingerprint=entry["thermodynamic_surface_fingerprint"],
            renewal_fingerprint=entry["renewal_fingerprint"],
            entropy_convention=entry["entropy_convention"],
            temperature_response_classification=entry["temperature_response_classification"],
            rising_resistance_status=entry["rising_resistance_status"],
            fatigue_status=entry["fatigue_status"],
            developed_fatigue_status=entry["developed_fatigue_status"],
            spatial_transfer_status=entry["spatial_transfer_status"],
            limitations=tuple(entry["limitations"]),
            full_precision_row=_freeze(dict(entry["full_precision_row"])),
            evidence=_freeze(dict(entry["response_evidence"])),
        )
        if record.registry_alias in records or candidate in exact:
            raise ValueError("duplicate named parameterization or source candidate")
        records[record.registry_alias] = record
        exact[candidate] = record
        for alias in record.aliases:
            if alias in aliases:
                raise ValueError(f"duplicate parameterization alias {alias!r}")
            aliases[alias] = record
        if record.generation == "LEGACY":
            option = record.full_precision_row["option_key"]
            if option in options:
                raise ValueError("duplicate legacy option key")
            options[option] = record
    if tuple(payload.get("default_four_aliases", ())) != DEFAULT_FOUR_ALIASES:
        raise ValueError("default four-class aliases changed")
    if set(aliases) != set(payload.get("alias_index", {})):
        raise ValueError("catalog alias index is stale")
    for alias, primary in payload["alias_index"].items():
        if aliases[alias].registry_alias != primary:
            raise ValueError("catalog alias index points to the wrong row")
    return _Catalog(
        MappingProxyType(records), MappingProxyType(aliases),
        MappingProxyType(exact), MappingProxyType(options),
    )


def load(alias: str, *, required_status: str | None = None) -> NamedParameterization:
    key = str(alias).strip()
    try:
        record = _catalog().aliases[key]
    except KeyError as exc:
        raise KeyError(f"unknown named parameterization alias {key!r}") from exc
    if required_status is not None and record.status != required_status:
        raise ValueError(
            f"parameterization {key!r} has status {record.status!r}, "
            f"not required status {required_status!r}"
        )
    return record


def load_exact_candidate(candidate_id: str) -> NamedParameterization:
    key = str(candidate_id).strip()
    try:
        return _catalog().exact_candidates[key]
    except KeyError as exc:
        raise KeyError(f"unknown exact parameterization candidate {key!r}") from exc


def load_for(alias: str, loader: str, *, required_status: str | None = None) -> NamedParameterization:
    if loader not in SUPPORTED_LOADERS:
        raise ValueError(f"unsupported parameterization loader {loader!r}")
    return load(alias, required_status=required_status)


def load_analytical(alias: str) -> NamedParameterization:
    return load_for(alias, "analytical")


def load_monotonic_reduced_1d(alias: str) -> NamedParameterization:
    return load_for(alias, "monotonic_reduced_1D")


def load_fatigue_reduced_1d(alias: str) -> NamedParameterization:
    return load_for(alias, "fatigue_reduced_1D")


def load_pf_sharp_front(alias: str) -> NamedParameterization:
    return load_for(alias, "PF_sharp_front")


def load_fem_czm(alias: str) -> NamedParameterization:
    return load_for(alias, "FEM_CZM")


def load_checkpoint_serialization(alias: str) -> NamedParameterization:
    return load_for(alias, "checkpoint_serialization")


def default_four_classes() -> tuple[NamedParameterization, ...]:
    return tuple(load(alias) for alias in DEFAULT_FOUR_ALIASES)


def load_checkpoint_reference(checkpoint: Mapping[str, Any]) -> NamedParameterization:
    """Resolve new identities and historical option/class checkpoint fields."""
    for field in ("parameterization_alias", "registry_alias"):
        value = checkpoint.get(field)
        if value:
            return load(str(value))
    for field in ("source_candidate_id", "candidate_id"):
        value = checkpoint.get(field)
        if value and str(value) in _catalog().exact_candidates:
            return load_exact_candidate(str(value))
    for field in ("parameter_option", "option_key"):
        value = checkpoint.get(field)
        if value and str(value) in _catalog().legacy_options:
            return _catalog().legacy_options[str(value)]
    material_class = str(checkpoint.get("material_class", "")).strip()
    class_alias = {"peak": "Peak", "dbtt": "DBTT", "weakt": "weakT", "ceramic": "ceramic"}
    if material_class.lower() in class_alias:
        return load(class_alias[material_class.lower()])
    raise KeyError("checkpoint contains no recognized parameterization identity")


def clear_cache_for_testing() -> None:
    _catalog.cache_clear()


__all__ = [
    "CATALOG_PATH", "DEFAULT_FOUR_ALIASES", "SUPPORTED_LOADERS", "ALLOWED_STATUSES",
    "NamedParameterization", "canonical_hash", "validate_entry", "load",
    "load_exact_candidate", "load_for", "load_analytical",
    "load_monotonic_reduced_1d", "load_fatigue_reduced_1d",
    "load_pf_sharp_front", "load_fem_czm", "load_checkpoint_serialization",
    "default_four_classes",
    "load_checkpoint_reference", "clear_cache_for_testing",
]
