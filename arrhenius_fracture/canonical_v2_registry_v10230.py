"""Fail-closed loader for the immutable V2.1 prospective material rows.

The CSV remains the single authoritative source.  Adapters may expose a row to
different reduced or spatial engines, but may not supply a missing field or
change the canonical serialized row.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer/production_candidate_rows.csv"
MANIFEST = ROOT / "analysis_outputs/corrected_joint_search_v2_1_and_1d_transfer/production_candidate_manifest.json"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load_rows(path: str | Path = REGISTRY) -> dict[str, dict[str, str]]:
    source = Path(path)
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 8:
        raise ValueError(f"canonical V2.1 registry must contain eight rows; found {len(rows)}")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = row.get("candidate_id", "")
        if not candidate_id or candidate_id in result:
            raise ValueError("canonical V2.1 registry has a missing or duplicate candidate_id")
        required = (
            "complete_bound_row_sha256", "opening_surface_json",
            "emission_surface_json", "parent_complete_registry_row_json",
            "parent_material_manifest_json", "physics__cleavage_hits",
            "physics__cleavage_correlation_time_s",
        )
        missing = [name for name in required if not row.get(name)]
        if missing:
            raise ValueError(f"{candidate_id} lacks canonical fields {missing}")
        result[candidate_id] = row
    manifest = json.loads(Path(MANIFEST).read_text())
    expected = {item["candidate_id"]: item["complete_bound_row_sha256"]
                for item in manifest["candidates"]}
    actual = {key: row["complete_bound_row_sha256"] for key, row in result.items()}
    if actual != expected:
        raise ValueError("canonical V2.1 row hashes differ from the sealed manifest")
    return result


def load_row(candidate_id: str, path: str | Path = REGISTRY) -> dict[str, str]:
    rows = load_rows(path)
    if candidate_id not in rows:
        raise KeyError(f"unknown canonical candidate alias {candidate_id!r}")
    return rows[candidate_id].copy()


def decoded_contract(row: dict[str, str]) -> dict[str, Any]:
    """Return the exact JSON-bound material contract used by every adapter."""
    return {
        "candidate_id": row["candidate_id"],
        "parent_id": row["parent_id"],
        "opening_surface": json.loads(row["opening_surface_json"]),
        "emission_surface": json.loads(row["emission_surface_json"]),
        "parent_row": json.loads(row["parent_complete_registry_row_json"]),
        "parent_manifest": json.loads(row["parent_material_manifest_json"]),
        "opening_attempt_frequency_s": float(row["opening_attempt_frequency_s"]),
        "emission_attempt_frequency_s": float(row["emission_attempt_frequency_s"]),
        "cleavage_hits": float(row["physics__cleavage_hits"]),
        "cleavage_correlation_time_s": float(row["physics__cleavage_correlation_time_s"]),
        "complete_bound_row_sha256": row["complete_bound_row_sha256"],
        "opening_surface_sha256": row["opening_surface_sha256"],
        "emission_surface_sha256": row["emission_surface_sha256"],
    }


def round_trip(row: dict[str, str]) -> dict[str, str]:
    """JSON round trip without numeric parsing, preserving CSV field bytes."""
    return json.loads(json.dumps(row, sort_keys=True, separators=(",", ":")))


class ThermodynamicBarrierAdapter:
    """One exact surface exposed through all supported barrier method names."""

    def __init__(self, surface, parent: dict[str, Any]):
        self.surface = surface
        self.parent = dict(parent)

    def values_eV(self, stress_Pa, temperature_K):
        return self.surface.G_eV(stress_Pa, temperature_K)

    def barrier_eV(self, stress_Pa, temperature_K):
        return self.surface.G_eV(stress_Pa, temperature_K)

    def rate(self, stress_Pa, temperature_K):
        return self.surface.raw_rate_s(stress_Pa, temperature_K)

    @property
    def Tref_K(self) -> float:
        return 300.0

    @property
    def sigc0_Pa(self) -> float:
        return float(self.parent["sigc0_Pa"])

    @property
    def floor_fraction(self) -> float:
        return float(self.parent["floor_fraction"])

    @property
    def floor_min_eV(self) -> float:
        return float(self.parent["floor_min_eV"])

    @property
    def floor_max_fraction(self) -> float:
        return float(self.parent["floor_max_fraction"])

    def parity(self, stress_Pa, temperature_K: float) -> bool:
        expected = np.asarray(self.surface.G_eV(stress_Pa, temperature_K))
        return bool(np.array_equal(expected, np.asarray(self.values_eV(stress_Pa, temperature_K)))
                    and np.array_equal(expected, np.asarray(self.barrier_eV(stress_Pa, temperature_K))))


def surface_adapters(row: dict[str, str]):
    from scripts.corrected_thermodynamic_joint_search_v10230 import deserialize_surface

    manifest = json.loads(row["parent_material_manifest_json"])
    opening = deserialize_surface(json.loads(row["opening_surface_json"]))
    emission = deserialize_surface(json.loads(row["emission_surface_json"]))
    return (
        ThermodynamicBarrierAdapter(opening, manifest["cleavage"]),
        ThermodynamicBarrierAdapter(emission, manifest["emission"]),
    )


__all__ = ["REGISTRY", "MANIFEST", "ThermodynamicBarrierAdapter", "canonical_hash",
           "decoded_contract", "load_row", "load_rows", "round_trip", "surface_adapters"]
