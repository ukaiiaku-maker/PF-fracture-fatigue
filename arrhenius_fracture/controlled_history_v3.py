"""Prospective classification helpers for the twelve controlled V5 histories."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np

SCHEMA = "v5.one-void-controlled-history/3"
DIFFUSION_SEPARATION_MARGIN = 0.05
DIFFUSION_OPENING_M = 1.05e-5
GROWTH_INTERVAL_COUNT = 8

EXPECTED = {
    "accommodation_limited": "STABLE_SUBGRID_VOID_WITH_PLASTIC_ACCOMMODATION_MINIMUM_ALL_INTERVALS",
    "centered": "DOWNSTREAM_FRONT_ACTIVE",
    "delayed_downstream": "DOWNSTREAM_FRONT_ACTIVE_AFTER_DORMANT_INTERVAL_AND_RELOAD",
    "diffusion_limited": "STABLE_SUBGRID_VOID_WITH_VACANCY_TRANSPORT_MINIMUM_ALL_INTERVALS",
    "downstream_zero_drive": "CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE",
    "embryo_healing": "HEALED_SITE",
    "fixed_mesh_oblique": "CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE",
    "local_remesh_refinement": "DOWNSTREAM_FRONT_ACTIVE_AFTER_QUALIFIED_LOCAL_REFINEMENT",
    "long_ligament": "DOWNSTREAM_FRONT_ACTIVE",
    "negative_offset": "CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE",
    "positive_offset": "CONNECTED_VOID_ZERO_DOWNSTREAM_DRIVE_WITH_QUALIFIED_SOURCE",
    "short_ligament": "DOWNSTREAM_FRONT_ACTIVE",
}


def _canonical(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if is_dataclass(value):
        return _canonical(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(_canonical(value), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_rate_channel(channel: str) -> str:
    """Remove the dimensional suffix before constructing a taxonomy enum."""
    return channel[:-2] if channel.endswith("_s") else channel


def growth_limiter_audit(operations: Sequence[Mapping], *, target_channel: str,
                         required_margin: float = DIFFUSION_SEPARATION_MARGIN) -> dict:
    channels = ("surface_reaction_s", "vacancy_transport_s", "plastic_accommodation_s")
    histories = [row for row in operations if row.get("api") == "accepted_load_growth_interval"]
    rows = []
    for index, operation in enumerate(histories):
        rates = operation.get("rates", {})
        if set(channels) - set(rates):
            rows.append({"interval": index, "passed": False, "failure": "MISSING_SERIES_RATE"})
            continue
        ordered = sorted(channels, key=lambda name: (float(rates[name]), name))
        minimum, next_slowest = ordered[:2]
        denominator = max(float(rates[next_slowest]), np.finfo(float).tiny)
        margin = (float(rates[next_slowest]) - float(rates[minimum])) / denominator
        rows.append({
            "interval": index,
            "minimum_channel": minimum,
            "next_slowest_channel": next_slowest,
            "minimum_rate_s-1": float(rates[minimum]),
            "next_slowest_rate_s-1": float(rates[next_slowest]),
            "relative_separation_margin": margin,
            "required_relative_separation_margin": float(required_margin),
            "passed": minimum == target_channel and margin >= required_margin,
        })
    limiter = canonical_rate_channel(target_channel).upper()
    expected = f"STABLE_SUBGRID_VOID_WITH_{limiter}_MINIMUM_ALL_INTERVALS"
    passed = len(histories) == GROWTH_INTERVAL_COUNT and all(row["passed"] for row in rows)
    return {
        "schema": SCHEMA,
        "target_channel": target_channel,
        "expected_terminal_classification": expected,
        "required_interval_count": GROWTH_INTERVAL_COUNT,
        "observed_interval_count": len(histories),
        "intervals": rows,
        "passed": passed,
    }


def dormant_state_projection(state) -> dict:
    """Exact quantities that a zero-rate interval is forbidden to change."""
    return {
        "competition_candidates": _canonical(state.competition.candidates),
        "hazard_states": _canonical(state.competition.hazard_states),
        "competition_event_index": state.competition.competition_event_index,
        "competition_consumed_event_ids": _canonical(state.competition.consumed_event_ids),
        "rng_state": _canonical(state.rng_state),
        "crack_network": _canonical(state.crack_network),
        "void_state_without_event_history": {
            "sites": _canonical(state.void_state.sites),
            "cavities": _canonical(state.void_state.cavities),
            "length_ledgers": _canonical(state.void_state.length_ledgers),
            "inventory": [state.void_state.available_defect_inventory_area_m2,
                          state.void_state.consumed_defect_inventory_area_m2],
        },
        "active_event_source": _canonical(state.junction_process_state.get("active_event_source", {})),
    }


def dormant_interval_audit(before, after, directional_rows: Sequence[Mapping]) -> dict:
    first = dormant_state_projection(before)
    second = dormant_state_projection(after)
    zero = bool(directional_rows) and all(
        float(row.get("effective_rate_s", math.nan)) == 0.0
        and not row.get("winner", False)
        and not row.get("emitted_event_ids", ())
        for row in directional_rows
    )
    return {
        "schema": SCHEMA,
        "projection_sha256": {"before": _digest(first), "after": _digest(second)},
        "hazard_threshold_rng_graph_topology_unchanged": first == second,
        "all_candidate_effective_rates_zero": zero,
        "no_pending_or_completed_event": zero and first == second,
        "passed": zero and first == second,
    }


def mirrored_source_audit(positive: Mapping, negative: Mapping, *, relative_limit: float = 0.05) -> dict:
    """Compare fixed-laboratory-crack source metrics under y-reflection."""
    transform = np.diag((1.0, -1.0))
    first = np.asarray(positive["tensor_Pa"], dtype=float)
    second = np.asarray(negative["tensor_Pa"], dtype=float)
    expected = transform @ first @ transform
    error = float(np.linalg.norm(second - expected) / max(np.linalg.norm(expected), np.finfo(float).tiny))
    scalar_names = (
        "eta_n_max", "eta_t_max", "local_aspect_ratio_max", "minimum_quality",
        "local_minimum_quality", "normalized_traction",
    )
    scalars = {
        name: {
            "positive": float(positive[name]),
            "negative": float(negative[name]),
            "relative_error": abs(float(positive[name]) - float(negative[name])) /
                              max(abs(float(positive[name])), abs(float(negative[name])), np.finfo(float).tiny),
        }
        for name in scalar_names
    }
    scalar_exact = all(row["positive"] == row["negative"] for row in scalars.values())
    return {
        "schema": SCHEMA,
        "reflection_matrix": transform.tolist(),
        "positive_tensor_Pa": first.tolist(),
        "expected_reflected_tensor_Pa": expected.tolist(),
        "negative_tensor_Pa": second.tolist(),
        "tensor_relative_error": error,
        "frozen_tensor_relative_limit": float(relative_limit),
        "geometry_and_quality_scalars": scalars,
        "geometry_and_quality_scalars_exact": scalar_exact,
        "passed": scalar_exact and error <= relative_limit,
    }


__all__ = [
    "DIFFUSION_OPENING_M", "DIFFUSION_SEPARATION_MARGIN", "EXPECTED", "SCHEMA",
    "canonical_rate_channel", "dormant_interval_audit", "dormant_state_projection",
    "growth_limiter_audit", "mirrored_source_audit",
]
