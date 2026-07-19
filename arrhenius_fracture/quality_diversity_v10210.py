"""Corrected quality-diversity promotion for staged parameter calibration.

v10.2.10 preserves the v10.2.9 quality/diversity policy while making two gate
semantics explicit:

* when fewer candidates pass than the promotion budget, every passer is selected
  before any near-pass candidate; and
* historical-anchor lineage protection is restricted to the analytical DBTT
  promotion, where those lineages have a physical interpretation.

The selector changes only promotion.  It does not alter material parameters,
constitutive responses, or stage scores.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

MODEL_ID = "v10.2.10_quality_diversity_hard_gate"

PARAMETER_FIELDS = {
    "cleave_G00_eV",
    "cleave_gT_eV_per_K",
    "cleave_sigc0_GPa",
    "cleave_sT_GPa_per_K",
    "cleave_exp_a",
    "cleave_exp_n",
    "cleave_floor_frac",
    "emit_G00_eV",
    "emit_gT_eV_per_K",
    "emit_sigc0_GPa",
    "emit_sT_GPa_per_K",
    "emit_exp_a",
    "emit_exp_n",
    "emit_floor_frac",
    "peierls_H0_eV",
    "peierls_activation_entropy_kB",
    "peierls_exp_a",
    "peierls_exp_n",
    "peierls_nu0_s",
    "taylor_H0_eV",
    "taylor_activation_entropy_kB",
    "taylor_exp_a",
    "taylor_exp_n",
    "taylor_nu0_s",
    "taylor_corr_rho_c_m2",
    "taylor_corr_scale",
    "source_sites_per_system",
    "encounter_efficiency",
    "retained_recovery_rate_s",
    "source_refresh_length_um",
    "c_blunt",
}

LOG_PARAMETER_FIELDS = {
    "cleave_G00_eV",
    "cleave_sigc0_GPa",
    "emit_G00_eV",
    "emit_sigc0_GPa",
    "peierls_H0_eV",
    "peierls_nu0_s",
    "taylor_H0_eV",
    "taylor_nu0_s",
    "taylor_corr_rho_c_m2",
    "taylor_corr_scale",
    "source_sites_per_system",
    "encounter_efficiency",
    "retained_recovery_rate_s",
    "source_refresh_length_um",
    "c_blunt",
}

ANALYTICAL_RESPONSE_FIELDS = {
    "low_emission_advantage_fraction",
    "high_emission_advantage_fraction",
    "emission_advantage_monotonic_fraction",
    "high_linearized_Kshield_MPa_sqrt_m",
    "cleavage_temperature_span_ratio",
    "emission_advantage_span",
    "mean_linearized_shield_fraction",
}

FIRST_PASSAGE_RESPONSE_FIELDS = {
    "first_passage_endpoint_ratio",
    "first_passage_monotonic_fraction",
    "first_passage_temperature_span_ratio",
}

RCURVE_RESPONSE_FIELDS = {
    "full_endpoint_ratio",
    "low_R_rise_fraction",
    "high_R_rise_fraction",
    "plasticity_off_endpoint_ratio",
    "shielding_fraction_of_temperature_rise",
    "shielding_fraction_of_high_T_R_rise",
    "backstress_off_high_T_R_rise_MPa_sqrt_m",
    "monotonic_temperature_fraction",
    "full_init_temperature_span_ratio",
    "full_final_temperature_span_ratio",
    "minimum_R_rise_MPa_sqrt_m",
    "maximum_R_rise_MPa_sqrt_m",
    "minimum_R_rise_fraction",
    "maximum_R_rise_fraction",
    "plasticity_fraction_of_mean_R_rise",
    "shielding_fraction_of_mean_R_rise",
}


@dataclass(frozen=True)
class QualityDiversityConfig:
    count: int
    quality_reserve_fraction: float = 0.25
    quality_weight: float = 0.35
    parameter_distance_weight: float = 0.45
    response_distance_weight: float = 0.55
    pool_factor: int = 12
    preserve_anchor_lineages: bool = True

    def validate(self) -> "QualityDiversityConfig":
        if self.count < 1:
            raise ValueError("promotion count must be positive")
        if not 0.0 <= self.quality_reserve_fraction <= 1.0:
            raise ValueError("quality_reserve_fraction must lie in [0, 1]")
        if not 0.0 <= self.quality_weight <= 1.0:
            raise ValueError("quality_weight must lie in [0, 1]")
        if self.parameter_distance_weight < 0.0:
            raise ValueError("parameter_distance_weight must be non-negative")
        if self.response_distance_weight < 0.0:
            raise ValueError("response_distance_weight must be non-negative")
        if self.parameter_distance_weight + self.response_distance_weight <= 0.0:
            raise ValueError("at least one distance weight must be positive")
        if self.pool_factor < 1:
            raise ValueError("pool_factor must be at least one")
        return self


def _finite(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result if math.isfinite(result) else math.nan


def _objective(row: dict[str, Any], key: str) -> float:
    value = _finite(row.get(key))
    return value if math.isfinite(value) else math.inf


def _parameter_fields(rows: list[dict[str, Any]]) -> list[str]:
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    return sorted(
        key
        for key in keys
        if key in PARAMETER_FIELDS or key.startswith("anchor_weight_")
    )


def _response_fields(rows: list[dict[str, Any]], stage: str) -> list[str]:
    keys = set().union(*(row.keys() for row in rows)) if rows else set()
    normalized = str(stage).strip().lower()
    if normalized == "analytical":
        explicit = ANALYTICAL_RESPONSE_FIELDS
        prefixes = (
            "analytical_K_cleave_",
            "analytical_K_first_emission_",
            "analytical_emission_advantage_",
            "analytical_Kshield_",
            "analytical_retained_fraction_",
            "analytical_expected_activations_",
            "analytical_signed_line_",
        )
    elif normalized in {"first", "first-passage", "first_passage"}:
        explicit = FIRST_PASSAGE_RESPONSE_FIELDS
        prefixes = ("K_first_",)
    elif normalized in {"rcurve", "r-curve", "r_curve"}:
        explicit = RCURVE_RESPONSE_FIELDS
        prefixes = ("K_init_", "K_final_")
    else:
        raise ValueError(f"unsupported selection stage {stage!r}")
    return sorted(
        key for key in keys if key in explicit or key.startswith(prefixes)
    )


def _robust_matrix(
    rows: list[dict[str, Any]],
    fields: Iterable[str],
    *,
    log_fields: set[str] | None = None,
) -> tuple[np.ndarray, list[str], dict[str, dict[str, float]]]:
    columns: list[np.ndarray] = []
    retained_fields: list[str] = []
    scaling: dict[str, dict[str, float]] = {}
    log_fields = set() if log_fields is None else set(log_fields)
    for field in fields:
        values = np.asarray([_finite(row.get(field)) for row in rows], dtype=float)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        transform = "linear"
        if field in log_fields and np.all(values[finite] > 0.0):
            values[finite] = np.log10(values[finite])
            transform = "log10"
        median = float(np.median(values[finite]))
        values[~finite] = median
        q10, q25, q75, q90 = np.quantile(values, [0.10, 0.25, 0.75, 0.90])
        scale = max(float(q75 - q25), 0.25 * float(q90 - q10), 1.0e-12)
        z = np.clip((values - median) / scale, -8.0, 8.0)
        if float(np.ptp(z)) <= 1.0e-12:
            continue
        columns.append(z)
        retained_fields.append(field)
        scaling[field] = {
            "median": median,
            "scale": scale,
            "transform": transform,
        }
    if not columns:
        return np.zeros((len(rows), 0), dtype=float), [], scaling
    matrix = np.column_stack(columns)
    matrix /= math.sqrt(matrix.shape[1])
    return matrix, retained_fields, scaling


def _distance(matrix: np.ndarray, index: int, selected: list[int]) -> float:
    if matrix.shape[1] == 0 or not selected:
        return 0.0
    return float(np.min(np.linalg.norm(matrix[selected] - matrix[index], axis=1)))


def _pairwise_summary(matrix: np.ndarray, indices: list[int]) -> dict[str, float]:
    if matrix.shape[1] == 0 or len(indices) < 2:
        return {"minimum": 0.0, "median": 0.0, "maximum": 0.0}
    values: list[float] = []
    for position, left in enumerate(indices[:-1]):
        values.extend(
            np.linalg.norm(matrix[indices[position + 1 :]] - matrix[left], axis=1)
            .astype(float)
            .tolist()
        )
    array = np.asarray(values, dtype=float)
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "maximum": float(np.max(array)),
    }


def _combined_novelty(
    parameter_distance: float,
    response_distance: float,
    parameter_available: bool,
    response_available: bool,
    config: QualityDiversityConfig,
) -> float:
    values: list[float] = []
    weights: list[float] = []
    if parameter_available:
        values.append(parameter_distance / (1.0 + parameter_distance))
        weights.append(config.parameter_distance_weight)
    if response_available:
        values.append(response_distance / (1.0 + response_distance))
        weights.append(config.response_distance_weight)
    if not weights or sum(weights) <= 0.0:
        return 0.0
    return float(np.dot(values, weights) / sum(weights))


def _dominant_anchor(
    row: dict[str, Any], anchor_fields: list[str]
) -> str | None:
    if not anchor_fields:
        return None
    values = np.asarray([_finite(row.get(field)) for field in anchor_fields])
    if not np.any(np.isfinite(values)):
        return None
    values[~np.isfinite(values)] = -math.inf
    return anchor_fields[int(np.argmax(values))]


def _lineage_reserve_active(
    *, stage: str, target_class: str, config: QualityDiversityConfig
) -> bool:
    return bool(
        config.preserve_anchor_lineages
        and str(stage).strip().lower() == "analytical"
        and str(target_class).strip().upper() == "DBTT"
    )


def select_quality_diverse(
    rows: list[dict[str, Any]],
    *,
    pass_key: str,
    objective_key: str,
    target_class: str,
    stage: str,
    config: QualityDiversityConfig,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select a high-quality, non-collapsed promotion set.

    If fewer candidates pass than the promotion budget, every passer is inserted
    first.  Diversity is then used only to fill the remaining slots from the
    highest-quality near-pass pool.
    """
    config = config.validate()
    subset = [
        dict(row)
        for row in rows
        if str(row.get("target_class")) == str(target_class)
    ]
    if not subset:
        return [], {
            "schema": MODEL_ID,
            "target_class": target_class,
            "stage": stage,
            "selected_count": 0,
            "reason": "no candidates",
        }

    quality_order = sorted(
        range(len(subset)),
        key=lambda index: (
            not bool(subset[index].get(pass_key, False)),
            _objective(subset[index], objective_key),
            str(subset[index].get("candidate_id", "")),
        ),
    )
    pass_order = [
        index for index in quality_order if bool(subset[index].get(pass_key, False))
    ]
    fail_order = [
        index for index in quality_order if not bool(subset[index].get(pass_key, False))
    ]
    count = min(int(config.count), len(subset))
    pool_limit = min(len(subset), max(count, int(config.pool_factor) * count))
    if len(pass_order) >= count:
        pool = pass_order[:pool_limit]
    else:
        pool = pass_order + fail_order[: max(pool_limit - len(pass_order), 0)]
    if len(pool) < count:
        pool = quality_order[:count]

    pool_rank = {index: rank for rank, index in enumerate(pool)}
    quality_score = {
        index: 1.0 - rank / max(len(pool) - 1, 1)
        for index, rank in pool_rank.items()
    }
    pool_rows = [subset[index] for index in pool]
    parameter_matrix, parameter_fields, parameter_scaling = _robust_matrix(
        pool_rows,
        _parameter_fields(pool_rows),
        log_fields=LOG_PARAMETER_FIELDS,
    )
    response_matrix, response_fields, response_scaling = _robust_matrix(
        pool_rows,
        _response_fields(pool_rows, stage),
    )
    pool_position = {index: position for position, index in enumerate(pool)}

    selected: list[int] = []
    selection_records: dict[int, dict[str, Any]] = {}

    def add(index: int, reason: str, utility: float) -> None:
        if index in selected or len(selected) >= count:
            return
        position = pool_position[index]
        selected_positions = [pool_position[item] for item in selected]
        d_parameter = _distance(parameter_matrix, position, selected_positions)
        d_response = _distance(response_matrix, position, selected_positions)
        novelty = _combined_novelty(
            d_parameter,
            d_response,
            parameter_matrix.shape[1] > 0,
            response_matrix.shape[1] > 0,
            config,
        )
        selected.append(index)
        selection_records[index] = {
            "selection_reason": reason,
            "selection_utility_at_pick": float(utility),
            "minimum_parameter_distance_at_pick": d_parameter,
            "minimum_response_distance_at_pick": d_response,
            "combined_novelty_at_pick": novelty,
        }

    # Hard gate: if the passing population cannot fill the budget, no passer may
    # be displaced by a novel near-pass candidate.
    if len(pass_order) < count:
        for index in pass_order:
            add(index, "all_passers_reserve", quality_score[index])
    else:
        reserve = min(
            count,
            max(1, int(math.ceil(config.quality_reserve_fraction * count))),
        )
        for index in pool[:reserve]:
            add(index, "quality_reserve", quality_score[index])

    anchor_fields = sorted(
        field for field in parameter_fields if field.startswith("anchor_weight_")
    )
    lineage_active = _lineage_reserve_active(
        stage=stage, target_class=target_class, config=config
    )
    if lineage_active and anchor_fields and len(selected) < count:
        best_by_lineage: dict[str, int] = {}
        for index in pool:
            lineage = _dominant_anchor(subset[index], anchor_fields)
            if lineage is not None and lineage not in best_by_lineage:
                best_by_lineage[lineage] = index
        represented = {
            _dominant_anchor(subset[index], anchor_fields) for index in selected
        }
        for lineage, index in sorted(
            best_by_lineage.items(), key=lambda item: pool_rank[item[1]]
        ):
            if lineage not in represented and len(selected) < count:
                add(index, "anchor_lineage_reserve", quality_score[index])
                represented.add(lineage)

    while len(selected) < count:
        selected_positions = [pool_position[item] for item in selected]
        best_index: int | None = None
        best_key: tuple[float, bool, float, int] | None = None
        for index in pool:
            if index in selected:
                continue
            position = pool_position[index]
            d_parameter = _distance(parameter_matrix, position, selected_positions)
            d_response = _distance(response_matrix, position, selected_positions)
            novelty = _combined_novelty(
                d_parameter,
                d_response,
                parameter_matrix.shape[1] > 0,
                response_matrix.shape[1] > 0,
                config,
            )
            utility = (
                config.quality_weight * quality_score[index]
                + (1.0 - config.quality_weight) * novelty
            )
            key = (
                float(utility),
                bool(subset[index].get(pass_key, False)),
                quality_score[index],
                -pool_rank[index],
            )
            if best_key is None or key > best_key:
                best_key = key
                best_index = index
        if best_index is None or best_key is None:
            break
        add(best_index, "quality_diversity", best_key[0])

    # Defensive deterministic fill for degenerate feature sets.
    for index in pool:
        if len(selected) >= count:
            break
        add(index, "quality_fallback", quality_score[index])

    selected_positions = [pool_position[index] for index in selected]
    for rank, index in enumerate(selected, start=1):
        position = pool_position[index]
        others = [candidate for candidate in selected_positions if candidate != position]
        d_parameter = _distance(parameter_matrix, position, others)
        d_response = _distance(response_matrix, position, others)
        subset[index].update(
            {
                "quality_rank_within_class": quality_order.index(index) + 1,
                "quality_diversity_rank": rank,
                "quality_score_percentile": quality_score[index],
                "dominant_anchor_lineage": _dominant_anchor(
                    subset[index], anchor_fields
                )
                or "",
                "nearest_selected_parameter_distance": d_parameter,
                "nearest_selected_response_distance": d_response,
                "nearest_selected_combined_distance": _combined_novelty(
                    d_parameter,
                    d_response,
                    parameter_matrix.shape[1] > 0,
                    response_matrix.shape[1] > 0,
                    config,
                ),
                **selection_records[index],
            }
        )

    quality_only = pool[:count]
    quality_only_positions = [pool_position[index] for index in quality_only]
    selected_ids = [str(subset[index].get("candidate_id")) for index in selected]
    passer_ids = {
        str(subset[index].get("candidate_id")) for index in pass_order[:count]
    }
    selected_id_set = set(selected_ids)
    audit = {
        "schema": MODEL_ID,
        "target_class": target_class,
        "stage": stage,
        "pass_key": pass_key,
        "objective_key": objective_key,
        "input_count": len(subset),
        "passed_count": len(pass_order),
        "pool_count": len(pool),
        "selected_count": len(selected),
        "config": asdict(config),
        "lineage_reserve_active": lineage_active,
        "all_passers_retained": bool(
            len(pass_order) >= count or passer_ids.issubset(selected_id_set)
        ),
        "parameter_fields": parameter_fields,
        "response_fields": response_fields,
        "parameter_scaling": parameter_scaling,
        "response_scaling": response_scaling,
        "selected_candidate_ids": selected_ids,
        "selected_reasons": {
            str(subset[index].get("candidate_id")): selection_records[index][
                "selection_reason"
            ]
            for index in selected
        },
        "selected_parameter_pairwise_distance": _pairwise_summary(
            parameter_matrix, selected_positions
        ),
        "selected_response_pairwise_distance": _pairwise_summary(
            response_matrix, selected_positions
        ),
        "quality_only_parameter_pairwise_distance": _pairwise_summary(
            parameter_matrix, quality_only_positions
        ),
        "quality_only_response_pairwise_distance": _pairwise_summary(
            response_matrix, quality_only_positions
        ),
        "selection_changes_promotion_set": {
            str(subset[index].get("candidate_id")) for index in selected
        }
        != {
            str(subset[index].get("candidate_id")) for index in quality_only
        },
    }
    if not audit["all_passers_retained"]:
        raise RuntimeError("quality-diversity selector displaced a stage passer")
    return [subset[index] for index in selected], audit


__all__ = [
    "MODEL_ID",
    "PARAMETER_FIELDS",
    "QualityDiversityConfig",
    "select_quality_diverse",
]
