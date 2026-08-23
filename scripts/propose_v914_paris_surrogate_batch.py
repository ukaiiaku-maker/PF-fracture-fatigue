#!/usr/bin/env python3
"""Propose a small adaptive Paris-objective batch from the historical pool."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import norm


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def finite(value) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def pairwise_sq(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.maximum(np.sum(a * a, axis=1)[:, None] + np.sum(b * b, axis=1)[None, :] - 2.0 * a @ b.T, 0.0)


def gp_predict(x: np.ndarray, y: np.ndarray, query: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    distances = np.sqrt(pairwise_sq(x, x))
    positive = distances[distances > 0.0]
    length = float(np.median(positive)) if positive.size else 1.0
    length = max(length, 1.0e-6)
    kernel = np.exp(-0.5 * pairwise_sq(x, x) / length**2)
    kernel.flat[:: len(x) + 1] += 1.0e-6
    factor = cho_factor(kernel, lower=True, check_finite=True)
    cross = np.exp(-0.5 * pairwise_sq(query, x) / length**2)
    mean = cross @ cho_solve(factor, y)
    solved = cho_solve(factor, cross.T)
    variance = np.maximum(1.0 - np.sum(cross * solved.T, axis=1), 1.0e-12)
    return mean, np.sqrt(variance), length


def descriptor_response(
    registry_ids: set[str], fits: list[dict[str, str]], fidelity: str,
    response_field: str = "objective",
) -> tuple[list[str], dict[str, float], set[str]]:
    """Separate measured local-slope descriptors from censored/invalid cases."""
    selected = [row for row in fits if row["fidelity"] == fidelity and row["candidate_id"] in registry_ids]
    evaluated = {row["candidate_id"] for row in selected}
    response = {
        row["candidate_id"]: finite(row[response_field])
        for row in selected
        if math.isfinite(finite(row.get(response_field)))
        and row.get("descriptor_status", "") in {
            "PROVISIONAL_THREE_POINT_LOCAL_SLOPE",
            "FOUR_POINT_INVALID_REGION",
            "VALID_PARIS_REGION",
        }
    }
    return sorted(evaluated), response, evaluated - set(response)


def select_exploitation_and_exploration(
    query: np.ndarray,
    acquisition: np.ndarray,
    observed: np.ndarray,
    *,
    exploitation_count: int,
    exploration_count: int,
    diversity_length: float,
) -> tuple[list[int], list[str]]:
    """Select slope-directed points plus a protected maximin exploration reserve."""
    selected: list[int] = []
    roles: list[str] = []
    remaining = acquisition.copy()
    for _ in range(min(exploitation_count, len(query))):
        chosen = int(np.argmax(remaining))
        selected.append(chosen); roles.append("LOW_M_EXPLOITATION")
        distance = np.sqrt(np.sum((query - query[chosen]) ** 2, axis=1))
        remaining *= 1.0 - np.exp(-0.5 * (distance / max(0.25 * diversity_length, 1.0e-6)) ** 2)
        remaining[selected] = -np.inf
    for _ in range(min(exploration_count, len(query) - len(selected))):
        anchors = observed if not selected else np.vstack((observed, query[selected]))
        distance = np.sqrt(pairwise_sq(query, anchors))
        score = np.min(distance, axis=1)
        score[selected] = -np.inf
        chosen = int(np.argmax(score))
        selected.append(chosen); roles.append("HISTORICAL_DOMAIN_MAXIMIN_EXPLORATION")
    return selected, roles


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("artifacts/v914_paris_reparameterization/candidate_pool.csv"))
    parser.add_argument("--registry", type=Path, default=Path("artifacts/v914_paris_reparameterization/candidate_registry.csv"))
    parser.add_argument("--fits", type=Path, default=Path("runs/v914_paris_reparameterization_v1/analysis/paris_candidate_fits.csv"))
    parser.add_argument("--fidelity", default="level1")
    parser.add_argument(
        "--response-mode", choices=("objective", "developed-m"), default="objective",
        help="Use developed-m for the shape-first, low-slope search.",
    )
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--exploration-count", type=int, default=2)
    parser.add_argument(
        "--minimum-descriptors",
        type=int,
        default=3,
        help="Minimum measured valid/provisional local-slope descriptors; use two only for an audited initial exploratory iteration.",
    )
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--target-m-min", type=float, default=2.0)
    parser.add_argument("--target-m-max", type=float, default=10.0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 4 <= args.batch_size <= 8:
        raise SystemExit("batch-size must lie in the requested 4-8 range")
    if not 2 <= args.exploration_count <= 3:
        raise SystemExit("exploration-count must lie in the requested 2-3 range")
    if not 4 <= args.batch_size - args.exploration_count <= 6:
        raise SystemExit("batch must retain 4-6 low-m exploitation candidates")
    if args.minimum_descriptors < 2:
        raise SystemExit("minimum-descriptors must be at least two")
    pool = read_csv(args.pool); registry = read_csv(args.registry); fits = read_csv(args.fits)
    coordinates = [field for field in pool[0] if field.startswith("u__")]
    pool_by_id = {row["candidate_id"]: row for row in pool}
    evaluated_ids = {row["candidate_id"] for row in registry}
    response_field = "m" if args.response_mode == "developed-m" else "objective"
    observed_ids, response, invalid_ids = descriptor_response(
        evaluated_ids, fits, args.fidelity, response_field
    )
    descriptor_ids = sorted(response)
    finite_values = [response[candidate] for candidate in descriptor_ids]
    if len(observed_ids) < 8 or len(finite_values) < args.minimum_descriptors:
        raise SystemExit(
            f"adaptive surrogate requires >=8 evaluated candidates and >={args.minimum_descriptors} "
            "measured local-slope descriptors"
        )
    y_raw = np.asarray([response[candidate] for candidate in descriptor_ids])
    y_mean, y_scale = float(np.mean(y_raw)), max(float(np.std(y_raw)), 1.0e-12)
    y = (y_raw - y_mean) / y_scale
    x = np.asarray([[finite(pool_by_id[candidate][field]) for field in coordinates] for candidate in descriptor_ids])
    available = [row for row in pool if row["candidate_id"] not in evaluated_ids]
    query = np.asarray([[finite(row[field]) for field in coordinates] for row in available])
    mean, std, length = gp_predict(x, y, query)
    # Feasibility is learned separately from all evaluated candidates.  This
    # prevents censored/invalid families from acquiring fabricated slopes or
    # objective penalties while still steering the acquisition away from
    # regions that only produced censors, bursts, or arrest.
    feasibility_x = np.asarray([[finite(pool_by_id[candidate][field]) for field in coordinates] for candidate in observed_ids])
    feasibility_y = np.asarray([1.0 if candidate in response else 0.0 for candidate in observed_ids])
    feasibility_mean, feasibility_std, feasibility_length = gp_predict(feasibility_x, feasibility_y, query)
    feasibility_probability = np.clip(feasibility_mean, 0.0, 1.0)
    incumbent = float(np.min(y))
    improvement = incumbent - mean
    z = improvement / std
    conditional_improvement = improvement * norm.cdf(z) + std * norm.pdf(z)
    acquisition = conditional_improvement * feasibility_probability
    extrapolation = np.zeros(len(query), dtype=float)
    low_slope_direction_fields: list[str] = []
    predicted_m = mean * y_scale + y_mean
    predicted_m_std = std * y_scale
    target_probability = np.zeros(len(query), dtype=float)
    if args.response_mode == "developed-m":
        # With only a few real slope descriptors, explicitly seek coordinates
        # beyond the current best in the direction away from the steep-slope
        # centroid.  Feasibility remains a separate model; censors/bursts never
        # receive fabricated slopes.
        best_id = min(descriptor_ids, key=lambda candidate: response[candidate])
        best_x = np.asarray([finite(pool_by_id[best_id][field]) for field in coordinates])
        other = np.asarray([
            [finite(pool_by_id[candidate][field]) for field in coordinates]
            for candidate in descriptor_ids if candidate != best_id
        ])
        # A's completed causal audit has zero blunting/shielding feedback and
        # attributes the majority of its slope to the cleavage EXP surface and
        # three-hit transform.  Define extrapolation in that responsible
        # five-coordinate subspace; other coordinates retain feasibility/level
        # roles but cannot compensate for pathological shape.
        low_slope_direction_fields = [
            field for field in coordinates if field in {
                "u__cleave_G00_eV", "u__cleave_sigc0_GPa", "u__cleave_exp_a",
                "u__cleave_exp_n", "u__cleave_floor_frac",
            }
        ]
        active = np.asarray([coordinates.index(field) for field in low_slope_direction_fields])
        direction = np.zeros_like(best_x)
        direction[active] = best_x[active] - np.mean(other[:, active], axis=0)
        norm2 = max(float(direction @ direction), 1.0e-12)
        extrapolation = (query - best_x) @ direction / norm2
        target_probability = np.maximum(
            norm.cdf((args.target_m_max - predicted_m) / predicted_m_std)
            - norm.cdf((args.target_m_min - predicted_m) / predicted_m_std),
            0.0,
        ) * feasibility_probability
        # Expected improvement is referenced to the desired upper slope, not
        # merely to the current (still pathological) incumbent.  The causal
        # extrapolation factor prioritizes movement beyond A without allowing
        # distance alone to outweigh credible low-m probability.
        desired_improvement = args.target_m_max - predicted_m
        desired_z = desired_improvement / predicted_m_std
        desired_ei = desired_improvement * norm.cdf(desired_z) + predicted_m_std * norm.pdf(desired_z)
        acquisition = desired_ei * feasibility_probability * (1.0 + np.maximum(extrapolation, 0.0))
        if int(np.sum(extrapolation > 0.0)) >= args.batch_size:
            acquisition[extrapolation <= 0.0] = -np.inf
    selected, acquisition_roles = select_exploitation_and_exploration(
        query,
        acquisition,
        feasibility_x,
        exploitation_count=args.batch_size - args.exploration_count,
        exploration_count=args.exploration_count,
        diversity_length=length,
    )
    batch = []
    for rank, (index, acquisition_role) in enumerate(zip(selected, acquisition_roles), 1):
        source = dict(available[index])
        source.update({
            "paris_initial_design_order": "",
            "paris_design_role": f"ADAPTIVE_SURROGATE_ITERATION_{args.iteration}",
            "surrogate_iteration": args.iteration,
            "surrogate_batch_rank": rank,
            "surrogate_predicted_objective": float(mean[index] * y_scale + y_mean),
            "surrogate_predicted_std": float(std[index] * y_scale),
            "surrogate_expected_improvement": float(acquisition[index] * y_scale),
            "surrogate_descriptor_expected_improvement": float(conditional_improvement[index] * y_scale),
            "surrogate_feasibility_probability": float(feasibility_probability[index]),
            "surrogate_feasibility_std": float(feasibility_std[index]),
            "surrogate_kernel_length": length,
            "surrogate_response_mode": args.response_mode,
            "surrogate_predicted_m": float(mean[index] * y_scale + y_mean) if args.response_mode == "developed-m" else "",
            "surrogate_predicted_m_std": float(std[index] * y_scale) if args.response_mode == "developed-m" else "",
            "surrogate_low_slope_extrapolation_score": float(extrapolation[index]),
            "surrogate_acquisition_role": acquisition_role,
        })
        batch.append(source)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "surrogate_batch.csv", batch, list(batch[0]))
    if args.response_mode == "developed-m":
        selected_ids = {row["candidate_id"] for row in batch}
        ranked_indices = sorted(
            range(len(available)),
            key=lambda index: (-target_probability[index], predicted_m[index]),
        )[:25]
        low_m_rows = [
            {
                "domain_rank": rank,
                "candidate_id": available[index]["candidate_id"],
                "selected_for_batch": available[index]["candidate_id"] in selected_ids,
                "predicted_m": predicted_m[index],
                "predicted_m_std": predicted_m_std[index],
                "predicted_m_lower_1p5sigma": predicted_m[index] - 1.5 * predicted_m_std[index],
                "credible_target_probability_times_feasibility": target_probability[index],
                "feasibility_probability": feasibility_probability[index],
                "low_slope_extrapolation_score": extrapolation[index],
            }
            for rank, index in enumerate(ranked_indices, 1)
        ]
        write_csv(
            args.out / "surrogate_predicted_low_m_region.csv",
            low_m_rows,
            list(low_m_rows[0]),
        )
    combined = registry + batch
    fields = list(dict.fromkeys(key for row in combined for key in row))
    write_csv(args.out / "candidate_registry.csv", combined, fields)
    audit = {
        "schema": "v914_paris_surrogate_batch_v2", "iteration": args.iteration,
        "batch_size": len(batch), "fidelity_response": args.fidelity,
        "observations": len(observed_ids), "finite_objectives": len(finite_values),
        "minimum_descriptors_required": args.minimum_descriptors,
        "initial_two_descriptor_exploration": args.minimum_descriptors == 2,
        "measured_descriptor_candidates": descriptor_ids,
        "censored_or_invalid_candidates": sorted(invalid_ids),
        "invalid_objective_penalty": None, "kernel": "isotropic_squared_exponential",
        "kernel_length": length, "acquisition": (
            "desired_m_expected_improvement_with_causal_extrapolation_and_batch_diversity"
            if args.response_mode == "developed-m" else "expected_improvement_with_batch_diversity_penalty"
        ),
        "feasibility_kernel_length": feasibility_length,
        "acquisition_feasibility_weighted": True,
        "candidate_ids": [row["candidate_id"] for row in batch],
        "exploitation_candidate_ids": [row["candidate_id"] for row in batch if row["surrogate_acquisition_role"] == "LOW_M_EXPLOITATION"],
        "exploration_candidate_ids": [row["candidate_id"] for row in batch if row["surrogate_acquisition_role"] == "HISTORICAL_DOMAIN_MAXIMIN_EXPLORATION"],
        "response_mode": args.response_mode,
        "response_field": response_field,
        "shape_objective_is_primary": args.response_mode == "developed-m",
        "level_objective_compensation_allowed": False,
        "low_slope_extrapolation_active": args.response_mode == "developed-m",
        "low_slope_direction_fields": low_slope_direction_fields,
    }
    if args.response_mode == "developed-m":
        minimum_index = int(np.argmin(predicted_m))
        credible_index = int(np.argmax(target_probability))
        audit["domain_predicted_minimum_m"] = float(predicted_m[minimum_index])
        audit["domain_predicted_minimum_m_candidate_id"] = available[minimum_index]["candidate_id"]
        audit["domain_best_credible_target_probability_times_feasibility"] = float(target_probability[credible_index])
        audit["domain_best_credible_target_candidate_id"] = available[credible_index]["candidate_id"]
        audit["provisional_target_m"] = [args.target_m_min, args.target_m_max]
    (args.out / "surrogate_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"V914_PARIS_SURROGATE_BATCH iteration={args.iteration} size={len(batch)} observations={len(observed_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
