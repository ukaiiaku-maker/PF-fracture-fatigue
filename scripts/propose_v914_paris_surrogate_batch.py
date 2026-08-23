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
    registry_ids: set[str], fits: list[dict[str, str]], fidelity: str
) -> tuple[list[str], dict[str, float], set[str]]:
    """Separate measured local-slope descriptors from censored/invalid cases."""
    selected = [row for row in fits if row["fidelity"] == fidelity and row["candidate_id"] in registry_ids]
    evaluated = {row["candidate_id"] for row in selected}
    response = {
        row["candidate_id"]: finite(row["objective"])
        for row in selected
        if math.isfinite(finite(row["objective"]))
        and row.get("descriptor_status", "") in {
            "PROVISIONAL_THREE_POINT_LOCAL_SLOPE",
            "FOUR_POINT_INVALID_REGION",
            "VALID_PARIS_REGION",
        }
    }
    return sorted(evaluated), response, evaluated - set(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", type=Path, default=Path("artifacts/v914_paris_reparameterization/candidate_pool.csv"))
    parser.add_argument("--registry", type=Path, default=Path("artifacts/v914_paris_reparameterization/candidate_registry.csv"))
    parser.add_argument("--fits", type=Path, default=Path("runs/v914_paris_reparameterization_v1/analysis/paris_candidate_fits.csv"))
    parser.add_argument("--fidelity", default="level1")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 4 <= args.batch_size <= 8:
        raise SystemExit("batch-size must lie in the requested 4-8 range")
    pool = read_csv(args.pool); registry = read_csv(args.registry); fits = read_csv(args.fits)
    coordinates = [field for field in pool[0] if field.startswith("u__")]
    pool_by_id = {row["candidate_id"]: row for row in pool}
    evaluated_ids = {row["candidate_id"] for row in registry}
    observed_ids, response, invalid_ids = descriptor_response(evaluated_ids, fits, args.fidelity)
    descriptor_ids = sorted(response)
    finite_values = [response[candidate] for candidate in descriptor_ids]
    if len(observed_ids) < 8 or len(finite_values) < 3:
        raise SystemExit("adaptive surrogate requires >=8 evaluated candidates and >=3 measured local-slope descriptors")
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
    selected: list[int] = []
    remaining = acquisition.copy()
    for _ in range(min(args.batch_size, len(available))):
        chosen = int(np.argmax(remaining))
        selected.append(chosen)
        # Deterministic local diversity penalty for sequential batch selection.
        distance = np.sqrt(np.sum((query - query[chosen]) ** 2, axis=1))
        remaining *= 1.0 - np.exp(-0.5 * (distance / max(0.25 * length, 1.0e-6)) ** 2)
        remaining[selected] = -np.inf
    batch = []
    for rank, index in enumerate(selected, 1):
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
        })
        batch.append(source)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "surrogate_batch.csv", batch, list(batch[0]))
    combined = registry + batch
    fields = list(dict.fromkeys(key for row in combined for key in row))
    write_csv(args.out / "candidate_registry.csv", combined, fields)
    audit = {
        "schema": "v914_paris_surrogate_batch_v1", "iteration": args.iteration,
        "batch_size": len(batch), "fidelity_response": args.fidelity,
        "observations": len(observed_ids), "finite_objectives": len(finite_values),
        "measured_descriptor_candidates": descriptor_ids,
        "censored_or_invalid_candidates": sorted(invalid_ids),
        "invalid_objective_penalty": None, "kernel": "isotropic_squared_exponential",
        "kernel_length": length, "acquisition": "expected_improvement_with_batch_diversity_penalty",
        "feasibility_kernel_length": feasibility_length,
        "acquisition_feasibility_weighted": True,
        "candidate_ids": [row["candidate_id"] for row in batch],
    }
    (args.out / "surrogate_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"V914_PARIS_SURROGATE_BATCH iteration={args.iteration} size={len(batch)} observations={len(observed_ids)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
