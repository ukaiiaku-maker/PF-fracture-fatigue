#!/usr/bin/env python3
"""Materialize candidate-specific supplemental and promotion load tables."""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def finite(value) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else math.nan
    except (TypeError, ValueError):
        return math.nan


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else ["candidate_id", "DeltaK_MPa_sqrt_m", "case_role"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("level0-supplement", "level1-search-descriptors", "level1-promotions", "level2-fidelity-pairs", "level2-promotions", "level3-finalists"), required=True)
    parser.add_argument("--registry", type=Path, default=Path("artifacts/v914_paris_reparameterization/candidate_registry.csv"))
    parser.add_argument("--config", type=Path, default=Path("config/v914_paris_search_target.json"))
    parser.add_argument("--promotions", type=Path, default=Path("runs/v914_paris_reparameterization_v1/analysis/candidate_promotions.csv"))
    parser.add_argument("--points", type=Path, default=Path("runs/v914_paris_reparameterization_v1/analysis/paris_search_points.csv"))
    parser.add_argument("--fits", type=Path, default=Path("runs/v914_paris_reparameterization_v1/analysis/paris_candidate_fits.csv"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--finalist-count", type=int, default=5)
    parser.add_argument("--finalist-seeds", default="1720,2718,3141")
    args = parser.parse_args()
    registry = read_csv(args.registry); config = json.loads(args.config.read_text())
    rows = []
    if args.mode == "level0-supplement":
        level0 = config["level0"]; maximum = max(level0["DeltaK_MPa_sqrt_m"])
        for candidate in registry:
            k50 = finite(candidate.get("historical_monotonic_K50_300K_MPa_sqrt_m"))
            if not math.isfinite(k50):
                continue
            load = min(level0["inert_supplement_max_DeltaK_MPa_sqrt_m"], level0["inert_supplement_historical_K50_factor"] * k50)
            if load - maximum >= level0["inert_supplement_minimum_separation_MPa_sqrt_m"]:
                rows.append({"candidate_id": candidate["candidate_id"], "DeltaK_MPa_sqrt_m": f"{load:.12g}", "case_role": "HIGH_K50_LEVEL0_SUPPLEMENT"})
    elif args.mode in {"level1-search-descriptors", "level1-promotions"}:
        promote = {row["candidate_id"] for row in read_csv(args.promotions) if row["next_fidelity"] == "level1" and row["promote"].lower() == "true"}
        points = read_csv(args.points)
        for candidate_id in sorted(promote):
            q = [row for row in points if row["candidate_id"] == candidate_id and row["fidelity"] == "level0"]
            growth = sorted(finite(row["DeltaK_MPa_sqrt_m"]) for row in q if row["status"] == "growth_target_reached")
            if not growth:
                continue
            first_growth = growth[0]
            lower_censors = [finite(row["DeltaK_MPa_sqrt_m"]) for row in q if row["status"] != "growth_target_reached" and finite(row["DeltaK_MPa_sqrt_m"]) < first_growth]
            low = 1.05 * max(lower_censors) if lower_censors else 0.75 * first_growth
            high = min(1.20 * first_growth, config["level0"]["inert_supplement_max_DeltaK_MPa_sqrt_m"])
            if high <= low:
                low = 0.8 * first_growth
            descriptor_only = args.mode == "level1-search-descriptors"
            count = 3 if descriptor_only else 6
            loads = [low * (high / low) ** (index / (count - 1.0)) for index in range(count)]
            for load in loads:
                rows.append({
                    "candidate_id": candidate_id,
                    "DeltaK_MPa_sqrt_m": f"{load:.12g}",
                    "case_role": "LEVEL1_SEARCH_DESCRIPTOR" if descriptor_only else "LEVEL1_ADAPTIVE_BRACKETED_PARIS_RANGE",
                })
    elif args.mode == "level2-fidelity-pairs":
        pair_ids = {row["candidate_id"] for row in registry if row["paris_design_role"].startswith("ABCD_") or row["paris_design_role"] == "DIVERSE_FIDELITY_PAIR"}
        for candidate_id in sorted(pair_ids):
            for load in config["level1"]["DeltaK_MPa_sqrt_m"]:
                rows.append({"candidate_id": candidate_id, "DeltaK_MPa_sqrt_m": load, "case_role": "N64_N128_RANKING_PAIR"})
    elif args.mode == "level2-promotions":
        fits = [row for row in read_csv(args.fits) if row["fidelity"] == "level1" and row["valid_Paris_region"].lower() == "true"]
        for fit in fits:
            low, high = finite(fit["DeltaK_min"]), finite(fit["DeltaK_max"])
            if low > 0.0 and high > low:
                for load in [low * (high / low) ** (index / 4.0) for index in range(5)]:
                    rows.append({"candidate_id": fit["candidate_id"], "DeltaK_MPa_sqrt_m": f"{load:.12g}", "case_role": "ADAPTIVE_LEVEL2_PARIS_INTERVAL"})
    else:
        fits = [row for row in read_csv(args.fits) if row["fidelity"] == "level2" and row["valid_Paris_region"].lower() == "true"]
        fits.sort(key=lambda row: finite(row["objective"]))
        seeds = [int(value) for value in args.finalist_seeds.split(",") if value.strip()]
        if not seeds:
            raise SystemExit("level3-finalists requires at least one seed")
        for fit in fits[:args.finalist_count]:
            low, high = finite(fit["DeltaK_min"]), finite(fit["DeltaK_max"])
            if not (low > 0.0 and high > low):
                continue
            for seed in seeds:
                for load in [low * (high / low) ** (index / 5.0) for index in range(6)]:
                    rows.append({
                        "candidate_id": fit["candidate_id"],
                        "DeltaK_MPa_sqrt_m": f"{load:.12g}",
                        "seed": seed,
                        "case_role": "LEVEL3_FINALIST_PRODUCTION_PARIS_INTERVAL",
                    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(args.out, rows)
    print(f"V914_PARIS_ADAPTIVE_CASES mode={args.mode} cases={len(rows)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
