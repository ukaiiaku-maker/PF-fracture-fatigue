#!/usr/bin/env python3
"""Finalize coverage products and the staged n64 queue for the v9.14 atlas."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

PARAMETER_COLUMNS = [
    "cleave_G00_eV", "cleave_sigc0_GPa", "cleave_exp_a", "cleave_exp_n",
    "cleave_floor_frac", "emit_G00_eV", "emit_sigc0_GPa", "emit_exp_a",
    "emit_exp_n", "emit_floor_frac", "peierls_H0_eV",
    "peierls_activation_entropy_kB", "peierls_exp_a", "peierls_exp_n",
    "taylor_H0_eV", "taylor_activation_entropy_kB", "taylor_exp_a",
    "taylor_exp_n", "rho_source0_m2", "taylor_corr_rho_c_m2",
    "taylor_corr_scale", "c_blunt",
]
PROPERTY_COLUMNS = [
    "minimum_local_m", "median_developed_m", "best_broad_window_m",
    "log10_best_window_C", "maximum_valid_window_width", "number_valid_windows",
    "minimum_predicted_developed_log10_da_dN",
    "maximum_predicted_developed_log10_da_dN", "sustained_growth_probability",
    "three_hit_amplification_median", "cleavage_emission_competition_median",
]


def robust_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median/IQR scaling with finite imputation, stable for inactive rows."""
    x = np.asarray(values, dtype=float).copy()
    finite = np.isfinite(x)
    med = np.nanmedian(np.where(finite, x, np.nan), axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    x = np.where(finite, x, med)
    q25, q75 = np.percentile(x, [25, 75], axis=0)
    scale = q75 - q25
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (x - med) / scale, med, scale


def clustered(values: np.ndarray, clusters: int, seed: int, sample_size: int = 65536):
    """Fit deterministic sample centroids and assign all rows in bounded chunks."""
    x, med, scale = robust_scale(values)
    rng = np.random.default_rng(seed)
    sample = rng.choice(len(x), min(sample_size, len(x)), replace=False)
    centroids, _ = kmeans2(x[sample], clusters, minit="++", iter=35, seed=seed)
    labels = np.empty(len(x), dtype=np.int16)
    distances = np.empty(len(x), dtype=np.float32)
    for start in range(0, len(x), 32768):
        stop = min(start + 32768, len(x))
        d = cdist(x[start:stop], centroids)
        labels[start:stop] = np.argmin(d, axis=1)
        distances[start:stop] = np.min(d, axis=1)
    return labels, distances, centroids, med, scale


def select_queue(frame: pd.DataFrame, total: int = 250) -> pd.DataFrame:
    """Diverse deterministic queue with the predeclared five scientific roles."""
    selected: list[int] = []
    role_for: dict[int, str] = {}

    def add(pool: pd.DataFrame, count: int, role: str, cluster: str, score: str,
            ascending: bool = True) -> None:
        pool = pool.loc[~pool.index.isin(selected)].copy()
        if pool.empty:
            return
        pool = pool.sort_values([cluster, score, "candidate_id"], ascending=[True, ascending, True])
        # Round-robin over clusters preserves disconnected basins.
        groups = {k: list(g.index) for k, g in pool.groupby(cluster, sort=True)}
        while len([i for i in selected if role_for[i] == role]) < count and groups:
            for key in list(groups):
                if groups[key]:
                    idx = groups[key].pop(0); selected.append(idx); role_for[idx] = role
                    if len([i for i in selected if role_for[i] == role]) >= count:
                        break
                if not groups[key]:
                    del groups[key]

    realistic = frame[frame.best_broad_window_m.between(2.0, 6.0) & (frame.number_valid_windows > 0)]
    add(realistic, 100, "realistic_paris_exploitation", "property_cluster_id", "maximum_valid_window_width", False)
    low = frame[(frame.best_broad_window_m < 2.0) & (frame.number_valid_windows > 0)]
    if len(low) < 50:
        low = frame[frame.number_valid_windows > 0].nsmallest(max(50, len(low)), "best_broad_window_m")
    add(low, 50, "distinct_low_m_basin", "parameter_cluster_id", "best_broad_window_m", True)
    add(frame, 50, "property_space_coverage", "property_cluster_id", "property_cluster_distance", False)
    uncertainty = frame.assign(uncertainty_score=(frame.sustained_growth_probability - 0.5).abs())
    add(uncertainty, 25, "analytical_uncertainty", "property_cluster_id", "uncertainty_score", True)
    add(frame, 25, "maximin_global_exploration", "parameter_cluster_id", "parameter_cluster_distance", False)
    if len(selected) < total:
        add(frame, total - len(selected), "coverage_backfill", "property_cluster_id", "property_cluster_distance", False)
    out = frame.loc[selected[:total]].copy()
    out.insert(1, "atlas_sampling_role", [role_for[i] for i in out.index])
    out.insert(2, "queue_priority", np.arange(1, len(out) + 1))
    return out


def empty_observation_table(path: Path) -> None:
    schema = pa.schema([
        ("candidate_id", pa.string()), ("physics_model", pa.string()),
        ("fidelity", pa.string()), ("condition", pa.string()),
        ("property_name", pa.string()), ("value", pa.float64()),
        ("uncertainty", pa.float64()), ("status", pa.string()),
        ("provenance", pa.string()),
    ])
    pq.write_table(pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema), path)


def empty_result_table(path: Path) -> None:
    schema = pa.schema([
        ("candidate_id", pa.string()), ("fidelity", pa.string()),
        ("DeltaK_MPa_sqrt_m", pa.float64()), ("status", pa.string()),
        ("developed_da_dN_m_per_cycle", pa.float64()), ("measured_m", pa.float64()),
        ("run_directory", pa.string()), ("producer_commit", pa.string()),
    ])
    pq.write_table(pa.Table.from_arrays([pa.array([], type=f.type) for f in schema], schema=schema), path)


def plot_projections(out: Path, frame: pd.DataFrame, seed: int) -> None:
    rng = np.random.default_rng(seed)
    valid = frame[np.isfinite(frame.best_broad_window_m) & (frame.best_window_C > 0)]
    if len(valid) > 100000:
        valid = valid.iloc[rng.choice(len(valid), 100000, replace=False)]
    specs = [
        ("best_broad_window_m", "log10_best_window_C", "m_vs_C.png", "Paris m", "log10 C"),
        ("best_broad_window_m", "maximum_valid_window_width", "m_vs_window_width.png", "Paris m", "window width"),
        ("best_broad_window_m", "three_hit_amplification_median", "m_vs_three_hit_amplification.png", "Paris m", "three-hit amplification"),
    ]
    for x, y, name, xl, yl in specs:
        fig, ax = plt.subplots(figsize=(7, 5)); ax.scatter(valid[x], valid[y], c=valid.property_cluster_id, s=2, alpha=.25, cmap="turbo", rasterized=True)
        ax.set(xlabel=xl, ylabel=yl); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out/name, dpi=180); plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--queue-size", type=int, default=250)
    ap.add_argument("--creation-commit", required=True)
    args = ap.parse_args(); out = args.atlas
    registry_cols = ["candidate_id", "candidate_sha256", "domain_label", "legacy_candidate_id", "sobol_sequence_index", *PARAMETER_COLUMNS]
    registry = pq.read_table(out/"master_candidate_registry.parquet", columns=registry_cols).to_pandas()
    props = pq.read_table(out/"fatigue_candidate_properties.parquet").to_pandas()
    props["log10_best_window_C"] = np.log10(props.best_window_C.where(props.best_window_C > 0))
    frame = registry.merge(props, on="candidate_id", validate="one_to_one")
    parameter_values = registry[PARAMETER_COLUMNS].to_numpy(float)
    property_values = frame[PROPERTY_COLUMNS].to_numpy(float)
    pl, pdist, pc, pmed, pscale = clustered(parameter_values, 64, 91464)
    ql, qdist, qc, qmed, qscale = clustered(property_values, 48, 91448)
    frame["parameter_cluster_id"] = pl; frame["parameter_cluster_distance"] = pdist
    frame["property_cluster_id"] = ql; frame["property_cluster_distance"] = qdist
    pq.write_table(pa.Table.from_pandas(frame[["candidate_id", "parameter_cluster_id", "parameter_cluster_distance"]], preserve_index=False), out/"parameter_space_clusters.parquet", compression="zstd")
    pq.write_table(pa.Table.from_pandas(frame[["candidate_id", "property_cluster_id", "property_cluster_distance"]], preserve_index=False), out/"property_space_clusters.parquet", compression="zstd")

    queue = select_queue(frame, args.queue_size)
    # Fixed v7 launch constants supplement, but never alter, the 22 candidate coordinates.
    queue["source_historical_sobol_index"] = queue["sobol_sequence_index"]
    queue["paris_design_role"] = queue["atlas_sampling_role"]
    fixed = {"Tref_K": 300.0, "cleave_gT_eV_per_K": 0.0,
             "cleave_sT_GPa_per_K": 0.0, "emit_gT_eV_per_K": 0.0,
             "emit_sT_GPa_per_K": 0.0, "peierls_nu0_s": 1.0e12,
             "taylor_nu0_s": 1.0e11, "n64_launch_status": "QUEUED_NOT_LAUNCHED"}
    for key, value in fixed.items(): queue[key] = value
    queue.to_csv(out/"n64_candidate_queue.csv", index=False, quoting=csv.QUOTE_MINIMAL)
    observation_path=out/"candidate_property_observations.parquet"
    n64_results_path=out/"n64_candidate_results.parquet"
    n128_results_path=out/"n128_candidate_results.parquet"
    if not observation_path.exists(): empty_observation_table(observation_path)
    if not n64_results_path.exists(): empty_result_table(n64_results_path)
    if not n128_results_path.exists(): empty_result_table(n128_results_path)

    valid = frame.number_valid_windows > 0
    preferred = frame.best_broad_window_m.between(2, 6) & valid
    exploratory = frame.best_broad_window_m.between(2, 8) & valid
    coverage = pd.DataFrame([
        {"metric": "atlas_candidates", "value": len(frame)},
        {"metric": "historical_candidates", "value": int((frame.domain_label == "HISTORICAL_DOMAIN").sum())},
        {"metric": "cleavage_extension_candidates", "value": int((frame.domain_label == "CLEAVAGE_EXTENSION").sum())},
        {"metric": "candidates_with_valid_windows", "value": int(valid.sum())},
        {"metric": "preferred_m_2_6_candidates", "value": int(preferred.sum())},
        {"metric": "exploratory_m_2_8_candidates", "value": int(exploratory.sum())},
        {"metric": "occupied_parameter_clusters", "value": int(frame.parameter_cluster_id.nunique())},
        {"metric": "occupied_property_clusters", "value": int(frame.property_cluster_id.nunique())},
        {"metric": "n64_queue_candidates", "value": len(queue)},
        {"metric": "distinct_queue_parameter_clusters", "value": int(queue.parameter_cluster_id.nunique())},
        {"metric": "distinct_queue_property_clusters", "value": int(queue.property_cluster_id.nunique())},
        {"metric": "best_m_min", "value": float(frame.loc[valid, "best_broad_window_m"].min())},
        {"metric": "best_m_max", "value": float(frame.loc[valid, "best_broad_window_m"].max())},
        {"metric": "log10_C_min", "value": float(frame.loc[valid, "log10_best_window_C"].min())},
        {"metric": "log10_C_max", "value": float(frame.loc[valid, "log10_best_window_C"].max())},
    ])
    coverage.to_csv(out/"candidate_atlas_coverage.csv", index=False)
    roles = queue.atlas_sampling_role.value_counts().to_dict()
    report = f"""# v9.14 candidate atlas coverage

This immutable analytical atlas contains **{len(frame):,}** candidates: **{int((frame.domain_label == 'HISTORICAL_DOMAIN').sum()):,}** historical-domain and **{int((frame.domain_label == 'CLEAVAGE_EXTENSION').sum()):,}** cleavage-extension candidates. It records all candidates, rather than pruning to a single Paris target.

## Fatigue-property coverage

- Candidates with at least one retained analytical window: {int(valid.sum()):,}
- Preferred realistic-Paris subset (`2 <= m <= 6`): {int(preferred.sum()):,}
- Exploratory subset (`2 <= m <= 8`): {int(exploratory.sum()):,}
- Occupied parameter/property clusters: {frame.parameter_cluster_id.nunique()} / {frame.property_cluster_id.nunique()}
- Valid-window `m` range: {frame.loc[valid, 'best_broad_window_m'].min():.5g} to {frame.loc[valid, 'best_broad_window_m'].max():.5g}
- Valid-window `log10(C)` range: {frame.loc[valid, 'log10_best_window_C'].min():.5g} to {frame.loc[valid, 'log10_best_window_C'].max():.5g}

## Staged n64 measurement queue

The {len(queue)}-candidate queue is deliberately multimodal and is **not launched by this analytical finalizer**. Role counts: `{json.dumps(roles, sort_keys=True)}`. Existing production workers remain authoritative and uninterrupted; terminal measurements are appended later through the common observation schema.

## Future fracture/R-curve join

The joint schema reserves quantitative fracture-class probabilities, toughness, matched-extension R-curve values, R-curve slopes/rises, and mechanism-state descriptors. Candidate identity is shared across analytical fatigue, n64/n128 fatigue, and future pure-fracture calculations.
"""
    (out/"candidate_atlas_coverage_report.md").write_text(report)
    plot_projections(out, frame, 914)
    metadata = {"schema": "v914_candidate_atlas_finalization_v1", "creation_commit": args.creation_commit,
                "parameter_clusters": 64, "property_clusters": 48, "queue_size": len(queue),
                "queue_launched": False, "production_physics_modified": False,
                "parameter_center": pmed.tolist(), "parameter_scale": pscale.tolist(),
                "property_center": qmed.tolist(), "property_scale": qscale.tolist(),
                "parameter_centroids": pc.tolist(), "property_centroids": qc.tolist()}
    (out/"candidate_atlas_finalization.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps({"candidates": len(frame), "queue": len(queue), "roles": roles}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
