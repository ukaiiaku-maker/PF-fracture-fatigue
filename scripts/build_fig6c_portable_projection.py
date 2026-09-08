"""Independent-verifier closure (review round 4): build a compact, portable
projection for Fig.6C so every reported AUC (not just one spot-checked row)
can be recomputed from bundled files alone.

Fig.6C's frozen manuscript rows (manuscript_statistics_table.csv, panel C,
39 rows) come from three analysis families, each requiring a per-surface
join between an S-N endurance classification and a fracture metric, at
T in {100, 300, 500} K, using the STRICT S-N definition only (knees
excluded). The full per-surface source tables are large (fourway_phenotype_
summary_v5_7.csv is 23,040 rows / ~8MB; fracture_monotonic_points_v5_7.csv
is 40MB). This script performs the exact join/filter logic from
lowT_conditional() in analyze_v57_final_integrated.py ONCE against the live
external files, keeps only the columns needed for AUC recomputation, and
writes three compact tables plus a provenance record (source hashes, row
counts before/after each filter, compact hashes).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"
BUNDLE = OUT_DIR / "source_bundle_figures_2_4"
FP_ROOT = Path("/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF")

PHENOTYPE_PATH = FP_ROOT / "runs" / "v5_7_extension" / "fourway_phenotype_summary_v5_7.csv"
MONO_PATH = FP_ROOT / "runs" / "v5_7_extension" / "fracture_monotonic_points_v5_7.csv"
THR_PATH = FP_ROOT / "runs" / "v5_7_extension" / "fatigue_thresholds_v5_7.csv"
STRENGTH_PATH = FP_ROOT / "runs" / "v5_7_final_analysis" / "temperature_relative_strength_metrics.csv"

TEMPERATURES = (100.0, 300.0, 500.0)
PRIMARY_RATE = 1e-10


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dkth_ordinal_score(group: pd.DataFrame) -> pd.Series:
    recs = []
    for idx, r in group.iterrows():
        st = str(r["threshold_status"])
        if st == "below_search_range":
            recs.append((idx, 0, 0.0))
        elif st == "bracketed" and np.isfinite(r["DeltaK_th_MPa_sqrtm"]):
            recs.append((idx, 1, float(r["DeltaK_th_MPa_sqrtm"])))
        elif st == "above_search_range":
            recs.append((idx, 2, 0.0))
    out = pd.Series(np.nan, index=group.index, dtype=float)
    if not recs:
        return out
    surrogate = np.array([band * 1.0e12 + value for _, band, value in recs], float)
    ranks = rankdata(surrogate, method="average")
    for (idx, _, _), rank in zip(recs, ranks):
        out.loc[idx] = float(rank)
    return out


def sn_binary_strict(frame: pd.DataFrame, sn_col: str) -> pd.DataFrame:
    x = frame.copy()
    keep = ["continuous_SN", "endurance_like"]
    x = x[x[sn_col].isin(keep)].copy()
    x["endurance_binary"] = (x[sn_col] == "endurance_like").astype(int)
    return x


def main() -> None:
    for p in [PHENOTYPE_PATH, MONO_PATH, THR_PATH, STRENGTH_PATH]:
        if not p.is_file():
            raise SystemExit(f"Full-source file not available: {p}")

    source_hashes = {p.name: _sha256(p) for p in [PHENOTYPE_PATH, MONO_PATH, THR_PATH, STRENGTH_PATH]}

    ph_cols = ["surface_id"] + [f"shielded_fatigue_class_T{int(T)}" for T in TEMPERATURES]
    ph = pd.read_csv(PHENOTYPE_PATH, usecols=ph_cols).drop_duplicates("surface_id")
    n_ph_total = len(ph)

    mono = pd.read_csv(MONO_PATH, usecols=["surface_id", "context_id", "T_K", "Kc_first_MPa_sqrtm"])
    n_mono_total = len(mono)

    thr = pd.read_csv(THR_PATH)
    n_thr_total = len(thr)
    thr_primary = thr[np.isclose(thr["rate_criterion_m_per_cycle"].astype(float), PRIMARY_RATE,
                                  rtol=1e-6, atol=0.0)].copy()

    strength = pd.read_csv(STRENGTH_PATH, usecols=["surface_id", "fatigue_T_K", "available_anomaly_gain_frac_exact"])
    n_strength_total = len(strength)

    rows_kc, rows_dkth, rows_strength = [], [], []
    # NOTE on accounting: `base` (the SN-classified surface set at a given T) is shared
    # across all 6 fracture-context groups; each group's join naturally keeps only the
    # subset of `base` that belongs to that context, so (len(base) - len(joined_group))
    # is NOT a data-quality exclusion -- it is mostly surfaces belonging to the OTHER 5
    # contexts. We therefore report per-family totals (rows kept) rather than a
    # cross-context subtraction, plus the genuine dropna-based exclusion (rows joined
    # but with a null score, e.g. an unbracketed DKth status) counted separately.
    n_dkth_pre_dropna = 0
    n_strength_pre_dropna = 0

    for T in TEMPERATURES:
        sn_col = f"shielded_fatigue_class_T{int(T)}"
        base = sn_binary_strict(ph[["surface_id", sn_col]].dropna(), sn_col)

        mT = mono[np.isclose(mono["T_K"].astype(float), T)]
        for cid, g in mT.groupby("context_id", sort=False):
            j = base.merge(g[["surface_id", "Kc_first_MPa_sqrtm"]], on="surface_id", how="inner")
            for _, r in j.iterrows():
                rows_kc.append(dict(temperature_K=T, context_id=cid, surface_id=r["surface_id"],
                                     endurance_binary=int(r["endurance_binary"]),
                                     score=float(r["Kc_first_MPa_sqrtm"])))

        tT = thr_primary[np.isclose(thr_primary["temperature_K"].astype(float), T)].copy()
        for cid, g in tT.groupby("fracture_context", sort=False):
            gg = g.copy()
            gg["DKth_order_score"] = dkth_ordinal_score(gg)
            j = base.merge(gg[["surface_id", "DKth_order_score"]], on="surface_id", how="inner")
            n_dkth_pre_dropna += len(j)
            for _, r in j.dropna(subset=["DKth_order_score"]).iterrows():
                rows_dkth.append(dict(temperature_K=T, context_id=cid, surface_id=r["surface_id"],
                                       endurance_binary=int(r["endurance_binary"]),
                                       score=float(r["DKth_order_score"])))

        sT = strength[np.isclose(strength["fatigue_T_K"].astype(float), T)].copy()
        j = base.merge(sT, on="surface_id", how="inner")
        n_strength_pre_dropna += len(j)
        for _, r in j.dropna(subset=["available_anomaly_gain_frac_exact"]).iterrows():
            rows_strength.append(dict(temperature_K=T, surface_id=r["surface_id"],
                                       endurance_binary=int(r["endurance_binary"]),
                                       score=float(r["available_anomaly_gain_frac_exact"])))

    excluded = dict(
        kc_rows_kept=len(rows_kc),
        dkth_rows_joined_before_null_score_drop=n_dkth_pre_dropna,
        dkth_rows_excluded_null_score_after_join=n_dkth_pre_dropna - len(rows_dkth),
        dkth_rows_kept=len(rows_dkth),
        strength_rows_joined_before_null_score_drop=n_strength_pre_dropna,
        strength_rows_excluded_null_score_after_join=n_strength_pre_dropna - len(rows_strength),
        strength_rows_kept=len(rows_strength),
    )

    kc_df = pd.DataFrame(rows_kc)
    dkth_df = pd.DataFrame(rows_dkth)
    strength_df = pd.DataFrame(rows_strength)

    kc_path = BUNDLE / "fig6C_compact_kc_endurance.csv"
    dkth_path = BUNDLE / "fig6C_compact_dkth_endurance.csv"
    strength_out_path = BUNDLE / "fig6C_compact_strength_endurance.csv"
    kc_df.to_csv(kc_path, index=False)
    dkth_df.to_csv(dkth_path, index=False)
    strength_df.to_csv(strength_out_path, index=False)

    provenance = dict(
        schema="v1_fig6c_portable_projection_provenance",
        full_source_files={name: dict(sha256=h) for name, h in source_hashes.items()},
        full_source_row_counts=dict(phenotype=n_ph_total, monotonic=n_mono_total,
                                     thresholds=n_thr_total, strength=n_strength_total),
        filter_and_join_keys=dict(
            sn_definition="strict_endurance_vs_continuous (endurance_like vs continuous_SN only; "
                          "knee_threshold_like excluded)",
            temperatures_K=list(TEMPERATURES),
            join_key="surface_id (+ context_id for Kc/DKth families)",
            dkth_rate_criterion_filter=f"rate_criterion_m_per_cycle == {PRIMARY_RATE}",
        ),
        excluded_row_counts_and_reasons=excluded,
        compact_tables=dict(
            kc=dict(path=str(kc_path.relative_to(REPO_ROOT)), sha256=_sha256(kc_path), n_rows=len(kc_df)),
            dkth=dict(path=str(dkth_path.relative_to(REPO_ROOT)), sha256=_sha256(dkth_path), n_rows=len(dkth_df)),
            strength=dict(path=str(strength_out_path.relative_to(REPO_ROOT)),
                          sha256=_sha256(strength_out_path), n_rows=len(strength_df)),
        ),
    )
    (OUT_DIR / "fig6c_portable_projection_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str))
    print(json.dumps(provenance, indent=2, default=str))


if __name__ == "__main__":
    main()
