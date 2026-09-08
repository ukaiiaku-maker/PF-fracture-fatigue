"""Independent-verifier closure (review round 4): build a compact, portable,
deterministic projection of Fig.6B's join so the verifier never needs the
external 40MB fracture_monotonic_points_v5_7.csv at check time.

The review's specific objection: Fig.6B's recomputation depended on a live
read of a 40MB external file not included in the bundle, so the branch was
not genuinely self-contained. This script performs the exact join once
(against the live external files, recording their SHA-256), keeps only the
minimal columns the correlation recomputation needs, and writes the
resulting compact table into the bundle with full provenance: full-source
hashes, filter/join keys, excluded-row counts and reasons, and the compact
table's own hash. A companion check (recompute_fig6b_from_compact_bundle in
claim_recompute_v4.py) then reproduces n=1360, the pooled Pearson r, and all
six per-context correlations from the compact table ALONE.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "artifacts" / "paper_simulation_completion"
BUNDLE = OUT_DIR / "source_bundle_figures_2_4"
FATIGUE_PF_ROOT = Path("/Volumes/Data/Data/Nanopillar_calculation/Fatigue-PF")

THR_PATH = FATIGUE_PF_ROOT / "runs" / "v5_7_extension" / "fatigue_thresholds_v5_7.csv"
MONO_PATH = FATIGUE_PF_ROOT / "runs" / "v5_7_extension" / "fracture_monotonic_points_v5_7.csv"
PRIMARY_RATE = 1e-10


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    if not THR_PATH.is_file() or not MONO_PATH.is_file():
        raise SystemExit(f"Full-source files not available: {THR_PATH} / {MONO_PATH}")

    thr_hash = _sha256(THR_PATH)
    mono_hash = _sha256(MONO_PATH)

    thr = pd.read_csv(THR_PATH)
    mono = pd.read_csv(MONO_PATH)
    n_thr_total = len(thr)
    n_mono_total = len(mono)

    # Step 1: rate-criterion + bracketed-status filter on the threshold table.
    rate_mask = np.isclose(thr["rate_criterion_m_per_cycle"].astype(float), PRIMARY_RATE,
                            rtol=1e-6, atol=0.0)
    bracket_mask = thr["threshold_status"] == "bracketed"
    t = thr[rate_mask & bracket_mask].copy()
    n_thr_wrong_rate = int((~rate_mask).sum())
    n_thr_not_bracketed = int((rate_mask & ~bracket_mask).sum())
    n_thr_after_filter = len(t)

    # Step 2: inner join to the monotonic-points table on the natural key.
    m = mono.rename(columns={"context_id": "fracture_context", "T_K": "temperature_K"})
    j = t.merge(m[["surface_id", "fracture_context", "temperature_K", "Kc_first_MPa_sqrtm"]],
                on=["surface_id", "fracture_context", "temperature_K"], how="inner")
    n_join_dropped_no_match = n_thr_after_filter - len(j)

    # Step 3: finite/positive filter on both correlated quantities.
    good = (np.isfinite(j["DeltaK_th_MPa_sqrtm"]) & np.isfinite(j["Kc_first_MPa_sqrtm"])
            & (j["DeltaK_th_MPa_sqrtm"] > 0) & (j["Kc_first_MPa_sqrtm"] > 0))
    compact = j.loc[good, ["surface_id", "fracture_context", "temperature_K",
                           "Kc_first_MPa_sqrtm", "DeltaK_th_MPa_sqrtm"]].copy()
    n_excluded_nonfinite_or_nonpositive = int((~good).sum())
    n_compact = len(compact)

    compact_path = BUNDLE / "fig6B_compact_joined_1360.csv"
    compact.to_csv(compact_path, index=False)
    compact_hash = _sha256(compact_path)

    provenance = dict(
        schema="v1_fig6b_portable_projection_provenance",
        full_source_files=dict(
            fatigue_thresholds_v5_7_csv=dict(path=str(THR_PATH), sha256=thr_hash, n_rows=n_thr_total),
            fracture_monotonic_points_v5_7_csv=dict(path=str(MONO_PATH), sha256=mono_hash, n_rows=n_mono_total),
        ),
        filter_and_join_keys=dict(
            rate_criterion_filter=f"np.isclose(rate_criterion_m_per_cycle, {PRIMARY_RATE}, rtol=1e-6, atol=0.0)",
            status_filter="threshold_status == 'bracketed'",
            join_keys=["surface_id", "fracture_context(=context_id)", "temperature_K(=T_K)"],
            join_type="inner",
            post_join_filter="finite(DeltaK_th_MPa_sqrtm) & finite(Kc_first_MPa_sqrtm) & "
                              "DeltaK_th_MPa_sqrtm>0 & Kc_first_MPa_sqrtm>0",
        ),
        excluded_row_counts_and_reasons=dict(
            threshold_rows_excluded_wrong_rate_criterion=n_thr_wrong_rate,
            threshold_rows_excluded_not_bracketed_at_primary_rate=n_thr_not_bracketed,
            threshold_rows_remaining_after_rate_and_status_filter=n_thr_after_filter,
            rows_dropped_no_join_match_in_monotonic_table=n_join_dropped_no_match,
            rows_excluded_nonfinite_or_nonpositive_after_join=n_excluded_nonfinite_or_nonpositive,
        ),
        compact_table=dict(
            path=str(compact_path.relative_to(REPO_ROOT)),
            sha256=compact_hash,
            n_rows=n_compact,
            columns=list(compact.columns),
        ),
        expected_n_rows=1360,
        n_rows_matches_expected=(n_compact == 1360),
    )
    (OUT_DIR / "fig6b_portable_projection_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str))
    print(json.dumps(provenance, indent=2, default=str))
    if n_compact != 1360:
        raise SystemExit(f"Compact table has {n_compact} rows, expected 1360")


if __name__ == "__main__":
    main()
