#!/usr/bin/env python3
"""Exact-parameter crosswalk from legacy/search candidate IDs to immutable V914_ATLAS base IDs.

Never renames or reindexes a base candidate; only records which existing
V914_ATLAS_* row a legacy/search identity exactly corresponds to, matched on
the shared 22-D parameter vector. Fails closed (no row, or an explicit
"multiple matches" error) rather than guessing.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd, pyarrow as pa, pyarrow.parquet as pq

BASE = ["cleave_G00_eV","cleave_sigc0_GPa","cleave_exp_a","cleave_exp_n","cleave_floor_frac","emit_G00_eV","emit_sigc0_GPa","emit_exp_a","emit_exp_n","emit_floor_frac","peierls_H0_eV","peierls_activation_entropy_kB","peierls_exp_a","peierls_exp_n","taylor_H0_eV","taylor_activation_entropy_kB","taylor_exp_a","taylor_exp_n","rho_source0_m2","taylor_corr_rho_c_m2","taylor_corr_scale","c_blunt"]


def legacy_bijection_rows(atlas: pd.DataFrame) -> list[dict]:
    legacy = atlas[atlas.legacy_candidate_id.ne("")]
    return [
        {
            "source_candidate_id": r.legacy_candidate_id,
            "base_candidate_id": r.candidate_id,
            "exact_parameter_match": True,
            "match_method": "legacy_candidate_id_bijection",
            "atlas_sha256": r.candidate_sha256,
            "provenance": "master_candidate_registry.parquet:legacy_candidate_id",
        }
        for r in legacy.itertuples(index=False)
    ]


def exact_parameter_match_rows(source: pd.DataFrame, atlas: pd.DataFrame, provenance: str) -> list[dict]:
    """Fail-closed exact join on the 22-D shared parameter vector."""
    atlas_base = atlas[["candidate_id", "candidate_sha256", *BASE]].rename(
        columns={"candidate_id": "base_candidate_id", "candidate_sha256": "atlas_sha256"}
    )
    rows = []
    for _, src in source.iterrows():
        hits = atlas_base
        for name in BASE:
            hits = hits[hits[name] == float(src[name])]
            if hits.empty:
                break
        if len(hits) == 0:
            rows.append({
                "source_candidate_id": src["candidate_id"], "base_candidate_id": pd.NA,
                "exact_parameter_match": False, "match_method": "no_exact_atlas_match",
                "atlas_sha256": pd.NA, "provenance": provenance,
            })
        elif len(hits) > 1:
            raise RuntimeError(f"multiple atlas rows share the exact 22-D parameter vector for {src['candidate_id']!r}")
        else:
            rows.append({
                "source_candidate_id": src["candidate_id"], "base_candidate_id": hits.iloc[0].base_candidate_id,
                "exact_parameter_match": True, "match_method": "parameter_vector_exact",
                "atlas_sha256": hits.iloc[0].atlas_sha256, "provenance": provenance,
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--atlas", type=Path, required=True)
    ap.add_argument("--abcd-registry", type=Path, default=Path("arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv"))
    a = ap.parse_args()
    atlas = pq.read_table(a.atlas / "master_candidate_registry.parquet", columns=["candidate_id", "candidate_sha256", "legacy_candidate_id", *BASE]).to_pandas()

    rows = legacy_bijection_rows(atlas)
    if a.abcd_registry.exists():
        abcd = pd.read_csv(a.abcd_registry)
        rows += exact_parameter_match_rows(abcd, atlas, str(a.abcd_registry.resolve()))

    out = pd.DataFrame(rows)
    if out.source_candidate_id.duplicated().any():
        raise RuntimeError("duplicate source_candidate_id across crosswalk sources")
    pq.write_table(pa.Table.from_pandas(out, preserve_index=False), a.atlas / "candidate_id_crosswalk.parquet", compression="zstd")
    out.to_csv(a.atlas / "candidate_id_crosswalk.csv", index=False)
    print(json.dumps({
        "rows": len(out),
        "exact_matches": int(out.exact_parameter_match.sum()),
        "unmatched": int((~out.exact_parameter_match).sum()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
