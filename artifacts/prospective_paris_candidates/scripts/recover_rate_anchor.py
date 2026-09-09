"""Recover the source-qualified A_NATIVE developed-rate anchor g_star at
Kstar=18 MPa*sqrt(m), R=0.1, T=300K, f=1000Hz, n_bins=80, seed=1720.

Per the amended campaign decision (this session), the historical
INV_OPENING_M4_V1 corrected-candidate row and its full physical audit trail
are unrecoverable from any tracked or portable evidence in this environment
(verified: git tree at the exact claimed HEAD has no result artifacts;
exhaustive filesystem search across /Volumes/Data finds none;
scripts/verify_v10_2_30_physical_slope_transfer.py fails immediately with
"missing archive_resolution_audit.csv"; 15/17 focused tests fail with
FileNotFoundError). The 2.543e-7 m/cycle historical value is retained ONLY
as UNVERIFIED_HISTORICAL_PRIOR and must never gate any decision.

This script reads ONLY the qualified, developed A_NATIVE row from the
already-verified R-ratio-nominal-deltaK study table (used and verified in
the prior focused-dadn-K-plots mission this session) and computes g_star
plus its full provenance record. No new simulation is run here.
"""
from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path

SRC = Path("/Volumes/Data/working-papers/fracture_and_fatigue/1A_PT03_PT08_R_developed_points.csv")
OUT_DIR = Path(__file__).resolve().parents[1]

TARGET = dict(option="A_NATIVE", R="0.1", Kmax_MPa_sqrt_m="18.0", seed="1720",
              n_bins="80", temperature_K="300.0", frequency_Hz="1000.0")


def row_hash(row: dict) -> str:
    canon = json.dumps(row, sort_keys=True)
    return hashlib.sha256(canon.encode()).hexdigest()


def main() -> None:
    src_sha256 = hashlib.sha256(SRC.read_bytes()).hexdigest()
    with SRC.open() as fh:
        rows = list(csv.DictReader(fh))
    matches = [r for r in rows if all(r[k] == v for k, v in TARGET.items())]
    if len(matches) != 1:
        raise SystemExit(f"expected exactly 1 matching row, found {len(matches)}")
    row = matches[0]

    if row["terminal_classification"] not in ("PHYSICAL_TARGET_REACHED", "REUSED_PHYSICAL_TARGET_REACHED"):
        raise SystemExit(f"anchor row is not a qualified terminal result: {row['terminal_classification']}")

    g_star = float(row["developed_da_dN"])

    record = dict(
        schema="v10.2.30_prospective_paris_rate_anchor_v1",
        role="gstar_common_rate_anchor_for_P25_P40_P55",
        Kstar_MPa_sqrt_m=18.0,
        R=0.1, temperature_K=300.0, frequency_Hz=1000.0, n_bins=80, seed=1720,
        option="A_NATIVE",
        gstar_m_per_cycle=g_star,
        terminal_classification=row["terminal_classification"],
        stationarity_ratio=float(row["stationarity_ratio"]),
        source_table=str(SRC),
        source_table_sha256=src_sha256,
        source_row=row,
        source_row_sha256=row_hash(row),
        branch=row.get("branch"),
        job_launch_head=row.get("head"),
        analysis_head=row.get("analysis_head"),
        production_solver_hash=row.get("production_solver_hash"),
        common_physics_hash=row.get("common_physics_hash"),
        composite_hash=row.get("composite_hash"),
        job_key=row.get("job_id"),
        result_path=row.get("result_path"),
        note=("This is the qualified A_NATIVE developed rate used ONLY as the "
              "common normalization anchor for the P25/P40/P55 design campaign. "
              "It is NOT the historical INV_OPENING_M4_V1 corrected-candidate "
              "rate (that candidate deliberately differs from A_NATIVE via 5 "
              "cleavage-field changes; its own rate at K=18 is unrecoverable)."),
    )
    (OUT_DIR / "rate_anchor_gstar.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: record[k] for k in ("gstar_m_per_cycle", "source_row_sha256", "job_key")}, indent=2))


if __name__ == "__main__":
    main()
