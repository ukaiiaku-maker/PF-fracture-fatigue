#!/usr/bin/env python3
"""Build the immutable, accepted-step PF canonical trajectory table.

This is an analysis-only reader.  It verifies every source steps file against
the published campaign manifest, joins the certified onset/avalanche tables,
and enriches accepted rows with the consolidated observer records.  It never
imports or invokes a production runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RAW_DEFAULT = Path(
    "/Volumes/Data/Data/Nanopillar_calculation/"
    "PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/"
    "canonical_pf_fracture_v2_20260826"
)
PUBLICATION_DEFAULT = Path(
    "analysis_outputs/pf_canonical_fracture_v2_final/publication"
)
OUTPUT_DEFAULT = Path(
    "analysis_outputs/pf_canonical_full_trajectory_and_mechanism_audit"
)

PROVENANCE = {
    "canonical_case_count": 288,
    "campaign_execution_commit": "c3f33fa7477ea44e612fa21b6b1b1fed0df73295",
    "final_publisher_producer_commit": "b06e7cbcfc535081c8836f988e601eeea620892b",
    "qualified_physical_pf_source": "9e884fb0b0845da621d2612bdf1042e481b8df49",
    "campaign_lock_fingerprint": "5928e6abb7dcd59e6387d5d479128fec83c3ba4d509bae3a0e757b9e9ece5dde",
    "scientific_plan_fingerprint": "f3928476f2564a3eb10ca4737780a38578d9517a860bd77a9321dcd94fd4df99",
}

CORE_TRAJECTORY_COLUMNS = (
    "case_id", "matrix", "material_class", "candidate_id", "temperature_K",
    "theta_deg", "rate_tag", "loading_rate_factor", "seed",
    "is_orientation_matrix_case", "is_rate_matrix_case", "accepted_step_index",
    "accepted_step_global_index", "raw_step", "physical_time_s", "accepted_dt_s",
    "applied_opening_m", "reaction_N", "projected_crack_extension_m",
    "projected_crack_extension_um", "projected_total_crack_length_m",
    "projected_total_crack_length_um", "native_J_J_per_m2",
    "native_signed_J_J_per_m2", "native_KJ_Pa_sqrt_m",
    "native_KJ_MPa_sqrt_m", "native_KJ_per_opening_Pa_sqrt_m_per_m",
    "numerical_events_in_row", "raw_steps_sha256", "crack_event_transaction_index",
    "is_crack_event_row", "physical_avalanche_index", "in_physical_avalanche_event",
    "onset_role", "is_initial_onset", "is_reload_separated_reinitiation_onset",
    "is_target_right_censored_endpoint", "event_state",
)

OBSERVER_FIELDS = (
    "time_s",
    "K_Pa_sqrt_m",
    "fired",
    "hazard_event_index",
    "hazard_action_current",
    "hazard_action_completed",
    "hazard_threshold_current_action",
    "lambda_c_s-1",
    "sigma_opening_tip_Pa",
    "sigma_cleave_eff_Pa",
    "persistent_tip_radius_m",
    "persistent_site_front_width_m",
    "persistent_site_multiplicity_per_system",
    "persistent_sigma_back_Pa",
    "active_K_shield_signed_Pa_sqrt_m",
    "developed_state_mobile_count",
    "developed_state_retained_count",
    "developed_state_retained_fraction",
    "developed_state_cumulative_emitted",
    "developed_state_cumulative_trapped",
    "developed_state_cumulative_released",
    "developed_state_cumulative_escaped",
    "developed_state_cumulative_recovered",
    "dN_emit_raw",
    "dN_trapped",
    "dN_released",
    "dN_escaped",
    "dN_recovered",
    "tip_source_emission_rate_s",
    "tip_source_aggregate_hazard_s",
    "tip_source_effective_multiplicity_total",
    "tip_source_clear_rate_s",
    "anisotropic_drive_reliable",
)

CHANNEL_FIELDS = (
    "anisotropic_tau_signed_Pa",
    "anisotropic_sigma_emit_by_system_Pa",
    "anisotropic_sigma_back_by_system_Pa",
    "anisotropic_lambda_emit_by_system_s",
    "anisotropic_emission_probability_by_system",
    "anisotropic_transport_velocity_by_system_m_s",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def load_observer(case_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    archive = case_dir / "canonical_pf_state_observer_v2.json.zst"
    manifest_path = case_dir / "canonical_pf_state_observer_v2_manifest.json"
    if not archive.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing consolidated observer archive in {case_dir}")
    manifest = json.loads(manifest_path.read_text())
    result = subprocess.run(
        ["zstd", "-dc", str(archive)], check=True, capture_output=True
    )
    payload = json.loads(result.stdout)
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"observer records are not a list: {archive}")
    return records, manifest


def select_system(record: dict[str, Any]) -> tuple[float, str | None]:
    rates = record.get("anisotropic_lambda_emit_by_system_s")
    names = record.get("anisotropic_channel_names")
    if not isinstance(rates, list) or not rates:
        return np.nan, None
    finite = np.asarray(rates, dtype=float)
    if not np.isfinite(finite).any():
        return np.nan, None
    index = int(np.nanargmax(finite))
    name = names[index] if isinstance(names, list) and index < len(names) else None
    return float(index), name


def find_steps(case_dir: Path, temperature: float) -> Path:
    expected = case_dir / f"steps_{int(round(temperature)):04d}K.csv"
    if expected.is_file():
        return expected
    matches = sorted(case_dir.glob("steps_*K.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"could not identify unique steps file in {case_dir}")
    return matches[0]


def build_case(
    manifest_row: pd.Series,
    raw_root: Path,
    onset: pd.DataFrame,
    avalanches: pd.DataFrame,
    global_offset: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    case_id = str(manifest_row.case_id)
    case_dir = raw_root / case_id
    steps_path = find_steps(case_dir, float(manifest_row.temperature_K))
    observed_hash = sha256(steps_path)
    expected_hash = str(manifest_row.steps_sha256)
    if observed_hash != expected_hash:
        raise ValueError(
            f"steps SHA-256 mismatch for {case_id}: {observed_hash} != {expected_hash}"
        )

    steps = pd.read_csv(steps_path)
    records, observer_manifest = load_observer(case_dir)
    if len(records) != len(steps):
        raise ValueError(
            f"observer/steps row mismatch for {case_id}: {len(records)} != {len(steps)}"
        )

    observer_k = np.asarray([r.get("K_Pa_sqrt_m", np.nan) for r in records], dtype=float)
    raw_k = steps["KJ_Pa_sqrtm"].to_numpy(dtype=float)
    if not np.allclose(observer_k, raw_k, rtol=2e-12, atol=1e-7, equal_nan=True):
        raise ValueError(f"observer K does not align with raw KJ for {case_id}")

    out = pd.DataFrame(
        {
            "case_id": case_id,
            "matrix": manifest_row.matrix,
            "material_class": manifest_row.material_class,
            "candidate_id": manifest_row.candidate_id,
            "temperature_K": float(manifest_row.temperature_K),
            "theta_deg": float(manifest_row.theta_deg),
            "rate_tag": manifest_row.rate_tag,
            "loading_rate_factor": float(manifest_row.loading_rate_factor),
            "seed": int(manifest_row.seed),
            "is_orientation_matrix_case": bool(manifest_row.is_orientation_matrix_case),
            "is_rate_matrix_case": bool(manifest_row.is_rate_matrix_case),
            "accepted_step_index": np.arange(len(steps), dtype=np.int64),
            "accepted_step_global_index": np.arange(
                global_offset, global_offset + len(steps), dtype=np.int64
            ),
            "raw_step": steps["step"].astype("int64"),
            "physical_time_s": [float(r["time_s"]) for r in records],
            "accepted_dt_s": steps["dt_cur_s"].astype(float),
            "applied_opening_m": steps["Uapp_m"].astype(float),
            "reaction_N": steps["Ftop_N"].astype(float),
            "projected_crack_extension_m": steps["crack_extension_m"].astype(float),
            "projected_crack_extension_um": steps["crack_extension_m"].astype(float) * 1e6,
            "projected_total_crack_length_m": steps["a_tip_m"].astype(float),
            "projected_total_crack_length_um": steps["a_tip_m"].astype(float) * 1e6,
            "native_J_J_per_m2": steps["J_effective_direct_J_per_m2"].astype(float),
            "native_signed_J_J_per_m2": steps["J_signed_direct_J_per_m2"].astype(float),
            "native_KJ_Pa_sqrt_m": steps["KJ_Pa_sqrtm"].astype(float),
            "native_KJ_MPa_sqrt_m": steps["KJ_Pa_sqrtm"].astype(float) / 1e6,
            "native_KJ_per_opening_Pa_sqrt_m_per_m": np.divide(
                steps["KJ_Pa_sqrtm"].astype(float),
                steps["Uapp_m"].astype(float),
                out=np.full(len(steps), np.nan),
                where=steps["Uapp_m"].to_numpy(dtype=float) != 0,
            ),
            "numerical_events_in_row": steps["n_fire"].fillna(0).astype("int64"),
            "raw_steps_sha256": observed_hash,
        }
    )

    for field in OBSERVER_FIELDS:
        values = [record.get(field) for record in records]
        out[f"observer_{field}"] = values
    for field in CHANNEL_FIELDS:
        for system in (0, 1):
            out[f"observer_{field}_system_{system}"] = [
                (record.get(field) or [np.nan, np.nan])[system]
                if len(record.get(field) or []) > system else np.nan
                for record in records
            ]
    selected = [select_system(record) for record in records]
    out["observer_selected_emission_system_index"] = [x[0] for x in selected]
    out["observer_selected_emission_system_name"] = [x[1] for x in selected]

    fire = out["numerical_events_in_row"].to_numpy() > 0
    if np.any(out.loc[fire, "numerical_events_in_row"].to_numpy() != 1):
        raise ValueError(f"multiple event transactions in one accepted row: {case_id}")
    tx = np.full(len(out), -1, dtype=np.int64)
    tx[fire] = np.arange(int(fire.sum()), dtype=np.int64)
    out["crack_event_transaction_index"] = pd.array(
        np.where(fire, tx, np.nan), dtype="Int64"
    )
    out["is_crack_event_row"] = fire
    out["physical_avalanche_index"] = pd.array([pd.NA] * len(out), dtype="Int64")

    case_avalanches = avalanches.loc[avalanches.case_id.eq(case_id)]
    for avalanche in case_avalanches.itertuples(index=False):
        member = fire & (tx >= int(avalanche.first_event_transaction_index)) & (
            tx <= int(avalanche.last_event_transaction_index)
        )
        out.loc[member, "physical_avalanche_index"] = int(
            avalanche.physical_avalanche_index
        )
    if out.loc[fire, "physical_avalanche_index"].isna().any():
        raise ValueError(f"unassigned event transaction in {case_id}")
    out["in_physical_avalanche_event"] = fire

    out["onset_role"] = None
    out["is_initial_onset"] = False
    out["is_reload_separated_reinitiation_onset"] = False
    case_onset = onset.loc[onset.case_id.eq(case_id)]
    for row in case_onset.itertuples(index=False):
        match = (out["raw_step"] == int(row.pre_event_step)) & (
            out["crack_event_transaction_index"].fillna(-1)
            == int(row.event_transaction_index)
        )
        if int(match.sum()) != 1:
            raise ValueError(
                f"onset did not map exactly once for {case_id}, transaction "
                f"{row.event_transaction_index}"
            )
        role = str(row.onset_role)
        out.loc[match, "onset_role"] = role
        out.loc[match, "is_initial_onset"] = role == "INITIAL_ONSET_PRE_EVENT"
        out.loc[match, "is_reload_separated_reinitiation_onset"] = role.startswith(
            "REINITIATION_ONSET"
        )
        kj = out.loc[match, "native_KJ_MPa_sqrt_m"].iloc[0]
        opening = out.loc[match, "applied_opening_m"].iloc[0]
        if not np.isclose(kj, row.pre_event_native_KJ_MPa_sqrt_m, rtol=2e-12):
            raise ValueError(f"onset KJ mismatch for {case_id}")
        if not np.isclose(opening, row.pre_event_opening_m, rtol=2e-12):
            raise ValueError(f"onset opening mismatch for {case_id}")

    out["is_target_right_censored_endpoint"] = False
    out.loc[out.index[-1], "is_target_right_censored_endpoint"] = True
    out["event_state"] = np.where(
        out["is_initial_onset"],
        "INITIAL_ONSET_EVENT",
        np.where(
            out["is_reload_separated_reinitiation_onset"],
            "REINITIATION_ONSET_EVENT",
            np.where(fire, "IN_AVALANCHE_EVENT", "ACCEPTED_LOADING_STATE"),
        ),
    )
    out.loc[out.index[-1], "event_state"] = (
        "TARGET_RIGHT_CENSORED_EVENT_ENDPOINT"
        if fire[-1]
        else "TARGET_RIGHT_CENSORED_ENDPOINT"
    )

    initial_crack = (
        out["projected_total_crack_length_m"]
        - out["projected_crack_extension_m"]
    )
    initial_spread = float(initial_crack.max() - initial_crack.min())
    if initial_spread > 5e-13:
        raise ValueError(f"initial projected crack length is inconsistent for {case_id}")
    if not np.all(np.diff(out["physical_time_s"].to_numpy()) >= -1e-12):
        raise ValueError(f"nonchronological observer time for {case_id}")
    if not np.array_equal(out["accepted_step_index"], np.arange(len(out))):
        raise ValueError(f"accepted-step order changed for {case_id}")

    duplicate_extension_rows = int(
        out["projected_crack_extension_m"].duplicated(keep=False).sum()
    )
    index_row = {
        "case_id": case_id,
        "material_class": manifest_row.material_class,
        "temperature_K": float(manifest_row.temperature_K),
        "theta_deg": float(manifest_row.theta_deg),
        "rate_tag": manifest_row.rate_tag,
        "seed": int(manifest_row.seed),
        "row_start": global_offset,
        "row_stop_exclusive": global_offset + len(out),
        "accepted_row_count": len(out),
        "numerical_event_count": int(fire.sum()),
        "physical_avalanche_count": len(case_avalanches),
        "onset_marker_count": len(case_onset),
        "initial_projected_crack_length_m": float(initial_crack.iloc[0]),
        "initial_projected_crack_length_explicit": True,
        "duplicate_extension_row_count": duplicate_extension_rows,
        "raw_steps_relative_path": str(steps_path.relative_to(raw_root)),
        "raw_steps_sha256": observed_hash,
        "observer_archive_present": True,
        "observer_record_count": len(records),
        "observer_manifest_schema": observer_manifest.get("schema"),
        "full_local_tensor_archived": False,
    }
    return out, index_row


def write_audit(output: Path, index: pd.DataFrame, table_hash: str) -> None:
    total_rows = int(index.accepted_row_count.sum())
    duplicate_rows = int(index.duplicate_extension_row_count.sum())
    text = f"""# PF Canonical Full-Trajectory Data Audit

## Decision

Complete chronological accepted-step PF model-native driving histories were
recovered and hash-verified for all **{len(index)}** canonical cases. The long
table contains **{total_rows:,}** accepted rows. No stochastic trajectory,
FEM/CZM run, or production-state mutation was performed.

## Source and join semantics

- Every `steps_*.csv` SHA-256 equals the corresponding published
  `steps_sha256` value.
- Row order is the raw accepted chronological order; the data were not sorted
  by crack extension.
- **{duplicate_rows:,}** rows participate in repeated-extension coordinates;
  these loading/reloading states are intentionally preserved.
- Native KJ is reported as **PF model-native KJ**, never as applied K or a
  conventional R-curve.
- Initial and reload-separated onset flags are exact joins on case,
  `pre_event_step`, and event-transaction index.
- Physical-avalanche membership is assigned only to event rows using the
  certified transaction-index ranges.
- The last accepted row of every target-reaching case is explicitly marked as
  a target-right-censored endpoint.
- Absolute projected crack length is retained because `a_tip_m` is explicit in
  every raw history; `a0 = a_tip - Delta a` was verified constant within each
  case.
- Physical time comes from the consolidated accepted-state observer. It is not
  reconstructed from `Kdot*time`.

## State availability

Accepted-state scalar histories include radius, front width, multiplicity,
mobile/retained populations, backstress, signed shielding, cleavage and
emission rates/actions, resolved signed shears, and channel-resolved emission
and transport values. The default-off observer did **not** archive complete
opening/channel tensor matrices. Such matrices must therefore be reported as
unavailable unless an exact deterministic frozen-state probe can reconstruct
the requested archived state; they must not be inferred from later states.

## Fingerprint

- `pf_canonical_full_step_trajectories.parquet`: `{table_hash}`
"""
    (output / "PF_CANONICAL_FULL_TRAJECTORY_DATA_AUDIT.md").write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=RAW_DEFAULT)
    parser.add_argument("--publication", type=Path, default=PUBLICATION_DEFAULT)
    parser.add_argument("--output", type=Path, default=OUTPUT_DEFAULT)
    args = parser.parse_args()

    manifest = pd.read_csv(args.publication / "pf_canonical_fracture_run_manifest.csv")
    onset = pd.read_csv(args.publication / "pf_canonical_onset_candidates_v2.csv")
    avalanches = pd.read_csv(
        args.publication / "pf_canonical_physical_avalanches_v2.csv"
    )
    if len(manifest) != PROVENANCE["canonical_case_count"]:
        raise ValueError(f"expected 288 canonical cases, found {len(manifest)}")
    if manifest.case_id.duplicated().any():
        raise ValueError("canonical manifest contains duplicate case IDs")

    args.output.mkdir(parents=True, exist_ok=True)
    tables: list[pd.DataFrame] = []
    index_rows: list[dict[str, Any]] = []
    offset = 0
    for number, row in enumerate(manifest.itertuples(index=False), start=1):
        table, index_row = build_case(
            pd.Series(row._asdict()), args.raw_root, onset, avalanches, offset
        )
        tables.append(table)
        index_rows.append(index_row)
        offset += len(table)
        if number % 24 == 0:
            print(f"verified {number}/{len(manifest)} cases; {offset:,} rows")

    combined_full_state = pd.concat(tables, ignore_index=True)
    index = pd.DataFrame(index_rows)
    peak_state_path = args.output / "pf_peak_theta0_rate_full_state_source.parquet"
    peak_state = combined_full_state.loc[
        combined_full_state.material_class.eq("Peak")
        & combined_full_state.theta_deg.eq(0.0)
        & combined_full_state.is_rate_matrix_case
    ]
    peak_state.to_parquet(
        peak_state_path, index=False, compression="zstd", compression_level=19
    )
    combined = combined_full_state[list(CORE_TRAJECTORY_COLUMNS)].copy()
    parquet = args.output / "pf_canonical_full_step_trajectories.parquet"
    combined.to_parquet(
        parquet, index=False, compression="zstd", compression_level=19
    )
    roundtrip = pd.read_parquet(parquet)
    if len(roundtrip) != len(combined) or list(roundtrip.columns) != list(combined.columns):
        raise ValueError("Parquet schema/row-count round trip failed")
    index_path = args.output / "pf_canonical_full_step_trajectories_index.csv"
    index.to_csv(index_path, index=False)

    table_hash = sha256(parquet)
    manifest_out = {
        "schema": "pf_canonical_full_trajectory_manifest_v1",
        "analysis_only": True,
        "stochastic_production_run_launched": False,
        "fem_czm_run_launched": False,
        "raw_canonical_artifacts_modified": False,
        "provenance": PROVENANCE,
        "raw_root": str(args.raw_root),
        "canonical_case_count": len(index),
        "accepted_row_count": len(combined),
        "unique_case_count": int(combined.case_id.nunique()),
        "shared_theta0_rate1_physical_case_count": int(
            manifest["is_orientation_matrix_case"].astype(bool)
            .mul(manifest["is_rate_matrix_case"].astype(bool))
            .sum()
        ),
        "all_steps_sha256_verified": True,
        "all_observer_rows_aligned": True,
        "duplicate_extension_rows_preserved": int(
            index.duplicate_extension_row_count.sum()
        ),
        "tensor_state_availability": {
            "resolved_signed_shears": "archived_at_every_accepted_step",
            "complete_opening_tensor": "not_archived",
            "complete_channel_tensors": "not_archived",
            "policy": "fail_closed_no_later_state_inference",
        },
        "artifacts": {
            parquet.name: {"sha256": table_hash, "rows": len(combined)},
            index_path.name: {"sha256": sha256(index_path), "rows": len(index)},
            peak_state_path.name: {
                "sha256": sha256(peak_state_path), "rows": len(peak_state)
            },
        },
    }
    manifest_out["scientific_fingerprint_sha256"] = canonical_json_sha256(
        {
            "provenance": PROVENANCE,
            "case_ids": index.case_id.tolist(),
            "steps_hashes": index.raw_steps_sha256.tolist(),
            "accepted_row_counts": index.accepted_row_count.tolist(),
            "parquet_sha256": table_hash,
        }
    )
    manifest_path = args.output / "pf_canonical_full_trajectory_manifest.json"
    manifest_path.write_text(json.dumps(manifest_out, indent=2, sort_keys=True) + "\n")
    write_audit(args.output, index, table_hash)
    print(f"wrote {parquet} ({len(combined):,} rows, {len(index)} cases)")


if __name__ == "__main__":
    main()
