#!/usr/bin/env python3
"""Run guarded DBTT and FCC-like weakT parameterization with v10.2.6 physics."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
from dataclasses import asdict
import json
import math
from pathlib import Path
import traceback
from typing import Any

import numpy as np
from scipy.stats import qmc

from arrhenius_fracture.reduced_campaign_v1024 import manifest_from_row
from arrhenius_fracture.signed_kernel_family_v1026 import (
    StateResolvedSignedShieldingKernelFamily,
)
from arrhenius_fracture.state_resolved_drive_family_v1027 import (
    StateResolvedSignedDriveFamily,
)
from arrhenius_fracture.state_resolved_reduced_campaign_v1027 import (
    DEFAULT_TEMPERATURES_K,
    MODEL_ID,
    ReducedCampaignControl,
    StateResolvedProductionConfig,
    run_reduced_r_curve,
    score_ceramic_reference,
    score_dbtt,
    score_weakt,
)

SIGNED_FIELDS = {
    "cleave_gT_eV_per_K",
    "cleave_sT_GPa_per_K",
    "emit_gT_eV_per_K",
    "emit_sT_GPa_per_K",
    "peierls_activation_entropy_kB",
    "taylor_activation_entropy_kB",
}

DBTT_SEARCH = {
    "cleave_G00_eV": ("factor", 1.30),
    "cleave_gT_eV_per_K": ("absolute", 0.0010),
    "cleave_sigc0_GPa": ("factor", 1.30),
    "emit_G00_eV": ("factor", 2.00),
    "emit_gT_eV_per_K": ("absolute", 0.0030),
    "emit_sigc0_GPa": ("factor", 2.00),
    "peierls_H0_eV": ("factor", 1.80),
    "peierls_activation_entropy_kB": ("absolute", 25.0),
    "taylor_H0_eV": ("factor", 1.80),
    "taylor_activation_entropy_kB": ("absolute", 25.0),
    "taylor_corr_rho_c_m2": ("factor", 30.0),
    "taylor_corr_scale": ("factor", 5.0),
    "encounter_efficiency": ("factor", 5.0),
    "retained_recovery_rate_s": ("factor", 30.0),
    "source_refresh_length_um": ("factor", 5.0),
    "c_blunt": ("factor", 3.0),
}

WEAKT_SEARCH = {
    "cleave_G00_eV": ("factor", 1.20),
    "cleave_gT_eV_per_K": ("absolute", 0.0005),
    "cleave_sigc0_GPa": ("factor", 1.20),
    "emit_G00_eV": ("factor", 1.70),
    "emit_gT_eV_per_K": ("absolute", 0.0010),
    "emit_sigc0_GPa": ("factor", 1.60),
    "peierls_H0_eV": ("factor", 1.50),
    "peierls_activation_entropy_kB": ("absolute", 12.0),
    "taylor_H0_eV": ("factor", 1.50),
    "taylor_activation_entropy_kB": ("absolute", 12.0),
    "taylor_corr_rho_c_m2": ("factor", 10.0),
    "taylor_corr_scale": ("factor", 3.0),
    "encounter_efficiency": ("factor", 3.0),
    "retained_recovery_rate_s": ("factor", 10.0),
    "source_refresh_length_um": ("factor", 3.0),
    "c_blunt": ("factor", 2.0),
}


def _read_one_row(path: str | Path) -> dict[str, Any]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"expected one row in {path}; found {len(rows)}")
    row: dict[str, Any] = dict(rows[0])
    for key, value in list(row.items()):
        if key in {"candidate_id", "target_class"}:
            continue
        row[key] = float(value)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _physical_source_interval(kernel) -> tuple[float, float]:
    bounds = np.asarray(kernel.source_capacity_bounds, dtype=float)
    low = float(np.max(bounds[:, 0]))
    high = float(np.min(bounds[:, 1]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= 0.0 or high < low:
        raise ValueError("signed-kernel source-capacity bounds have no common scalar interval")
    return max(low, 1.0e-12), high


def _generate_rows(
    *,
    target_class: str,
    anchors: list[dict[str, Any]],
    samples: int,
    seed: int,
    source_interval: tuple[float, float],
) -> list[dict[str, Any]]:
    if not anchors:
        raise ValueError(f"no anchors supplied for {target_class}")
    search = DBTT_SEARCH if target_class == "DBTT" else WEAKT_SEARCH
    numeric_fields = [
        key
        for key in anchors[0]
        if key not in {"candidate_id", "target_class"}
    ]
    n_anchor = len(anchors)
    dimension = n_anchor + len(search) + 1
    exponent = int(math.ceil(math.log2(max(int(samples), 1))))
    points = qmc.Sobol(d=dimension, scramble=True, seed=int(seed)).random_base2(exponent)
    points = points[: int(samples)]
    rows: list[dict[str, Any]] = []
    search_names = list(search)
    low_source, high_source = source_interval

    for index, point in enumerate(points):
        weights = np.maximum(point[:n_anchor], 1.0e-12)
        weights /= np.sum(weights)
        prefix = "DBTT" if target_class == "DBTT" else "WEAKT"
        row: dict[str, Any] = {
            "candidate_id": f"{prefix}_V1027_{index:07d}",
            "target_class": target_class,
        }
        for anchor_index, weight in enumerate(weights):
            row[f"anchor_weight_{anchor_index}"] = float(weight)
        for field in numeric_fields:
            values = np.asarray([float(anchor[field]) for anchor in anchors], dtype=float)
            if field not in SIGNED_FIELDS and np.all(values > 0.0):
                value = float(np.exp(weights @ np.log(values)))
            else:
                value = float(weights @ values)
            row[field] = value
        for offset, field in enumerate(search_names, start=n_anchor):
            kind, magnitude = search[field]
            coordinate = 2.0 * float(point[offset]) - 1.0
            if kind == "factor":
                row[field] = max(
                    float(row[field]) * math.exp(coordinate * math.log(magnitude)),
                    1.0e-30,
                )
            else:
                row[field] = float(row[field]) + coordinate * magnitude
        source_coordinate = float(point[-1])
        if high_source / low_source > 1.000001:
            source = math.exp(
                math.log(low_source)
                + source_coordinate * (math.log(high_source) - math.log(low_source))
            )
        else:
            source = low_source
        row["source_sites_per_system"] = source
        row["max_K_shield_MPa_sqrt_m"] = 0.0
        rows.append(row)
    return rows


def _evaluate_manifest(
    manifest,
    production: StateResolvedProductionConfig,
    control: ReducedCampaignControl,
    target_class: str,
):
    runs: dict[tuple[str, float], dict[str, Any]] = {}
    for T in DEFAULT_TEMPERATURES_K:
        runs[("full", T)] = run_reduced_r_curve(
            manifest, T, production, control, mode="full"
        )
    if target_class != "ceramic":
        for mode in ("plasticity_off", "shielding_off", "backstress_off"):
            for T in (DEFAULT_TEMPERATURES_K[0], DEFAULT_TEMPERATURES_K[-1]):
                runs[(mode, T)] = run_reduced_r_curve(
                    manifest, T, production, control, mode=mode
                )
    if target_class == "DBTT":
        score = score_dbtt(runs)
    elif target_class == "weakT":
        score = score_weakt(runs)
    else:
        score = score_ceramic_reference(runs)
    return score, runs


def _worker(payload):
    row, production_dict, control_dict, target_class = payload
    try:
        manifest = manifest_from_row(row)
        score, _runs = _evaluate_manifest(
            manifest,
            StateResolvedProductionConfig(**production_dict),
            ReducedCampaignControl(**control_dict),
            target_class,
        )
        return {"ok": True, "row": {**row, **score}}
    except Exception as exc:
        return {
            "ok": False,
            "candidate_id": row.get("candidate_id"),
            "target_class": target_class,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "row": row,
        }


def _run_class(
    rows: list[dict[str, Any]],
    *,
    production: StateResolvedProductionConfig,
    control: ReducedCampaignControl,
    workers: int,
    progress_path: Path,
):
    completed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    payloads = [
        (row, asdict(production), asdict(control), str(row["target_class"]))
        for row in rows
    ]
    with progress_path.open("w") as progress:
        if workers <= 1:
            iterator = enumerate(map(_worker, payloads), start=1)
            for count, result in iterator:
                progress.write(json.dumps(result) + "\n")
                progress.flush()
                if result["ok"]:
                    completed.append(result["row"])
                else:
                    failures.append(result)
                print(
                    f"[{count}/{len(payloads)}] "
                    f"{result.get('candidate_id', result.get('row', {}).get('candidate_id'))} "
                    f"ok={int(result['ok'])}",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(_worker, payload): payload[0]["candidate_id"]
                    for payload in payloads
                }
                for count, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    progress.write(json.dumps(result) + "\n")
                    progress.flush()
                    if result["ok"]:
                        completed.append(result["row"])
                    else:
                        failures.append(result)
                    print(
                        f"[{count}/{len(payloads)}] {futures[future]} "
                        f"ok={int(result['ok'])}",
                        flush=True,
                    )
    completed.sort(
        key=lambda row: (
            not bool(row.get("strict_reduced_pass", False)),
            float(row.get("objective", math.inf)),
        )
    )
    return completed, failures


def _write_manifest(path: Path, row: dict[str, Any]) -> None:
    excluded_prefixes = ("anchor_weight_", "full_K_")
    excluded = {
        "strict_reduced_pass",
        "objective",
        "full_endpoint_ratio",
        "low_R_rise_fraction",
        "high_R_rise_fraction",
        "plasticity_off_endpoint_ratio",
        "shielding_fraction_of_temperature_rise",
        "shielding_fraction_of_high_T_R_rise",
        "backstress_off_high_T_R_rise_MPa_sqrt_m",
        "monotonic_temperature_fraction",
        "full_final_temperature_span_ratio",
        "full_init_temperature_span_ratio",
        "minimum_R_rise_MPa_sqrt_m",
        "maximum_R_rise_MPa_sqrt_m",
        "minimum_R_rise_fraction",
        "maximum_R_rise_fraction",
        "plasticity_fraction_of_mean_R_rise",
        "shielding_fraction_of_mean_R_rise",
    }
    fields = [
        key
        for key in row
        if key not in excluded and not key.startswith(excluded_prefixes)
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({key: row[key] for key in fields})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kernel-family", type=Path, required=True)
    parser.add_argument("--drive-family", type=Path, required=True)
    parser.add_argument("--engine-template", type=Path, required=True)
    parser.add_argument("--dbtt-anchor", type=Path, action="append", required=True)
    parser.add_argument("--weakt-anchor", type=Path, action="append", required=True)
    parser.add_argument("--ceramic-reference", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-dbtt", type=int, default=128)
    parser.add_argument("--samples-weakt", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--promote-per-class", type=int, default=12)
    parser.add_argument("--Kdot", type=float, default=0.005)
    parser.add_argument("--Kmax", type=float, default=80.0)
    parser.add_argument("--dK", type=float, default=0.05)
    parser.add_argument("--target-extension-um", type=float, default=50.0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")

    kernel = StateResolvedSignedShieldingKernelFamily.from_json(args.kernel_family)
    drive = StateResolvedSignedDriveFamily.from_json(args.drive_family)
    drive.validate_against_kernel_family(kernel)
    if not bool(kernel.metadata.get("production_parameterization_allowed", False)):
        raise SystemExit("kernel family is not authorized for production parameterization")
    if not bool(drive.metadata.get("production_parameterization_allowed", False)):
        raise SystemExit("drive family is not authorized for production parameterization")
    template = json.loads(args.engine_template.read_text())
    production = StateResolvedProductionConfig.from_trace(
        template,
        kernel_family_path=args.kernel_family,
        drive_family_path=args.drive_family,
    )
    control = ReducedCampaignControl(
        Kdot_MPa_sqrt_m_s=float(args.Kdot),
        Kmax_MPa_sqrt_m=float(args.Kmax),
        dK_MPa_sqrt_m=float(args.dK),
        target_extension_um=float(args.target_extension_um),
    ).validate()

    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "promoted_manifests" / "DBTT").mkdir(parents=True)
    (args.out / "promoted_manifests" / "weakT").mkdir(parents=True)
    source_interval = _physical_source_interval(kernel)

    ceramic_row = _read_one_row(args.ceramic_reference)
    ceramic_manifest = manifest_from_row(ceramic_row)
    ceramic_score, _ = _evaluate_manifest(
        ceramic_manifest, production, control, "ceramic"
    )
    (args.out / "ceramic_frozen_reference.json").write_text(
        json.dumps(ceramic_score, indent=2)
    )
    if not bool(ceramic_score.get("frozen_reference_pass", False)):
        raise SystemExit(
            "frozen ceramic-like control failed under the new shared physics; "
            "do not start DBTT/weakT optimization"
        )

    dbtt_rows = _generate_rows(
        target_class="DBTT",
        anchors=[_read_one_row(path) for path in args.dbtt_anchor],
        samples=int(args.samples_dbtt),
        seed=int(args.seed),
        source_interval=source_interval,
    )
    weakt_rows = _generate_rows(
        target_class="weakT",
        anchors=[_read_one_row(path) for path in args.weakt_anchor],
        samples=int(args.samples_weakt),
        seed=int(args.seed) + 1,
        source_interval=source_interval,
    )
    _write_csv(args.out / "generated_DBTT_candidates.csv", dbtt_rows)
    _write_csv(args.out / "generated_weakT_candidates.csv", weakt_rows)

    dbtt_completed, dbtt_failures = _run_class(
        dbtt_rows,
        production=production,
        control=control,
        workers=max(int(args.workers), 1),
        progress_path=args.out / "DBTT_progress.jsonl",
    )
    weakt_completed, weakt_failures = _run_class(
        weakt_rows,
        production=production,
        control=control,
        workers=max(int(args.workers), 1),
        progress_path=args.out / "weakT_progress.jsonl",
    )
    _write_csv(args.out / "DBTT_candidate_scores.csv", dbtt_completed)
    _write_csv(args.out / "weakT_candidate_scores.csv", weakt_completed)
    (args.out / "candidate_failures.json").write_text(
        json.dumps(dbtt_failures + weakt_failures, indent=2)
    )

    promote = max(int(args.promote_per_class), 0)
    dbtt_promoted = dbtt_completed[: min(promote, len(dbtt_completed))]
    weakt_promoted = weakt_completed[: min(promote, len(weakt_completed))]
    _write_csv(args.out / "promoted_DBTT_candidates.csv", dbtt_promoted)
    _write_csv(args.out / "promoted_weakT_candidates.csv", weakt_promoted)
    for label, promoted in (("DBTT", dbtt_promoted), ("weakT", weakt_promoted)):
        for rank, row in enumerate(promoted, start=1):
            _write_manifest(
                args.out
                / "promoted_manifests"
                / label
                / f"rank_{rank:03d}_{row['candidate_id']}.csv",
                row,
            )

    assessment = {
        "schema": MODEL_ID,
        "status": "complete",
        "kernel_family": str(args.kernel_family.resolve()),
        "drive_family": str(args.drive_family.resolve()),
        "engine_template": str(args.engine_template.resolve()),
        "source_capacity_interval_per_system": list(source_interval),
        "ceramic_reference_frozen": True,
        "ceramic_reference_pass": True,
        "DBTT_samples_completed": len(dbtt_completed),
        "DBTT_samples_failed": len(dbtt_failures),
        "DBTT_strict_pass_count": sum(
            bool(row.get("strict_reduced_pass", False)) for row in dbtt_completed
        ),
        "weakT_samples_completed": len(weakt_completed),
        "weakT_samples_failed": len(weakt_failures),
        "weakT_strict_pass_count": sum(
            bool(row.get("strict_reduced_pass", False)) for row in weakt_completed
        ),
        "reduced_candidates_are_final_parameterizations": False,
        "next_required_gate": (
            "cap-free state-resolved signed 2-D endpoint ablations and 100 um R-curves"
        ),
    }
    (args.out / "campaign_assessment.json").write_text(
        json.dumps(assessment, indent=2)
    )
    print(json.dumps(assessment, indent=2))


if __name__ == "__main__":
    main()
