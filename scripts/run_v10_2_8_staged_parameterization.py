#!/usr/bin/env python3
"""Run the v10.2.8 analytical -> first-passage -> R-curve -> 2-D gates."""
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

from arrhenius_fracture.analytical_screen_v1028 import (
    AnalyticalControl,
    AnalyticalMechanics,
    DBTT_FIRST_PASSAGE_TEMPERATURES_K,
    WEAKT_FIRST_PASSAGE_TEMPERATURES_K,
    analytical_screen,
)
from arrhenius_fracture.reduced_campaign_v1024 import manifest_from_row
from arrhenius_fracture.signed_kernel_family_v1026 import (
    StateResolvedSignedShieldingKernelFamily,
)
from arrhenius_fracture.staged_parameterization_v1028 import (
    FirstPassageControl,
    run_first_passage,
    score_first_passage_dbtt,
    score_first_passage_weakt,
    score_rcurve_dbtt,
    score_rcurve_weakt,
    two_d_validation_cases,
)
from arrhenius_fracture.state_resolved_drive_family_v1027 import (
    StateResolvedSignedDriveFamily,
)
from arrhenius_fracture.state_resolved_reduced_campaign_v1027 import (
    ReducedCampaignControl,
    StateResolvedProductionConfig,
    run_reduced_r_curve,
    score_ceramic_reference,
)

MODEL_ID = "v10.2.8_staged_parameterization_campaign"
SIGNED_FIELDS = {
    "cleave_gT_eV_per_K",
    "cleave_sT_GPa_per_K",
    "emit_gT_eV_per_K",
    "emit_sT_GPa_per_K",
    "peierls_activation_entropy_kB",
    "taylor_activation_entropy_kB",
}
DBTT_SEARCH = {
    "cleave_G00_eV": ("factor", 1.35),
    "cleave_gT_eV_per_K": ("absolute", 0.0012),
    "cleave_sigc0_GPa": ("factor", 1.35),
    "emit_G00_eV": ("factor", 2.20),
    "emit_gT_eV_per_K": ("absolute", 0.0035),
    "emit_sigc0_GPa": ("factor", 2.20),
    "peierls_H0_eV": ("factor", 2.0),
    "peierls_activation_entropy_kB": ("absolute", 30.0),
    "taylor_H0_eV": ("factor", 2.0),
    "taylor_activation_entropy_kB": ("absolute", 30.0),
    "taylor_corr_rho_c_m2": ("factor", 50.0),
    "taylor_corr_scale": ("factor", 6.0),
    "encounter_efficiency": ("factor", 6.0),
    "retained_recovery_rate_s": ("factor", 50.0),
    "source_refresh_length_um": ("factor", 6.0),
    "c_blunt": ("factor", 3.5),
}
WEAKT_SEARCH = {
    "cleave_G00_eV": ("factor", 1.25),
    "cleave_gT_eV_per_K": ("absolute", 0.0007),
    "cleave_sigc0_GPa": ("factor", 1.25),
    "emit_G00_eV": ("factor", 1.90),
    "emit_gT_eV_per_K": ("absolute", 0.0013),
    "emit_sigc0_GPa": ("factor", 1.80),
    "peierls_H0_eV": ("factor", 1.70),
    "peierls_activation_entropy_kB": ("absolute", 15.0),
    "taylor_H0_eV": ("factor", 1.70),
    "taylor_activation_entropy_kB": ("absolute", 15.0),
    "taylor_corr_rho_c_m2": ("factor", 15.0),
    "taylor_corr_scale": ("factor", 4.0),
    "encounter_efficiency": ("factor", 4.0),
    "retained_recovery_rate_s": ("factor", 15.0),
    "source_refresh_length_um": ("factor", 4.0),
    "c_blunt": ("factor", 2.5),
}


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    converted = []
    for raw in rows:
        row: dict[str, Any] = dict(raw)
        for key, value in list(row.items()):
            if key in {"candidate_id", "target_class"}:
                continue
            if value == "":
                continue
            try:
                row[key] = float(value)
            except ValueError:
                pass
        converted.append(row)
    return converted


def _read_one(path: str | Path) -> dict[str, Any]:
    rows = _read_rows(path)
    if len(rows) != 1:
        raise ValueError(f"expected one manifest row in {path}; found {len(rows)}")
    return rows[0]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (dict, list, tuple, np.ndarray)):
                continue
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _physical_source_interval(kernel) -> tuple[float, float]:
    bounds = np.asarray(kernel.source_capacity_bounds, dtype=float)
    low = float(np.max(bounds[:, 0]))
    high = float(np.min(bounds[:, 1]))
    if not math.isfinite(low) or not math.isfinite(high) or high <= 0.0 or high < low:
        raise ValueError("kernel source-capacity bounds have no common scalar interval")
    return max(low, 1.0e-12), high


def _generate_rows(target_class: str, anchors: list[dict[str, Any]], samples: int,
                   seed: int, source_interval: tuple[float, float]) -> list[dict[str, Any]]:
    search = DBTT_SEARCH if target_class == "DBTT" else WEAKT_SEARCH
    n_anchor = len(anchors)
    if n_anchor < 1:
        raise ValueError(f"no {target_class} anchors supplied")
    numeric_fields = [
        key for key in anchors[0]
        if key not in {"candidate_id", "target_class"}
        and isinstance(anchors[0][key], (int, float))
    ]
    dimension = n_anchor + len(search) + 1
    exponent = int(math.ceil(math.log2(max(samples, 1))))
    points = qmc.Sobol(d=dimension, scramble=True, seed=seed).random_base2(exponent)[:samples]
    low_source, high_source = source_interval
    rows = []
    for index, point in enumerate(points):
        weights = np.maximum(point[:n_anchor], 1.0e-12)
        weights /= np.sum(weights)
        prefix = "DBTT" if target_class == "DBTT" else "WEAKT"
        row: dict[str, Any] = {
            "candidate_id": f"{prefix}_V1028_{index:07d}",
            "target_class": target_class,
        }
        for i, weight in enumerate(weights):
            row[f"anchor_weight_{i}"] = float(weight)
        for field in numeric_fields:
            values = np.asarray([float(anchor[field]) for anchor in anchors], dtype=float)
            row[field] = (
                float(np.exp(weights @ np.log(values)))
                if field not in SIGNED_FIELDS and np.all(values > 0.0)
                else float(weights @ values)
            )
        for offset, (field, (kind, magnitude)) in enumerate(search.items(), start=n_anchor):
            coordinate = 2.0 * float(point[offset]) - 1.0
            row[field] = (
                max(float(row[field]) * math.exp(coordinate * math.log(magnitude)), 1.0e-30)
                if kind == "factor"
                else float(row[field]) + coordinate * magnitude
            )
        coordinate = float(point[-1])
        row["source_sites_per_system"] = (
            math.exp(math.log(low_source) + coordinate * (math.log(high_source) - math.log(low_source)))
            if high_source / low_source > 1.000001 else low_source
        )
        row["max_K_shield_MPa_sqrt_m"] = 0.0
        rows.append(row)
    return rows


def _authorized_families(kernel_path: Path, drive_path: Path):
    kernel = StateResolvedSignedShieldingKernelFamily.from_json(kernel_path)
    drive = StateResolvedSignedDriveFamily.from_json(drive_path)
    drive.validate_against_kernel_family(kernel)
    if not bool(kernel.metadata.get("production_parameterization_allowed", False)):
        raise ValueError("shielding-kernel family is not authorized for parameterization")
    if not bool(drive.metadata.get("production_parameterization_allowed", False)):
        raise ValueError("signed-drive family is not authorized for parameterization")
    return kernel, drive


def _production(args) -> StateResolvedProductionConfig:
    payload = json.loads(args.engine_template.read_text())
    return StateResolvedProductionConfig.from_trace(
        payload,
        kernel_family_path=args.kernel_family,
        drive_family_path=args.drive_family,
    )


def _select(rows: list[dict[str, Any]], pass_key: str, objective_key: str,
            target_class: str, count: int) -> list[dict[str, Any]]:
    subset = [row for row in rows if str(row.get("target_class")) == target_class]
    subset.sort(key=lambda row: (
        not bool(row.get(pass_key, False)),
        float(row.get(objective_key, math.inf)),
    ))
    return subset[:count]


def _worker_analytical(payload):
    row, kernel_path, drive_path, engine_path, control_dict = payload
    try:
        kernel = StateResolvedSignedShieldingKernelFamily.from_json(kernel_path)
        drive = StateResolvedSignedDriveFamily.from_json(drive_path)
        mechanics = AnalyticalMechanics.from_engine_config(engine_path)
        result = analytical_screen(
            manifest_from_row(row), mechanics, AnalyticalControl(**control_dict),
            kernel, drive, target_class=str(row["target_class"]),
        )
        summary = {key: value for key, value in result.items() if key != "details"}
        return {"ok": True, "row": {**row, **summary}, "details": result}
    except Exception as exc:
        return {"ok": False, "candidate_id": row.get("candidate_id"),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _worker_first(payload):
    row, production_dict, control_dict = payload
    try:
        production = StateResolvedProductionConfig(**production_dict)
        control = FirstPassageControl(**control_dict)
        target = str(row["target_class"])
        temperatures = (
            DBTT_FIRST_PASSAGE_TEMPERATURES_K if target == "DBTT"
            else WEAKT_FIRST_PASSAGE_TEMPERATURES_K
        )
        manifest = manifest_from_row(row)
        runs = {("full", T): run_first_passage(manifest, T, production, control) for T in temperatures}
        score = score_first_passage_dbtt(runs) if target == "DBTT" else score_first_passage_weakt(runs)
        return {"ok": True, "row": {**row, **score}}
    except Exception as exc:
        return {"ok": False, "candidate_id": row.get("candidate_id"),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _worker_rcurve(payload):
    row, production_dict, control_dict = payload
    try:
        production = StateResolvedProductionConfig(**production_dict)
        control = ReducedCampaignControl(**control_dict)
        target = str(row["target_class"])
        temperatures = (
            DBTT_FIRST_PASSAGE_TEMPERATURES_K if target == "DBTT"
            else WEAKT_FIRST_PASSAGE_TEMPERATURES_K
        )
        manifest = manifest_from_row(row)
        runs = {("full", T): run_reduced_r_curve(manifest, T, production, control, mode="full") for T in temperatures}
        for mode in ("plasticity_off", "shielding_off", "backstress_off"):
            for T in (temperatures[0], temperatures[-1]):
                runs[(mode, T)] = run_reduced_r_curve(manifest, T, production, control, mode=mode)
        score = score_rcurve_dbtt(runs) if target == "DBTT" else score_rcurve_weakt(runs)
        return {"ok": True, "row": {**row, **score}}
    except Exception as exc:
        return {"ok": False, "candidate_id": row.get("candidate_id"),
                "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _parallel(payloads, worker, workers: int, progress_path: Path):
    completed, failures = [], []
    with progress_path.open("w") as progress:
        if workers <= 1:
            iterator = (worker(payload) for payload in payloads)
            for index, result in enumerate(iterator, start=1):
                progress.write(json.dumps(result) + "\n"); progress.flush()
                (completed if result["ok"] else failures).append(result.get("row", result))
                print(f"[{index}/{len(payloads)}] {result.get('candidate_id', result.get('row', {}).get('candidate_id'))} ok={int(result['ok'])}", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(worker, payload): payload[0]["candidate_id"] for payload in payloads}
                for index, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    progress.write(json.dumps(result) + "\n"); progress.flush()
                    (completed if result["ok"] else failures).append(result.get("row", result))
                    print(f"[{index}/{len(payloads)}] {futures[future]} ok={int(result['ok'])}", flush=True)
    return completed, failures


def _write_manifests(root: Path, rows: list[dict[str, Any]]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    manifest_fields = {
        "candidate_id", "target_class",
        "cleave_G00_eV", "cleave_gT_eV_per_K", "cleave_sigc0_GPa",
        "cleave_sT_GPa_per_K", "cleave_exp_a", "cleave_exp_n", "cleave_floor_frac",
        "emit_G00_eV", "emit_gT_eV_per_K", "emit_sigc0_GPa", "emit_sT_GPa_per_K",
        "emit_exp_a", "emit_exp_n", "emit_floor_frac",
        "peierls_H0_eV", "peierls_activation_entropy_kB", "peierls_exp_a",
        "peierls_exp_n", "peierls_nu0_s", "taylor_H0_eV",
        "taylor_activation_entropy_kB", "taylor_exp_a", "taylor_exp_n",
        "taylor_nu0_s", "taylor_corr_rho_c_m2", "taylor_corr_scale",
        "source_sites_per_system", "encounter_efficiency", "retained_recovery_rate_s",
        "source_refresh_length_um", "c_blunt", "max_K_shield_MPa_sqrt_m",
    }
    for row in rows:
        fields = [key for key in row if key in manifest_fields]
        path = root / f"{row['candidate_id']}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader(); writer.writerow({key: row[key] for key in fields})


def stage_analytical(args) -> None:
    kernel, _drive = _authorized_families(args.kernel_family, args.drive_family)
    source_interval = _physical_source_interval(kernel)
    dbtt = _generate_rows("DBTT", [_read_one(path) for path in args.dbtt_anchor],
                          args.samples_dbtt, args.seed, source_interval)
    weakt = _generate_rows("weakT", [_read_one(path) for path in args.weakt_anchor],
                           args.samples_weakt, args.seed + 1, source_interval)
    candidates = dbtt + weakt
    _write_csv(args.out / "generated_candidates.csv", candidates)
    control = AnalyticalControl(
        Kdot_MPa_sqrt_m_s=args.Kdot,
        Kmax_MPa_sqrt_m=args.Kmax,
        dK_MPa_sqrt_m=args.analytical_dK,
    )
    payloads = [
        (row, str(args.kernel_family), str(args.drive_family), str(args.engine_template), asdict(control))
        for row in candidates
    ]
    completed, failures = _parallel(payloads, _worker_analytical, args.workers,
                                    args.out / "analytical_progress.jsonl")
    _write_csv(args.out / "analytical_scores.csv", completed)
    (args.out / "analytical_failures.json").write_text(json.dumps(failures, indent=2))
    promoted = _select(completed, "analytical_pass", "analytical_objective", "DBTT", args.promote_dbtt)
    promoted += _select(completed, "analytical_pass", "analytical_objective", "weakT", args.promote_weakt)
    _write_csv(args.out / "promoted_to_first_passage.csv", promoted)
    _write_manifests(args.out / "promoted_manifests", promoted)


def _first_passage_ceramic(args, production, control):
    if args.ceramic_reference is None:
        return None
    row = _read_one(args.ceramic_reference)
    manifest = manifest_from_row(row)
    runs = [run_first_passage(manifest, T, production, control) for T in DBTT_FIRST_PASSAGE_TEMPERATURES_K]
    values = np.asarray([run["K_init_MPa_sqrt_m"] for run in runs], dtype=float)
    passed = bool(all(run["status"] == "complete" for run in runs)
                  and np.min(values) > 0.0 and np.max(values) / np.min(values) <= 1.20)
    result = {"passed": passed, "temperature_span_ratio": float(np.max(values) / np.min(values)),
              "values": values.tolist()}
    (args.out / "ceramic_first_passage_control.json").write_text(json.dumps(result, indent=2))
    if not passed:
        raise RuntimeError("frozen ceramic first-passage control failed")
    return result


def stage_first(args) -> None:
    _authorized_families(args.kernel_family, args.drive_family)
    production = _production(args)
    control = FirstPassageControl(Kdot_MPa_sqrt_m_s=args.Kdot,
                                  Kmax_MPa_sqrt_m=args.Kmax,
                                  dK_MPa_sqrt_m=args.dK)
    _first_passage_ceramic(args, production, control)
    rows = _read_rows(args.candidates)
    payloads = [(row, asdict(production), asdict(control)) for row in rows]
    completed, failures = _parallel(payloads, _worker_first, args.workers,
                                    args.out / "first_passage_progress.jsonl")
    _write_csv(args.out / "first_passage_scores.csv", completed)
    (args.out / "first_passage_failures.json").write_text(json.dumps(failures, indent=2))
    promoted = _select(completed, "first_passage_pass", "first_passage_objective", "DBTT", args.promote_dbtt)
    promoted += _select(completed, "first_passage_pass", "first_passage_objective", "weakT", args.promote_weakt)
    _write_csv(args.out / "promoted_to_rcurve.csv", promoted)
    _write_manifests(args.out / "promoted_manifests", promoted)


def _rcurve_ceramic(args, production, control):
    if args.ceramic_reference is None:
        raise ValueError("R-curve stage requires --ceramic-reference")
    manifest = manifest_from_row(_read_one(args.ceramic_reference))
    runs = {("full", T): run_reduced_r_curve(manifest, T, production, control, mode="full")
            for T in DBTT_FIRST_PASSAGE_TEMPERATURES_K}
    result = score_ceramic_reference(runs)
    (args.out / "ceramic_rcurve_control.json").write_text(json.dumps(result, indent=2))
    if not bool(result.get("strict_reduced_pass", False)):
        raise RuntimeError("frozen ceramic R-curve control failed")


def stage_rcurve(args) -> None:
    _authorized_families(args.kernel_family, args.drive_family)
    production = _production(args)
    control = ReducedCampaignControl(
        Kdot_MPa_sqrt_m_s=args.Kdot, Kmax_MPa_sqrt_m=args.Kmax,
        dK_MPa_sqrt_m=args.dK, target_extension_um=args.target_extension_um,
    ).validate()
    _rcurve_ceramic(args, production, control)
    rows = _read_rows(args.candidates)
    payloads = [(row, asdict(production), asdict(control)) for row in rows]
    completed, failures = _parallel(payloads, _worker_rcurve, args.workers,
                                    args.out / "rcurve_progress.jsonl")
    _write_csv(args.out / "rcurve_scores.csv", completed)
    (args.out / "rcurve_failures.json").write_text(json.dumps(failures, indent=2))
    promoted = _select(completed, "rcurve_pass", "rcurve_objective", "DBTT", args.promote_dbtt)
    promoted += _select(completed, "rcurve_pass", "rcurve_objective", "weakT", args.promote_weakt)
    _write_csv(args.out / "promoted_to_2d.csv", promoted)
    _write_manifests(args.out / "promoted_manifests", promoted)
    dbtt_ids = [row["candidate_id"] for row in promoted if row["target_class"] == "DBTT"]
    weakt_ids = [row["candidate_id"] for row in promoted if row["target_class"] == "weakT"]
    _write_csv(args.out / "two_d_validation_cases.csv",
               two_d_validation_cases(dbtt_ids, weakt_ids, target_extension_um=args.two_d_extension_um))


def stage_2d_plan(args) -> None:
    rows = _read_rows(args.candidates)
    dbtt_ids = [str(row["candidate_id"]) for row in rows if row["target_class"] == "DBTT"]
    weakt_ids = [str(row["candidate_id"]) for row in rows if row["target_class"] == "weakT"]
    _write_csv(args.out / "two_d_validation_cases.csv",
               two_d_validation_cases(dbtt_ids, weakt_ids, target_extension_um=args.two_d_extension_um))
    (args.out / "README.txt").write_text(
        "Run every row with arrhenius_fracture.sharp_front_v10_2_6 using the same "
        "authorized kernel family. Full cases use the production signed engine; "
        "endpoint ablations must alter only the named mechanism. Validate 100 um "
        "R-curves before 500 um production.\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("analytical", "first-passage", "rcurve", "2d-plan"), required=True)
    parser.add_argument("--kernel-family", type=Path)
    parser.add_argument("--drive-family", type=Path)
    parser.add_argument("--engine-template", type=Path)
    parser.add_argument("--dbtt-anchor", type=Path, action="append", default=[])
    parser.add_argument("--weakt-anchor", type=Path, action="append", default=[])
    parser.add_argument("--ceramic-reference", type=Path)
    parser.add_argument("--candidates", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--samples-dbtt", type=int, default=4096)
    parser.add_argument("--samples-weakt", type=int, default=2048)
    parser.add_argument("--promote-dbtt", type=int, default=256)
    parser.add_argument("--promote-weakt", type=int, default=256)
    parser.add_argument("--seed", type=int, default=1201)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--Kdot", type=float, default=0.005)
    parser.add_argument("--Kmax", type=float, default=80.0)
    parser.add_argument("--analytical-dK", type=float, default=0.20)
    parser.add_argument("--dK", type=float, default=0.05)
    parser.add_argument("--target-extension-um", type=float, default=50.0)
    parser.add_argument("--two-d-extension-um", type=float, default=100.0)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.mkdir(parents=True)
    if args.stage != "2d-plan":
        for name in ("kernel_family", "drive_family", "engine_template"):
            path = getattr(args, name)
            if path is None or not path.is_file():
                raise SystemExit(f"--{name.replace('_', '-')} must name an existing file")
    if args.stage in {"first-passage", "rcurve", "2d-plan"}:
        if args.candidates is None or not args.candidates.is_file():
            raise SystemExit("--candidates must name the previous stage promotion CSV")
    if args.stage == "analytical":
        if not args.dbtt_anchor or not args.weakt_anchor:
            raise SystemExit("analytical stage requires DBTT and weakT anchors")
        stage_analytical(args)
    elif args.stage == "first-passage":
        stage_first(args)
    elif args.stage == "rcurve":
        stage_rcurve(args)
    else:
        stage_2d_plan(args)
    (args.out / "stage_complete.json").write_text(json.dumps({
        "schema": MODEL_ID, "stage": args.stage, "complete": True,
    }, indent=2))


if __name__ == "__main__":
    main()
