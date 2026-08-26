#!/usr/bin/env python3
"""Fail-closed launcher for the canonical V2 monotonic PF campaign.

The launcher consumes the generated campaign plan and the authoritative PF
adapter registry.  It never edits either input, never enables fatigue, and
limits concurrency to two processes.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python")
EXPECTED_CLASSES = {
    "Peak": "v913_zeroD_sobol_0242980",
    "DBTT": "v913_zeroD_sobol_0202500",
    "weak-T": "oneD_v2_focused_weak_T_0016",
    "ceramic-like": "oneD_v2_focused_ceramic_like_0018",
}
CLASS_ALIASES = {"peak": "Peak", "DBTT": "DBTT", "weakT": "weak-T", "ceramic": "ceramic-like"}
LARGE_OBSERVER_ARTIFACTS = (
    "anisotropic_emission_audit_v10174.json",
    "kinetic_tip_cell_audit_v101.json",
    "v10_2_17_final_signed_stochastic_stack.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def load_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        material_class = CLASS_ALIASES.get(row["material_class"], row["material_class"])
        if material_class in result:
            raise ValueError(f"duplicate material class in PF registry: {material_class}")
        if EXPECTED_CLASSES.get(material_class) != row["candidate_id"]:
            raise ValueError(f"noncanonical material row for {material_class}: {row['candidate_id']}")
        result[material_class] = row
    if set(result) != set(EXPECTED_CLASSES):
        raise ValueError(f"PF registry classes differ from required classes: {sorted(result)}")
    return result


def family_for_theta(cache: Path, theta: float) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in cache.glob("*/family.json"):
        try:
            payload = json.loads(path.read_text())
            config = payload["mechanical_configuration"]
            actual = float(config["theta_deg"])
            coverage = 1.0e6 * float(payload["cumulative_crack_path_extension_levels_m"][-1])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
        if math.isclose(actual, theta, abs_tol=1e-12) and coverage >= 1000.0 / max(math.cos(math.radians(theta)), 1e-12):
            matches.append((path.resolve(), payload))
    if len(matches) != 1:
        raise ValueError(f"expected one qualified kernel family for theta={theta}, found {len(matches)}")
    path, payload = matches[0]
    if not payload.get("production_parameterization_allowed", False):
        raise ValueError(f"kernel family is not production-qualified: {path}")
    return path, payload


def case_id(row: dict[str, str]) -> str:
    cls = row["material_class"].replace("-", "").lower()
    return (
        f"{row['matrix'].lower()}__{cls}__T{int(float(row['temperature_K'])):04d}K"
        f"__theta{float(row['theta_deg']):g}__{row['rate_tag']}__seed{row['seed']}"
    )


def select_rows(rows: list[dict[str, str]], stage: str) -> list[dict[str, str]]:
    if stage == "all":
        return rows
    if stage == "theta":
        return [row for row in rows if row["matrix"] == "CANONICAL_SINGLE_CRACK_THETA"]
    if stage == "rate":
        return [row for row in rows if row["matrix"] == "CANONICAL_STRAIN_RATE"]
    if stage == "smoke":
        chosen: list[dict[str, str]] = []
        for cls in EXPECTED_CLASSES:
            candidates = [r for r in rows if r["material_class"] == cls and r["matrix"] == "CANONICAL_SINGLE_CRACK_THETA"]
            chosen.append(next(r for r in candidates if float(r["temperature_K"]) == 1100.0 and float(r["theta_deg"]) == 30.0))
        return chosen
    raise ValueError(stage)


def canonical_env(registry: Path, selection: Path, family: Path, seed: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "PYTHONPATH": str(ROOT),
        "PYTHONUNBUFFERED": "1",
        "CONDA_ENV": "arrhenius-sharp-front-v10",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10",
        "PARAMETER_CAMPAIGN": "1",
        "CLEAVAGE_HAZARD_MODE": "exponential",
        "CLEAVAGE_HAZARD_SEED": str(seed),
        "CLEAVAGE_EVENT_LENGTH_MODE": "threshold_scaled",
        "CLEAVAGE_EVENT_MIN_FACTOR": "0.5",
        "CLEAVAGE_EVENT_MAX_FACTOR": "4.0",
        "CLEAVAGE_EVENT_SUBSEGMENT_FRACTION": "0.1",
        "ANISOTROPIC_TRANSPORT_MODE": "validated_scalar",
        "ANISOTROPIC_USE_AVALANCHE_BACKEND": "1",
        "ANISOTROPIC_EMISSION_ENABLED": "1",
        "KERNEL_STRICT_FAMILY_OVERRIDE": "1",
        "SIGNED_KERNEL_FAMILY_JSON": str(family),
        "PERSISTENT_SOURCE_MIN_WIDTH_UM": "0",
        "ONED_V2_TRANSFER_REGISTRY": str(registry),
        "ONED_V2_TRANSFER_SELECTION": str(selection),
        "ONED_V2_TP_STATE_DIAGNOSTICS": "1",
        "MPLCONFIGDIR": "/private/tmp/pf-canonical-mpl",
    })
    for forbidden in ("V10229_FATIGUE_ENABLED", "V10230_ENERGY_GATE_ENABLED"):
        env.pop(forbidden, None)
    return env


def build_command(row: dict[str, str], option: str, family: Path, out: Path, target_um: float, save_snapshots: int) -> list[str]:
    return [
        str(PYTHON), "-u", str(ROOT / "scripts/run_oneD_v2_terminal_pf_transfer.py"),
        "--signed-kernel-family", str(family), "--mode", "2d",
        "--parameter-option", option, "--temperatures", f"{float(row['temperature_K']):g}",
        "--steps", "2000000", "--nx", "36", "--ny", "72",
        "--dU", f"{float(row['nominal_dU_m']):.17g}", "--dt", f"{float(row['nominal_dt_s']):.17g}",
        "--n-stagger", "2", "--tip-h-fine", "1e-6", "--tip-ratio", "1.2",
        "--da-phys", "5e-6", "--target-crack-extension-um", f"{target_um:.17g}",
        "--mpz-length-um", "50", "--mpz-n-bins", "80",
        "--front-state-model", "moving_pz", "--tip-source-model", "continuum",
        "--tip-kinetics-mode", "moving_velocity", "--bulk-plasticity-mode", "tip_only",
        "--directional-j-mode", "root_signed", "--tip-plasticity", "--active-shielding",
        "--signed-active-shielding", "--mobile-shield-fraction", "0", "--no-wake-shielding",
        "--crystal-aniso", "--crystal-compete", "--crystal-theta-deg", f"{float(row['theta_deg']):g}",
        "--crystal-material", "w", "--j-decomposition", "cluster", "--max-fronts", "1",
        "--crack-backend", "sharp_wake", "--adaptive-events", "--adaptive-event-target", "0.15",
        "--print-every", "200", "--save-snapshots", str(save_snapshots), "--no-plots", "--out", str(out),
    ]


def completed(case_root: Path, temperature: float, target_um: float) -> bool:
    steps = case_root / f"steps_{int(temperature):04d}K.csv"
    if not steps.is_file():
        return False
    with steps.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return False
    last = rows[-1]
    if last.get("crack_extension_m"):
        return 1.0e6 * float(last["crack_extension_m"]) >= target_um - 1e-6
    for key in ("crack_extension_um", "projected_crack_extension_um", "a_extension_um"):
        if key in last and last[key]:
            return float(last[key]) >= target_um - 1e-6
    return False


def compress_observer_artifacts(case_root: Path) -> list[dict[str, Any]]:
    """Losslessly replace large observer JSON with verified zstd members."""
    records: list[dict[str, Any]] = []
    for name in LARGE_OBSERVER_ARTIFACTS:
        source = case_root / name
        packed = case_root / f"{name}.zst"
        if not source.is_file():
            if packed.is_file():
                records.append({"source_name": name, "archive_name": packed.name,
                                "archive_sha256": sha256(packed), "status": "REUSED_VERIFIED_ARCHIVE"})
            continue
        source_hash = sha256(source)
        subprocess.run(["zstd", "-q", "-9", "-f", str(source), "-o", str(packed)], check=True)
        subprocess.run(["zstd", "-q", "-t", str(packed)], check=True)
        source_size = source.stat().st_size
        archive_size = packed.stat().st_size
        source.unlink()
        records.append({"source_name": name, "source_sha256": source_hash,
                        "source_size_bytes": source_size, "archive_name": packed.name,
                        "archive_sha256": sha256(packed), "archive_size_bytes": archive_size,
                        "status": "LOSSLESS_ZSTD_VERIFIED_SOURCE_REMOVED"})
    if records:
        (case_root / "observer_artifact_compression.json").write_text(
            json.dumps({"schema": "pf_observer_artifact_compression_v1", "artifacts": records},
                       indent=2, sort_keys=True) + "\n")
    return records


def launch_one(row: dict[str, str], *, outroot: Path, registry: Path, selection: Path,
               cache: Path, target_um: float, save_snapshots: int, source_commit: str,
               force: bool) -> dict[str, Any]:
    identifier = case_id(row)
    case_root = outroot / identifier
    case_root.mkdir(parents=True, exist_ok=True)
    if completed(case_root, float(row["temperature_K"]), target_um) and not force:
        return {"case_id": identifier, "status": "REUSED_COMPLETE", "returncode": 0, "path": str(case_root)}
    registry_rows = load_registry(registry)
    registry_row = registry_rows[row["material_class"]]
    family, family_payload = family_for_theta(cache, float(row["theta_deg"]))
    command = build_command(row, registry_row["option_key"], family, case_root, target_um, save_snapshots)
    contract = {
        "schema": "pf_canonical_fracture_case_launch_v1",
        "case_id": identifier,
        "matrix": row["matrix"],
        "material_class": row["material_class"],
        "candidate_id": registry_row["candidate_id"],
        "parameter_option": registry_row["option_key"],
        "temperature_K": float(row["temperature_K"]),
        "theta_deg": float(row["theta_deg"]),
        "rate_tag": row["rate_tag"],
        "loading_rate_factor": float(row["loading_rate_factor"]),
        "dU_m": float(row["nominal_dU_m"]),
        "dt_s": float(row["nominal_dt_s"]),
        "seed": int(row["seed"]),
        "target_extension_um": target_um,
        "physical_source_commit": "9e884fb0b0845da621d2612bdf1042e481b8df49",
        "runner_commit": source_commit,
        "registry_sha256": sha256(registry),
        "selection_sha256": sha256(selection),
        "kernel_family": str(family),
        "kernel_family_sha256": sha256(family),
        "kernel_configuration_fingerprint": family_payload.get("mechanical_configuration_fingerprint"),
        "analysis_only_observer": "ONED_V2_TP_STATE_DIAGNOSTICS=1",
        "observer_feedback": False,
        "fatigue_enabled": False,
        "command": command,
        "launched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (case_root / "canonical_case_launch.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")
    environment = canonical_env(registry, selection, family, int(row["seed"]))
    with (case_root / "run.stdout.log").open("w") as stdout, (case_root / "run.stderr.log").open("w") as stderr:
        proc = subprocess.run(command, cwd=ROOT, env=environment, stdout=stdout, stderr=stderr)
    status = "COMPLETE" if proc.returncode == 0 and completed(case_root, float(row["temperature_K"]), target_um) else "FAILED_OR_CENSORED"
    compressed = compress_observer_artifacts(case_root) if proc.returncode == 0 else []
    result = {**contract, "status": status, "returncode": proc.returncode, "path": str(case_root),
              "observer_artifact_compression": compressed,
              "finished_at_utc": datetime.now(timezone.utc).isoformat()}
    (case_root / "canonical_case_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--kernel-cache", type=Path, required=True)
    parser.add_argument("--outroot", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "theta", "rate", "all"), required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--target-extension-um", type=float, default=1000.0)
    parser.add_argument("--save-snapshots", type=int, default=20)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 2:
        raise SystemExit("workers must be one or two")
    for path in (args.plan, args.registry, args.selection):
        if not path.is_file():
            raise SystemExit(f"required input missing: {path}")
    if not PYTHON.is_file():
        raise SystemExit(f"qualified Python is unavailable: {PYTHON}")
    with args.plan.open(newline="") as stream:
        rows = select_rows(list(csv.DictReader(stream)), args.stage)
    if not rows:
        raise SystemExit("selected stage has no rows")
    source_commit = git_commit(ROOT)
    args.outroot.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    stopped_after_failure = False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        row_iter = iter(rows)
        running = {}

        def submit_next() -> bool:
            try:
                row = next(row_iter)
            except StopIteration:
                return False
            future = pool.submit(
                launch_one, row, outroot=args.outroot, registry=args.registry.resolve(),
                selection=args.selection.resolve(), cache=args.kernel_cache.resolve(),
                target_um=args.target_extension_um, save_snapshots=args.save_snapshots,
                source_commit=source_commit, force=args.force,
            )
            running[future] = row
            return True

        for _ in range(args.workers):
            submit_next()
        while running:
            done, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in done:
                running.pop(future)
                result = future.result()
                results.append(result)
                print(json.dumps({k: result.get(k) for k in ("case_id", "status", "returncode", "path")}), flush=True)
                if result["status"] not in {"COMPLETE", "REUSED_COMPLETE"}:
                    stopped_after_failure = True
            if not stopped_after_failure:
                for _ in range(len(done)):
                    submit_next()
    results.sort(key=lambda row: row["case_id"])
    manifest = args.outroot / f"canonical_{args.stage}_launch_manifest.json"
    manifest.write_text(json.dumps({
        "schema": "pf_canonical_fracture_launch_manifest_v1",
        "stage": args.stage,
        "runner_commit": source_commit,
        "workers": args.workers,
        "target_extension_um": args.target_extension_um,
        "stopped_after_failure": stopped_after_failure,
        "planned_case_count": len(rows),
        "executed_case_count": len(results),
        "cases": results,
    }, indent=2, sort_keys=True) + "\n")
    failures = [row for row in results if row["status"] not in {"COMPLETE", "REUSED_COMPLETE"}]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
