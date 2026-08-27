#!/usr/bin/env python3
"""Persistent, fresh-run controller for the bounded A-native + 8PT study.

The controller is restartable, but physical trajectories are not.  A missing
terminal record from an earlier controller invocation is classified and
quarantined before a fresh attempt is launched.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PYTHON = "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
BRANCH = "codex/v10.2.30-A-native-TP-panel"
FAMILY = "/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json"
BASE_LOADS = (10.8, 13.5, 16.2, 21.6)
NATIVE = "A_NATIVE"
RETENTION_DONOR = "A_PT_03_oneD_v2_dbtt_TP_4895f9e5b44deea5"
RETURN_DONOR = "A_PT_08_oneD_v2_dbtt_TP_f2817e7998cb7be6"
MEANINGFUL_RATE_LOG10 = 0.05  # 12.2%, below this is not a material response shift.
MEANINGFUL_M_SHIFT = 0.25


@dataclass(frozen=True)
class Job:
    job_id: str
    stage: str
    option: str
    delta_k: float
    R: float
    n_bins: int
    seed: int
    mode: str
    output: str


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def atomic_csv(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    os.replace(tmp, path)


def terminal_summary(out: Path) -> dict | None:
    path = out / "developed_fatigue_growth_summary.json"
    if not path.is_file():
        return None
    data = json.loads(path.read_text())
    if not data.get("target_reached"):
        return None
    return data


def classify(out: Path, returncode: int | None) -> str:
    data = terminal_summary(out)
    if returncode == 0 and data:
        return "PHYSICAL_TARGET_REACHED" if data.get("stable_growth_provisional") else "DEVELOPED_UNSTABLE"
    if returncode == 2 and not (out / "high_cycle_live_checkpoint.json").exists():
        return "LAUNCH_PREFLIGHT_FAILURE"
    if (out / "high_cycle_live_checkpoint.json").exists():
        return "WATCHDOG_OR_OPERATIONAL_TERMINATION"
    return "NUMERICAL_FAILURE"


def base_analysis(root: Path) -> dict:
    jobs = pd.read_csv(root / "A_native_plus_8PT_developed_job_registry.csv")
    if len(jobs) != 36 or set(jobs.status) != {"COMPLETE"}:
        raise RuntimeError("base panel is not 36/36 terminal")
    rows: list[dict] = []
    for j in jobs.itertuples():
        out = Path(j.result_path)
        data = terminal_summary(out)
        if data is None:
            raise RuntimeError(f"invalid base terminal: {out}")
        dev = data["developed_interval"]
        first = data["event_measurements"][0]
        status = "PHYSICAL_TARGET_REACHED" if data["stable_growth_provisional"] else "DEVELOPED_UNSTABLE"
        rows.append({
            "campaign_stage": "DEVELOPED_N80_BASE_PANEL", "parameter_option": j.parameter_option,
            "deltaK_MPa_sqrt_m": float(j.deltaK_MPa_sqrt_m), "R": float(j.R),
            "n_bins": int(j.n_bins), "seed": int(j.seed), "status": status,
            "target_reached": bool(data["target_reached"]), "numerically_valid": True,
            "stable_growth": bool(data["stable_growth_provisional"]),
            "event_count": int(data["event_count"]),
            "cycle_to_first_event": float(first["cycles_post"]),
            "final_extension_um": float(data["final_projected_extension_um"]),
            "developed_interval_count": int(dev["event_count"]),
            "early_developed_da_dN": data["stability_early_interval"].get("da_dN"),
            "late_developed_da_dN": data["stability_late_interval"].get("da_dN"),
            "stationarity_ratio": data.get("late_to_early_rate_ratio"),
            "developed_da_dN_m_per_cycle": dev.get("da_dN") if data["stable_growth_provisional"] else None,
            "censor_or_failure_reason": data.get("censor_or_failure_reason"),
            "result_path": str(out.resolve()),
        })
    points = pd.DataFrame(rows)
    points.to_csv(root / "A_native_plus_8PT_developed_fatigue_points.csv", index=False)
    stable_counts = points.groupby("parameter_option").stable_growth.sum()
    adaptive = [str(x) for x in stable_counts[stable_counts < 3].index]
    decision = {
        "schema": "A8PT_conditional_followup_decision_v1",
        "base_terminal_count": len(points), "base_stable_count": int(points.stable_growth.sum()),
        "adaptive_max_additions_per_variant": 4, "adaptive_required_variants": adaptive,
    }
    if adaptive:
        raise RuntimeError("bounded adaptive refinement is required; controller has not admitted unstable descriptors")

    native = points[points.parameter_option == NATIVE].set_index("deltaK_MPa_sqrt_m")
    effects = []
    for option, group in points.groupby("parameter_option"):
        if option == NATIVE:
            continue
        g = group.set_index("deltaK_MPa_sqrt_m")
        common = sorted(set(g.index) & set(native.index))
        ratios = [float(g.loc[k, "developed_da_dN_m_per_cycle"] / native.loc[k, "developed_da_dN_m_per_cycle"]) for k in common]
        effects.append({"parameter_option": option, "max_abs_log10_rate_ratio": max(abs(math.log10(x)) for x in ratios),
                        "max_rate_ratio": max(ratios), "min_rate_ratio": min(ratios)})
    effect_df = pd.DataFrame(effects)
    fits = _fits(points)
    native_m = float(fits.loc[fits.parameter_option == NATIVE, "m"].iloc[0])
    max_dm = float((fits.m - native_m).abs().max())
    max_log = float(effect_df.max_abs_log10_rate_ratio.max())
    meaningful = max_log >= MEANINGFUL_RATE_LOG10 or max_dm >= MEANINGFUL_M_SHIFT
    decision.update({
        "max_abs_log10_rate_ratio": max_log, "max_relative_rate_difference": 10**max_log - 1,
        "max_abs_global_m_shift": max_dm,
        "meaningful_gate": {"abs_log10_rate_ratio": MEANINGFUL_RATE_LOG10, "abs_global_m_shift": MEANINGFUL_M_SHIFT},
        "meaningful_PT_fatigue_divergence": meaningful,
        "n128_required": meaningful,
        "n128_classification": "NOT_TRIGGERED_NO_MEANINGFUL_N80_FATIGUE_DIVERGENCE" if not meaningful else "REQUIRED",
        "multiseed_required": True,
        "multiseed_reason": "independent-seed verification of the key PT-insensitivity conclusion",
        "multiseed_variants": [NATIVE, RETENTION_DONOR, RETURN_DONOR],
    })
    effect_df.to_csv(root / "A_native_plus_8PT_n80_pairwise_effects.csv", index=False)
    atomic_json(root / "A_native_plus_8PT_conditional_followup_decision.json", decision)
    return decision


def _fits(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for option, g in points[(points.stable_growth) & points.developed_da_dN_m_per_cycle.notna()].groupby("parameter_option"):
        g = g.sort_values("deltaK_MPa_sqrt_m")
        x = np.log(g.deltaK_MPa_sqrt_m.to_numpy(float)); y = np.log(g.developed_da_dN_m_per_cycle.to_numpy(float))
        m, b = np.polyfit(x, y, 1); pred = m*x+b
        rows.append({"parameter_option": option, "m": m, "C": math.exp(b),
                     "R2": 1-float(np.sum((y-pred)**2))/max(float(np.sum((y-y.mean())**2)), 1e-300),
                     "admissible_points": len(g), "rate_span": float(np.exp(y.max()-y.min()))})
    return pd.DataFrame(rows)


def make_followup_jobs(root: Path, decision: dict) -> list[Job]:
    jobs: list[Job] = []
    for option in decision["multiseed_variants"]:
        for dk in BASE_LOADS:
            jid = f"multiseed__{option}__DK{dk:g}__seed1001723"
            jobs.append(Job(jid, "CONDITIONAL_MULTISEED", option, dk, .1, 80, 1001723, "normal",
                            str((root / "multiseed" / "n80" / option / f"DK_{dk:g}" / "seed_1001723").resolve())))
    parity = (
        (NATIVE, 13.5, .1, "pos"),
        (RETENTION_DONOR, 13.5, .1, "pos"),
        (RETURN_DONOR, 35.1, -.95, "neg"),
    )
    for option, dk, R, condition in parity:
        for mode in ("explicit", "normal"):
            jid = f"parity__{condition}__{option}__{mode}"
            jobs.append(Job(jid, "TRUE_ACCELERATOR_PARITY", option, dk, R, 80, 1720, mode,
                            str((root / "true_accelerator_parity" / condition / option / mode).resolve())))
    return jobs


def environment(job: Job, root: Path, head: str) -> dict[str, str]:
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": PYTHON, "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex", "EXPECTED_BRANCH": BRANCH,
        "EXPECTED_HEAD": head, "FAMILY_JSON": FAMILY,
        "V10230_ENTRY_MODULE": "arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK",
        "V10230_CANDIDATE_REGISTRY": str((root / "A_native_plus_8PT_registry.csv").resolve()),
        "V10230_CANDIDATE_SELECTION": str((root / "A_native_plus_8PT_selection.json").resolve()),
        "PARAMETER_OPTION": job.option, "TARGET_DELTAK": f"{job.delta_k:g}", "R_RATIO": f"{job.R:g}",
        "TARGET_FRACTION": job.stage.lower(), "RUN_LABEL": job.job_id, "TARGET_EXT_UM": "100",
        "CYCLES_MAX": "1000000000000", "HAZARD_SEED": str(job.seed), "MAX_WALL_SECONDS": "43200",
        "OUTROOT": job.output,
    })
    if job.mode == "explicit":
        env["V10230_HIGH_CYCLE_EXPLICIT_ONLY"] = "1"
    return env


def run_job(job: Job, root: Path, head: str) -> dict:
    out = Path(job.output)
    if out.exists():
        data = terminal_summary(out)
        if data:
            old = json.loads((out / "controller_job_contract.json").read_text()) if (out / "controller_job_contract.json").is_file() else {}
            if old.get("job_id") == job.job_id and old.get("solver_head") == head:
                return asdict(job) | {"status": classify(out, 0), "exit_code": 0, "reused_terminal": True}
        quarantine = root / "quarantine" / f"{job.job_id}__{int(time.time())}"
        quarantine.parent.mkdir(parents=True, exist_ok=True)
        os.replace(out, quarantine)
    out.parent.mkdir(parents=True, exist_ok=True)
    logdir = root / "controller_logs"; logdir.mkdir(exist_ok=True)
    contract = asdict(job) | {"solver_head": head, "fresh_physical_trajectory": True, "resume": False}
    start = time.time()
    with (logdir / f"{job.job_id}.log").open("w") as log:
        proc = subprocess.run(["bash", "scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"],
                              env=environment(job, root, head), stdout=log, stderr=subprocess.STDOUT)
    if out.exists():
        atomic_json(out / "controller_job_contract.json", contract)
    return asdict(job) | {"status": classify(out, proc.returncode), "exit_code": proc.returncode,
                          "wall_seconds": time.time()-start, "reused_terminal": False}


def run_followups(root: Path, jobs: list[Job], workers: int, head: str) -> None:
    path = root / "A_native_plus_8PT_followup_job_registry.csv"
    prior: dict[str, dict] = {}
    if path.is_file():
        prior = {str(x["job_id"]): x for x in pd.read_csv(path).to_dict("records")}
    rows = []
    for j in jobs:
        old = prior.get(j.job_id)
        if old and old.get("status") == "PHYSICAL_TARGET_REACHED" and terminal_summary(Path(j.output)):
            rows.append(old)
        else:
            rows.append(asdict(j) | {"status": "PENDING", "exit_code": None, "wall_seconds": None})
    atomic_csv(path, rows)
    pending = [j for j in jobs if next(r for r in rows if r["job_id"] == j.job_id)["status"] != "PHYSICAL_TARGET_REACHED"]
    byid = {str(r["job_id"]): r for r in rows}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_job, j, root, head): j for j in pending}
        for future in as_completed(futures):
            row = future.result(); byid[row["job_id"]] = row; atomic_csv(path, list(byid.values()))
            if row["status"] != "PHYSICAL_TARGET_REACHED":
                raise RuntimeError(f"follow-up failed closed: {row}")


def state(root: Path, phase: str, **extra: object) -> None:
    path = root / "A_native_plus_8PT_study_controller_state.json"
    old = json.loads(path.read_text()) if path.is_file() else {"history": []}
    history = old.get("history", [])
    if not history or history[-1].get("phase") != phase:
        history.append({"phase": phase, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    atomic_json(path, {"schema": "A8PT_persistent_study_controller_v1", "phase": phase,
                       "history": history, **extra})


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--root", type=Path, default=Path("runs/A_native_plus_8PT_fatigue_v1")); ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args(); root = args.root.resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    branch = subprocess.check_output(["git", "branch", "--show-current"], text=True).strip()
    if branch != BRANCH or subprocess.check_output(["git", "status", "--short"], text=True).strip():
        raise SystemExit("controller requires the clean mechanism-study branch")
    state(root, "DEVELOPED_N80_BASE_PANEL", solver_head=head)
    decision = base_analysis(root)
    state(root, "DEVELOPED_N80_ANALYSIS", decision=decision, solver_head=head)
    state(root, "ADAPTIVE_LOAD_REFINEMENT", required=False, bounded_max_per_variant=4, solver_head=head)
    state(root, "PT_DIVERGENCE_CLASSIFICATION", decision=decision, solver_head=head)
    state(root, "CONDITIONAL_N128_SELECTION", required=decision["n128_required"], solver_head=head)
    if decision["n128_required"]:
        raise RuntimeError("n128 was triggered; fail closed pending mechanistically distinct selection")
    state(root, "N128_RUNS", required=False, classification=decision["n128_classification"], solver_head=head)
    jobs = make_followup_jobs(root, decision)
    state(root, "CONDITIONAL_MULTISEED", required=True, job_count=sum(j.stage == "CONDITIONAL_MULTISEED" for j in jobs), solver_head=head)
    run_followups(root, jobs, args.workers, head)
    state(root, "TRUE_ACCELERATOR_PARITY", job_count=sum(j.stage == "TRUE_ACCELERATOR_PARITY" for j in jobs), solver_head=head)
    subprocess.run([PYTHON, "scripts/analyze_v10_2_30_A_native_plus_8PT_final.py", "--root", str(root)], check=True)
    state(root, "FINAL_ANALYSIS", solver_head=head)
    subprocess.run([PYTHON, "scripts/verify_v10_2_30_A_native_plus_8PT_study.py", "--root", str(root)], check=True)
    state(root, "FINAL_VERIFIER", result="PASS", solver_head=head)
    # Re-run after the terminal state is durable so the verifier checks COMPLETE.
    state(root, "COMPLETE", result="PASS", solver_head=head)
    subprocess.run([PYTHON, "scripts/verify_v10_2_30_A_native_plus_8PT_study.py", "--root", str(root)], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
