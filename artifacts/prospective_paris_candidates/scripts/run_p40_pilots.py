"""Launch the three frozen P40 pilot trajectories (Kmax = 13.5, 18, 21).

Enforces the physical contract: fresh virgin result paths only (a job whose
result path already exists is refused, never resumed), at most three workers,
clean worktree, exact expected branch/HEAD, and the frozen registry/selection
artifacts. Attempts are recorded; an interrupted attempt is preserved as
INTERRUPTED_NOT_SCIENCE and must be relaunched into a NEW virgin path.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parents[1]
PY = "/opt/homebrew/Caskroom/miniconda/base/envs/arrhenius-sharp-front-v10-codex/bin/python"
FAMILY = ("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_v10_2_21_persistent_sites_top1/"
          "runs/v10_2_28_kernel_cache/4fa015d77f1aadf05f77f550366f64cd611f537ae716bbd47870bf9e6fe2f873/family.json")
BRANCH = "codex/v10.2.30-prospective-paris-candidate-design"
MIN_FREE_GIB = 3.0
ATTEMPTS = OUT / "physical_attempt_registry.csv"
PROGRESS = ROOT / "runs" / "prospective_paris_p40_pilot_v1" / "attempt_progress.csv"


def _write_attempts(results, path) -> None:
    import csv
    if not results:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in results for k in row})
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, "") for k in fields})


def git(*a) -> str:
    return subprocess.check_output(["git", *a], cwd=ROOT, text=True).strip()


def free_gib() -> float:
    return shutil.disk_usage(ROOT).free / 2 ** 30


def next_virgin_path(base: Path) -> Path:
    """Return a never-used attempt directory for this frozen job key.

    The low-level launcher refuses to write into an existing OUTROOT, and the
    contract forbids resuming, so every attempt gets its own virgin directory
    and the launcher (not this controller) creates it.
    """
    for n in range(1, 100):
        cand = base.parent / f"{base.name}__attempt{n}"
        if not cand.exists():
            return cand
    raise SystemExit(f"too many attempts for {base}")


def run_one(job: dict, head: str) -> dict:
    rec = dict(job)
    if free_gib() < MIN_FREE_GIB:
        rec["status"] = "REFUSED_INSUFFICIENT_DISK"
        return rec
    out = next_virgin_path(Path(job["result_path"]))
    rec["attempt_path"] = str(out)
    # create only the PARENT; the launcher must create OUTROOT itself
    out.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTHON_BIN": PY, "CONDA_ENV": "arrhenius-sharp-front-v10-codex",
        "CONDA_DEFAULT_ENV": "arrhenius-sharp-front-v10-codex",
        "EXPECTED_BRANCH": BRANCH, "EXPECTED_HEAD": head, "FAMILY_JSON": FAMILY,
        "V10230_ENTRY_MODULE": "arrhenius_fracture.sharp_front_v10_2_30_candidate_fixed_deltaK",
        "V10230_CANDIDATE_REGISTRY": str((OUT / "p40_candidate_registry.csv").resolve()),
        "V10230_CANDIDATE_SELECTION": str((OUT / "p40_candidate_selection.json").resolve()),
        "TARGET_FRACTION": "p40_pilot", "TARGET_EXT_UM": "100",
        "CYCLES_MAX": "1000000000000", "HAZARD_SEED": str(job["seed"]),
        "MAX_WALL_SECONDS": "43200",
        "PARAMETER_OPTION": job["parameter_option"],
        "TARGET_DELTAK": str(job["deltaK_MPa_sqrt_m"]), "R_RATIO": str(job["R"]),
        "RUN_LABEL": job["composite_id"], "OUTROOT": str(out.resolve()),
    })
    t0 = time.time()
    proc = subprocess.run(["bash", "scripts/run_v10_2_30_weakt_high_cycle_1e12.sh"],
                          cwd=ROOT, env=env, capture_output=True, text=True)
    rec["exit_code"] = proc.returncode
    rec["wall_seconds"] = time.time() - t0
    # the launcher creates OUTROOT; if it failed before that, keep the logs beside it
    log_dir = out if out.is_dir() else out.parent
    prefix = "" if out.is_dir() else f"{out.name}__"
    (log_dir / f"{prefix}launch_stdout.log").write_text(proc.stdout[-200000:])
    (log_dir / f"{prefix}launch_stderr.log").write_text(proc.stderr[-200000:])
    summary = out / "developed_fatigue_growth_summary.json"
    checkpoint = out / "high_cycle_live_checkpoint.json"
    if proc.returncode == 0 and summary.is_file():
        d = json.loads(summary.read_text())
        if d.get("target_reached"):
            rec["status"] = "COMPLETE"
        elif d.get("status") == "cycle_censor":
            rec["status"] = "PHYSICAL_CENSOR"
        else:
            rec["status"] = "COMPLETE_PARTIAL_GROWTH"
        rec["developed_da_dN"] = (d.get("developed_interval") or {}).get("da_dN")
        rec["event_count"] = d.get("event_count")
        rec["final_extension_um"] = d.get("final_projected_extension_um")
    elif checkpoint.is_file():
        rec["status"] = "WALL_LIMIT_NONTERMINAL"
    elif proc.returncode == 2:
        rec["status"] = "LAUNCH_PREFLIGHT_FAILURE"
    else:
        rec["status"] = "NUMERICAL_FAILURE"
    return rec


def main() -> None:
    freeze = json.loads((OUT / "prediction_freeze_manifest.json").read_text())
    head = git("rev-parse", "HEAD")
    frozen = freeze["frozen_at_head"]
    # HEAD may legitimately advance past the freeze commit (e.g. to add this
    # launcher). What must NOT change is any frozen prediction input, so the
    # invariant enforced here is: the freeze commit is an ancestor of HEAD, and
    # every frozen launcher artifact still hashes to its recorded value.
    if subprocess.run(["git", "merge-base", "--is-ancestor", frozen, head],
                      cwd=ROOT).returncode != 0:
        raise SystemExit(f"freeze commit {frozen} is not an ancestor of HEAD {head}")
    import hashlib
    for key, path_key in (("registry_csv_sha256", "registry_csv"),
                          ("selection_json_sha256", "selection_json")):
        p = Path(freeze["launcher_artifacts"][path_key])
        actual = hashlib.sha256(p.read_bytes()).hexdigest()
        if actual != freeze["launcher_artifacts"][key]:
            raise SystemExit(f"frozen artifact changed after freeze: {p} ({actual})")
    if git("branch", "--show-current") != BRANCH:
        raise SystemExit("wrong branch")
    if git("status", "--porcelain"):
        raise SystemExit("worktree not clean")
    if free_gib() < MIN_FREE_GIB:
        raise SystemExit(f"insufficient disk: {free_gib():.1f} GiB")
    jobs = freeze["physical_jobs"]
    print(json.dumps(dict(head=head, jobs=len(jobs), free_gib=round(free_gib(), 1))), flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = {ex.submit(run_one, j, head): j for j in jobs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            print(json.dumps({k: r.get(k) for k in
                              ("composite_id", "status", "exit_code", "wall_seconds",
                               "developed_da_dN", "event_count", "final_extension_um")},
                             default=str), flush=True)
            # progress goes to the gitignored runs/ tree: writing into the
            # tracked worktree while jobs are live would dirty it and trip the
            # launcher's clean-tree gate for any job that starts afterwards.
            _write_attempts(results, PROGRESS)
    _write_attempts(results, ATTEMPTS)
    print(json.dumps(dict(completed=len(results),
                          statuses={r["composite_id"]: r["status"] for r in results}),
                     indent=2, default=str))


if __name__ == "__main__":
    main()
