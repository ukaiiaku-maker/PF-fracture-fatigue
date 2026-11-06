"""Resumable bounded search for the declared one-condition Ast capability test."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import time

import numpy as np

from .single_condition import (Evaluator, SEED_PATHS, BOUNDS, KB_EV_PER_K,
                               diverse, from_isothermal, isothermal_coordinates,
                               rank, seed_material)
from .objective import physically_admissible

SOURCE = "/private/tmp/ast2018_native_source"
DATA = "runs/ast2018_shape_fit_v5/production/input/Ast2018_joint_fit_data"
RUN = Path("runs/ast2018_single_21c_rate0p5_v1")
_EVALUATORS = {}


def worker(branch, x):
    if branch not in _EVALUATORS:
        _EVALUATORS[branch] = Evaluator(SOURCE, DATA, branch)
    return _EVALUATORS[branch].run(x)


def bounds(ev):
    c = ev.completion
    dt = 294.15 - ev.material.emission.Tref_K
    shifts = (c["opening_gT"] * dt, c["opening_sT"] * dt,
              c["emission_gT"] * dt, c["emission_sT"] * dt,
              -c["peierls_entropy"] * KB_EV_PER_K * dt,
              -c["taylor_entropy"] * KB_EV_PER_K * dt)
    out = np.array([(lo+s, hi+s) for (lo, hi), s in zip(BOUNDS, shifts)])
    if ev.branch == "floor":
        sf = ev.material.cleavage.Sf_over_kB
        shift = -sf * KB_EV_PER_K * dt
        out = np.vstack((out, (max(0., shift), 5.+shift)))
    return out


def admissible(ev, x):
    try:
        if np.any(x < bounds(ev)[:, 0]) or np.any(x > bounds(ev)[:, 1]):
            return False
        return physically_admissible(from_isothermal(ev.material, x, ev.completion))
    except ValueError:
        return False


def load_rows():
    path = RUN / "evaluations.jsonl"
    return [json.loads(l) for l in path.open()] if path.exists() else []


def budget_remaining(limit, completed_keys, submitted_keys):
    """Reserve the call cap at submission, including interrupted jobs."""
    unresolved = set(submitted_keys) - set(completed_keys)
    if unresolved:
        raise RuntimeError(f"{len(unresolved)} submitted evaluations lack records; refusing restart")
    return max(0, limit-len(completed_keys))


def qualify(row, scale):
    m = row.get("metrics")
    if not m or row.get("status", "").startswith("numerical"):
        return False
    return (m["shortfall_nm"] <= 50 and m["raw_rms"] <= 2*scale
            and abs(m["level"]) <= 2*scale
            and abs(m["rise_error"]) <= .25*1040.441
            and sum(abs(v) <= 4*scale for v in m["window_errors"]) >= 3)


def main():
    cfg = json.loads((RUN / "qualification_config.json").read_text())
    started = cfg["started_unix_s"]
    scale = cfg["objective"]["J_scale_N_per_m"]
    rng = np.random.default_rng(20182510)
    evaluators = {branch: Evaluator(SOURCE, DATA, branch) for branch in ("native", "floor")}
    bases = {branch: [isothermal_coordinates(seed_material(ev.base, name))
                      for name in SEED_PATHS if (name.startswith("native") == (branch == "native"))]
             for branch, ev in evaluators.items()}
    existing = load_rows()
    known = {r["key"] for r in existing}
    audit_path=RUN/"interrupted_batch_reconstruction.json"
    if audit_path.exists() and json.loads(audit_path.read_text()).get("unlogged_submitted_total",0):
        raise RuntimeError("interrupted search has unlogged submitted work; budget exhausted")
    journal=RUN/"submission_journal.jsonl"
    submitted=[json.loads(l)["key"] for l in journal.open()] if journal.exists() else []
    remaining=budget_remaining(512,[r["key"] for r in existing],submitted)
    generation = 0
    with ProcessPoolExecutor(max_workers=4) as pool, (RUN / "evaluations.jsonl").open("a") as output, journal.open("a") as submissions:
        while time.time() - started < 2700 and remaining > 0:
            proposals = []
            for branch, ev in evaluators.items():
                old = [r for r in existing if r.get("branch") == branch and r.get("metrics")]
                parents = [np.asarray(r["x"]) for r in diverse(old, 12)] or bases[branch]
                width = np.diff(bounds(ev), axis=1)[:, 0]
                for _ in range(32):
                    if rng.random() < .2:
                        parent = bases[branch][int(rng.integers(len(bases[branch])))]
                    else:
                        parent = parents[int(rng.integers(len(parents)))]
                    x = parent.copy()
                    dims = rng.choice(len(x), size=int(rng.integers(1, min(4, len(x)+1))), replace=False)
                    sigma = .16 if generation < 2 else .07 if generation < 5 else .035
                    x[dims] += rng.normal(size=len(dims)) * sigma * width[dims]
                    x = np.clip(x, bounds(ev)[:, 0], bounds(ev)[:, 1])
                    if not admissible(ev, x):
                        continue
                    key = branch + ":" + ",".join(f"{v:.12g}" for v in x)
                    if key not in known:
                        known.add(key); proposals.append((branch, x.tolist()))
            proposals = proposals[:min(remaining, 64)]
            if not proposals:
                break
            for branch,x in proposals:
                key=branch+":"+",".join(f"{v:.12g}" for v in x)
                submissions.write(json.dumps({"key":key,"generation":generation,
                                               "submitted_unix_s":time.time()})+"\n")
                submitted.append(key)
            submissions.flush()
            futures = {pool.submit(worker, branch, x): (branch, x) for branch, x in proposals}
            for future in as_completed(futures):
                row = future.result(); row["generation"] = generation
                row["completed_unix_s"] = time.time()
                output.write(json.dumps(row, default=float) + "\n"); output.flush()
                existing.append(row)
                if qualify(row, scale):
                    print("ANALYTICAL_QUALIFIED", row["key"], flush=True)
            good = [r for r in existing if r.get("metrics")]
            best = min(good, key=rank)
            print("generation", generation, "evaluated", len(existing),
                  "elapsed_s", round(time.time()-started), "best", best["key"],
                  "metrics", best["metrics"], flush=True)
            generation += 1
            remaining=budget_remaining(512,[r["key"] for r in existing],submitted)
            if any(qualify(r, scale) for r in good):
                break
    print("SEARCH_DONE", len(existing), round(time.time()-started), flush=True)


if __name__ == "__main__":
    main()
