"""Three regime-spanning conditional, local leave-one-condition-out checks."""
from __future__ import annotations

import json
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import AstAssumptions
from .campaign import _append_jsonl, _load_existing
from .data import AstData
from .native import historical_start, load_native
from .objective import WorkingObjective


HOLDOUTS = ("Tm90C_rate0p2", "Tp100C_rate1p6", "Tp500C_rate1p6")
ACTIVE_LOCAL_INDICES = (0, 3, 6)


def _make_objective(source, data_dir, scenario, assumptions_dict, held):
    mods, _ = load_native(Path(source))
    base, physics, _ = historical_start(Path(source), mods[2], mods[3].CommonPhysics)
    objective = WorkingObjective(AstData.load(Path(data_dir)), scenario,
                                 AstAssumptions(**assumptions_dict),
                                 mods[0], base, physics)
    if held not in objective.families:
        raise RuntimeError(f"holdout {held} has no supported fit family")
    return objective


def _evaluate(spec):
    source, data_dir, scenario, assumptions_dict, held, x, subset = spec
    start = perf_counter()
    objective = _make_objective(source, data_dir, scenario, assumptions_dict, held)
    if subset == "training":
        objective.condition_ids = tuple(cid for cid in objective.condition_ids if cid != held)
    elif subset == "held_out":
        objective.condition_ids = (held,)
    else:
        raise ValueError(subset)
    objective.n_residuals = sum(len(f.values) for cid in objective.condition_ids
                                for f in objective.families.get(cid, ()))
    residual, detail = objective.evaluate(np.asarray(x, dtype=float), detail=True)
    return {"x": x, "subset": subset, "cost": float(.5*np.sum(residual**2)),
            "detail": detail, "runtime_s": perf_counter()-start,
            "forward_objective_calls": 1,
            "forward_case_calls": objective.case_calls}


def analyze(source: Path, run_dir: Path, scenario: str, vector: Path,
            *, workers: int = 7, step: float = .05):
    if workers < 1 or workers > 8:
        raise ValueError("workers must be 1..8")
    cfg = json.loads((run_dir / f"config_{scenario.lower()}.json").read_text())
    x0 = np.asarray(json.loads(vector.read_text())["x"], dtype=float)
    common = (str(source.resolve()),
              str((run_dir / "input/Ast2018_joint_fit_data").resolve()),
              scenario, cfg["ast_assumptions"])
    checkpoint=run_dir/f"holdout_calls_{scenario.lower()}.jsonl"
    completed=_load_existing(checkpoint)
    def key_for(held, x, subset):
        payload={"condition_id":held,"x":[float(v).hex() for v in x],
                 "subset":subset,"scenario":scenario}
        return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
    def score_one(held,x,subset):
        key=key_for(held,x,subset)
        if key not in completed:
            row=_evaluate(common+(held,x,subset))
            row.update({"key":key,"condition_id":held,"scenario":scenario})
            _append_jsonl(checkpoint,row)
            completed[key]=row
        return completed[key]
    output = []
    for held in HOLDOUTS:
        proposals = [x0.tolist()]
        for index in ACTIVE_LOCAL_INDICES:
            for direction in (-1, 1):
                trial = x0.copy()
                trial[index] = np.clip(trial[index] + direction*step, 0, 1)
                if trial.tolist() not in proposals:
                    proposals.append(trial.tolist())
        training = [completed[key_for(held,x,"training")] for x in proposals
                    if key_for(held,x,"training") in completed]
        pending=[x for x in proposals if key_for(held,x,"training") not in completed]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_evaluate, common+(held,x,"training")): x
                       for x in pending}
            for future in as_completed(futures):
                row=future.result()
                key=key_for(held,row["x"],"training")
                row.update({"key":key,"condition_id":held,"scenario":scenario})
                _append_jsonl(checkpoint,row)
                completed[key]=row
                training.append(row)
        winner = min(training, key=lambda row: row["cost"])
        full_test = score_one(held,x0.tolist(),"held_out")
        held_test = (score_one(held,winner["x"],"held_out")
                     if winner["x"] != x0.tolist() else full_test)
        output.append({"condition_id": held, "scenario": scenario,
                       "source_fit_x": x0.tolist(),
                       "local_training_best_x": winner["x"],
                       "source_training_cost": next(row["cost"] for row in training
                                                    if row["x"] == x0.tolist()),
                       "local_training_cost": winner["cost"],
                       "source_held_out": full_test,
                       "local_refit_held_out": held_test,
                       "training_trials": training,
                       "method": "one-step local conditional leave-one-condition-out; "
                                 "not a global refit or out-of-sample probability score"})
        print(f"{scenario} holdout {held} local train {winner['cost']:.6g} "
              f"held-out {held_test['cost']:.6g}", flush=True)
    path = run_dir / f"holdouts_{scenario.lower()}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output
