"""Re-evaluate finalists with smaller load steps and tighter first passage."""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np

from .config import AstAssumptions
from .data import AstData
from .native import historical_start, load_native
from .objective import WorkingObjective


def _evaluate(spec):
    source_dir, data_dir, scenario, x, assumptions_dict, step, bisections = spec
    start = perf_counter()
    source = Path(source_dir)
    mods, _ = load_native(source)
    base, physics, _ = historical_start(source, mods[2], mods[3].CommonPhysics)
    assumptions = replace(AstAssumptions(**assumptions_dict),
                          nominal_K_step_MPa_sqrt_m=step,
                          localization_bisections=bisections)
    objective = WorkingObjective(AstData.load(Path(data_dir)), scenario,
                                 assumptions, mods[0], base, physics)
    residual, detail = objective.evaluate(np.asarray(x, dtype=float), detail=True)
    return {"nominal_K_step_MPa_sqrt_m": step,
            "localization_bisections": bisections,
            "cost": float(.5 * np.sum(residual**2)),
            "residual": residual.tolist(),
            "condition_detail": detail,
            "forward_objective_calls": 1,
            "forward_case_calls": objective.case_calls,
            "runtime_s": perf_counter() - start}


def audit(source: Path, run_dir: Path, scenario: str, vector: Path,
          *, workers: int = 2):
    if workers < 1 or workers > 2:
        raise ValueError("convergence audit permits one or two workers")
    config = json.loads((run_dir / f"config_{scenario.lower()}.json").read_text())
    x = json.loads(vector.read_text())["x"]
    base = (str(source.resolve()),
            str((run_dir / "input/Ast2018_joint_fit_data").resolve()),
            scenario, x, config["ast_assumptions"])
    plans = [(0.05, 12), (0.025, 14), (0.0125, 16)]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_evaluate, base + plan): plan for plan in plans}
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(f"{scenario} convergence {len(results)}/{len(plans)} "
                  f"step={row['nominal_K_step_MPa_sqrt_m']} "
                  f"cost={row['cost']:.6g}", flush=True)
    results.sort(key=lambda row: -row["nominal_K_step_MPa_sqrt_m"])
    coarse, middle, fine = results
    differences = {
        "coarse_to_middle_cost": middle["cost"] - coarse["cost"],
        "middle_to_fine_cost": fine["cost"] - middle["cost"],
        "coarse_to_fine_max_abs_residual": float(np.max(np.abs(
            np.asarray(fine["residual"]) - np.asarray(coarse["residual"])))),
        "middle_to_fine_max_abs_residual": float(np.max(np.abs(
            np.asarray(fine["residual"]) - np.asarray(middle["residual"])))),
    }
    output = {"scenario": scenario, "vector": x,
              "same_unit_action_event_sequence": True,
              "results": results, "differences": differences,
              "interpretation": "direct stateful 0-D recalculation; differences are numerical, not fitted error"}
    path = run_dir / f"convergence_{scenario.lower()}.json"
    path.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    return output
