"""Aggregate actual objective, forward-case and confirmation runtimes."""
from __future__ import annotations

import json
from pathlib import Path


def _jsonl(path):
    if not path.exists():return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def account(run_dir:Path):
    stages={}
    for scenario in ("table_primary","figure5_alternative"):
        for stage in ("search","refine","spatial_local"):
            path=run_dir/f"{stage}_{scenario}.jsonl"
            rows=_jsonl(path)
            if not rows:continue
            key=f"{stage}_{scenario}"
            stages[key]={"completed_objective_calls":len(rows),
                         "failed_calls":sum(r.get("error") is not None for r in rows),
                         "forward_case_calls":sum(int(r.get("case_calls",0)) for r in rows),
                         "sum_process_runtime_s":sum(float(r.get("runtime_s",0)) for r in rows),
                         "first_started_unix_s":min(float(r["completed_unix_s"])-
                            float(r.get("runtime_s",0)) for r in rows),
                         "last_completed_unix_s":max(float(r["completed_unix_s"])
                            for r in rows),
                         "completion_span_including_pauses_s":
                            max(float(r["completed_unix_s"]) for r in rows)-
                            min(float(r["completed_unix_s"])-float(r.get("runtime_s",0))
                                for r in rows),
                         "artifact":str(path)}
        for suffix in ("","_priority11","_endpoint_validation"):
            path=run_dir/f"confirmation_{scenario}{suffix}/summary.json"
            if not path.exists():continue
            rows=json.loads(path.read_text())
            key=f"confirmation_{scenario}{suffix}"
            stages[key]={"forward_case_calls":len(rows),
                         "sum_process_runtime_s":sum(float(r["runtime_s"]) for r in rows),
                         "numerical_failures":sum(r["status"].startswith("numerical") for r in rows),
                         "artifact":str(path)}
        path=run_dir/f"identifiability_{scenario}/sensitivity.json"
        if path.exists():
            row=json.loads(path.read_text())
            stages[f"sensitivity_{scenario}"]={"objective_calls":row["forward_objective_calls"],
                "forward_case_calls":row["case_calls"],
                "sum_process_runtime_s":row.get("sum_process_runtime_s",0),
                "artifact":str(path)}
        path=run_dir/f"identifiability_{scenario}/profiles.json"
        if path.exists():
            rows=json.loads(path.read_text())
            stages[f"profiles_{scenario}"]={"profile_points":len(rows),
                "objective_calls":sum(r["forward_objective_calls"] for r in rows),
                "forward_case_calls":sum(r.get("forward_case_calls",0) for r in rows),
                "sum_process_runtime_s":sum(r.get("runtime_s",0) for r in rows),
                "artifact":str(path)}
        path=run_dir/f"convergence_{scenario}.json"
        if path.exists():
            rows=json.loads(path.read_text())["results"]
            stages[f"convergence_{scenario}"]={
                "objective_calls":sum(r["forward_objective_calls"] for r in rows),
                "forward_case_calls":sum(r["forward_case_calls"] for r in rows),
                "sum_process_runtime_s":sum(r["runtime_s"] for r in rows),
                "artifact":str(path)}
        path=run_dir/f"holdouts_{scenario}.json"
        if path.exists():
            rows=json.loads(path.read_text())
            tasks=[trial for row in rows for trial in row["training_trials"]]
            tasks += [row["source_held_out"] for row in rows]
            tasks += [row["local_refit_held_out"] for row in rows
                      if row["local_training_best_x"] != row["source_fit_x"]]
            stages[f"holdouts_{scenario}"]={
                "objective_calls":sum(r["forward_objective_calls"] for r in tasks),
                "forward_case_calls":sum(r["forward_case_calls"] for r in tasks),
                "sum_process_runtime_s":sum(r["runtime_s"] for r in tasks),
                "artifact":str(path)}
        path=run_dir/f"sensitivity_{scenario}.json"
        if path.exists():
            rows=json.loads(path.read_text())
            stages[f"assumption_sensitivity_{scenario}"]={
                "objective_calls":sum(r["kind"]!="working_weight" for r in rows),
                "forward_case_calls":sum(r["forward_case_calls"] for r in rows),
                "sum_process_runtime_s":sum(r["runtime_s"] for r in rows),
                "artifact":str(path)}
        for level in ("reduced","spatial"):
            path=run_dir/f"ensemble_{scenario}_{level}/runs.json"
            if path.exists():
                rows=json.loads(path.read_text())
                stages[f"ensemble_{scenario}_{level}"]={
                    "forward_case_calls":len(rows),
                    "sum_process_runtime_s":sum(r["runtime_s"] for r in rows),
                    "numerical_failures":sum(r["status"].startswith("numerical") for r in rows),
                    "artifact":str(path)}
    result={"stages":stages,
            "total_completed_search_refine_case_calls":sum(
                v.get("forward_case_calls",0) for k,v in stages.items()
                if k.startswith(("search_","refine_","spatial_local_"))),
            "total_confirmation_case_calls":sum(v.get("forward_case_calls",0)
                for k,v in stages.items() if k.startswith("confirmation_")),
            "interpretation":"sum_process_runtime is CPU/wall time accumulated across workers, not elapsed campaign wall time"}
    output=run_dir/"runtime_accounting.json"
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result
