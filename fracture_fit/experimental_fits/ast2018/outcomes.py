"""Qualitative no-cleavage endpoint checks without invented subgroup likelihoods."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .data import AstData


def _history(path):
    with path.open(newline="",encoding="utf-8") as stream:return list(csv.DictReader(stream))


def check_endpoints(data_dir:Path,run_dir:Path,scenario:str):
    data=AstData.load(data_dir)
    results=[]
    for record in data.no_cleavage:
        observed_max=float(record["K_preferred_MPa_sqrt_m"])
        condition=record["condition_id"]
        for level in ("reduced","spatial"):
            file=(run_dir/f"confirmation_{scenario.lower()}_endpoint_validation/"
                  f"{level}/{condition}/ast_observer.csv")
            rows=_history(file)
            reached=next((i for i,r in enumerate(rows)
                if float(r["K_nominal_MPa_sqrt_m"])>=observed_max),None)
            failure=next((i for i,r in enumerate(rows)
                if r["physical_failure"]=="True"),None)
            if failure is not None and (reached is None or failure<reached):
                verdict="ligament_failure_before_observed_nominal_K"
            elif reached is not None and (failure is None or reached<failure):
                verdict="survived_through_observed_nominal_K"
            else:verdict="model_history_does_not_cover_endpoint"
            results.append({"observation_id":record["observation_id"],
                "condition_id":condition,"level":level,
                "observed_max_nominal_K_MPa_sqrt_m":observed_max,
                "check_status":verdict,
                "model_max_nominal_K_MPa_sqrt_m":max(
                    (float(r["K_nominal_MPa_sqrt_m"]) for r in rows),default=None),
                "validation_horizon":"observed endpoint initial-elastic displacement; independent rate-controlled trajectory",
                "interpretation":"qualitative outcome check; unknown subgroup count"})
    out=run_dir/f"no_cleavage_outcomes_{scenario.lower()}.json"
    out.write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    return results
