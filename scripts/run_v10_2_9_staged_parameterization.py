#!/usr/bin/env python3
"""Run v10.2.8 physics with v10.2.9 quality-diversity promotion.

The constitutive calculations are unchanged.  This wrapper replaces only the
promotion policy and exposes complete analytical response trajectories to that
policy so later-stage behavior is not discarded merely because several candidates
share nearly identical scalar objective values.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

from arrhenius_fracture.quality_diversity_v1029 import (
    MODEL_ID as SELECTION_MODEL_ID,
    QualityDiversityConfig,
    select_quality_diverse,
)
from scripts import run_v10_2_8_staged_parameterization as base

_ORIGINAL_ANALYTICAL_WORKER = base._worker_analytical
_SELECTION_AUDITS: dict[str, dict] = {}


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, str(default)))


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return bool(default)
    return raw.strip().lower() not in {"0", "false", "no", "off"}


def _selection_config(count: int) -> QualityDiversityConfig:
    return QualityDiversityConfig(
        count=int(count),
        quality_reserve_fraction=_env_float(
            "QD_QUALITY_RESERVE_FRACTION", 0.25
        ),
        quality_weight=_env_float("QD_QUALITY_WEIGHT", 0.35),
        parameter_distance_weight=_env_float("QD_PARAMETER_WEIGHT", 0.45),
        response_distance_weight=_env_float("QD_RESPONSE_WEIGHT", 0.55),
        pool_factor=_env_int("QD_POOL_FACTOR", 12),
        preserve_anchor_lineages=_env_bool(
            "QD_PRESERVE_ANCHOR_LINEAGES", True
        ),
    ).validate()


def _stage_from_pass_key(pass_key: str) -> str:
    mapping = {
        "analytical_pass": "analytical",
        "first_passage_pass": "first-passage",
        "rcurve_pass": "rcurve",
    }
    try:
        return mapping[str(pass_key)]
    except KeyError as exc:
        raise ValueError(f"unsupported promotion pass key {pass_key!r}") from exc


def _flatten_analytical_worker(payload):
    result = _ORIGINAL_ANALYTICAL_WORKER(payload)
    if not bool(result.get("ok", False)):
        return result
    row = result["row"]
    full = result.get("details", {})
    details = full.get("details", []) if isinstance(full, dict) else []
    for state in details:
        temperature = int(round(float(state["temperature_K"])))
        suffix = f"{temperature}K"
        row[f"analytical_K_cleave_{suffix}"] = state.get(
            "K_cleave_no_plastic_MPa_sqrt_m"
        )
        row[f"analytical_K_first_emission_{suffix}"] = state.get(
            "K_first_emission_MPa_sqrt_m"
        )
        row[f"analytical_emission_advantage_{suffix}"] = state.get(
            "emission_advantage_fraction"
        )
        row[f"analytical_Kshield_{suffix}"] = state.get(
            "linearized_source_bin_Kshield_MPa_sqrt_m"
        )
        row[f"analytical_retained_fraction_{suffix}"] = state.get(
            "mean_retained_fraction_indicator"
        )
        row[f"analytical_expected_activations_{suffix}"] = state.get(
            "expected_source_activations"
        )
        row[f"analytical_signed_line_{suffix}"] = state.get(
            "expected_signed_line_content"
        )
    return result


def _quality_diverse_select(
    rows,
    pass_key: str,
    objective_key: str,
    target_class: str,
    count: int,
):
    stage = _stage_from_pass_key(pass_key)
    selected, audit = select_quality_diverse(
        rows,
        pass_key=pass_key,
        objective_key=objective_key,
        target_class=target_class,
        stage=stage,
        config=_selection_config(count),
    )
    _SELECTION_AUDITS[f"{stage}:{target_class}"] = audit
    return selected


def _option_value(argv: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(argv):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(argv):
            return argv[index + 1]
    return None


def main() -> None:
    base._worker_analytical = _flatten_analytical_worker
    base._select = _quality_diverse_select
    argv = list(sys.argv[1:])
    out_value = _option_value(argv, "--out")
    base.main()
    if out_value is None:
        raise RuntimeError("base campaign completed without an --out path")
    out = Path(out_value)
    payload = {
        "schema": SELECTION_MODEL_ID,
        "constitutive_campaign_schema": base.MODEL_ID,
        "selection_only_change": True,
        "material_parameters_modified_by_selector": False,
        "selection_audits": _SELECTION_AUDITS,
    }
    (out / "quality_diversity_selection.json").write_text(
        json.dumps(payload, indent=2)
    )
    complete_path = out / "stage_complete.json"
    complete = json.loads(complete_path.read_text())
    complete.update(
        {
            "quality_diversity_selection": True,
            "selection_schema": SELECTION_MODEL_ID,
        }
    )
    complete_path.write_text(json.dumps(complete, indent=2))


if __name__ == "__main__":
    main()
