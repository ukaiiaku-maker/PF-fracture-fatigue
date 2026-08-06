#!/usr/bin/env python3
"""Build the v10.4.3 reuse-aware plastic-dominance production launcher."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


OLD_ENTRY = "arrhenius_fracture.sharp_front_v10_4_2_plastic_flow_audited"
MODEL_ENTRY = (
    "arrhenius_fracture."
    "sharp_front_v10_4_3_plastic_dominance_audited"
)
OLD_MODEL_SCHEMA = "v10.4.2_bulk_detailed_balance_plastic_flow_terminal"
MODEL_SCHEMA = "v10.4.3_bulk_detailed_balance_plastic_dominance_censor"


def _load_v1042_builder():
    path = Path(__file__).with_name("build_v10_4_2_reuse_aware_launcher.py")
    spec = importlib.util.spec_from_file_location("v1042_reuse_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v10.4.2 reuse builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    text = _load_v1042_builder().transform(source)

    entry_count = text.count(OLD_ENTRY)
    if entry_count < 1:
        raise RuntimeError("v10.4.2 production model entry is missing")
    text = text.replace(OLD_ENTRY, MODEL_ENTRY)

    schema_count = text.count(OLD_MODEL_SCHEMA)
    if schema_count < 1:
        raise RuntimeError("v10.4.2 native model schema is missing")
    text = text.replace(OLD_MODEL_SCHEMA, MODEL_SCHEMA)

    schema_replacements = {
        "v10.4.2_bulk_plastic_flow_orientation_rate_campaign_v1":
            "v10.4.3_plastic_dominance_orientation_rate_campaign_v1",
        "v10.4.2_bulk_plastic_flow_orientation_rate_case_contract_v1":
            "v10.4.3_plastic_dominance_orientation_rate_case_contract_v1",
        "v10.4.2_bulk_plastic_flow_orientation_rate_lock_v1":
            "v10.4.3_plastic_dominance_orientation_rate_lock_v1",
    }
    for old, new in schema_replacements.items():
        if old not in text:
            raise RuntimeError(f"v10.4.2 launcher token is missing: {old}")
        text = text.replace(old, new)

    scheduler_adapter = r"""
replace_scheduler_exact(
    '''RESTART_INCOMPLETE=${RESTART_INCOMPLETE:-1}''',
    '''RESTART_INCOMPLETE=${RESTART_INCOMPLETE:-1}
CASE_FILTER_OPTION=${CASE_FILTER_OPTION:-}
CASE_FILTER_TEMPERATURE=${CASE_FILTER_TEMPERATURE:-}''',
    label="v10.4.3 manifest-preserving execution filters",
)

replace_scheduler_exact(
    '''    while [[ ${#pids[@]} -ge $MAX_JOBS ]]; do
      wait_one
    done
    run_case "$option" "$T" "$case_seed" &
    pids+=("$!")
    labels+=("${option}:T${T}K:seed${case_seed}")
    temperature_index=$((temperature_index + 1))''',
    '''    if [[ -n "$CASE_FILTER_OPTION" && "$option" != "$CASE_FILTER_OPTION" ]]; then
      temperature_index=$((temperature_index + 1))
      continue
    fi
    if [[ -n "$CASE_FILTER_TEMPERATURE" && "$T" != "$CASE_FILTER_TEMPERATURE" ]]; then
      temperature_index=$((temperature_index + 1))
      continue
    fi

    while [[ ${#pids[@]} -ge $MAX_JOBS ]]; do
      wait_one
    done
    run_case "$option" "$T" "$case_seed" &
    pids+=("$!")
    labels+=("${option}:T${T}K:seed${case_seed}")
    temperature_index=$((temperature_index + 1))''',
    label="v10.4.3 filtered case scheduling with canonical seed indices",
)

replace_scheduler_exact(
    '''if find "$OUTROOT" \\( -type f -o -type l \\) -name COMPLETE -print -quit | grep -q .; then''',
    '''if [[ -z "$CASE_FILTER_OPTION" && -z "$CASE_FILTER_TEMPERATURE" ]] \\
  && find "$OUTROOT" \\( -type f -o -type l \\) -name COMPLETE -print -quit | grep -q .; then''',
    label="v10.4.3 skip campaign plotting during filtered smoke or pilot",
)

replace_scheduler_exact(
    '''OUTROOT="$OUTROOT" "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
manifest = json.loads((root / "v10_2_27_campaign_manifest.json").read_text())
with (root / "v10_2_27_case_seed_map.csv").open(newline="") as stream:
    cases = list(csv.DictReader(stream))''',
    '''OUTROOT="$OUTROOT" CASE_FILTER_OPTION="$CASE_FILTER_OPTION" \\
CASE_FILTER_TEMPERATURE="$CASE_FILTER_TEMPERATURE" "$PYTHON_BIN" - <<'PY'
import csv
import json
import os
from pathlib import Path

root = Path(os.environ["OUTROOT"]).resolve()
manifest = json.loads((root / "v10_2_27_campaign_manifest.json").read_text())
with (root / "v10_2_27_case_seed_map.csv").open(newline="") as stream:
    cases = list(csv.DictReader(stream))
filter_option = os.environ.get("CASE_FILTER_OPTION", "").strip()
filter_temperature = os.environ.get("CASE_FILTER_TEMPERATURE", "").strip()
if filter_option:
    cases = [case for case in cases if case["option"] == filter_option]
if filter_temperature:
    requested_temperature = float(filter_temperature)
    cases = [
        case for case in cases
        if float(case["temperature_K"]) == requested_temperature
    ]
if not cases:
    raise SystemExit(
        "ERROR: v10.4.3 execution filter selected zero canonical cases"
    )''',
    label="v10.4.3 filtered campaign acceptance",
)

replace_scheduler_exact(
    '''    --plastic-flow-contour-multipliers "1 2 4 8"''',
    '''    --plastic-flow-contour-multipliers "1 2 4 8"
    --plastic-flow-min-plastic-fraction 0.50
    --plastic-flow-min-cumulative-plastic-fraction 0.10
    --plastic-flow-max-elastic-fraction 0.50
    --plastic-flow-max-tangent-fraction 0.50
    --plastic-flow-energy-balance-tolerance 0.01''',
    label="v10.4.3 plastic-dominance command thresholds",
)

_v1043_required = [
    "arrhenius_fracture.sharp_front_v10_4_3_plastic_dominance_audited",
    "--plastic-flow-min-plastic-fraction 0.50",
    "--plastic-flow-min-cumulative-plastic-fraction 0.10",
    "--plastic-flow-max-elastic-fraction 0.50",
    "--plastic-flow-max-tangent-fraction 0.50",
    "--plastic-flow-energy-balance-tolerance 0.01",
    "CASE_FILTER_OPTION",
    "CASE_FILTER_TEMPERATURE",
    "SKIP_REUSED_VERIFIED",
]
for _token in _v1043_required:
    if _token not in scheduler:
        raise SystemExit(
            f"ERROR: final v10.4.3 scheduler is missing required token: {_token}"
        )
if scheduler.index("SKIP_REUSED_VERIFIED") > scheduler.index("expected = {"):
    raise SystemExit(
        "ERROR: v10.4.3 inherited-case reuse remains after native checks"
    )
"""

    marker = "plotter = source_plotter.read_text()"
    text = _replace_once(
        text,
        marker,
        scheduler_adapter + "\n" + marker,
        "v10.4.3 final-scheduler adapter",
    )

    text = text.replace(
        "Plastic-flow terminal: enabled; 2000 accepted-step persistence window",
        (
            "Plastic-dominance censor: enabled; sustained majority plastic "
            "accommodation with energy-balance and convergence gates"
        ),
    )
    text = text.replace(
        "v10.4.2 terminal model",
        "v10.4.3 plastic-dominance model-limit censor",
    )
    return text


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
