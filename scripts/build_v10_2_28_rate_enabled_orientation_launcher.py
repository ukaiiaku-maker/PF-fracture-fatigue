#!/usr/bin/env python3
"""Build a temporary rate-aware v10.2.28 orientation launcher.

The validated base orientation launcher remains unchanged. This builder adds
loading-rate provenance to its campaign lock and to the inherited per-case
scheduler, while replacing only the nominal monotonic ``dU``/``dt`` inputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def _replace_exact(
    text: str,
    old: str,
    new: str,
    *,
    count: int = 1,
    label: str,
) -> str:
    actual = text.count(old)
    if actual != count:
        raise RuntimeError(
            f"{label} changed: expected {count} occurrence(s), found {actual}"
        )
    return text.replace(old, new, count if count == 1 else -1)


def transform(source: str) -> str:
    text = source

    text = _replace_exact(
        text,
        'SIGNED_KERNEL_NOMINAL_FORWARD_COS="$SIGNED_KERNEL_NOMINAL_FORWARD_COS" \\\n"$PYTHON_BIN" - <<\'PY\'',
        'SIGNED_KERNEL_NOMINAL_FORWARD_COS="$SIGNED_KERNEL_NOMINAL_FORWARD_COS" \\\nLOADING_RATE_FACTOR="$LOADING_RATE_FACTOR" DU_M="$DU_M" BASE_DT_S="$BASE_DT_S" \\\nDT_S="$DT_S" NOMINAL_OPENING_RATE_M_PER_S="$NOMINAL_OPENING_RATE_M_PER_S" \\\nRATE_TAG="$RATE_TAG" \\\n"$PYTHON_BIN" - <<\'PY\'',
        label="campaign-lock environment",
    )
    text = _replace_exact(
        text,
        '"schema": "v10.2.28_paper_four_class_orientation_1000um_campaign_lock_v2",',
        '"schema": "v10.2.28_paper_four_class_orientation_1000um_campaign_lock_v3",',
        label="campaign-lock schema",
    )
    text = _replace_exact(
        text,
        '    "crystal_theta_deg": float(os.environ["THETA"]),\n',
        '    "crystal_theta_deg": float(os.environ["THETA"]),\n'
        '    "loading_rate_factor": float(os.environ["LOADING_RATE_FACTOR"]),\n'
        '    "nominal_dU_m": float(os.environ["DU_M"]),\n'
        '    "base_dt_s": float(os.environ["BASE_DT_S"]),\n'
        '    "nominal_dt_s": float(os.environ["DT_S"]),\n'
        '    "nominal_opening_rate_m_per_s": float(\n'
        '        os.environ["NOMINAL_OPENING_RATE_M_PER_S"]\n'
        '    ),\n'
        '    "loading_rate_tag": os.environ["RATE_TAG"],\n',
        label="campaign-lock rate fields",
    )

    adapter_marker = '''scheduler = scheduler.replace(increment, "", 1)

for old, new in replacements.items():'''
    adapter_replacement = """scheduler = scheduler.replace(increment, "", 1)


def replace_scheduler_exact(old, new, expected_count=1, label="scheduler token"):
    global scheduler
    actual = scheduler.count(old)
    if actual != expected_count:
        raise SystemExit(
            f"ERROR: {label} changed: expected {expected_count}, found {actual}"
        )
    scheduler = scheduler.replace(old, new)


replace_scheduler_exact(
    '    "maximum_steps": int(os.environ["STEPS"]),',
    '''    "maximum_steps": int(os.environ["STEPS"]),
    "loading_rate_factor": float(os.environ["LOADING_RATE_FACTOR"]),
    "nominal_dU_m": float(os.environ["DU_M"]),
    "base_dt_s": float(os.environ["BASE_DT_S"]),
    "nominal_dt_s": float(os.environ["DT_S"]),
    "nominal_opening_rate_m_per_s": float(
        os.environ["NOMINAL_OPENING_RATE_M_PER_S"]
    ),
    "loading_rate_tag": os.environ["RATE_TAG"],
    "common_random_numbers_across_loading_rates": True,''',
    label="campaign manifest rate insertion",
)
replace_scheduler_exact(
    '    "theta_deg": float(os.environ["THETA"]),',
    '''    "theta_deg": float(os.environ["THETA"]),
    "loading_rate_factor": float(os.environ["LOADING_RATE_FACTOR"]),
    "nominal_dU_m": float(os.environ["DU_M"]),
    "base_dt_s": float(os.environ["BASE_DT_S"]),
    "nominal_dt_s": float(os.environ["DT_S"]),
    "nominal_opening_rate_m_per_s": float(
        os.environ["NOMINAL_OPENING_RATE_M_PER_S"]
    ),
    "loading_rate_tag": os.environ["RATE_TAG"],''',
    expected_count=2,
    label="case contract rate insertion",
)
replace_scheduler_exact(
    '    f"--crystal-theta-deg {os.environ[\'THETA\']}",',
    '''    f"--crystal-theta-deg {os.environ['THETA']}",
    f"--dU {os.environ['DU_M']}",
    f"--dt {os.environ['DT_S']}",''',
    label="completed-case command rate verification",
)
replace_scheduler_exact(
    '    --dU 2e-7 --dt 8.4 --n-stagger 2',
    '    --dU "$DU_M" --dt "$DT_S" --n-stagger 2',
    label="simulation loading-rate command",
)
replace_scheduler_exact(
    '"v10.2.27_paper_four_class_30deg_long_rcurve_campaign_v1"',
    '"v10.2.28_paper_four_class_orientation_loading_rate_campaign_v1"',
    label="campaign manifest schema",
)
replace_scheduler_exact(
    '"v10.2.27_case_contract_v1"',
    '"v10.2.28_orientation_loading_rate_case_contract_v1"',
    label="case contract schema",
)

for old, new in replacements.items():"""
    text = _replace_exact(
        text,
        adapter_marker,
        adapter_replacement,
        label="embedded scheduler rate adapter",
    )

    text = _replace_exact(
        text,
        '  RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \\\n  SIGNED_KERNEL_EXTENSION_COORDINATE="$SIGNED_KERNEL_EXTENSION_COORDINATE" \\',
        '  RESTART_INCOMPLETE="$RESTART_INCOMPLETE" \\\n  LOADING_RATE_FACTOR="$LOADING_RATE_FACTOR" \\\n  DU_M="$DU_M" \\\n  BASE_DT_S="$BASE_DT_S" \\\n  DT_S="$DT_S" \\\n  NOMINAL_OPENING_RATE_M_PER_S="$NOMINAL_OPENING_RATE_M_PER_S" \\\n  RATE_TAG="$RATE_TAG" \\\n  SIGNED_KERNEL_EXTENSION_COORDINATE="$SIGNED_KERNEL_EXTENSION_COORDINATE" \\',
        label="generated scheduler rate environment",
    )
    return text


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = transform(args.source.read_text())
    args.output.write_text(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
