#!/usr/bin/env python3
"""Harden the generated v10.4.2/v10.4.3 production scheduler.

The builder inserts audited-reuse verification before native contract checks,
makes marker handling symlink-aware, and reconciles the shell job counter with
the authoritative filesystem acceptance record.  v10.4.3 disables legacy
v10.4.1 reuse by default because the stagger-time correction changes the
plastic state and therefore the coupled fracture trajectory.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_positive_j_builder():
    path = Path(__file__).with_name("build_v10_4_2_positive_J_launcher.py")
    spec = importlib.util.spec_from_file_location("v1042_positive_j_builder", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load v10.4.2 positive-J builder: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    text = _load_positive_j_builder().transform(source)

    # Edit the nested final-scheduler generator.  The old anchor at bulk_audit
    # was too late: inherited v10.4.1 contracts had already failed native v10.4.2
    # comparisons.  Keep an explicit opt-in only for historical v10.4.2 reuse
    # verification; corrected v10.4.3 production must use a fresh run root.
    scheduler_hardening = (
        'replace_scheduler_exact(\n'
        '    \'contract = json.loads((root / "v10_2_27_case_contract.json").read_text())\',\n'
        '    \'\'\'v1042_reuse_path = root / "v10_4_2_reuse_audit.json"\n'
        'if v1042_reuse_path.is_file():\n'
        '    if os.environ.get("ALLOW_V1041_REUSE_AFTER_STAGGER_FIX", "0") != "1":\n'
        '        print(\n'
        '            f"RERUN_REQUIRED_STAGGER_TIME_CORRECTION {root}: "\n'
        '            "legacy v10.4.1 plastic trajectories are not reusable by v10.4.3"\n'
        '        )\n'
        '        raise SystemExit(1)\n'
        '    from arrhenius_fracture.reuse_v1041_v1042 import (\n'
        '        verify_materialized_case,\n'
        '        verify_source_case,\n'
        '    )\n'
        '\n'
        '    try:\n'
        '        reuse_audit = verify_materialized_case(root)\n'
        '        verify_source_case(Path(reuse_audit["source_case"]))\n'
        '    except BaseException as exc:\n'
        '        print(f"FAILED_REUSE_VERIFICATION {root}: {exc}")\n'
        '        raise\n'
        '    print(f"SKIP_REUSED_VERIFIED {root}")\n'
        '    raise SystemExit(0)\n'
        '\n'
        'contract = json.loads((root / "v10_2_27_case_contract.json").read_text())\'\'\',\n'
        '    label="audited-reuse short-circuit before native contract checks",\n'
        ')\n\n'
        'replace_scheduler_exact(\n'
        '    \'if find "$OUTROOT" -type f -name COMPLETE -print -quit | grep -q .; then\',\n'
        '    \'if find "$OUTROOT" -name COMPLETE -print -quit | grep -q .; then\',\n'
        '    label="symlink-aware COMPLETE postprocessing gate",\n'
        ')\n\n'
        'replace_scheduler_exact(\n'
        '    \'\'\'echo "Campaign complete: failures=$failures output=$OUTROOT"\n'
        '[[ "$failures" -eq 0 ]] || exit 1\'\'\',\n'
        '    \'\'\'acceptance_failures=$(\n'
        '  "$PYTHON_BIN" -c \'import json,sys; print(int(json.load(open(sys.argv[1]))["failed_or_incomplete_cases"]))\' \\\n'
        '    "$OUTROOT/v10_2_27_campaign_acceptance.json"\n'
        ') || {\n'
        '  echo "ERROR: could not read authoritative campaign acceptance count" >&2\n'
        '  exit 1\n'
        '}\n'
        'if [[ "$failures" -ne "$acceptance_failures" ]]; then\n'
        '  echo "INFO: reconciling shell failures=$failures with filesystem failures=$acceptance_failures" >&2\n'
        'fi\n'
        'failures=$acceptance_failures\n'
        'echo "Campaign complete: failures=$failures output=$OUTROOT"\n'
        '[[ "$failures" -eq 0 ]] || exit 1\'\'\',\n'
        '    label="authoritative campaign failure reconciliation",\n'
        ')\n\n'
    )

    tail_marker = "plotter = source_plotter.read_text()"
    text = _replace_once(
        text,
        tail_marker,
        scheduler_hardening + tail_marker,
        "generated scheduler hardening",
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
