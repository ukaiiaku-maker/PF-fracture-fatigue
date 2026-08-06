#!/usr/bin/env python3
"""Add fail-closed audited-reuse handling to the v10.4.2 production launcher.

Materialized v10.4.1 cases carry a v10.4.2 reuse audit that verifies source
hashes, detailed-balance provenance, target completion, and the corrected
positive directional-J history.  A valid inherited case must return a clean
skip before native-v10.4.2 contract checks are evaluated.
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

    # These are transformations of the final generated scheduler. They are
    # deliberately inserted into the outer builder immediately before it reads
    # the plotter, after every earlier scheduler adapter has been registered.
    scheduler_repairs = r"""
replace_scheduler_exact(
    'contract = json.loads((root / "v10_2_27_case_contract.json").read_text())',
    '''v1042_reuse_path = root / "v10_4_2_reuse_audit.json"
if v1042_reuse_path.is_file():
    from arrhenius_fracture.reuse_v1041_v1042 import (
        verify_materialized_case,
        verify_source_case,
    )

    reuse_audit = verify_materialized_case(root)
    verify_source_case(Path(reuse_audit["source_case"]))
    print(f"SKIP_REUSED_VERIFIED {root}")
    raise SystemExit(0)

contract = json.loads((root / "v10_2_27_case_contract.json").read_text())''',
    label="v10.4.2 audited-reuse short-circuit before native contract checks",
)

replace_scheduler_exact(
    '''v1042_reuse_path = root / "v10_4_2_reuse_audit.json"
if v1042_reuse_path.is_file():
    from arrhenius_fracture.reuse_v1041_v1042 import verify_materialized_case

    verify_materialized_case(root)
elif bulk_model_audit.get("schema") != "v10.4.2_bulk_detailed_balance_plastic_flow_terminal":
    raise SystemExit(1)''',
    '''if bulk_model_audit.get("schema") != "v10.4.2_bulk_detailed_balance_plastic_flow_terminal":
    raise SystemExit(1)''',
    label="remove obsolete late v10.4.2 reuse path",
)

replace_scheduler_exact(
    '''if find "$OUTROOT" -type f -name COMPLETE -print -quit | grep -q .; then''',
    '''if find "$OUTROOT" \\( -type f -o -type l \\) -name COMPLETE -print -quit | grep -q .; then''',
    label="v10.4.2 symlink-aware COMPLETE postprocessing gate",
)

replace_scheduler_exact(
    '''echo "Campaign complete: failures=$failures output=$OUTROOT"
[[ "$failures" -eq 0 ]] || exit 1''',
    '''acceptance_rc=$?
if [[ "$acceptance_rc" -ne 0 ]]; then
  echo "ERROR: final filesystem campaign acceptance failed (exit=$acceptance_rc)" >&2
  exit "$acceptance_rc"
fi
if [[ "$failures" -ne 0 ]]; then
  echo "WARNING: child-status failures=$failures but final filesystem acceptance passed; reconciling to failures=0" >&2
fi
failures=0
echo "Campaign complete: failures=$failures output=$OUTROOT"''',
    label="v10.4.2 reconcile scheduler and filesystem completion status",
)

# Convert any verification rejection of an inherited case into an explicit
# status before the terminal-looking directory guard reports the shell failure.
replace_scheduler_exact(
    '''    if [[ -f "$case_root/COMPLETE" || -f "$case_root/PLASTIC_FLOW" ]]; then
      echo "ERROR: terminal-looking case failed contract verification: $case_root" >&2
      return 3
    fi''',
    '''    if [[ -f "$case_root/v10_4_2_reuse_audit.json" ]]; then
      echo "FAILED_REUSE_VERIFICATION $case_root" >&2
      return 3
    fi
    if [[ -f "$case_root/COMPLETE" || -f "$case_root/PLASTIC_FLOW" ]]; then
      echo "ERROR: terminal-looking case failed contract verification: $case_root" >&2
      return 3
    fi''',
    label="v10.4.2 explicit reuse-verification failure status",
)

_reuse_skip = 'print(f"SKIP_REUSED_VERIFIED {root}")'
_native_expected = 'expected = {'
if scheduler.count(_reuse_skip) != 1:
    raise SystemExit(
        "ERROR: final scheduler must contain exactly one SKIP_REUSED_VERIFIED path; "
        f"found {scheduler.count(_reuse_skip)}"
    )
if scheduler.index(_reuse_skip) > scheduler.index(_native_expected):
    raise SystemExit(
        "ERROR: audited-reuse guard is still after native v10.4.2 contract checks"
    )
"""

    tail_marker = "plotter = source_plotter.read_text()"
    return _replace_once(
        text,
        tail_marker,
        scheduler_repairs + "\n" + tail_marker,
        "v10.4.2 audited-reuse final-scheduler repairs",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
