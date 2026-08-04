#!/usr/bin/env python3
"""Patch the v10.4.4 scheduler with fail-closed nonzero-exit bookkeeping."""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path


def _load_base_patcher():
    path = Path(__file__).with_name("patch_v10_4_4_generated_scheduler.py")
    spec = importlib.util.spec_from_file_location("v1044_scheduler_patcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load scheduler patcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _replace_exact(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{label} changed: expected one occurrence, found {count}"
        )
    return text.replace(old, new, 1)


def transform(source: str) -> str:
    text = _load_base_patcher().transform(source)

    text = _replace_exact(
        text,
        '''  echo "START: option=${option} T=${T}K seed=${case_seed} target=${TARGET_EXT_UM}um theta=${THETA}"
  env \\
''',
        '''  echo "START: option=${option} T=${T}K seed=${case_seed} target=${TARGET_EXT_UM}um theta=${THETA}"
  local rc=0
  if env \\
''',
        label="solver command guarded start",
    )

    text = _replace_exact(
        text,
        '''    "${cmd[@]}" > "$log" 2>&1
  local rc=$?
  echo "$rc" > "$case_root/exit_code.txt"
''',
        '''    "${cmd[@]}" > "$log" 2>&1; then
    rc=0
  else
    rc=$?
  fi
  echo "$rc" > "$case_root/exit_code.txt"
''',
        label="solver nonzero-exit capture",
    )

    if 'echo "$rc" > "$case_root/exit_code.txt"' not in text:
        raise RuntimeError("v10.4.8 exit-code bookkeeping is missing")
    if 'echo "simulation_exit_$rc" > "$case_root/RUN_FAILED"' not in text:
        raise RuntimeError("v10.4.8 RUN_FAILED bookkeeping is missing")
    return text


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.output.write_text(transform(args.source.read_text()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
