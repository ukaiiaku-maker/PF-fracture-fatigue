#!/usr/bin/env python3
"""Resolve or execute a qualified shared fracture/fatigue option set."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from arrhenius_fracture.shared_execution import build_execution_plan, execute_shared
from arrhenius_fracture.shared_options import PRESETS, SharedOptions

def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=sorted(PRESETS))
    source.add_argument("--options", type=Path)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--native-args-json", default="[]")
    parser.add_argument("--source-policy", choices=("qualified", "development_snapshot"),
                        default="qualified")
    parser.add_argument("--restart-from", type=Path)
    args = parser.parse_args(argv)
    options = PRESETS[args.preset] if args.preset else SharedOptions.load(args.options)
    plan = build_execution_plan(options)
    if args.plan_only:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0
    if args.out is None:
        parser.error("--out is required for execution")
    native_args = json.loads(args.native_args_json)
    if not isinstance(native_args, list) or not all(isinstance(item, str) for item in native_args):
        parser.error("--native-args-json must encode a list of strings")
    execute_shared(options, native_args, args.out,
                   source_policy=args.source_policy, restart_from=args.restart_from)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
