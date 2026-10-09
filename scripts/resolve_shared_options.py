#!/usr/bin/env python3
"""Resolve a shared option JSON or named preset into a provider receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.shared_options import PRESETS, SharedOptions
from arrhenius_fracture.shared_runtime import resolve_runtime


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--preset", choices=sorted(PRESETS))
    source.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    options = PRESETS[args.preset] if args.preset else SharedOptions.load(args.config)
    payload = resolve_runtime(options).to_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
