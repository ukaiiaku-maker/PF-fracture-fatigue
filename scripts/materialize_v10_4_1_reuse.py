#!/usr/bin/env python3
"""Materialize approved v10.4.0 cases into a fresh v10.4.1 root."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from arrhenius_fracture.reuse_v1040_v1041 import materialize_reuse_cases


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit_json", type=Path)
    parser.add_argument("destination_root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    payload = materialize_reuse_cases(args.audit_json, args.destination_root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
