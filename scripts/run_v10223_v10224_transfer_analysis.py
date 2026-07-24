#!/usr/bin/env python3
"""User-facing safe launcher for analyze_v10223_v10224_transfer.py."""
from __future__ import annotations

import math
from pathlib import Path
import sys

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import analyze_v10223_v10224_transfer as core


def json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def main() -> int:
    original_dumps = core.json.dumps

    def safe_dumps(value, *args, **kwargs):
        kwargs["allow_nan"] = False
        kwargs.pop("default", None)
        return original_dumps(json_safe(value), *args, **kwargs)

    core.json.dumps = safe_dumps
    try:
        return core.main()
    finally:
        core.json.dumps = original_dumps


if __name__ == "__main__":
    raise SystemExit(main())
