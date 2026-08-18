#!/usr/bin/env python3
"""Run matched baseline/reversible cases with v4 physical-return semantics."""
from __future__ import annotations

import run_v914_minimal_reversible_case as _runner
from v914_minimal_reversible_explicit_v4 import (
    run_minimal_reversible_explicit,
)


_runner.run_minimal_reversible_explicit = run_minimal_reversible_explicit


if __name__ == "__main__":
    raise SystemExit(_runner.main())
