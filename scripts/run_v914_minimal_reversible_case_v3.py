#!/usr/bin/env python3
"""Run matched baseline/reversible cases with v3 reversal diagnostics.

This wrapper reuses the existing authoritative case runner and changes only the
reversible explicit integrator binding.  Baseline execution is unchanged.
"""
from __future__ import annotations

import run_v914_minimal_reversible_case as _runner
from v914_minimal_reversible_explicit_v3 import (
    run_minimal_reversible_explicit,
)


_runner.run_minimal_reversible_explicit = run_minimal_reversible_explicit


if __name__ == "__main__":
    raise SystemExit(_runner.main())
