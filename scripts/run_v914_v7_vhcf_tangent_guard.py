#!/usr/bin/env python3
"""Run the existing v7 VHCF CLI with constitutive-tangent block admission."""
from __future__ import annotations

import run_v914_v7_vhcf_event_to_event as _runner
from v914_v7_vhcf_tangent_guard import (
    ENGINE_ID,
    GUARD_ID,
    run_v7_vhcf_event_to_event_tangent_guarded,
)


# Reuse the qualified parser, provenance checks, candidate loading, output schema,
# and command-line contract.  Replace only the event engine callable and engine
# identifier used by that runner.
_runner.run_v7_vhcf_event_to_event = run_v7_vhcf_event_to_event_tangent_guarded
_runner.ENGINE_ID = ENGINE_ID


if __name__ == "__main__":
    raise SystemExit(_runner.main())
