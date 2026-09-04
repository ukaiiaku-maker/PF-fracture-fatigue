#!/usr/bin/env python3
"""Reserved entry point for the replacement V5 campaign.

The broad campaign remains prohibited until the audit-repair checkpoint is
qualified at an exact clean head.  Keeping this entry point fail-closed also
prevents the historical output directory from being reused.
"""
from __future__ import annotations

raise SystemExit(
    "AUDIT_REPAIR_CHECKPOINT_REQUIRED: replacement output root will be "
    "artifacts/voiding_v5_finalization_v2 after prerequisite qualification"
)
