"""Audited v10.2.27 entry with reversible transport on the live MPZ state."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_27_audited as _audited
from . import sharp_front_v10_2_27 as _entry
from .persistent_site_monotonic_reversible_v10230 import (
    MODEL_ID,
    MonotonicReversiblePersistentSiteTipEngine,
)


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    MonotonicReversiblePersistentSiteTipEngine.reset_runtime_bindings()
    original = _audited.AuditedPersistentSiteStateResolvedTipEngine
    _audited.AuditedPersistentSiteStateResolvedTipEngine = (
        MonotonicReversiblePersistentSiteTipEngine
    )
    try:
        result = _audited.main(args)
        out = _option_value(args, "--out")
        if out:
            receipts = MonotonicReversiblePersistentSiteTipEngine.reversible_runtime_bindings
            payload = {
                "schema": "v10.2.30-monotonic-reversible-runtime/1",
                "model_id": MODEL_ID,
                "constructed_object_count": len(receipts),
                "all_expected_methods_bound": bool(receipts) and all(
                    row["all_expected_methods_bound"] for row in receipts
                ),
                "bindings": receipts,
            }
            target = Path(out) / "monotonic_reversible_runtime_binding.json"
            target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return result
    finally:
        _audited.AuditedPersistentSiteStateResolvedTipEngine = original


if __name__ == "__main__":
    main()


__all__ = ["main"]
