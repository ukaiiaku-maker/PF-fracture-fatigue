"""Fresh spatial entry using the R3 current persistent/reversible law."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_27_audited as _selected
from .persistent_site_bracket_fix_v10221 import (
    install_backstress_complementarity_fix,
)
from .reduced_persistent_reference_r2 import (
    ReducedPersistentReversibleTipEngine, TAYLOR_LAW_ID,
    install_mean_rate_taylor,
)


MODEL_ID = "r4.spatial-current-persistent-reversible-taylor/1"


class CurrentLawSpatialTipEngine(ReducedPersistentReversibleTipEngine):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        install_mean_rate_taylor(self.mpz)


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    original = _selected.AuditedPersistentSiteStateResolvedTipEngine
    install_backstress_complementarity_fix()
    _selected.AuditedPersistentSiteStateResolvedTipEngine = CurrentLawSpatialTipEngine
    try:
        result = _selected.main(args)
        out = _selected._option_value(args, "--out")
        if out:
            payload = {
                "schema": MODEL_ID,
                "engine": (
                    f"{CurrentLawSpatialTipEngine.__module__}."
                    f"{CurrentLawSpatialTipEngine.__qualname__}"
                ),
                "source_lifecycle": "persistent_reusable",
                "transport": "reversible_signed",
                "taylor_law_id": TAYLOR_LAW_ID,
                "prescribed_opening_history_m": __import__("os").environ.get(
                    "ARRHENIUS_PRESCRIBED_OPENING_HISTORY_M"
                ),
            }
            (Path(out) / "r4_current_law_spatial.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        return result
    finally:
        _selected.AuditedPersistentSiteStateResolvedTipEngine = original


if __name__ == "__main__":
    main()


__all__ = ["CurrentLawSpatialTipEngine", "MODEL_ID", "main"]
