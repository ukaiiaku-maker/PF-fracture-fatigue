"""Worker-start hook used only by the explicit Taylor consequence diagnostic."""
from __future__ import annotations

import json
import os
from pathlib import Path


if os.environ.get("FRACTURE_DIAGNOSTIC_TAYLOR_COMPLETION") == "historical_implicit_1s":
    from arrhenius_fracture.taylor_release import historical_legacy_implicit_1s
    import arrhenius_fracture.unified_mpz as unified_mpz

    def diagnostic_historical_completion(net_elementary_rate_s, order):
        return historical_legacy_implicit_1s(
            net_elementary_rate_s, order, rate_cap_s=1.0e12
        )

    before = unified_mpz.UnifiedMPZState._transport_rates.__globals__["completion_rate"]
    unified_mpz.completion_rate = diagnostic_historical_completion
    after = unified_mpz.UnifiedMPZState._transport_rates.__globals__["completion_rate"]
    if after is not diagnostic_historical_completion:
        raise RuntimeError("worker Taylor diagnostic overlay did not bind")
    receipt_path = Path(os.environ["FRACTURE_DIAGNOSTIC_TAYLOR_RECEIPT"])
    receipt_path.write_text(json.dumps({
        "schema": "taylor-completion-worker-overlay/1",
        "classification": "DIAGNOSTIC_ONLY_NOT_PRODUCTION_SELECTABLE",
        "pid": os.getpid(),
        "changed_symbol": "arrhenius_fracture.unified_mpz.completion_rate",
        "transport_constructor": (
            "arrhenius_fracture.unified_mpz.UnifiedMPZState._transport_rates"
        ),
        "before_callable": f"{before.__module__}.{before.__qualname__}",
        "after_callable": f"{after.__module__}.{after.__qualname__}",
        "exact_historical_callable": (
            "arrhenius_fracture.taylor_release.historical_legacy_implicit_1s"
        ),
        "implicit_window_s": 1.0,
        "rate_cap_s": 1.0e12,
    }, indent=2, sort_keys=True) + "\n")
