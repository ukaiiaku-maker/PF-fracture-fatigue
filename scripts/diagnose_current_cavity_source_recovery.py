#!/usr/bin/env python3
"""Evaluate the exact retained enabled cavity source recovery and emit one compact record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arrhenius_fracture.embryo_growth_v1 import (
    CavityLoopSourceRecoveryQualificationError,
    RESOLVED,
    source_local_fem_tensors,
)
from arrhenius_fracture.shared_v5_cavity_adapter import _finite, _restore_pair
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint

DEFAULT_PAIR = ROOT / "review_packet" / "current_science_pilots_6abbf10" / "v11_migration_pair.json"
DEFAULT_OUT = ROOT / "integration_evidence" / "final_fracture_campaign" / "cavity_source_recovery_diagnostic.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pair", type=Path, default=DEFAULT_PAIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    states, ancestry = _restore_pair(args.pair)
    state = states["enabled"]
    growth = next(item for item in state.void_state.embryo_growth_states
                  if item.representation == RESOLVED)
    before = complete_accepted_state_fingerprint(state)
    try:
        tensors, sigma_zz = source_local_fem_tensors(state, growth)
    except CavityLoopSourceRecoveryQualificationError as error:
        status = "FAILED_QUALIFICATION"
        result = error.diagnostic
    except RuntimeError as error:
        status = "UNAVAILABLE"
        result = {"exception_class": type(error).__name__, "exception_message": str(error)}
    else:
        status = "QUALIFIED"
        result = {"source_tensors_Pa": tensors, "source_sigma_zz_Pa": sigma_zz}
    after = complete_accepted_state_fingerprint(state)
    payload = _finite({
        "schema": "retained-cavity-source-recovery-diagnostic/1",
        "status": status,
        "pair_manifest": str(args.pair.resolve()),
        "pair_generation": ancestry.get("generation"),
        "physical_time_s": ancestry["physical_time_s"],
        "accepted_state_fingerprint": before,
        "accepted_state_unchanged": before == after,
        "growth_site_id": growth.site_id,
        "growth_cavity_id": growth.cavity_id,
        "growth_representation": growth.representation,
        "source_count": len(growth.cavity_emission_state.sources),
        "result": result,
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite diagnostic: {args.out}")
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if status == "QUALIFIED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
