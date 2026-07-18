"""v10.2.3 2-D entry point with cap-free state-equivalence tracing.

This wrapper preserves the v10.2.2 physical shielding law and v10.1.7.5
anisotropic transport selection.  It adds only a nonintrusive record of the
production engine's K/T/dt/channel-factor calls and final spatial MPZ state for
replay through ``reduced_shared_state_v1023``.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_1_7_5 as _transport
from .physical_shielding_v1022 import (
    install_uncapped_physical_shielding,
    reset_physical_shielding_audit,
    write_physical_shielding_audit,
)
from .state_equivalence_trace_v1023 import (
    capture_state_equivalence_trace,
    write_state_equivalence_trace,
)


MODEL_ID = "v10.2.3_cap_free_2d_state_equivalence_trace"


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
    out = _option_value(args, "--out")
    if not out:
        raise SystemExit("v10.2.3 requires --out for trace and shielding audits")
    root = Path(out)

    reset_physical_shielding_audit()
    print(
        "  v10.2.3 state-equivalence trace: shielding_cap=off "
        "state=production_spatial_MPZ trace=K+T+dt+channel_factors"
    )
    with capture_state_equivalence_trace() as trace:
        with install_uncapped_physical_shielding():
            result = _transport.main(args)

    shielding = write_physical_shielding_audit(root)
    state_trace = write_state_equivalence_trace(trace, root)
    payload = {
        "schema": MODEL_ID,
        "constitutive_state_modified": False,
        "mechanical_solver_modified": False,
        "anisotropic_drive_modified": False,
        "transport_modified": False,
        "shielding_cap_applied": False,
        "state_trace": state_trace,
        "physical_shielding": {
            "maximum_abs_raw_K_shield_Pa_sqrt_m": shielding[
                "maximum_abs_raw_K_shield_Pa_sqrt_m"
            ],
            "raw_equals_effective_within_relative_1e_12": shielding[
                "raw_equals_effective_within_relative_1e_12"
            ],
        },
    }
    (root / "v10_2_3_state_equivalence_driver.json").write_text(
        json.dumps(payload, indent=2)
    )
    print(
        "  v10.2.3 trace audit: "
        f"records={state_trace['n_records']} "
        f"engines={state_trace['n_engines']} "
        "raw==effective="
        f"{int(state_trace['raw_equals_effective_within_relative_1e_12'])}"
    )
    return result


if __name__ == "__main__":
    main()
