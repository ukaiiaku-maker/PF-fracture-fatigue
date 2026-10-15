"""Audited executable wrapper for the v10.2.27 four-class campaign."""
from __future__ import annotations

import json
from pathlib import Path
import sys

from . import sharp_front_v10_2_27 as _entry
from .anisotropic_front_direction_fix_v10227 import (
    install_front_direction_fix,
)
from .energy_ledger_output_v10227 import (
    install_energy_ledger_output,
    restore_energy_ledger_output,
    write_energy_ledger_audit,
)
from .geometry_override_v10227 import (
    install_geometry_override,
    restore_geometry_override,
)
from .persistent_site_audited_engine_v10221 import (
    AuditedPersistentSiteStateResolvedTipEngine,
)
from .persistent_site_bracket_fix_v10221 import (
    install_backstress_complementarity_fix,
)
from .persistent_site_physical_width_v10222 import install_physical_front_width


def _option_value(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            return token[len(prefix):]
        if token == name and index + 1 < len(args):
            return args[index + 1]
    return None


def _persistent_execution_binding(records: list[dict]) -> dict:
    transport_modes = sorted({
        str(row.get("signed_transport_mode", "")) for row in records
    })
    exchange_modes = sorted({
        str(row.get("exchange_transport_mode", "")) for row in records
    })
    return {
        "schema": "persistent-native-execution-binding/1",
        "requested_lifecycle": "persistent_reusable",
        "finite_inventory_substitute_started": False,
        "all_expected_methods_bound": bool(records) and all(
            row["all_expected_methods_bound"] for row in records
        ),
        "signed_transport_mode": (
            transport_modes[0] if len(transport_modes) == 1 else None
        ),
        "observed_signed_transport_modes": transport_modes,
        "exchange_transport_mode": (
            exchange_modes[0] if len(exchange_modes) == 1 else None
        ),
        "observed_exchange_transport_modes": exchange_modes,
        "records": records,
    }


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    install_front_direction_fix()
    install_backstress_complementarity_fix()
    install_physical_front_width()
    install_geometry_override()
    install_energy_ledger_output()
    original = _entry.PersistentSiteStateResolvedTipEngine
    AuditedPersistentSiteStateResolvedTipEngine.reset_runtime_binding_records()
    _entry.PersistentSiteStateResolvedTipEngine = (
        AuditedPersistentSiteStateResolvedTipEngine
    )
    try:
        result = _entry.main(args)
        out = _option_value(args, "--out")
        if out:
            write_energy_ledger_audit(Path(out))
            records = AuditedPersistentSiteStateResolvedTipEngine.persistent_runtime_binding_records
            (Path(out) / "persistent_native_runtime_binding.json").write_text(
                json.dumps(
                    _persistent_execution_binding(records),
                    indent=2,
                    sort_keys=True,
                ) + "\n"
            )
        return result
    finally:
        restore_energy_ledger_output()
        restore_geometry_override()
        _entry.PersistentSiteStateResolvedTipEngine = original


if __name__ == "__main__":
    main()
