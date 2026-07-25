"""Audited executable wrapper for the v10.2.27 four-class campaign."""
from __future__ import annotations

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


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    install_front_direction_fix()
    install_backstress_complementarity_fix()
    install_physical_front_width()
    install_energy_ledger_output()
    original = _entry.PersistentSiteStateResolvedTipEngine
    _entry.PersistentSiteStateResolvedTipEngine = (
        AuditedPersistentSiteStateResolvedTipEngine
    )
    try:
        result = _entry.main(args)
        out = _option_value(args, "--out")
        if out:
            write_energy_ledger_audit(Path(out))
        return result
    finally:
        restore_energy_ledger_output()
        _entry.PersistentSiteStateResolvedTipEngine = original


if __name__ == "__main__":
    main()
