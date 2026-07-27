"""Audited entry for v10.2.29 persistent-site cyclic fatigue."""
from __future__ import annotations

from pathlib import Path
import sys

from . import sharp_front_v10_2_28_audited as _monotonic_audited
from . import sharp_front_v10_2_29_fatigue as _entry
from .anisotropic_front_direction_fix_v10227 import install_front_direction_fix
from .energy_ledger_output_v10227 import (
    install_energy_ledger_output,
    restore_energy_ledger_output,
    write_energy_ledger_audit,
)
from .geometry_override_v10227 import install_geometry_override, restore_geometry_override
from .persistent_site_bracket_fix_v10221 import install_backstress_complementarity_fix
from .persistent_site_cyclic_coupled_audited_v10229 import (
    AuditedCoupledPersistentSiteCyclicTipEngine,
)
from .persistent_site_physical_width_v10222 import install_physical_front_width


def _has_option(args: list[str], name: str) -> bool:
    return any(token == name or token.startswith(name + "=") for token in args)


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
    if not _has_option(args, "--fatigue-cycles"):
        return _monotonic_audited.main(args)

    install_front_direction_fix()
    install_backstress_complementarity_fix()
    install_physical_front_width()
    install_geometry_override()
    install_energy_ledger_output()
    original = _entry.PersistentSiteStateResolvedTipEngine
    _entry.PersistentSiteStateResolvedTipEngine = (
        AuditedCoupledPersistentSiteCyclicTipEngine
    )
    try:
        result = _entry.main(args)
        out = _option_value(args, "--out")
        if out:
            write_energy_ledger_audit(Path(out))
        return result
    finally:
        _entry.PersistentSiteStateResolvedTipEngine = original
        restore_energy_ledger_output()
        restore_geometry_override()


if __name__ == "__main__":
    main()
