"""Audited executable wrapper for the v10.2.27 four-class campaign."""
from __future__ import annotations

from . import sharp_front_v10_2_27 as _entry
from .anisotropic_front_direction_fix_v10227 import (
    install_front_direction_fix,
)
from .persistent_site_audited_engine_v10221 import (
    AuditedPersistentSiteStateResolvedTipEngine,
)
from .persistent_site_bracket_fix_v10221 import (
    install_backstress_complementarity_fix,
)
from .persistent_site_physical_width_v10222 import install_physical_front_width


def main(argv=None):
    install_front_direction_fix()
    install_backstress_complementarity_fix()
    install_physical_front_width()
    original = _entry.PersistentSiteStateResolvedTipEngine
    _entry.PersistentSiteStateResolvedTipEngine = (
        AuditedPersistentSiteStateResolvedTipEngine
    )
    try:
        return _entry.main(argv)
    finally:
        _entry.PersistentSiteStateResolvedTipEngine = original


if __name__ == "__main__":
    main()
