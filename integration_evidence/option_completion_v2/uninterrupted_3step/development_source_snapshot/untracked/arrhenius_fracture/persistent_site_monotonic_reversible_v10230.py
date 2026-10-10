"""Monotonic-loading engine with native local-drive reversible transport."""
from __future__ import annotations

import copy
from typing import Any

from .persistent_site_audited_engine_v10221 import (
    AuditedPersistentSiteStateResolvedTipEngine,
)
from .persistent_site_reversible_transport_v10230 import (
    install_reversible_transport,
    reversible_binding_receipt,
)


MODEL_ID = "v10.2.30_monotonic_local_drive_reversible_transport_v1"


class MonotonicReversiblePersistentSiteTipEngine(
    AuditedPersistentSiteStateResolvedTipEngine
):
    """Install reversal only after the final signed persistent MPZ exists."""

    reversible_runtime_bindings: list[dict[str, Any]] = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        receipt = install_reversible_transport(self.mpz)
        receipt = {**receipt, "engine_type": type(self).__qualname__}
        type(self).reversible_runtime_bindings.append(copy.deepcopy(receipt))

    @classmethod
    def reset_runtime_bindings(cls) -> None:
        cls.reversible_runtime_bindings = []

    def reversible_binding_receipt(self) -> dict[str, Any]:
        return reversible_binding_receipt(self.mpz)


__all__ = ["MODEL_ID", "MonotonicReversiblePersistentSiteTipEngine"]
