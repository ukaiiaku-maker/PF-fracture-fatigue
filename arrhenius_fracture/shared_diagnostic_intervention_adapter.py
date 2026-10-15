"""Native populated-state adapter for qualification-only P3 interventions."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from . import sharp_front_v10_2_27 as _entry
from . import sharp_front_v10_2_27_audited as _audited
from .diagnostic_interventions_v1 import install_diagnostic_intervention
from .persistent_site_audited_engine_v10221 import (
    AuditedPersistentSiteStateResolvedTipEngine,
)


SUPPORTED = frozenset({
    "retained_shielding_off", "source_backstress_off", "blunting_fixed",
    "transport_retention_off",
})


def _extract_option(args: list[str], name: str) -> str | None:
    prefix = name + "="
    for index, token in enumerate(args):
        if token.startswith(prefix):
            value = token[len(prefix):]
            del args[index]
            return value
        if token == name:
            if index + 1 >= len(args):
                raise ValueError(f"{name} requires a value")
            value = args[index + 1]
            del args[index:index + 2]
            return value
    return None


def _population(state) -> float:
    names = (
        "mobile_positive", "mobile_negative", "retained_positive",
        "retained_negative", "accumulated_slip_positive",
        "accumulated_slip_negative", "wake_mobile_positive",
        "wake_mobile_negative", "wake_retained_positive",
        "wake_retained_negative", "wake_slip_positive", "wake_slip_negative",
    )
    return float(sum(
        np.sum(np.abs(np.asarray(getattr(state, name), dtype=float)))
        for name in names if hasattr(state, name)
    ))


def _engine(base, intervention: str):
    class DiagnosticEngine(base):
        diagnostic_runtime_records: list[dict] = []

        @classmethod
        def reset_runtime_binding_records(cls):
            super().reset_runtime_binding_records()
            cls.diagnostic_runtime_records = []

        def step(self, K, T, dt):
            result = super().step(K, T, dt)
            if (not hasattr(self.mpz, "diagnostic_intervention_receipt")
                    and _population(self.mpz) > 0.0):
                receipt = install_diagnostic_intervention(
                    self.mpz, intervention,
                    r0_m=float(self.f.r0), b_m=float(self.b),
                )
                receipt = dict(receipt)
                receipt.update({
                    "installation_stage": "after_first_populated_accepted_native_step",
                    "applies_from": "next_native_step",
                    "population_measure_at_installation": _population(self.mpz),
                })
                self.mpz.diagnostic_intervention_receipt = receipt
                type(self).diagnostic_runtime_records.append(receipt)
            return result

    DiagnosticEngine.__name__ = f"Diagnostic{base.__name__}"
    return DiagnosticEngine


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    intervention = _extract_option(args, "--diagnostic-intervention")
    if intervention not in SUPPORTED:
        raise ValueError(
            "the populated-state diagnostic adapter requires exactly one of "
            + ", ".join(sorted(SUPPORTED))
        )
    out = _audited._option_value(args, "--out")
    engine = _engine(AuditedPersistentSiteStateResolvedTipEngine, intervention)
    original = _entry.PersistentSiteStateResolvedTipEngine
    engine.reset_runtime_binding_records()
    _entry.PersistentSiteStateResolvedTipEngine = engine
    try:
        # Reproduce the audited wrapper's native patch stack and receipts while
        # substituting only the engine class selected above.
        _audited.install_front_direction_fix()
        _audited.install_backstress_complementarity_fix()
        _audited.install_physical_front_width()
        _audited.install_geometry_override()
        _audited.install_energy_ledger_output()
        result = _entry.main(args)
        if out:
            destination = Path(out)
            _audited.write_energy_ledger_audit(destination)
            (destination / "persistent_native_runtime_binding.json").write_text(
                json.dumps(_audited._persistent_execution_binding(
                    engine.persistent_runtime_binding_records
                ), indent=2, sort_keys=True) + "\n"
            )
            if not engine.diagnostic_runtime_records:
                raise RuntimeError(
                    "native run ended before a populated tip state was accepted; "
                    "the requested diagnostic intervention was not applied"
                )
            payload = {
                "schema": "shared-diagnostic-intervention-runtime-binding/1",
                "requested_intervention": intervention,
                "installed": True,
                "records": engine.diagnostic_runtime_records,
                "finite_inventory_substitute_started": False,
            }
            (destination / "diagnostic_intervention_runtime_binding.json").write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n"
            )
        return result
    finally:
        _audited.restore_energy_ledger_output()
        _audited.restore_geometry_override()
        _entry.PersistentSiteStateResolvedTipEngine = original


if __name__ == "__main__":
    raise SystemExit(main())
