"""Propagate v10.2.30 gate metadata into stochastic geometry diagnostics."""
from __future__ import annotations

import copy
from typing import Any

from . import stochastic_avalanche_backend as _backend

MODEL_ID = "v10.2.30_hazard_energy_geometry_audit"
_ORIGINAL_POP = None
_ORIGINAL_ADVANCE = None
_LAST_DESCRIPTOR: dict[str, Any] | None = None


def install_hazard_energy_backend_audit() -> None:
    global _ORIGINAL_POP, _ORIGINAL_ADVANCE
    if _ORIGINAL_POP is not None:
        return
    _ORIGINAL_POP = _backend.pop_pending_geometry_event
    _ORIGINAL_ADVANCE = _backend.AvalancheSubsegmentBackend.advance

    def observed_pop():
        global _LAST_DESCRIPTOR
        descriptor = _ORIGINAL_POP()
        _LAST_DESCRIPTOR = copy.deepcopy(descriptor)
        return descriptor

    def audited_advance(self, **kwargs):
        result = _ORIGINAL_ADVANCE(self, **kwargs)
        descriptor = copy.deepcopy(_LAST_DESCRIPTOR)
        if result.inserted and self.advance_log and isinstance(descriptor, dict):
            gate = descriptor.get("hazard_energy_gate")
            if isinstance(gate, dict):
                row = self.advance_log[-1]
                row["hazard_energy_gate_model_id"] = MODEL_ID
                row["hazard_energy_gate_active"] = True
                row["hazard_energy_gate"] = copy.deepcopy(gate)
                fields = {
                    "proposed_event_advance_m": "energy_gate_proposed_event_advance_m",
                    "accepted_event_advance_m": "energy_gate_accepted_event_advance_m",
                    "rejected_event_advance_m": "energy_gate_rejected_event_advance_m",
                    "gate_fraction": "energy_gate_fraction",
                    "Gamma_haz_J_per_m2": "energy_gate_Gamma_haz_J_per_m2",
                    "J_probe_J_per_m2": "energy_gate_J_probe_J_per_m2",
                    "J_event_scaled_J_per_m2": "energy_gate_J_event_scaled_J_per_m2",
                    "K_probe_Pa_sqrt_m": "energy_gate_K_probe_Pa_sqrt_m",
                    "K_event_Pa_sqrt_m": "energy_gate_K_event_Pa_sqrt_m",
                    "probe_to_event_energy_scale": "energy_gate_probe_to_event_scale",
                    "gamma_rel": "energy_gate_gamma_rel",
                    "DeltaG_cleave_eff_eV": "energy_gate_DeltaG_cleave_eff_eV",
                    "energy_available_integrated_J_per_m": (
                        "energy_gate_available_integrated_J_per_m"
                    ),
                    "energy_dissipated_integrated_J_per_m": (
                        "energy_gate_dissipated_integrated_J_per_m"
                    ),
                    "energy_margin_integrated_J_per_m": (
                        "energy_gate_margin_integrated_J_per_m"
                    ),
                }
                for source, target in fields.items():
                    if source in gate:
                        row[target] = gate[source]
        return result

    _backend.pop_pending_geometry_event = observed_pop
    _backend.AvalancheSubsegmentBackend.advance = audited_advance


def restore_hazard_energy_backend_audit() -> None:
    global _ORIGINAL_POP, _ORIGINAL_ADVANCE, _LAST_DESCRIPTOR
    if _ORIGINAL_POP is None:
        return
    _backend.pop_pending_geometry_event = _ORIGINAL_POP
    _backend.AvalancheSubsegmentBackend.advance = _ORIGINAL_ADVANCE
    _ORIGINAL_POP = None
    _ORIGINAL_ADVANCE = None
    _LAST_DESCRIPTOR = None


def audit_payload() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "installed": _ORIGINAL_POP is not None,
        "nested_gate_payload_preserved": True,
        "flat_geometry_event_fields_added": True,
    }


__all__ = [
    "MODEL_ID",
    "audit_payload",
    "install_hazard_energy_backend_audit",
    "restore_hazard_energy_backend_audit",
]
