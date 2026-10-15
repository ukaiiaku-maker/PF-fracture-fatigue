"""Orthogonal diagnostic interventions for the persistent native tip state.

These switches are counterfactual study operators.  They never become a
production default and each removes only its named coupling.  Population
transfer between bulk and tip is deliberately absent because no physical
coefficient has been authorized.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Mapping

import numpy as np

from .signed_burgers_shared_v1025 import _sync_active, _sync_wake


MODEL_ID="persistent-native-diagnostic-interventions/1"
INTERVENTIONS=frozenset({
    "native","retained_shielding_off","source_backstress_off",
    "blunting_fixed","transport_retention_off","history_reset_reequilibrated",
})


def _population_arrays(state):
    names=(
        "mobile_positive","mobile_negative","retained_positive","retained_negative",
        "accumulated_slip_positive","accumulated_slip_negative",
        "wake_mobile_positive","wake_mobile_negative","wake_retained_positive",
        "wake_retained_negative","wake_slip_positive","wake_slip_negative",
    )
    return {name:np.asarray(getattr(state,name),float) for name in names if hasattr(state,name)}


def _fingerprint(state) -> str:
    digest=hashlib.sha256()
    for name,value in sorted(_population_arrays(state).items()):
        digest.update(name.encode()); digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def install_diagnostic_intervention(state,intervention: str,*,r0_m=None,b_m=None) -> dict[str,Any]:
    """Install one named intervention without replacing native method bodies."""
    name=str(intervention)
    if name not in INTERVENTIONS:
        raise ValueError(f"unknown diagnostic intervention {name!r}")
    if name=="history_reset_reequilibrated":
        raise ValueError("history reset requires apply_history_reset_reequilibrated")
    before=_fingerprint(state)
    state._diagnostic_intervention=name
    state._diagnostic_retained_shielding_enabled=name!="retained_shielding_off"
    state._diagnostic_source_backstress_off=name=="source_backstress_off"
    state._diagnostic_transport_retention_off=name=="transport_retention_off"
    state._diagnostic_fixed_blunting=name=="blunting_fixed"
    fixed=None
    if name=="blunting_fixed":
        r0=float(state._persistent_r0_m if r0_m is None else r0_m)
        b=float(state._persistent_b if b_m is None else b_m)
        # Capture the developed radius before enabling the switch.
        state._diagnostic_fixed_blunting=False
        fixed=float(state.blunted_radius(r0,b))
        state._diagnostic_fixed_blunting_radius_m=fixed
        state._diagnostic_fixed_blunting=True
    receipt={
        "schema":MODEL_ID,"intervention":name,"qualification_only":True,
        "production_default":False,"population_state_unchanged":before==_fingerprint(state),
        "fixed_blunting_radius_m":fixed,
        "removed_terms":{
            "retained_shielding":name=="retained_shielding_off",
            "source_backstress":name=="source_backstress_off",
            "blunting_evolution":name=="blunting_fixed",
            "exchange_retention_transport":name=="transport_retention_off",
        },
        "tip_bulk_population_transfer_added":False,
    }
    state.diagnostic_intervention_receipt=receipt
    return receipt


def apply_history_reset_reequilibrated(
    state,*,reequilibrate:Callable[[],Mapping[str,Any]],
    accumulated_work:Mapping[str,float]|None=None,
) -> dict[str,Any]:
    """Reset process history, inventory what was removed, then re-equilibrate.

    The reset is explicitly counterfactual: it does not increment recovery or
    escape ledgers, and historical work is recorded rather than erased.
    """
    if not callable(reequilibrate):
        raise TypeError("history reset requires a native re-equilibration callback")
    arrays=_population_arrays(state)
    before=_fingerprint(state)
    removed={name:float(np.sum(value)) for name,value in arrays.items()}
    ledgers={name:float(getattr(state,name)) for name in (
        "recovered_total","escaped_total","wake_discarded_mobile_total",
        "wake_discarded_retained_total","wake_discarded_slip_total",
    ) if hasattr(state,name)}
    for value in arrays.values():
        value[...] = 0.0
    _sync_active(state); _sync_wake(state)
    after_reset=_fingerprint(state)
    equilibrium=dict(reequilibrate())
    if not equilibrium.get("converged",False):
        raise RuntimeError("native post-reset re-equilibration did not converge")
    for name,value in ledgers.items():
        if float(getattr(state,name))!=value:
            raise RuntimeError("diagnostic reset was incorrectly booked as physical loss")
    receipt={
        "schema":MODEL_ID,"intervention":"history_reset_reequilibrated",
        "qualification_only":True,"production_default":False,
        "pre_reset_population_sha256":before,"post_reset_population_sha256":after_reset,
        "removed_population":removed,
        "removed_population_total":float(sum(removed.values())),
        "counterfactual_removed_work":dict(accumulated_work or {}),
        "physical_loss_ledgers_unchanged":True,
        "native_reequilibration":equilibrium,
        "tip_bulk_population_transfer_added":False,
    }
    # Ensure the receipt itself is immediately serialization-safe.
    json.dumps(receipt,sort_keys=True)
    state._diagnostic_intervention="history_reset_reequilibrated"
    state.diagnostic_intervention_receipt=receipt
    return receipt


__all__=["INTERVENTIONS","MODEL_ID","apply_history_reset_reequilibrated",
         "install_diagnostic_intervention"]
