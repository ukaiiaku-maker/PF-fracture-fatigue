"""Prospective V1 fingerprint of state capable of changing future evolution.

The raw checkpoint remains authoritative and is never rewritten.  This view
excludes only solver residual diagnostics, empty representation-only boundary
certificate fields, audit commit provenance, and the historical absence versus
explicit ``None`` representation of disabled void state.  Physical reaction,
energy, mechanics, hazard, RNG, source, remesh, and topology state is retained.
"""
from copy import deepcopy
import hashlib
import json

SCHEMA = "v5.future-causal-state-fingerprint/1"
DIAGNOSTIC_ENERGY_FIELDS = (
    "latest_free_dof_residual_l2_N_per_m",
    "latest_constrained_reaction_l2_N_per_m",
    "latest_top_bottom_reaction_balance",
    "latest_energy_reaction_identity",
)


def components(value):
    """Return the future-causal view and an explicit exclusion ledger."""
    result = deepcopy(value)
    excluded = {}
    if result.get("void_state") is not None:
        raise ValueError("future-causal disabled fingerprint requires void_state absent or None")
    result["void_state"] = None
    energy = result.get("energy_ledgers", {})
    for key in DIAGNOSTIC_ENERGY_FIELDS:
        if key in energy:
            excluded["/energy_ledgers/" + key] = energy.pop(key)
    junction = result.get("junction_process_state", {})
    for parent, key, path in (
        (junction, "v12_boundary_terminal_certificates", "/junction_process_state"),
        (junction.get("v12_graph_support_audit", {}), "boundary_terminal_certificates",
         "/junction_process_state/v12_graph_support_audit"),
    ):
        if key in parent and parent[key] in ([], ()):
            excluded[path + "/" + key] = parent.pop(key)
    support = result.get("v12_support_state")
    if support is not None and "source_commit" in support:
        excluded["/v12_support_state/source_commit"] = support.pop("source_commit")
    return result, excluded


def fingerprint(value):
    causal, _ = components(value)
    encoded = json.dumps(causal, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
