"""Declared hazard-coupled material-field translation for Ast comparison."""
from __future__ import annotations


TRANSLATION_POLICY = "hazard_coupled_power_0p95"


def partial_path_increment(action, predicted_action, threshold, accepted_path,
                           event_advance, exponent=0.95):
    if threshold <= 0 or event_advance <= 0 or exponent <= 0:
        raise ValueError("invalid hazard path parameters")
    fraction = min(max((action + predicted_action) / threshold, 0.), 1.)
    target = event_advance * fraction**exponent
    return min(max(target - accepted_path, 0.),
               max(event_advance - accepted_path, 0.))
