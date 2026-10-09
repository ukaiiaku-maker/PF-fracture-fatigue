#!/usr/bin/env python3
"""Canonical FEM confirmation for the prospectively selected natural protocol."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arrhenius_fracture.bulk_plasticity_v104 import PlasticityScope, accepted_bulk_load_step
from scripts.run_active_two_mechanism_natural_void_study import (
    TEMPERATURE_K, canonical_drive_check, finite, make_initial,
)


def main():
    out = ROOT/"artifacts/informative_natural_void_selected_protocol"
    out.mkdir(parents=True, exist_ok=True)
    initial = make_initial("combined")
    mechanical, plastic = accepted_bulk_load_step(
        initial, scope=PlasticityScope.FULL_FIELD_PT_EVOLVING_RHO,
        load_factor=6.0, duration_s=1.0e-3,
        temperature_K=TEMPERATURE_K, maximum_stagger_iterations=80,
    )
    _, K, canonical, _ = canonical_drive_check(mechanical)
    payload = finite({
        "selection_frozen_before_confirmation": True,
        "vacancy_energy_variant": "combined", "material_family": "DBTT",
        "temperature_K": TEMPERATURE_K, "load_factor": 6.0,
        "duration_s": 1.0e-3,
        "plasticity": plastic,
        "canonical_state_bound_K_effective_Pa_sqrt_m": K,
        "canonical_drive": canonical,
    })
    (out/"canonical_confirmation.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True)+"\n")
    print("SELECTED_NATURAL_CANONICAL_CONFIRMATION", canonical["interpretation"], K)


if __name__ == "__main__":
    main()
