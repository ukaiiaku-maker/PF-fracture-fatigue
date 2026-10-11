"""Bounded, conservative V5 cavity lifecycle fixture for R1 restoration.

This is an initial-condition-driven full-order functional fixture.  It uses the
native V5 birth, stabilization, vacancy debit, growth, promotion, remesh, FEM,
and checkpoint owners with no rate scaling.  It is not a natural-nucleation or
material-calibration claim.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .checkpoint_v11 import restore_checkpoint, write_checkpoint
from .topology_transaction_v11 import complete_accepted_state_fingerprint
from .unified_fracture_material_v5 import material_bundle
from .vacancy_coupling_v1 import vacancy_conservation
from .voiding_production_v5 import deterministic_trajectory


MODEL_ID = "r1.seeded-finite-activity-native-v5/1"


def run_seeded_finite_activity(output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite {root}")
    root.mkdir(parents=True)
    trace = []
    state, rows = deterministic_trajectory(
        bundle=material_bundle("DBTT"),
        stop_before_ligament=True,
        state_trace=trace,
        sentinel_initial_vacancy_fraction_gp=0.01,
        vacancy_coupled=True,
        boundary_segments=32,
        radial_layers=12,
        subgrid_radius_m=1.0e-6,
        promotion_radius_m=2.0e-6,
        resolved_radius_m=3.0e-6,
    )
    cavity = state.void_state.cavities[0]
    conservation = vacancy_conservation(state)
    stages = [str(row["operation"]) for row in rows]
    required = {
        "plastic_vacancy_production", "multi_hit_1", "multi_hit_2",
        "stabilization", "subgrid_void", "subgrid_growth",
        "geometric_promotion", "resolved_plastic_vacancy_production",
        "resolved_growth",
    }
    if not required <= set(stages):
        raise RuntimeError("seeded finite-activity V5 lifecycle is incomplete")
    if not (cavity.radius_m == 3.0e-6 and cavity.geometry_generation > 0):
        raise RuntimeError("seeded finite-activity V5 cavity did not grow and remesh")
    if abs(float(conservation["vacancy_conservation_residual_m2"])) > 1.0e-22:
        raise RuntimeError("seeded finite-activity V5 vacancy conservation failed")

    checkpoint = root / "terminal_checkpoint.json"
    write_checkpoint(state, checkpoint, compression="gzip")
    restored = restore_checkpoint(checkpoint)
    restart_exact = (
        complete_accepted_state_fingerprint(restored)
        == complete_accepted_state_fingerprint(state)
    )
    if not restart_exact:
        raise RuntimeError("seeded finite-activity V5 restart changed accepted state")

    compact_rows = []
    for row in rows:
        compact_rows.append({
            key: row.get(key)
            for key in (
                "operation", "site_phase", "cavity_phase", "cavity_count",
                "cavity_radius_m", "vacancy_fraction_min",
                "vacancy_fraction_max", "mesh_generation",
            )
            if key in row
        })
    payload = {
        "schema": MODEL_ID,
        "classification": "INITIAL_CONDITION_DRIVEN_FULL_ORDER_FUNCTIONAL_FIXTURE",
        "production_eligible_material_evidence": False,
        "natural_nucleation_claim": False,
        "rate_multiplier": 1.0,
        "initial_vacancy_fraction": 0.01,
        "dilute_transport_bound_respected": True,
        "native_owners": [
            "vacancy/plastic production", "multi-hit birth", "stabilization",
            "local conservative vacancy debit", "subgrid growth", "promotion",
            "body-fitted remesh", "resolved growth", "full spatial FEM",
        ],
        "stages": compact_rows,
        "terminal_cavity_radius_m": float(cavity.radius_m),
        "terminal_cavity_area_m2": float(cavity.area_m2),
        "terminal_cavity_geometry_generation": int(cavity.geometry_generation),
        "vacancy_ledgers": dict(state.vacancy_ledgers),
        "vacancy_conservation": conservation,
        "restart_exact": restart_exact,
        "terminal_checkpoint": str(checkpoint),
    }
    for value in conservation.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError("nonfinite seeded V5 conservation output")
    (root / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )
    return payload


__all__ = ["MODEL_ID", "run_seeded_finite_activity"]
