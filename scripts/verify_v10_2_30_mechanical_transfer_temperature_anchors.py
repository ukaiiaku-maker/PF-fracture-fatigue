#!/usr/bin/env python3
"""Strict verifier for the archive-only transfer/temperature preflight bundle."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/mechanical_transfer_temperature_anchors_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    decision = json.loads((OUT / "campaign_decision.json").read_text())
    reconstruction = pd.read_parquet(OUT / "spatial_mechanical_transfer_reconstruction.parquet")
    coverage = pd.read_csv(OUT / "mechanical_transfer_coverage.csv")
    inventory = pd.read_csv(OUT / "archived_local_history_inventory_and_decomposition.csv")
    passage = pd.read_csv(OUT / "archived_local_history_first_passage.csv")
    state = pd.read_parquet(OUT / "transient_state_semantic_comparison.parquet")
    monotonic = pd.read_csv(OUT / "named_row_monotonic_temperature_screen.csv")
    preflight = pd.read_csv(OUT / "bounded_temperature_fatigue_anchor_preflight.csv")
    canonical_monotonic = pd.read_csv(OUT / "canonical_monotonic_temperature_screen.csv")
    canonical_preflight = pd.read_csv(OUT / "canonical_temperature_fatigue_anchor_preflight.csv")

    require(len(reconstruction) == 1780, "unexpected spatial reconstruction row count")
    require(reconstruction.result_id.is_unique, "spatial result IDs are not unique")
    require(len(coverage) == reconstruction.result_kind.nunique(), "coverage is incomplete")
    for source in decision["archive_sources"]:
        path = Path(source["source_file"])
        require(path.is_file(), f"missing source {path}")
        require(sha256(path) == source["source_sha256"], f"source hash drift {path}")
    energetic = reconstruction.dropna(subset=[
        "J_front_J_per_m2", "K_J_front_MPa_sqrt_m", "E_effective_prime_Pa"
    ])
    reconstructed_K = np.sqrt(
        energetic.J_front_J_per_m2 * energetic.E_effective_prime_Pa
    ) * 1e-6
    require(
        np.allclose(reconstructed_K, energetic.K_J_front_MPa_sqrt_m, rtol=2e-12, atol=1e-12),
        "J/Eprime/KJ reconstruction does not close",
    )
    require((reconstruction.constraint_convention == "PLANE_STRAIN").all(), "constraint drift")
    require(not reconstruction.contour_or_domain_stability.isna().any(), "missing contour/domain status")

    qualified = inventory[inventory.history_status == "ARCHIVED_LOCAL_K_HISTORY_DRIVEN_NO_FIT"]
    require(len(qualified) == 29, "unexpected exact local-history count")
    require((inventory.history_status == "UNAVAILABLE_FAIL_CLOSED").sum() == 6, "missing histories not failed closed")
    require(len(passage) == 87 and set(passage.level) == {"F0", "F1", "F2"}, "first-passage hierarchy incomplete")
    require(passage.no_transfer_coefficient_fitted.all(), "a transfer fit was admitted")
    reached = passage.groupby("level").predicted_first_passage_reached_by_physical_event.sum().to_dict()
    require(reached == {"F0": 29, "F1": 16, "F2": 0}, f"unexpected first-passage result {reached}")
    require(set(state.level) == {"F1", "F2", "F2B"}, "state hierarchy incomplete")
    require(set(state.state_timing) == {"PRE_FIRST_EVENT_PRE_TRANSLATION"}, "state timing mismatch")
    require(not state.coefficient_fit_to_Kinit.any(), "state coefficient fit to Kinit was admitted")

    require(len(monotonic) == 36, "named-row monotonic grid incomplete")
    require(len(preflight) == 54, "bounded temperature preflight incomplete")
    require(set(preflight.n_bins) == {80}, "temperature anchors must use the qualified n80 kernel")
    require(preflight.registry_role.nunique() == 6, "named/optional control row missing")
    require(preflight.groupby(["registry_role", "temperature_K"]).size().eq(3).all(), "Kmax triplets incomplete")
    require(int(preflight.asymptotic_gate_passed.sum()) == 54, "asymptotic gate population drift")
    require(int(preflight.accessibility_gate_passed.sum()) == 40, "named accessibility population drift")
    require(not preflight.current_production_row_complete.any(), "legacy named row was silently completed")
    require(preflight.current_persistent_site_density_m2.isna().all(), "persistent density was fabricated")
    require(int(preflight.preflight_launch_eligible.sum()) == 0, "incomplete row admitted to production")
    require(not preflight.physical_launch_status.str.startswith("LAUNCHED").any(), "unexpected physical launch")
    require((~preflight.fixed_point_converged).all(), "stationary state result unexpectedly promoted")
    asymptotic = preflight[preflight.asymptotic_gate_passed]
    require((asymptotic.cooperative_ceiling_fraction < .95).all(), "ceiling-dominated point admitted")
    require((asymptotic.phase_fraction_near_barrier_floor < .5).all(), "floor-dominated point admitted")
    require((asymptotic.phase_fraction_at_stress_cap < .5).all(), "stress-cap-dominated point admitted")
    require(
        set(preflight[~preflight.accessibility_gate_passed].prospective_accessibility_class)
        == {"EXPECTED_PHYSICAL_CYCLE_CENSOR"},
        "sub-onset point was not retained as a prospective censor",
    )
    legacy = Path(decision["named_row_source_registry"])
    require(legacy.is_file(), "named-row source registry missing")
    require(sha256(legacy) == decision["named_row_source_registry_sha256"], "named-row source hash drift")
    canonical = Path(decision["canonical_row_source_registry"])
    require(canonical.is_file(), "canonical-row source registry missing")
    require(sha256(canonical) == decision["canonical_row_source_registry_sha256"], "canonical-row source hash drift")
    require(len(canonical_monotonic) == 24, "canonical monotonic grid incomplete")
    require(len(canonical_preflight) == 36, "canonical temperature preflight incomplete")
    require(canonical_preflight.registry_role.nunique() == 4, "canonical class missing")
    require(canonical_preflight.groupby(["registry_role", "temperature_K"]).size().eq(3).all(), "canonical Kmax triplets incomplete")
    require(set(canonical_preflight.n_bins) == {80}, "canonical anchors must use n80")
    require(canonical_preflight.current_production_row_complete.all(), "canonical production row incomplete")
    require(int(canonical_preflight.asymptotic_gate_passed.sum()) == 36, "canonical asymptotic population drift")
    require(int(canonical_preflight.accessibility_gate_passed.sum()) == 25, "canonical accessibility population drift")
    require(int(canonical_preflight.preflight_launch_eligible.sum()) == 36, "canonical launch population drift")

    require(decision["Kinit_transfer_fit_performed"] is False, "Kinit transfer fit present")
    require(decision["new_PF_or_FEM_calculations"] is False, "new spatial calculation present")
    require(decision["new_fatigue_trajectories"] is False, "new fatigue trajectory present")
    require(decision["FEM_mechanical_transfer_status"].endswith("FULL_LOCAL_TENSOR_HISTORY_UNAVAILABLE"), "FEM gap hidden")
    for figure in (
        "PF_APPLIED_VERSUS_SOURCE_LOCAL_K.png",
        "TEMPERATURE_FATIGUE_ANCHOR_PREFLIGHT.png",
    ):
        require((OUT / "figures" / figure).stat().st_size > 1000, f"missing figure {figure}")
    print(json.dumps({
        "status": "PASS", "spatial_rows": len(reconstruction),
        "qualified_histories": len(qualified), "asymptotically_admissible_anchors": len(asymptotic),
        "eligible_temperature_anchors": 36,
        "physical_launches": 0,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        raise
