from pathlib import Path

import numpy as np

from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.qste340tm_restored_binding import (
    PlasticCoordinates,
    build_engine,
    load_parent_binding,
    material_binding,
)
from arrhenius_fracture.reduced_persistent_reference_r2 import TAYLOR_LAW_ID, runtime_receipt


ROOT = Path(__file__).resolve().parents[1]
ALT = ROOT / "integration_evidence/r10_alternative_gstar_joint_search/alternative_gstar_registry.json"
PARENT = Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/v3-fatigue-informed-joint-barrier-search/analysis_outputs/v3_fatigue_informed_joint_barrier_search/v3_complete_candidate_registry.json")


def _engine():
    parent = load_parent_binding(ALT, PARENT, candidate_id="R10_ALT_GSTAR_02506")
    binding = material_binding(parent, PlasticCoordinates(2.0, 1.2e9, 1.8, 1.8))
    return build_engine(binding, kernel_family_path=R4_KERNEL_FAMILY)


def test_restored_qste_binding_uses_post_install_runtime_not_legacy_fixture():
    receipt = runtime_receipt(_engine(), stage="test")
    assert receipt["taylor_contract"]["law_id"] == TAYLOR_LAW_ID
    assert receipt["taylor_contract"]["finite_window_s"] is None
    assert receipt["source_lifecycle"] == "persistent_reusable"
    assert receipt["source_geometry"]["multiplicity_per_system"] > 1.0
    assert receipt["reversible_binding"]["all_expected_methods_bound"] is True


def test_steel_mechanics_and_density_order_are_preserved_together():
    engine = _engine()
    assert engine.G == 80.0e9
    assert engine.nu == 0.30
    assert engine.b == 2.48e-10
    assert engine.manifest.taylor_corr_scale > 0.0
    assert engine.manifest.retained_recovery_rate_s == 0.0
    rates = engine.mpz._transport_rates(
        np.full(engine.mpz.n_bins, 1.0e9),
        np.full(engine.mpz.n_bins, 5.0e12),
        298.15,
        engine.b,
    )
    assert np.all(rates["taylor_single"] >= rates["taylor"])
    assert np.all(rates["m"] > 1.0)
