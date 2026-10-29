from pathlib import Path

import numpy as np

from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.qste340tm_restored_binding import (
    PlasticCoordinates,
    build_engine,
    load_parent_binding,
    material_binding,
    material_binding_with_opening_surface,
)
from arrhenius_fracture.qste340tm_common_envelope import (
    CommonEnvelopeCoordinates, CommonEnvelopeSurface,
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


def test_common_envelope_replaces_only_complete_opening_provider():
    parent = load_parent_binding(ALT, PARENT, candidate_id="R10_ALT_GSTAR_02506")
    plastic = PlasticCoordinates(2.0, 1.2e9, 1.8, 1.8)
    model_a = material_binding(parent, plastic)
    surface = CommonEnvelopeSurface(
        process="cleavage", attempt_frequency_s=1.0e12,
        coordinates=CommonEnvelopeCoordinates(.62, .4, 4.0e9, 40.0e9),
        hits=3.273241435177624, tau_s=6.992153587194454e-7,
    )
    model_b = material_binding_with_opening_surface(
        parent, plastic, surface, family="QSTE340TM_COMMON_ENVELOPE",
        model_id="qste340tm.common-envelope-reference-temperature/1",
        opening_family="MODEL_B_COMMON_THREE_HALVES_ENVELOPE",
        opening_provenance=surface.serialize(),
    )
    assert model_b.manifest.emission == model_a.manifest.emission
    assert model_b.manifest.peierls == model_a.manifest.peierls
    assert model_b.manifest.taylor == model_a.manifest.taylor
    assert model_b.persistent_sites == model_a.persistent_sites
    engine = build_engine(model_b, kernel_family_path=R4_KERNEL_FAMILY)
    effective, raw, barrier_J = engine.lambda_cleave(10.0e9, 298.15)
    assert 0.0 < effective < raw
    assert barrier_J > 0.0
    diagnostic = engine.cleavage_diagnostics(10.0e9, 298.15)
    assert np.isnan(diagnostic["S_cleave_kB"])
    assert diagnostic["dGcleave_dsigma_eV_per_GPa"] < 0.0
    assert engine._r10_final_bound_implementation_identity.endswith("/1")
