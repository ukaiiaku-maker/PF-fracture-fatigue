from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from arrhenius_fracture.adaptive_multitip_mesh_v11 import refine_accepted_state
from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.fem import assemble_mechanics
from arrhenius_fracture.mesh import rebuild_tri_mesh
from arrhenius_fracture.crack_network_v11 import ROOT_BRANCH_ID
from arrhenius_fracture.sharp_front import build_engine
from arrhenius_fracture.unified_control_matrix_v5 import (
    DELTA_DEFINITIONS, PaperControl, paper_control_matrix,
)
from arrhenius_fracture.unified_fracture_material_v5 import (
    CORE_MODEL_ID, FRACTURE_ROWS, active_field_mapping, all_material_bundles,
    bind_identity, canonical_front_engine, elastic_material, identity_record, material_bundle,
    require_bound_identity,
)
from arrhenius_fracture.voiding_production_v5 import (
    _canonical_front_process_owner, build_production_void_state, crack_tip_tensor,
    capture_sharp_front_engine, directional_clock_rates, fresh_sharp_front_engine,
    remesh_cavity, restore_sharp_front_engine, sharp_front_constitutive_response,
    sharp_front_engine_fingerprint,
)
from arrhenius_fracture.voiding_v5 import (
    ProductionVoidState, VoidPhase, VoidSite, HazardClock,
    create_subgrid_cavity, promote_cavity,
)


FAMILIES = tuple(FRACTURE_ROWS)
TEMPERATURES_K = (300.0, 750.0, 1200.0)
STRESSES_PA = (0.5e9, 2.0e9, 5.0e9)


def _material(bundle):
    return elastic_material(bundle)


def _response(engine, stress, temperature):
    cleavage = engine.lambda_cleave(stress, temperature)
    emission = engine.lambda_emit(stress, temperature)
    return np.asarray((cleavage[0], cleavage[1], cleavage[2], emission[0], emission[2]))


def test_01_four_family_barrier_rate_parity_and_one_factory():
    """Single, multi and void drivers consume one bundle and constructor."""
    for family, bundle in all_material_bundles().items():
        material = _material(bundle)
        single = build_engine(SimpleNamespace(unified_material_bundle=bundle), material)
        multi = build_engine(SimpleNamespace(unified_material_bundle=bundle), material)
        void = fresh_sharp_front_engine(material, bundle)
        assert type(single) is type(multi) is type(void)
        assert single._unified_bundle_id == multi._unified_bundle_id == void._unified_bundle_id
        assert single._unified_core_model_id == CORE_MODEL_ID
        for temperature in TEMPERATURES_K:
            for stress in STRESSES_PA:
                np.testing.assert_array_equal(_response(single, stress, temperature), _response(multi, stress, temperature))
                np.testing.assert_array_equal(_response(single, stress, temperature), _response(void, stress, temperature))
                assert single.cleavage_diagnostics(stress, temperature) == multi.cleavage_diagnostics(stress, temperature) == void.cleavage_diagnostics(stress, temperature)
                ds = 1.0e5
                emission_derivatives = tuple(
                    (float(engine.manifest.emission.values_eV(stress + ds, temperature))
                     - float(engine.manifest.emission.values_eV(stress - ds, temperature))) / (2.0 * ds)
                    for engine in (single, multi, void)
                )
                assert emission_derivatives[0] == emission_derivatives[1] == emission_derivatives[2]
        assert single.r_eff() == multi.r_eff() == void.r_eff()
        assert single.sigma_back() == multi.sigma_back() == void.sigma_back() == 0.0
        assert single.e_stored() == multi.e_stored() == void.e_stored() == 0.0
        for engine in (single, multi, void):
            engine.B = 0.999
        renewal = [engine.step(2.0e6, 750.0, 1.0e-12) for engine in (single, multi, void)]
        for key in ("n_fire", "B", "N_em", "r_eff", "dB_step", "peierls_rate_s", "taylor_completion_rate_s"):
            assert renewal[0][key] == renewal[1][key] == renewal[2][key]
        for name in ("available_sites", "mobile", "retained", "accumulated_slip", "wake_mobile", "wake_retained"):
            np.testing.assert_array_equal(getattr(single.mpz, name), getattr(multi.mpz, name))
            np.testing.assert_array_equal(getattr(single.mpz, name), getattr(void.mpz, name))
        assert active_field_mapping(bundle)
        assert single.manifest.candidate_id == FRACTURE_ROWS[family]


def test_02_no_default_fallback_and_exact_four_rows():
    assert tuple(all_material_bundles()) == FAMILIES
    assert len({bundle.bundle_hash for bundle in all_material_bundles().values()}) == 4
    with pytest.raises(TypeError, match="bundle"):
        build_production_void_state()
    with pytest.raises(ValueError, match="NO_DEFAULT_FALLBACK"):
        build_engine(SimpleNamespace(unified_scientific_execution=True), elastic_material(material_bundle("Peak")))
    with pytest.raises(KeyError, match="unknown unified material family"):
        material_bundle("not-a-family")


@pytest.mark.parametrize("family", FAMILIES)
def test_03_no_void_single_tip_and_multitip_one_front_sentinel(family):
    bundle = material_bundle(family)
    state, hole = build_production_void_state(bundle=bundle, enabled=False)
    assert state.void_state is None
    assert state.crack_network.active_tip_ids == (ROOT_BRANCH_ID,)
    multi_network = replace(state.crack_network, branching_enabled=True)
    multi = replace(state, crack_network=multi_network)
    assert multi.crack_network.active_tip_ids == state.crack_network.active_tip_ids
    for name in ("elasticity_D", "ep_gp", "rho_gp", "damage", "displacement"):
        np.testing.assert_array_equal(getattr(multi, name), getattr(state, name))
    assert multi.stored_energy_J_per_m == state.stored_energy_J_per_m
    assert dict(require_bound_identity(multi)) == dict(require_bound_identity(state))
    engine = restore_sharp_front_engine(
        state.material, bundle, state.tip_process_state["by_branch"][ROOT_BRANCH_ID],
    )
    assert engine.f.r0 != hole.radius_m


def test_04_void_disabled_no_active_and_far_inactive_limits_are_mechanics_exact():
    bundle = material_bundle("Peak")
    no_void, _ = build_production_void_state(bundle=bundle, enabled=False)
    empty = replace(no_void, void_state=ProductionVoidState(()))
    far_site = VoidSite(
        "far-site", (0.999e-3, 0.499e-3), VoidPhase.AVAILABLE_SITE, 0, 2, 0.8,
        HazardClock(0.0, 2.0), HazardClock(0.0, 2.0), HazardClock(0.0, 2.0),
    )
    far = replace(no_void, void_state=ProductionVoidState((far_site,)))
    for other in (empty, far):
        for name in ("elasticity_D", "ep_gp", "rho_gp", "damage", "displacement"):
            np.testing.assert_array_equal(getattr(other, name), getattr(no_void, name))
        assert other.stored_energy_J_per_m == no_void.stored_energy_J_per_m
        assert dict(require_bound_identity(other)) == dict(require_bound_identity(no_void))


def _resolved_far_void_error(bundle, center_x):
    path = ((0.0, 0.0), (0.0005725993004046688, 0.0))
    no_void, hole = build_production_void_state(
        bundle=bundle, enabled=True, cavity_center_m=(center_x, 0.0), crack_path_m=path,
    )
    site = replace(no_void.void_state.sites[0], phase=VoidPhase.STABLE_SUBGRID_VOID)
    void_state = replace(no_void.void_state, sites=(site,))
    void_state = create_subgrid_cavity(void_state, site.site_id, 5.0e-5)
    void_state = promote_cavity(void_state, void_state.cavities[0].cavity_id, 5.0e-5)
    resolved = remesh_cavity(no_void, hole, void_state, "bounded-far-void-limit")

    def reaction(state):
        _, residual, *_ = assemble_mechanics(
            state.mesh, state.displacement, state.ep_gp, state.rho_gp, state.damage,
            state.elasticity_D, state.material, cohesive_network=state.cohesive_network,
        )
        return abs(float(np.sum(residual[2 * np.asarray(state.boundary.top_nodes) + 1])))

    base_reaction = reaction(no_void)
    void_reaction = reaction(resolved)
    opening = 4.0e-7
    base_tensor, _ = crack_tip_tensor(no_void, branch_id=ROOT_BRANCH_ID)
    void_tensor, _ = crack_tip_tensor(resolved, branch_id=ROOT_BRANCH_ID)
    base_drive = directional_clock_rates(no_void, base_tensor)[0]["rate_s"]
    void_drive = directional_clock_rates(resolved, void_tensor)[0]["rate_s"]
    cavity = resolved.void_state.cavities[0]
    tip = np.asarray(resolved.crack_network.branch(ROOT_BRANCH_ID).tip)
    return {
        "separation_m": float(np.linalg.norm(np.asarray(cavity.center_m) - tip) - cavity.radius_m),
        "reaction": abs(void_reaction - base_reaction) / base_reaction,
        "compliance": abs(opening / void_reaction - opening / base_reaction) / (opening / base_reaction),
        "energy": abs(resolved.stored_energy_J_per_m - no_void.stored_energy_J_per_m) / no_void.stored_energy_J_per_m,
        "tensor": float(np.linalg.norm(void_tensor - base_tensor) / np.linalg.norm(base_tensor)),
        "candidate_drive": abs(void_drive - base_drive) / base_drive,
    }


def test_05_resolved_far_void_converges_for_all_required_observables():
    bundle = material_bundle("Peak")
    near = _resolved_far_void_error(bundle, 0.00070)
    far = _resolved_far_void_error(bundle, 0.00075)
    assert near["separation_m"] > canonical_front_engine(bundle, _material(bundle)).f.L_pz
    assert far["separation_m"] > near["separation_m"]
    for observable in ("reaction", "compliance", "energy", "tensor", "candidate_drive"):
        assert far[observable] < near[observable]


def test_06_root_branch_child_material_and_process_owner_identity():
    bundle = material_bundle("DBTT")
    state, _ = build_production_void_state(bundle=bundle, enabled=False)
    owner = _canonical_front_process_owner(state, ROOT_BRANCH_ID)
    assert owner == state.tip_process_state["owner_by_front"][ROOT_BRANCH_ID]
    root = restore_sharp_front_engine(state.material, bundle, state.tip_process_state["by_branch"][owner])
    branch = canonical_front_engine(bundle, state.material)
    child = fresh_sharp_front_engine(state.material, bundle)
    assert root._unified_bundle_hash == branch._unified_bundle_hash == child._unified_bundle_hash
    assert sharp_front_engine_fingerprint(branch) == sharp_front_engine_fingerprint(child)
    with pytest.raises(ValueError, match="material/core identity"):
        restore_sharp_front_engine(state.material, material_bundle("Peak"), capture_sharp_front_engine(child))


def test_07_checkpoint_and_remesh_preserve_complete_material_identity(tmp_path):
    bundle = material_bundle("weak-T")
    state, _ = build_production_void_state(bundle=bundle, enabled=False)
    before = dict(require_bound_identity(state))
    refined, _ = refine_accepted_state(
        state, marked_parent_elements=(0,), active_tip_ids=state.crack_network.active_tip_ids,
        generation=1, operation_index=1,
    )
    assert dict(require_bound_identity(refined)) == before
    target = tmp_path / "unified.json"
    manifest = write_checkpoint(refined, target)
    restored = restore_checkpoint(target)
    assert dict(require_bound_identity(restored)) == before
    assert manifest["has_production_void_state"] is False
    engine = restore_sharp_front_engine(
        restored.material, bundle, restored.tip_process_state["by_branch"][ROOT_BRANCH_ID],
    )
    original = restore_sharp_front_engine(
        state.material, bundle, state.tip_process_state["by_branch"][ROOT_BRANCH_ID],
    )
    assert sharp_front_engine_fingerprint(engine) == sharp_front_engine_fingerprint(original)


def test_08_cross_material_cache_identity_cannot_alias():
    identities = {
        family: dict(identity_record(bundle, _material(bundle)))
        for family, bundle in all_material_bundles().items()
    }
    for key in ("material_bundle_id", "material_bundle_sha256", "plasticity_fingerprint", "cleavage_barrier_fingerprint", "emission_barrier_fingerprint"):
        assert len({identity[key] for identity in identities.values()}) == 4
    responses = {
        family: tuple(_response(canonical_front_engine(bundle, _material(bundle)), 3.0e9, 750.0))
        for family, bundle in all_material_bundles().items()
    }
    assert len(set(responses.values())) == 4


def test_09_paper_control_matrix_and_delta_definitions_are_closed():
    for bundle in all_material_bundles().values():
        controls = paper_control_matrix(bundle)
        assert set(controls) == set(PaperControl)
        assert controls[PaperControl.SINGLE_TIP].branching_enabled is False
        assert controls[PaperControl.SINGLE_TIP].voiding_enabled is False
        assert controls[PaperControl.MULTI_TIP].branching_enabled is True
        assert controls[PaperControl.MULTI_TIP].voiding_enabled is False
        assert controls[PaperControl.MULTI_TIP_PLUS_VOIDING].voiding_enabled is True
        assert len({(row.composite_identity["material_bundle_id"], row.composite_identity["extensions_enabled"]) for row in controls.values()}) == 3
        hashes = {row.front_engine(_material(bundle))._unified_bundle_hash for row in controls.values()}
        assert hashes == {bundle.bundle_hash}
    assert DELTA_DEFINITIONS == {
        "branching_increment": "O[MULTI_TIP] - O[SINGLE_TIP]",
        "voiding_increment": "O[MULTI_TIP_PLUS_VOIDING] - O[MULTI_TIP]",
        "combined_increment": "O[MULTI_TIP_PLUS_VOIDING] - O[SINGLE_TIP]",
    }


def test_10_reports_encode_terminal_passes_and_no_unresolved_core_divergence():
    import json
    root = Path(__file__).resolve().parents[1]
    lineage = json.loads((root / "v5_unified_model_lineage.json").read_text())
    delta = json.loads((root / "v5_voiding_extension_delta.json").read_text())
    assert lineage["decision"] == "TRANSPLANT_VOID_EXTENSION_ONTO_QUALIFIED_MULTITIP_CORE"
    assert lineage["unresolved_unintended_core_divergence"] == []
    assert all(value == "PASS" for value in lineage["terminal_gates"].values())
    assert delta["decision"] == "VOIDING_EXTENSION_DELTA_ONLY"
    assert delta["scientific_tolerances_changed"] is False
    assert delta["r_tip_equals_void_radius"] is False


def test_11_connected_topology_mesh_rebuild_accepts_no_active_tip_centers():
    state, _ = build_production_void_state(bundle=material_bundle("DBTT"), enabled=False)
    rebuilt = rebuild_tri_mesh(state.mesh.nodes, state.mesh.elems, tip_centers=())
    assert rebuilt.hbar_tip == rebuilt.hbar
