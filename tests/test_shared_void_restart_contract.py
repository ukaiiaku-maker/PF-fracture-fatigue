import pytest

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import build_production_void_state, observables
from arrhenius_fracture.shared_v5_cavity_adapter import _common_start_audit


def test_void_off_on_and_current_checkpoint_round_trip(tmp_path):
    bundle = material_bundle("DBTT")
    disabled, _ = build_production_void_state(bundle=bundle, enabled=False, seed=3621)
    enabled, _ = build_production_void_state(bundle=bundle, enabled=True, seed=3621)
    assert disabled.void_state is None
    assert enabled.void_state is not None
    assert observables(enabled, "initial")["site_phase"] == "AVAILABLE_SITE"
    before = complete_accepted_state_fingerprint(enabled)
    manifest = write_checkpoint(enabled, tmp_path / "void_state.json", compression="gzip")
    restored = restore_checkpoint(tmp_path / "void_state.json")
    assert manifest["has_production_void_state"] is True
    assert complete_accepted_state_fingerprint(restored) == before
    published = (tmp_path / "void_state.json").read_bytes()
    with pytest.raises(RuntimeError, match="injected"):
        write_checkpoint(
            enabled, tmp_path / "void_state.json", compression="gzip",
            failure_injector=lambda *_: (_ for _ in ()).throw(RuntimeError("injected")),
        )
    assert (tmp_path / "void_state.json").read_bytes() == published
    assert complete_accepted_state_fingerprint(
        restore_checkpoint(tmp_path / "void_state.json")
    ) == before


def test_physical_pair_has_exact_common_nonvoid_start_and_separate_cavity_owner():
    bundle = material_bundle("DBTT")
    common = dict(
        bundle=bundle, seed=1733, temperature_K=1200.0, stochastic=True,
        void_site_centers_m=((6e-4, 6e-5), (7.5e-4, -6e-5)),
        direct_interaction_birth_enabled=True,
        cavity_surface_loop_emission_enabled=True,
        interaction_population_partition=(0.5, 0.1, 0.4),
        bulk_initial_density_m2=float(bundle.fracture_row["rho_forest_floor_m2"]),
        initial_plastic_strain_gp=0.0,
    )
    disabled, _ = build_production_void_state(enabled=False, **common)
    enabled, _ = build_production_void_state(enabled=True, **common)
    assert all(_common_start_audit(disabled, enabled).values())
