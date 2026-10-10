import pytest

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import (
    _directional_rate_source_identity, build_production_void_state, observables,
)
from arrhenius_fracture.shared_v5_cavity_adapter import (
    TRANSPORT_BOUND_FAILURE, _advance_paired_with_refinement, _common_start_audit,
)


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


def test_paired_transport_refinement_uses_same_accepted_partition_for_both_arms():
    def advance(state, duration, arm):
        if duration > 0.1:
            raise RuntimeError(TRANSPORT_BOUND_FAILURE)
        return state + duration, {"arm": arm, "duration_s": duration}

    terminal, segments = _advance_paired_with_refinement(
        {"disabled": 0.0, "enabled": 0.0}, 0.2, advance,
        minimum_duration_s=1.0e-6,
    )
    assert terminal == pytest.approx({"disabled": 0.2, "enabled": 0.2})
    assert [duration for duration, _, _ in segments] == pytest.approx([0.1, 0.1])
    assert all(set(records) == {"disabled", "enabled"} for _, records, _ in segments)


def test_directional_clock_identity_excludes_numerical_cache_diagnostics():
    physical = {
        "candidate_id": "c1", "process_owner_id": "b0",
        "directional_drive_status": "AVAILABLE", "drive_source": "MARGINAL_G",
        "G_kinetic_used_J_per_m2": 12.0, "signed_J_J_per_m2": -2.0,
        "positive_J_J_per_m2": 12.0, "K_directional_Pa_sqrt_m": 3.0,
        "K_effective_Pa_sqrt_m": 3.0, "resolved_opening_stress_Pa": 4.0,
        "rate_s": 5.0, "effective_rate_s": 5.0,
        "clock_increment_per_second": 5.0,
    }
    first = {**physical, "exact_factorization_cache": {"hits": 1}}
    second = {**physical, "exact_factorization_cache": {"hits": 99}}
    assert _directional_rate_source_identity([first]) == (
        _directional_rate_source_identity([second])
    )


@pytest.mark.parametrize("field,replacement", [
    ("candidate_id", "c2"),
    ("process_owner_id", "b1"),
    ("G_kinetic_used_J_per_m2", 13.0),
    ("signed_J_J_per_m2", -3.0),
    ("K_directional_Pa_sqrt_m", 3.5),
    ("resolved_opening_stress_Pa", 4.5),
    ("rate_s", 6.0),
])
def test_directional_clock_identity_changes_with_physical_inputs(field, replacement):
    physical = {
        "candidate_id": "c1", "process_owner_id": "b0",
        "directional_drive_status": "AVAILABLE", "drive_source": "MARGINAL_G",
        "G_kinetic_used_J_per_m2": 12.0, "signed_J_J_per_m2": -2.0,
        "positive_J_J_per_m2": 12.0, "K_directional_Pa_sqrt_m": 3.0,
        "K_effective_Pa_sqrt_m": 3.0, "resolved_opening_stress_Pa": 4.0,
        "rate_s": 5.0, "effective_rate_s": 5.0,
        "clock_increment_per_second": 5.0,
    }
    changed = {**physical, field: replacement}
    assert _directional_rate_source_identity([physical]) != (
        _directional_rate_source_identity([changed])
    )
