from __future__ import annotations

from dataclasses import replace
import subprocess
import sys

import numpy as np
import pytest

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import (
    complete_accepted_state_fingerprint, equilibrate_fixed_load_with_production_fem,
)
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.vacancy_transport_phase_field_v2 import (
    VacancyTransportStateV2, accepted_elastoplastic_stress_3d,
)
from arrhenius_fracture.voiding_production_v5 import (
    advance_production_vacancy_transport_interval, build_production_void_state,
)
from arrhenius_fracture.embryo_growth_v1 import transport_flux_certificate


def production_pair():
    accepted, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0,
    )
    previous = equilibrate_fixed_load_with_production_fem(
        replace(accepted, ep_gp=np.zeros_like(accepted.ep_gp))
    )
    return replace(previous, vacancy_transport_state_v2=accepted.vacancy_transport_state_v2), accepted


def test_transport_is_part_of_isolated_accepted_state_and_main_checkpoint(tmp_path):
    _, state = production_pair()
    assert isinstance(state.vacancy_transport_state_v2, VacancyTransportStateV2)
    isolated = state.isolated_copy()
    assert isolated.vacancy_transport_state_v2 is state.vacancy_transport_state_v2
    assert complete_accepted_state_fingerprint(isolated) == complete_accepted_state_fingerprint(state)
    path = tmp_path/"accepted.json"
    write_checkpoint(state, path)
    restored = restore_checkpoint(path)
    assert complete_accepted_state_fingerprint(restored) == complete_accepted_state_fingerprint(state)
    assert np.array_equal(restored.vacancy_transport_state_v2.vacancy_fraction_gp,
                          state.vacancy_transport_state_v2.vacancy_fraction_gp)


def test_energy_variant_and_distributed_sites_survive_main_checkpoint(tmp_path):
    state, _ = build_production_void_state(
        bundle=material_bundle("DBTT"), temperature_K=900.0,
        vacancy_energy_variant="combined",
        void_site_centers_m=((6.0e-4, 6.0e-5), (7.5e-4, -6.0e-5)),
        stochastic=True, seed=3621)
    transport = state.vacancy_transport_state_v2
    assert transport.parameter_row.vacancy_formation_energy_eV == 2.4
    assert transport.parameter_row.vacancy_migration_energy_eV == 0.9
    assert [site.site_id for site in state.void_state.sites] == ["site-1", "site-2"]
    assert [site.site_id for site in transport.site_states] == ["site-1", "site-2"]
    path = tmp_path/"combined-two-site.json"
    write_checkpoint(state, path)
    restored = restore_checkpoint(path)
    assert complete_accepted_state_fingerprint(restored) == complete_accepted_state_fingerprint(state)
    assert restored.vacancy_transport_state_v2.parameter_row_fingerprint == \
        transport.parameter_row_fingerprint
    assert [site.site_id for site in restored.vacancy_transport_state_v2.site_states] == \
        ["site-1", "site-2"]


def test_production_adapter_uses_measured_increment_and_full_3d_stress():
    previous, accepted = production_pair()
    result, audit = advance_production_vacancy_transport_interval(
        previous, accepted, duration_s=1.0,
    )
    assert audit["accepted_plastic_increment_rate_max_s"] > 0.0
    assert audit["mean_stress_source"] == "accepted_3d_elastoplastic_constitutive_state"
    stress = accepted_elastoplastic_stress_3d(accepted)
    assert np.all(np.isfinite(stress["sigma_zz_Pa"]))
    assert result.vacancy_transport_state_v2.ledgers["plastic_production_area_m2"] > 0.0


def test_forged_and_stale_transport_certificates_are_rejected():
    previous, accepted = production_pair()
    result, _ = advance_production_vacancy_transport_interval(
        previous, accepted, duration_s=1.0e-6)
    before = previous.vacancy_transport_state_v2
    after = result.vacancy_transport_state_v2
    certificate = transport_flux_certificate(
        before, after, owner_id="site-1", owner_kind="SUBGRID_SITE")
    certificate.verify(before, after)
    forged = replace(certificate, transport_before_fingerprint="forged")
    with pytest.raises(RuntimeError, match="forged or stale"):
        forged.verify(before, after)
    later, _ = advance_production_vacancy_transport_interval(
        result, result, duration_s=1.0e-6)
    with pytest.raises(RuntimeError, match="forged or stale"):
        certificate.verify(before, later.vacancy_transport_state_v2)


def test_production_terminal_reports_required_contract():
    completed = subprocess.run(
        [sys.executable, "scripts/run_production_vacancy_transport_one_void.py"],
        check=True, capture_output=True, text=True,
    )
    required = (
        "MANUFACTURED_PLASTIC_INCREMENT_ADAPTER_CONTROL",
        "VACANCY_TRANSPORT_STATE_OWNED_BY_FRACTURE_STATE",
        "FREE_SURFACE_AND_CRACK_SINKS_ACTIVE",
        "MEASURED_ZERO_CROSS_CRACK_TRANSPORT",
        "CONTROLLED_ACTIVE_CAVITY_PLUMBING_SENTINEL",
        "GLOBAL_EVENT_COMPETITION",
        "CHECKPOINT_RESTART", "ROLLBACK",
        "PRODUCTION_VACANCY_TRANSPORT_ONE_VOID = PASS",
    )
    assert all(item in completed.stdout for item in required)
    assert completed.stdout.rstrip().endswith((
        "DIRECT_CRACK_BEFORE_VOID", "VOID_BIRTH_BEFORE_CRACK", "HEALED_EMBRYO",
        "EMISSION_MEDIATED_DELAY", "INVENTORY_LIMITED",
        "HAZARD_LIMITED", "NO_EVENT_WITHIN_PHYSICAL_WINDOW",
    ))
