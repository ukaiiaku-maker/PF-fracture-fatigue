from pathlib import Path
from types import SimpleNamespace

import pytest

from arrhenius_fracture.prospective_campaign_v3 import REGISTRY, build_cases_v3
from arrhenius_fracture.shared_v5_cavity_adapter import _requested_bundle
from arrhenius_fracture.unified_fracture_material_v5 import (
    active_field_mapping,
    material_bundle,
    material_manifest,
    mpz_config,
    registry_material_bundle,
)


ROOT = Path(__file__).resolve().parents[1]
OPTION = "v913_paper_dbtt01_0202500_persistent_sites__taylor_margin_0p10eV"


def _value(args, flag):
    return args[args.index(flag) + 1]


def test_current_registry_row_is_the_consumed_v5_fracture_owner():
    current = registry_material_bundle(ROOT / REGISTRY, OPTION, material_class="DBTT")
    historical = material_bundle("DBTT")

    assert float(current.fracture_row["taylor_H0_eV"]) == pytest.approx(2.3127889714162735)
    assert float(historical.fracture_row["taylor_H0_eV"]) == pytest.approx(1.4998139148950578)
    assert current.fracture_material_row_id == current.fracture_row["candidate_id"]
    assert current.void_kinetics_row_id == historical.void_kinetics_row_id
    assert current.elastic_row_id == historical.elastic_row_id
    assert current.site_population_row_id == historical.site_population_row_id
    assert material_manifest(current).max_K_shield_MPa_sqrt_m == 0.0
    assert mpz_config(current).n_bins == 80
    assert mpz_config(current).n_systems == 2
    assert mpz_config(current).source_bin_count == 3
    assert set(current.fracture_row) - {
        "material_class", "candidate_id", "source_or_search_provenance",
        "parameter_decision", "reduced_selection_status",
        "same_material_row_both_providers", "option_key", "role",
        "mechanism_summary", "validation_status", "qualification_parent_option_key",
        "qualification_taylor_construction", "target_class",
        "legacy_source_sites_active", "legacy_source_refresh_active",
        "explicit_recovery_active", "mobile_shield_fraction", "source_recovery_rate_s",
        "retained_recovery_rate_s", "source_refresh_length_um", "recovery_nu0_s",
        "recovery_H0_eV", "recovery_activation_entropy_kB",
    } <= set(active_field_mapping(current))


def test_adapter_requires_explicit_current_identity_or_named_historical_mode():
    args = SimpleNamespace(
        material_registry=str(ROOT / REGISTRY), parameter_option=OPTION,
        historical_v5_material=False, material_class="DBTT",
    )
    assert _requested_bundle(args).fracture_row["option_key"] == OPTION
    args.parameter_option = None
    with pytest.raises(SystemExit, match="requires --material-registry"):
        _requested_bundle(args)
    args.material_registry = None
    args.historical_v5_material = True
    assert _requested_bundle(args).fracture_material_row_id == "v913_zeroD_sobol_0202500"


def test_p7_primary_requests_reach_physical_endpoint_and_compact_runs_are_separate():
    rows = [row for row in build_cases_v3(ROOT) if row["block"] == "P7"]
    assert len(rows) == 44
    assert len({row["paired_execution_id"] for row in rows}) == 22
    for row in rows:
        args = row["native_args"]
        assert _value(args, "--parameter-option") == row["parameter_option"]
        assert Path(_value(args, "--material-registry")).resolve() == (ROOT / REGISTRY).resolve()
        assert float(_value(args, "--target-extension-um")) == row["target_extension_um"]
        assert float(_value(args, "--exposure-limit-s")) == pytest.approx(
            row["physical_endpoint"]["no_event_exposure_ceiling_s"]
        )
        assert "--steps" not in args
        qualifier = row["qualification_execution"]
        assert qualifier["physical_exposure_s"] == 0.2
        assert qualifier["production_case_substitution"] is False
        assert _value(qualifier["launcher"], "--steps") == "1"
