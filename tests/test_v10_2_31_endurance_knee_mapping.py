import numpy as np
import pytest
import json
import csv

from arrhenius_fracture.material_manifest import MaterialManifest
from arrhenius_fracture.unified_mpz import MPZConfig, UnifiedMPZState
from arrhenius_fracture.persistent_site_source_v10221 import PersistentSiteConfig
from arrhenius_fracture import sharp_front_v10_2_31_endurance_knee as mapped


def test_new_spatial_controls_have_legacy_neutral_defaults():
    cfg=MPZConfig()
    assert cfg.blunting_slip_fraction == 1.0
    assert np.isinf(cfg.taylor_phi_max)
    assert cfg.mobile_transport_velocity_scale == 1.0
    assert PersistentSiteConfig(rho_site0_m2=1e12).backstress_scale == 1.0


def test_explicit_registry_option_is_read_from_stage3_namespace(monkeypatch):
    seen={}
    monkeypatch.setattr(mapped,"_PRODUCTION_MAIN",lambda args:seen.setdefault("args",args))
    monkeypatch.setattr(mapped._registry_entry,"_prepare_option",lambda args:None)
    mapped.main(["--parameter-registry","registry.csv"])
    assert seen["args"] == ["--parameter-registry","registry.csv"]


def test_selection_record_exactly_matches_mapped_option_order():
    payload=json.loads(mapped.SELECTION_RECORD.read_text())
    assert payload["canonical_option_order"] == list(mapped.VALID)
    assert payload["parameter_refit"] is False


def test_transferred_registry_maps_active_encounter_efficiency():
    path="arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv"
    for row in csv.DictReader(open(path)):
        assert float(row["encounter_efficiency"]) == float(row["physics__encounter_efficiency"])
        assert float(row["source_sites_per_system"]) > 0.0
        assert row["legacy_source_sites_active"] == "0"
        assert row["exact_spatial_Tref_active"] == "1"
        assert row["material_class"] == "DBTT"
        assert row["endurance_mechanism_class"] in "ABCD"


def test_manifest_reads_candidate_reference_temperature(tmp_path):
    source="arrhenius_fracture/data/materials/v10_2_31_endurance_knee_ABCD_registry.csv"
    row=next(csv.DictReader(open(source)))
    target=tmp_path/"one.csv"
    with target.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(row));w.writeheader();w.writerow(row)
    manifest=MaterialManifest.from_csv(target)
    assert manifest.cleavage.Tref_K == 300.0
    assert manifest.emission.Tref_K == 300.0


def test_blunting_slip_fraction_maps_multiplicatively():
    manifest=MaterialManifest.from_csv("arrhenius_fracture/data/materials/weakT/spatial_promotion_manifest.csv")
    state=UnifiedMPZState(manifest,MPZConfig(n_bins=4,blunting_slip_fraction=0.25))
    state.accumulated_slip[:]=2.0
    reference=UnifiedMPZState(manifest,MPZConfig(n_bins=4,blunting_slip_fraction=1.0))
    reference.accumulated_slip[:]=2.0
    assert state.local_slip_count() == pytest.approx(0.25*reference.local_slip_count())


def test_transport_scale_does_not_change_encounter_storage_rate():
    manifest=MaterialManifest.from_csv("arrhenius_fracture/data/materials/weakT/spatial_promotion_manifest.csv")
    a=UnifiedMPZState(manifest,MPZConfig(n_bins=4,mobile_transport_velocity_scale=1.0))
    b=UnifiedMPZState(manifest,MPZConfig(n_bins=4,mobile_transport_velocity_scale=0.0))
    stress=np.full(4,1e9); rho=np.full(4,5e12)
    ra=a._transport_rates(stress,rho,300.0,2.74e-10); rb=b._transport_rates(stress,rho,300.0,2.74e-10)
    assert np.allclose(ra["encounter"],rb["encounter"])
    assert np.all(rb["velocity"]==0.0)
