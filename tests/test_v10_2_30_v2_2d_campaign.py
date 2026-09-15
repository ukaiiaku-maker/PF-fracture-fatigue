from __future__ import annotations
import json
from pathlib import Path
import numpy as np

from arrhenius_fracture import sharp_front as driver
from arrhenius_fracture import sharp_front_v10_2_30_v2_named_single_crack as adapter
from arrhenius_fracture.run_state_checkpoint_v10230 import write_combined_checkpoint
from arrhenius_fracture.sparse_accepted_field_export_v10230 import export_latest_checkpoint
from arrhenius_fracture.v2_named_parameterizations import NAMED_ALIASES, load_exact_candidate, load_for


def test_every_named_alias_has_exact_pf_identity():
    for alias in NAMED_ALIASES:
        row = load_for(alias, "PF_sharp_front")
        exact = load_exact_candidate(row.source_candidate_id)
        assert dict(row.full_precision_row) == dict(exact.full_precision_row)
        assert row.complete_bound_row_sha256 == row.full_precision_row["complete_bound_row_sha256"]


def test_adapter_binds_direct_surface_and_exact_renewal(tmp_path, monkeypatch):
    seen = {}
    def fake_main(args):
        registry = Path(args[args.index("--parameter-registry") + 1])
        manifest = driver.MaterialManifest.from_csv(registry)
        seen["candidate"] = manifest.candidate_id
        seen["surface"] = float(manifest.cleavage.values_eV(1.5e9, 1200.0))
        seen["m"] = float(args[args.index("--multihit-m") + 1])
        seen["tau"] = float(args[args.index("--multihit-tau") + 1])
    monkeypatch.setattr(adapter._audited, "main", fake_main)
    out = tmp_path / "case"
    adapter.main(["--v2-parameter-alias", "DBTT_V2", "--out", str(out)])
    row = dict(load_for("DBTT_V2", "PF_sharp_front").full_precision_row)
    assert seen["candidate"] == row["candidate_id"]
    assert seen["m"] == float(row["physics__cleavage_hits"])
    assert seen["tau"] == float(row["physics__cleavage_correlation_time_s"])
    assert np.isfinite(seen["surface"])
    audit = json.loads((out / "v2_exact_parameter_binding.json").read_text())
    assert audit["direct_opening_surface"] is True


def _checkpoint(root, *, step, events, extension):
    nodes=np.array([[0.,0.],[1.,0.],[0.,1.]])
    arrays={"mesh_nodes":nodes,"mesh_elems":np.array([[0,1,2]]),"damage":np.zeros(3),
      "displacement":np.zeros(6),"ep_gp":np.zeros((3,1)),"rho_gp":np.ones(1),
      "pz_store_gp":np.zeros(1),"pz_mobile_gp":np.zeros(1),"pz_escape_gp":np.zeros(1),
      "pz_emit_gp":np.zeros(1),"sigma_gp":np.array([[1.],[2.],[.1]]),"sigma1_gp":np.array([2.]),
      "von_mises_gp":np.array([1.]),"elastic_energy_density_gp":np.array([.2]),"plastic_rate_gp":np.zeros(1)}
    outer={"case":{},"driver":{"step":step,"a_tip":extension,"crack_extension_start_a":0.},
      "geometry":{"committed_event_count":events,"front_paths":[[[0.,0.],[extension,0.]]],"crack_tip_m":[extension,0.]}}
    write_combined_checkpoint(root,outer=outer,arrays=arrays,kinetic={},kinetic_vector=np.zeros(1))


def test_sparse_export_roles_are_milestone_accurate_and_reloadable(tmp_path):
    _checkpoint(tmp_path,step=0,events=0,extension=0.)
    export_latest_checkpoint(tmp_path,reason="outer_driver_initial_committed_state")
    assert (tmp_path/"portable_fields/initial.npz").is_file()
    _checkpoint(tmp_path,step=9,events=1,extension=500e-6)
    export_latest_checkpoint(tmp_path,reason="outer_driver_geometry_committed")
    meta=json.loads((tmp_path/"portable_fields/first_post_event.json").read_text())
    assert set(meta["roles"]) == {"FIRST_ACCEPTED_POST_EVENT","FIRST_ACCEPTED_AT_OR_BEYOND_250UM","FIRST_ACCEPTED_AT_OR_BEYOND_500UM"}
    with np.load(tmp_path/"portable_fields/first_post_event.npz") as z:
        assert z["sigma_xx_gp"].shape == (1,)
        assert z["u_magnitude"].shape == (3,)
    assert not (tmp_path/"portable_fields/milestone_750um.npz").exists()
