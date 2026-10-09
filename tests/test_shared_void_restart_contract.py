import pytest

from arrhenius_fracture.checkpoint_v11 import restore_checkpoint, write_checkpoint
from arrhenius_fracture.topology_transaction_v11 import complete_accepted_state_fingerprint
from arrhenius_fracture.unified_fracture_material_v5 import material_bundle
from arrhenius_fracture.voiding_production_v5 import build_production_void_state, observables


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
