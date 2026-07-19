import json
from pathlib import Path


def test_v10214_legacy_snapshot_forces_wake_disabled(tmp_path: Path):
    payload = {
        "schema": "v10.2.12_serialized_physical_fixed_crack_fem_state",
        "state_id": "legacy",
        "r_eff_over_r0": 1.0,
        "opening_strength_fraction": 0.1,
        "crack_extension_m": 0.0,
        "temperature_K": 700.0,
        "Uy_top_m": 0.0,
        "Uy_bot_m": 0.0,
        "crack_tip_xy_m": [0.0, 0.0],
        "crack_direction": [1.0, 0.0],
        "interaction_ell_m": 1e-6,
        "exclude_radius_m": 0.0,
        "active_x_m": [1e-6],
        "wake_x_m": [1e-6],
        "channel_directions": [[1.0, 0.0]],
        "channel_normals": [[0.0, 1.0]],
        "material": {"E": 1.0, "nu": 0.25, "b": 1e-10, "Tm": 1.0},
        "engine_config": {},
        "arrays": "missing.npz",
        "wake_kernel_supported": True,
    }
    root = tmp_path / "legacy"
    root.mkdir()
    (root / "snapshot.json").write_text(json.dumps(payload))
    from arrhenius_fracture.physical_fem_snapshot_v10212 import SnapshotMetadata
    data = json.loads((root / "snapshot.json").read_text())
    data["wake_kernel_supported"] = False
    metadata = SnapshotMetadata(
        **{
            key: data[key]
            for key in SnapshotMetadata.__dataclass_fields__
            if key in data
        }
    ).validate()
    assert metadata.wake_kernel_supported is False
