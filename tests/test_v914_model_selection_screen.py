from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "screen", ROOT / "scripts/run_v914_model_selection_screen.py"
)
screen = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(screen)


def test_primary_screen_has_eight_models_two_R_and_four_Kmax():
    cases = screen.cases()
    assert len(cases) == 64
    assert len({row["candidate"] for row in cases}) == 8
    assert {row["R"] for row in cases} == {0.1, -0.5}
    assert {row["Kmax_MPa_sqrt_m"] for row in cases} == {12.0, 15.0, 18.0, 21.0}
    assert all(row["deltaK_MPa_sqrt_m"] == row["Kmax_MPa_sqrt_m"] * (1-row["R"]) for row in cases)


def test_fatigue_specific_models_retain_original_registry_and_physics():
    for row in screen.cases():
        if row["family"] in "ABCD":
            assert row["registry"].endswith("v10_2_31_endurance_knee_ABCD_registry.csv")
            assert row["physics"].endswith("v9_13_R_curve_calibrated_common_physics.json")
            assert row["configuration_label"] == "LEGACY1D-V7"


def test_provenance_captures_exact_rows_barriers_and_overrides():
    rows = screen.model_provenance()
    assert len(rows) == 8
    assert all(len(row["source_row_sha256"]) == 64 for row in rows)
    assert all(row["barrier_parameters"] for row in rows)
    abcd = [row for row in rows if row["family"] in "ABCD"]
    assert all("physics__mobile_transport_velocity_scale" in row["model_specific_physics_overrides"] for row in abcd)
    assert all(not row["model_specific_physics_overrides"] for row in rows if row["family"] not in "ABCD")
