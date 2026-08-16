import pandas as pd

from scripts.materialize_v914_prospective_fatigue_registry import (
    physical_fingerprint,
    read_k300_batches,
    validate_runtime_registry_roundtrip,
)


def test_physical_fingerprint_is_id_independent_and_roundtrips():
    fields = ["cleave_G00_eV", "emit_G00_eV", "c_blunt"]
    source = pd.Series(
        {
            "candidate_id": "fracture-id",
            "cleave_G00_eV": 4.0,
            "emit_G00_eV": 3.0,
            "c_blunt": 0.25,
        }
    )
    fatigue = source.copy()
    fatigue["candidate_id"] = "fatigue-id"
    assert physical_fingerprint(source, fields) == physical_fingerprint(fatigue, fields)
    fatigue["c_blunt"] = 0.26
    assert physical_fingerprint(source, fields) != physical_fingerprint(fatigue, fields)


def test_disjoint_k300_batches_are_combined_exactly(tmp_path):
    paths = []
    for index in range(2):
        path = tmp_path / f"batch{index}.csv"
        pd.DataFrame(
            {
                "candidate_id": [f"c{index}"],
                "temperature_K": [300.0],
                "K_50um_MPa_sqrt_m": [25.0 + index],
            }
        ).to_csv(path, index=False)
        paths.append(path)
    combined = read_k300_batches(paths)
    assert combined.candidate_id.tolist() == ["c0", "c1"]
    assert combined.K_50um_MPa_sqrt_m.tolist() == [25.0, 26.0]


def test_written_registry_matches_v914_runtime_parser_bit_for_bit(tmp_path):
    fields = ["cleave_G00_eV", "emit_G00_eV", "c_blunt"]
    expected = pd.DataFrame(
        [{
            "candidate_id": "p",
            "cleave_G00_eV": float("2.357721022795886"),
            "emit_G00_eV": float("3.344896922819316"),
            "c_blunt": float("0.1229003664302577"),
        }]
    )
    expected["parameter_fingerprint"] = physical_fingerprint(expected.iloc[0], fields)
    path = tmp_path / "registry.csv"
    expected.to_csv(path, index=False, float_format="%.17g")
    validate_runtime_registry_roundtrip(path, expected, fields)
