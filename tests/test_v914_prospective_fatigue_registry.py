import pandas as pd

from scripts.materialize_v914_prospective_fatigue_registry import (
    physical_fingerprint,
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
