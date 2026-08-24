import importlib.util
from pathlib import Path
import pandas as pd

P = Path(__file__).parents[1] / "scripts/build_v914_candidate_id_crosswalk.py"
S = importlib.util.spec_from_file_location("x", P)
M = importlib.util.module_from_spec(S)
S.loader.exec_module(M)


def test_legacy_bijection_uses_atlas_identity_verbatim():
    atlas = pd.DataFrame({
        "candidate_id": ["V914_ATLAS_000000001", "V914_ATLAS_000000002"],
        "candidate_sha256": ["a", "b"],
        "legacy_candidate_id": ["v914_large_analytical_0000000", ""],
        **{name: [0.0, 1.0] for name in M.BASE},
    })
    rows = M.legacy_bijection_rows(atlas)
    assert len(rows) == 1
    assert rows[0]["base_candidate_id"] == "V914_ATLAS_000000001"
    assert rows[0]["exact_parameter_match"] is True


def test_exact_parameter_match_fails_closed_on_no_match():
    atlas = pd.DataFrame({
        "candidate_id": ["V914_ATLAS_000000001"],
        "candidate_sha256": ["a"],
        **{name: [0.0] for name in M.BASE},
    })
    source = pd.DataFrame({"candidate_id": ["legacy_x"], **{name: [1.0] for name in M.BASE}})
    rows = M.exact_parameter_match_rows(source, atlas, "test")
    assert rows[0]["base_candidate_id"] is pd.NA
    assert rows[0]["exact_parameter_match"] is False


def test_exact_parameter_match_raises_on_duplicate_atlas_rows():
    atlas = pd.DataFrame({
        "candidate_id": ["V914_ATLAS_000000001", "V914_ATLAS_000000002"],
        "candidate_sha256": ["a", "b"],
        **{name: [0.0, 0.0] for name in M.BASE},
    })
    source = pd.DataFrame({"candidate_id": ["legacy_x"], **{name: [0.0] for name in M.BASE}})
    try:
        M.exact_parameter_match_rows(source, atlas, "test")
        assert False, "expected RuntimeError for duplicate exact matches"
    except RuntimeError:
        pass
