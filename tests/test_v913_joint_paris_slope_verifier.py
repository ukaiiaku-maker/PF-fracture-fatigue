from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]


def test_verifier_requires_all_contract_artifacts_and_semantics():
    source=(ROOT/"scripts/verify_v913_joint_paris_slope_study.py").read_text()
    assert "len(fracture)!=30" in source
    assert "len(prospective_hazard)!=330" in source
    assert "len(overlap)!=30" in source
    assert "status.developed_da_dN_m_per_cycle" in source
    assert "NO_DEFENSIBLE_QUANTITATIVE_LOCAL_ENVELOPE" in source
