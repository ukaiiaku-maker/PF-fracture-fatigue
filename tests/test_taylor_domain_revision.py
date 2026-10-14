import math
from pathlib import Path

from arrhenius_fracture.taylor_domain_revision import (
    T_MAX_K, T_MIN_K, assert_amplitude_only, construct_amplitude_rows,
    minimal_intercept_stress_candidate, raw_taylor_extrema, read_rows,
    required_taylor_reference,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"


def _rows():
    return read_rows(REGISTRY)[1]


def test_continuous_taylor_minimum_is_endpoint_and_margin_is_exact():
    for row in _rows():
        parent = raw_taylor_extrema(row)
        assert parent["minimum_at_K"] in {T_MIN_K, T_MAX_K}
        for margin in (0.10, 0.25, 0.50):
            result = required_taylor_reference(row, margin)
            assert result["revised_minimum_eV"] >= margin
            assert math.isclose(
                result["revised_minimum_eV"], max(parent["minimum"], margin),
                rel_tol=1e-13, abs_tol=1e-13,
            )


def test_variants_change_only_amplitude_and_identity_metadata():
    parents = _rows()
    variants, records = construct_amplitude_rows(parents, (0.10, 0.25, 0.50))
    assert len(variants) == 12
    assert len(records) == 12
    for parent in parents:
        selected = [row for row in variants if row["option_key"].startswith(parent["option_key"] + "__")]
        assert len(selected) == 3
        for variant in selected:
            assert_amplitude_only(parent, variant)


def test_dbtt_minimal_intercept_is_separate_shared_law_candidate():
    dbtt = next(row for row in _rows() if row["material_class"] == "DBTT")
    candidate = minimal_intercept_stress_candidate(dbtt)
    revised = candidate["row"]
    endpoint = (
        float(revised["emit_sigc0_GPa"])
        + float(revised["emit_sT_GPa_per_K"]) * (1300.0 - float(revised["Tref_K"]))
    ) * 1.0e9
    assert math.isclose(endpoint, 1.0, rel_tol=0.0, abs_tol=1e-6)
    assert candidate["affected_mechanisms"] == [
        "emission characteristic stress",
        "Peierls characteristic stress inherited through peierls_stress_fraction",
        "Taylor characteristic stress inherited through taylor_stress_fraction",
    ]
