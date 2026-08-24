from scripts.audit_v914_historical_analytical_model import COORDINATES, equations_text, mapping_rows


def test_mapping_covers_exact_22_coordinates_once():
    rows = mapping_rows()
    assert len(rows) == 22
    assert tuple(row["coordinate"] for row in rows) == COORDINATES
    assert len(set(COORDINATES)) == 22
    assert all(row["varied_in_global_300K_population"] == "True" for row in rows)


def test_equations_distinguish_ancestral_proxy_from_abcd_screen():
    text = equations_text()
    assert "not be conflated" in text
    assert "sampled no stochastic" in text
    assert "not a developed" in text
