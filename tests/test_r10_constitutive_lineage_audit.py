from pathlib import Path

from scripts.reconcile_r10_constitutive_temperature_lineage import transform_audit


ROOT = Path(__file__).resolve().parents[1]


def _row(sigc0: float, sT: float) -> dict:
    return {
        "Tref_K": 481.33,
        "cleave_G00_eV": 1.1060847791228108,
        "cleave_gT_eV_per_K": 0.0019625,
        "cleave_sigc0_GPa": 1.1038020018699726,
        "cleave_sT_GPa_per_K": 0.00472567102860844,
        "cleave_exp_a": 0.5887163921130266,
        "cleave_exp_n": 0.25,
        "cleave_floor_frac": 0.48726421565281924,
        "emit_G00_eV": 1.096369176369715,
        "emit_gT_eV_per_K": -0.0007602504297796134,
        "emit_sigc0_GPa": sigc0,
        "emit_sT_GPa_per_K": sT,
        "emit_exp_a": 0.6522800833545626,
        "emit_exp_n": 0.8178969756700099,
        "emit_floor_frac": 0.001963405800003,
        "peierls_H0_eV": 5.107608093612558,
        "peierls_activation_entropy_kB": 80.85891837414806,
        "peierls_exp_a": 0.2264547564554959,
        "peierls_exp_n": 1.9374389592558143,
        "peierls_nu0_s": 1.0e12,
        "taylor_H0_eV": 3.1092449876738772,
        "taylor_activation_entropy_kB": 49.721227072445636,
        "taylor_exp_a": 1.055419295541942,
        "taylor_exp_n": 1.0061173520516604,
        "taylor_nu0_s": 1.0e11,
    }


def test_final_scale_maps_energy_and_entropy_in_stress_not_same_stress():
    parent = _row(0.878234268782052, 0.0008651616614272865)
    final = _row(3.65243827982399, 0.0035980713606352517)
    audit = transform_audit(parent, final)
    assert audit["named_energy_and_entropy_coefficients_preserved"]
    assert not audit["derived_entropy_at_same_physical_stress_preserved"]
    assert audit["max_abs_G_identity_error_eV"] < 1.0e-12
    assert audit["max_abs_S_identity_error_kB"] < 1.0e-7
    assert abs(audit["sT_scale_error_GPa_per_K"]) < 1.0e-15


def test_continuation_analysis_uses_evidence_scoped_labels():
    source = (ROOT / "scripts/analyze_r10_constitutive_coupled_continuation.py").read_text()
    assert "unevaluated_predicted_uncompetitive_rows" in source
    assert "new_rate_span_decades" in source
    assert "P1/P2-only" in source
    assert "stopped_dominated_rows" not in source
