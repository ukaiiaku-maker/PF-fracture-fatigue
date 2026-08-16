from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import analyze_v1032_transition_refinement as analysis


def sample_rows():
    base = dict(family="DBTT", candidate_id="id", parameter_option="opt",
                original_temperature_class="DBTT", deltaK_MPa_sqrt_m=20.0,
                normalized_f=1.0, cycles_to_target=100.0, extension_um=100.0,
                event_count=10.0, median_event_interval_cycles=5.0,
                minimum_event_interval_cycles=1.0, mean_event_interval_cycles=10.0,
                subcycle_fraction=0.0, fraction_below_10_cycles=0.5,
                fraction_below_0p1_cycle=0.0, seed=1, source_run_root="x",
                result_path="x", run_contract_path="x", repository_head="h",
                registry_sha256="r", candidate_fingerprint_sha256="f")
    rows = []
    for dim, mode, rate in (("1D", "accelerated", 1e-7), ("1D", "explicit", 2e-7),
                            ("2D", "explicit", 6e-7)):
        rows.append({**base, "dimensionality": dim, "integration_mode": mode,
                     "da_dN_m_per_cycle": rate, "censor_status": "developed",
                     "plot_kind": "resolved"})
    rows.append({**base, "dimensionality": "2D", "integration_mode": "accelerated",
                 "da_dN_m_per_cycle": np.nan, "censor_status": "cycle_censor",
                 "plot_kind": "censor"})
    return pd.DataFrame(rows)


def test_enrichment_computes_mode_and_spatial_ratios_without_censor_rate():
    data = analysis.enrich(sample_rows())
    explicit_1d = data[(data.dimensionality == "1D") & (data.integration_mode == "explicit")].iloc[0]
    explicit_2d = data[(data.dimensionality == "2D") & (data.integration_mode == "explicit")].iloc[0]
    censor = data[data.plot_kind == "censor"].iloc[0]
    assert explicit_1d.accelerated_explicit_ratio == 2
    assert explicit_2d.spatial_enhancement_ratio == 3
    assert explicit_2d.regime_classification == "SPATIAL_LCF"
    assert censor.regime_classification == "CYCLE_CENSOR"
    assert np.isnan(censor.da_dN_m_per_cycle)


def test_declared_parity_thresholds_are_logarithmic():
    assert analysis.STRICT_LOG_TOL == 0.10
    assert analysis.ENGINEERING_LOG_TOL == 0.30
    data = analysis.enrich(sample_rows())
    parity, spatial = analysis.diagnostics(data)
    row = parity[(parity.dimensionality == "1D")].iloc[0]
    assert not row.strict_parity
    assert not row.engineering_parity  # exactly log10(2), boundary is strict
    assert spatial.iloc[0].strong_spatial_divergence


def test_required_figure_contract_is_complete():
    import verify_v1032_transition_refinement as verifier
    assert len(verifier.STEMS) == 14
    assert set(verifier.MATERIALS) == set(analysis.MATERIALS)
