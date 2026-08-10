from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_v10_2_30_driving_force_ladder.py"


def load_module():
    spec = importlib.util.spec_from_file_location("ladder_analysis", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fraction_uses_existing_case_name_contract():
    module = load_module()
    assert module.fraction("/x/peak_f0p975_seed1720/output") == ("peak", .975)
    assert module.fraction("/x/dbtt_f1p100_seed1001723/output") == ("dbtt", 1.1)
    assert module.fraction("/x/weakT_f1p145_seed2001726/output") == ("weakt", 1.145)
    assert module.fraction("/x/ceramic_f1p205_seed3001729/output") == ("ceramic", 1.205)


def test_regime_uses_event_spacing_distribution_for_near_monotonic():
    module = load_module()
    assert module.regime({"status": "completed", "developed_da_dN_m_per_cycle": 1e-5,
                          "fraction_subcycle_intervals": .75,
                          "median_event_spacing_cycles": .2}) == \
        "NEAR_MONOTONIC_CYCLIC_FAILURE"
    assert module.regime({"status": "completed", "developed_da_dN_m_per_cycle": 1e-5,
                          "fraction_subcycle_intervals": .25,
                          "median_event_spacing_cycles": 3.0}) == "LCF"
    assert module.regime({"status": "censored"}) == "VHCF_OR_CENSORED"


def test_rate_endpoint_reports_measured_neighbor_bracket():
    module = load_module()
    rows = [
        {"class": "peak", "f": 1.135, "developed_da_dN_m_per_cycle": 8e-6,
         "deltaK_MPa_sqrt_m": 24.1, "cycles_to_target": 12, "regime": "LCF"},
        {"class": "peak", "f": 1.14, "developed_da_dN_m_per_cycle": 1.2e-5,
         "deltaK_MPa_sqrt_m": 24.2, "cycles_to_target": 8,
         "regime": "NEAR_MONOTONIC_CYCLIC_FAILURE"},
    ]
    endpoint = module.rate_endpoints(rows)[0]
    assert endpoint["lower_bracket_f"] == 1.135
    assert endpoint["upper_bracket_f"] == 1.14
