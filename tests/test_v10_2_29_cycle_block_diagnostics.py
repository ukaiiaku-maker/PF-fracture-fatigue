import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from arrhenius_fracture.persistent_site_cyclic_audited_v10229 import (
    _cycle_block_audit_fields,
)


class FakeEngine:
    def __init__(self):
        self.mpz = SimpleNamespace(
            mobile_count=2.0,
            retained_count=3.0,
            emitted_total=4.0,
            escaped_total=5.0,
        )
        self.micro_advance_total_m = 6.0


def _controller():
    return SimpleNamespace(
        cfg=SimpleNamespace(
            cycle_block_mode="hazard_limited",
            target_dB=0.1,
            target_dN_store=0.2,
            target_dN_emit=0.3,
            target_dN_mobile=0.4,
            target_dN_escape=0.5,
            target_dN_peierls=0.6,
            target_dN_taylor=0.7,
            min_block_cycles=1.0e-6,
            max_block_cycles=1.0e6,
            block_cycles=1.0e4,
        )
    )


def test_cycle_block_audit_records_active_limiter_and_state():
    result = {
        "cycles_requested": 0.25,
        "cycles_consumed": 0.20,
        "cycle_limiter": "emitted_pz",
        "cycle_unlimited": 0.25,
        "cycle_candidate_limits": {"emitted_pz": 0.25, "stored_pz": 0.5},
        "mu_cleave_pred": 0.01,
        "store_per_cycle": 0.02,
        "mu_emit": 0.03,
        "mobile_per_cycle": 0.04,
        "escape_per_cycle": 0.05,
        "peierls_per_cycle": 0.06,
        "taylor_per_cycle": 0.07,
        "N_em": 8.0,
        "kinetic_active_K_shield_signed_Pa_sqrt_m": -9.0,
        "kinetic_wake_K_shield_signed_Pa_sqrt_m": 0.0,
    }
    audit = _cycle_block_audit_fields(_controller(), FakeEngine(), result)
    assert audit["cycle_limiter"] == "emitted_pz"
    assert audit["cycle_candidate_limits"]["emitted_pz"] == 0.25
    assert audit["cycle_target_increments"]["taylor_clock"] == 0.7
    assert audit["cycle_predicted_increments_per_cycle"]["mobile_pz"] == 0.04
    assert audit["cycles_consumed_fraction"] == 0.8
    assert audit["state_N_em"] == 8.0
    assert audit["state_mobile_count"] == 2.0
    assert audit["state_retained_count"] == 3.0
    assert audit["state_active_K_shield_signed_Pa_sqrt_m"] == -9.0


def test_cycle_block_summarizer_counts_limiters(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "summarize_v10_2_29_cycle_blocks.py"
    spec = importlib.util.spec_from_file_location("cycle_summary", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    root = tmp_path / "run"
    root.mkdir()
    records = []
    for limiter, cycles in (("emitted_pz", 0.2), ("stored_pz", 0.3)):
        records.append(
            {
                "loading_mode": "cyclic",
                "cycles_requested": cycles,
                "cycles_consumed": cycles,
                "cycle_limiter": limiter,
                "cycle_unlimited": cycles,
                "cycle_candidate_limits": {limiter: cycles},
                "B": 0.0,
                "state_N_em": 1.0,
                "state_mobile_count": 2.0,
                "state_retained_count": 3.0,
                "persistent_sigma_back_Pa": 4.0,
                "state_micro_advance_total_m": 0.0,
            }
        )
    (root / "kinetic_tip_cell_audit_v101.json").write_text(
        json.dumps({"records": records})
    )
    loaded = module.load_cyclic_records(root)
    summary, rows = module.summarize(loaded, root)
    assert summary["cycles_consumed_total"] == 0.5
    assert summary["limiter_summary"]["emitted_pz"]["count"] == 1
    assert summary["limiter_summary"]["stored_pz"]["count"] == 1
    assert len(rows) == 2
