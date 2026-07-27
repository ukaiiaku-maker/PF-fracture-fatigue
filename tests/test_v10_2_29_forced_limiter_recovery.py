import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from arrhenius_fracture.persistent_site_cyclic_audited_v10229 import (
    _cycle_block_audit_fields,
)


class FakeEngine:
    def __init__(self):
        self.B = 0.0
        self.mpz = SimpleNamespace(
            mobile_count=0.0,
            retained_count=0.0,
            emitted_total=0.0,
            escaped_total=0.0,
        )
        self.micro_advance_total_m = 0.0


def controller(target=0.1):
    return SimpleNamespace(
        cfg=SimpleNamespace(
            cycle_block_mode="hazard_limited",
            target_dB=target,
            target_dN_store=target,
            target_dN_emit=target,
            target_dN_mobile=target,
            target_dN_escape=target,
            target_dN_peierls=target,
            target_dN_taylor=target,
            min_block_cycles=1.0e-6,
            max_block_cycles=1.0e6,
            block_cycles=1.0e4,
        )
    )


def forced_result():
    return {
        "cycles_requested": 0.00025,
        "cycles_consumed": 0.00025,
        "cycle_limiter": "global_forced",
        "cycle_unlimited": 0.00025,
        "cycle_candidate_limits": {"global_forced": 0.00025},
        "B_pre": 0.0,
        "B": 0.0,
        "mu_cleave_pred": 1.0e-9,
        "store_per_cycle": 1.0e-6,
        "mu_emit": 400.0,
        "mobile_per_cycle": 399.0,
        "escape_per_cycle": 1.0e-12,
        "peierls_per_cycle": 1.0e-8,
        "taylor_per_cycle": 1.0e-3,
        "N_em": 0.0,
        "kinetic_active_K_shield_signed_Pa_sqrt_m": 0.0,
        "kinetic_wake_K_shield_signed_Pa_sqrt_m": 0.0,
    }


def test_audit_recovers_emission_limiter_from_driver_force_cycles():
    audit = _cycle_block_audit_fields(controller(), FakeEngine(), forced_result())
    assert audit["cycle_applied_limiter"] == "global_forced"
    assert audit["cycle_limiter"] == "emitted_pz"
    assert audit["cycle_candidate_limits"]["emitted_pz"] == 0.00025
    assert audit["cycle_candidate_limits"]["mobile_pz"] > 0.00025


def test_summarizer_recovers_limiter_in_existing_audit(tmp_path):
    script = Path(__file__).parents[1] / "scripts" / "summarize_v10_2_29_cycle_blocks.py"
    spec = importlib.util.spec_from_file_location("cycle_summary_v2", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    record = {
        "loading_mode": "cyclic",
        "cycles_requested": 0.00025,
        "cycles_consumed": 0.00025,
        "cycle_limiter": "global_forced",
        "cycle_unlimited": 0.00025,
        "cycle_candidate_limits": {"global_forced": 0.00025},
        "cycle_block_mode": "hazard_limited",
        "cycle_target_increments": {
            "cleavage_clock": 0.1,
            "stored_pz": 0.1,
            "emitted_pz": 0.1,
            "mobile_pz": 0.1,
            "escape_pz": 0.1,
            "peierls_clock": 0.1,
            "taylor_clock": 0.1,
        },
        "cycle_predicted_increments_per_cycle": {
            "cleavage_clock": 1.0e-9,
            "stored_pz": 1.0e-6,
            "emitted_pz": 400.0,
            "mobile_pz": 399.0,
            "escape_pz": 1.0e-12,
            "peierls_clock": 1.0e-8,
            "taylor_clock": 1.0e-3,
        },
        "cycle_max_block_cycles": 1.0e6,
        "cycle_nominal_block_cycles": 1.0e4,
        "B": 0.0,
        "state_N_em": 0.0,
        "state_mobile_count": 0.0,
        "state_retained_count": 0.0,
        "persistent_sigma_back_Pa": 0.0,
        "state_micro_advance_total_m": 0.0,
    }
    root = tmp_path / "run"
    root.mkdir()
    (root / "kinetic_tip_cell_audit_v101.json").write_text(
        json.dumps({"records": [record]})
    )
    records = module.load_cyclic_records(root)
    summary, rows = module.summarize(records, root)
    assert summary["limiter_summary"]["emitted_pz"]["count"] == 1
    assert summary["applied_limiter_summary"]["global_forced"]["count"] == 1
    assert rows[0]["cycle_limiter"] == "emitted_pz"
