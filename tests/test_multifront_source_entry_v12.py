from pathlib import Path
from types import SimpleNamespace

import pytest

from arrhenius_fracture.sharp_front_current_source_multifront_v12 import validate_args
from arrhenius_fracture.stateful_multifront_production_v12 import (
    production_directional_rate_adapter_v12,
)
from scripts.prepare_pf_general_multifront_v6_parity_pair import commands


def test_default_general_source_has_no_finite_front_physics_cap():
    policy = validate_args([
        "--source-only-preflight", "--branching-mode", "mechanistic",
        "--front-resource-limit", "none", "--branch-transaction-limit", "none",
    ])
    assert policy.front_resource_limit is None
    assert policy.branch_transaction_limit is None


def test_source_only_entry_cannot_launch_pf_or_fem():
    with pytest.raises(RuntimeError, match="NO_PF_FEM_EXECUTION"):
        validate_args(["--branching-mode", "mechanistic"])


def test_future_parity_pair_uses_one_generic_entry_with_policy_only():
    values = commands(Path("/python"), Path("control.json"), Path("enabled.json"))
    assert values["control_n1"][:3] == values["enabled_n2"][:3]
    assert "--branching-mode" in values["control_n1"]
    assert "disabled" in values["control_n1"]
    assert "mechanistic" in values["enabled_n2"]
    assert "--front-resource-limit" in values["enabled_n2"]


def test_v11_tip_observation_vocabulary_binds_to_v12_front_rate(monkeypatch):
    captured = {}

    def preview(engine, candidate, **values):
        captured.update(values)
        return SimpleNamespace(lambda_per_s=17.0)

    monkeypatch.setattr(
        "arrhenius_fracture.directional_competition_v11.preview_production_cleavage_rate",
        preview,
    )
    context = SimpleNamespace(
        provisional_runtime=SimpleNamespace(
            owner_by_front={"front:1": "owner:1"},
            process_regions={
                "owner:1": SimpleNamespace(process_engine_id="engine:1")
            },
        ),
        actual_engines={"engine:1": object()},
        provisional_solved_state=SimpleNamespace(
            fem_state=SimpleNamespace(material=SimpleNamespace(Eprime=4.0))
        ),
        adapter_configuration={"temperature_K": 700.0},
    )
    observation = SimpleNamespace(
        tip_id="front:1", kinetic_J_J_per_m2=9.0,
    )
    assert production_directional_rate_adapter_v12(
        observation, object(), context,
    ) == 17.0
    assert captured == {
        "signed_J_J_per_m2": 9.0, "Eprime_Pa": 4.0,
        "temperature_K": 700.0,
    }
