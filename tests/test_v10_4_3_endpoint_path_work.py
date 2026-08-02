from __future__ import annotations

from pathlib import Path

import numpy as np

from arrhenius_fracture.energy_ledger_output_v10227 import augment_steps_table
from arrhenius_fracture.plastic_flow_path_work_v1043 import transform_source


def _source() -> str:
    return Path("arrhenius_fracture/sharp_front.py").read_text()


def test_endpoint_path_transform_compiles() -> None:
    transformed = transform_source(_source())
    compile(transformed, "sharp_front.py[v10.4.3-endpoint-path-work-test]", "exec")


def test_endpoint_path_transform_uses_accepted_endpoint_states() -> None:
    transformed = transform_source(_source())
    required = [
        "sigma_gp_step0_path_v1043",
        "ep_gp_step0_path_v1043",
        "_v1043_sigma_path_avg_gp = 0.5",
        "_v1043_dep_path_gp",
        "_v1043_sigma_path_avg_gp * _v1043_dep_path_gp",
        "equilibrated_endpoint_trapezoid_sigma_colon_delta_ep",
        "W_p_constitutive_acc_v1043",
        "hist['W_p_constitutive']",
        "_v1043_prefracture_path_step = Kc_first is None",
    ]
    missing = [token for token in required if token not in transformed]
    assert not missing, missing

    # The snapshots must be outside the rejected-trial loop so subdivision
    # retries retain the same beginning-of-accepted-step endpoint.
    snapshot = transformed.index("sigma_gp_step0_path_v1043")
    retry_loop = transformed.index("            while True:\n", snapshot)
    assert snapshot < retry_loop


def test_energy_output_retains_constitutive_comparison() -> None:
    steps = np.zeros((2, 15), dtype=float)
    steps[:, 0] = [1.0, 2.0]
    steps[:, 3] = [2.0, 3.0]
    hist = {
        "W_ext": [10.0, 20.0],
        "U_el": [8.0, 12.0],
        "W_p": [2.0, 8.0],
        "W_p_constitutive": [1.0, 2.0],
        "W_emit": [0.0, 0.0],
    }

    augmented, header, audit = augment_steps_table(
        steps,
        "legacy," * 14 + "legacy",
        hist,
        [],
        effective_modulus_pa=4.0,
    )

    names = header.split(",")
    assert "W_bulk_plastic_constitutive_cumulative_J_per_m" in names
    assert "W_bulk_plastic_path_minus_constitutive_cumulative_J_per_m" in names
    assert "W_fracture_residual_constitutive_cumulative_J_per_m" in names
    assert audit["constitutive_comparison_history_present"] is True

    index = {name: i for i, name in enumerate(names)}
    np.testing.assert_allclose(
        augmented[:, index["W_bulk_plastic_cumulative_J_per_m"]],
        [2.0, 8.0],
    )
    np.testing.assert_allclose(
        augmented[:, index["W_bulk_plastic_constitutive_cumulative_J_per_m"]],
        [1.0, 2.0],
    )
    np.testing.assert_allclose(
        augmented[:, index["W_bulk_plastic_path_minus_constitutive_cumulative_J_per_m"]],
        [1.0, 6.0],
    )
    np.testing.assert_allclose(
        augmented[:, index["W_fracture_residual_cumulative_J_per_m"]],
        [0.0, 0.0],
    )
    np.testing.assert_allclose(
        augmented[:, index["W_fracture_residual_constitutive_cumulative_J_per_m"]],
        [1.0, 6.0],
    )
