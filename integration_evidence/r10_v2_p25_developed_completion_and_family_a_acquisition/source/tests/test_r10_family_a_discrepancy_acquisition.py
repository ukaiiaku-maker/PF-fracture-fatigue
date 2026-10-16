from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts.design_r10_family_a_discrepancy_acquisition import (
    PARENT,
    constant_300k_opening_shift,
    reference_energy_shift,
    surface,
)


REGISTRY = Path("integration_evidence/r10_joint_search_amendment/inputs/original_four_registry.csv")


def parent_row() -> dict:
    with REGISTRY.open(newline="") as stream:
        return next(row for row in csv.DictReader(stream) if row["candidate_id"] == PARENT)


@pytest.mark.parametrize("prefix", ["cleave", "emit"])
@pytest.mark.parametrize("delta", [-0.12, -0.06, 0.0, 0.06, 0.12])
def test_reference_shift_preserves_thermal_entropy_and_shape_coordinates(prefix, delta):
    parent = parent_row()
    child = reference_energy_shift(parent, prefix, delta)
    nu0 = 1e12 if prefix == "cleave" else 1e11
    before = surface(parent, prefix, nu0)
    after = surface(child, prefix, nu0)

    assert after.G00_eV - before.G00_eV == pytest.approx(delta)
    assert child[f"{prefix}_floor_frac"] == parent[f"{prefix}_floor_frac"]
    assert child[f"{prefix}_exp_a"] == parent[f"{prefix}_exp_a"]
    assert child[f"{prefix}_exp_n"] == parent[f"{prefix}_exp_n"]
    assert child[f"{prefix}_gT_eV_per_K"] == parent[f"{prefix}_gT_eV_per_K"]
    assert child[f"{prefix}_sT_GPa_per_K"] == parent[f"{prefix}_sT_GPa_per_K"]
    assert child["peierls_activation_entropy_kB"] == parent["peierls_activation_entropy_kB"]
    assert child["taylor_activation_entropy_kB"] == parent["taylor_activation_entropy_kB"]


def test_positive_opening_level_shift_preserves_original_thermal_slopes():
    parent = parent_row()
    child = constant_300k_opening_shift(parent, 0.036)
    before = surface(parent, "cleave", 1e12)
    after = surface(child, "cleave", 1e12)
    for stress in (0.0, 0.25e9, 0.5e9):
        assert after.values_eV(stress, 300.0) - before.values_eV(stress, 300.0) == pytest.approx(0.036)
    assert child["cleave_gT_eV_per_K"] == parent["cleave_gT_eV_per_K"]
    assert child["cleave_sT_GPa_per_K"] == parent["cleave_sT_GPa_per_K"]
