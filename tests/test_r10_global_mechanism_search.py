from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.run_r10_global_mechanism_search import (
    FAMILY, TREF, assign_relative_mechanism_strata, domain, generate_row,
    read_sources, rereference,
)


ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = ROOT / "integration_evidence/r10_joint_search_amendment/inputs/original_four_registry.csv"
PARENTS = ROOT / "integration_evidence/r10_joint_search_amendment/inputs/fatigue_parent_registry.csv"


def test_rereference_preserves_linear_surface_coordinates():
    raw = next(csv.DictReader(ORIGINAL.open(newline="")))
    old_tref = float(raw["Tref_K"])
    converted = rereference(raw, TREF)
    for prefix in ("cleave", "emit"):
        for temperature in (300.0, 750.0, 1200.0):
            old_g = float(raw[f"{prefix}_G00_eV"]) + float(raw[f"{prefix}_gT_eV_per_K"]) * (temperature - old_tref)
            new_g = float(converted[f"{prefix}_G00_eV"]) + float(converted[f"{prefix}_gT_eV_per_K"]) * (temperature - TREF)
            old_s = float(raw[f"{prefix}_sigc0_GPa"]) + float(raw[f"{prefix}_sT_GPa_per_K"]) * (temperature - old_tref)
            new_s = float(converted[f"{prefix}_sigc0_GPa"]) + float(converted[f"{prefix}_sT_GPa_per_K"]) * (temperature - TREF)
            assert new_g == pytest.approx(old_g)
            assert new_s == pytest.approx(old_s)


def test_generated_row_is_strict_family_and_does_not_change_nonbarrier_contract():
    sources = read_sources(ORIGINAL, PARENTS)
    child = generate_row(0, np.full(22, 0.5), sources)
    parent = sources[4]
    assert child["family"] == FAMILY
    assert "guard" not in " ".join(child).lower()
    for field in ("rho_source0_m2", "reference_source_area_um2",
                  "reference_front_width_um", "source_zone_length_um"):
        assert child[field] == parent[field]
    assert child["peierls_nu0_s"] == parent["peierls_nu0_s"]
    assert child["taylor_nu0_s"] == parent["taylor_nu0_s"]
    assert domain(child)[0]


def test_acquisition_strata_are_relative_and_not_final_mechanism_labels():
    rows = [{"descriptor_complete": True, "candidate_id": str(i),
             "median_log10_state_drive_ratio": float(i)} for i in range(9)]
    assign_relative_mechanism_strata(rows)
    labels = [row["mechanism_acquisition_stratum"] for row in rows]
    assert labels.count("OPENING_DIRECTED") == 3
    assert labels.count("MIXED_DIRECTED") == 3
    assert labels.count("PLASTICITY_DIRECTED") == 3
    assert all("NOT_FINAL_CLASSIFICATION" in row["mechanism_stratum_basis"] for row in rows)
