import csv
import json
from pathlib import Path

import pytest

from arrhenius_fracture.archived_fatigue_material_r9 import build_archived_current_engine
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.retained_material_r10 import (
    STRICT_FAMILY,
    load_offspring_binding,
    load_parent_binding,
    load_strict_original_binding,
)


REGISTRY = (
    Path(__file__).resolve().parents[1]
    / "arrhenius_fracture/data/materials/v10_2_27_v913_four_class_paper_registry.csv"
)
PEAK = "v913_paper_peak01_0242980_persistent_sites"


def test_original_complete_row_binds_without_replacing_legacy_cooperativity():
    binding = load_strict_original_binding(REGISTRY, PEAK)
    assert binding.provenance["family"] == STRICT_FAMILY
    assert binding.provenance["barrier_changed"] is False
    assert binding.provenance["mechanics_changed"] is False
    assert binding.reduced_config.cleavage_hits == pytest.approx(3.0)
    assert binding.reduced_config.cleavage_tau_s == pytest.approx(1.0e-6)
    engine = build_archived_current_engine(binding, kernel_family_path=R4_KERNEL_FAMILY)
    assert engine.manifest.candidate_id == "v913_zeroD_sobol_0242980"
    assert engine.f.sigma_cap == pytest.approx(30.0e9)


def test_parent_complete_row_preserves_serialized_q_and_tau(tmp_path):
    with REGISTRY.open(newline="") as stream:
        row = next(r for r in csv.DictReader(stream) if r["option_key"] == PEAK)
    row = dict(row)
    row["candidate_id"] = "TEST_RETAINED_PARENT"
    row["physics__cleavage_hits"] = "3.2732414351776242"
    row["physics__cleavage_correlation_time_s"] = "6.992153587194454e-7"
    parent = tmp_path / "parents.csv"
    with parent.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("candidate_id", "complete_row_sha256", "complete_parameter_vector"),
        )
        writer.writeheader()
        writer.writerow({
            "candidate_id": row["candidate_id"],
            "complete_row_sha256": "fixture-row-hash",
            "complete_parameter_vector": json.dumps(row, sort_keys=True),
        })
    binding = load_parent_binding(parent, row["candidate_id"])
    assert binding.reduced_config.cleavage_hits == pytest.approx(3.2732414351776242)
    assert binding.reduced_config.cleavage_tau_s == pytest.approx(6.992153587194454e-7)
    assert binding.provenance["complete_row_sha256"] == "fixture-row-hash"


def test_retained_lookup_fails_closed_on_unknown_identity():
    with pytest.raises(KeyError, match="expected one strict-family option"):
        load_strict_original_binding(REGISTRY, "not-a-row")


def test_offspring_loader_requires_explicit_strict_family(tmp_path):
    child = tmp_path / "offspring.csv"
    with child.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            "candidate_id", "family", "complete_row_sha256", "cleavage_hits",
            "cleavage_tau_s", "complete_parameter_vector",
        ))
        writer.writeheader()
        writer.writerow({"candidate_id": "bad", "family": "another-family",
            "complete_row_sha256": "hash", "cleavage_hits": 3,
            "cleavage_tau_s": 1e-6, "complete_parameter_vector": "{}"})
    with pytest.raises(ValueError, match="only the strict family"):
        load_offspring_binding(child, "bad")
