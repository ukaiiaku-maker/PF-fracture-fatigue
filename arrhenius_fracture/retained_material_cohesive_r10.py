"""Fail-closed binding for additive R10 cohesive--Arrhenius offspring."""
from __future__ import annotations

import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from .archived_fatigue_material_r9 import (
    ArchivedMaterialBinding, build_archived_current_engine,
)
from .cohesive_arrhenius import (
    COHESIVE_BARRIER_ONLY, COHESIVE_KINETIC_CZM, FAMILY,
    CohesiveThermalLaw, ExponentialCohesiveBarrier,
)
from .retained_material_r10 import _binding, _simple_manifest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cohesive_binding(registry: str | Path, candidate_id: str,
                          *, execution_identity: str = COHESIVE_BARRIER_ONLY
                          ) -> ArchivedMaterialBinding:
    if execution_identity not in {COHESIVE_BARRIER_ONLY, COHESIVE_KINETIC_CZM}:
        raise ValueError("unsupported cohesive Arrhenius execution identity")
    source = Path(registry).resolve()
    with source.open(newline="") as stream:
        matches = [row for row in csv.DictReader(stream)
                   if row["candidate_id"] == candidate_id]
    if len(matches) != 1:
        raise KeyError(f"expected one cohesive-family candidate {candidate_id!r}")
    record = matches[0]
    if record.get("family") != FAMILY:
        raise ValueError("cohesive loader accepts only its separately named family")
    parent = json.loads(record["parent_complete_parameter_vector"])
    cohesive = json.loads(record["cohesive_parameter_vector"])
    thermal = CohesiveThermalLaw(
        Gamma_ref_J_m2=float(cohesive["Gamma_ref_J_m2"]),
        t_hat_ref_Pa=float(cohesive["t_hat_ref_Pa"]),
        S_Gamma_J_m2_K=float(cohesive["S_Gamma_J_m2_K"]),
        T_ref_K=float(cohesive["T_ref_K"]),
        compact_b=float(cohesive.get("compact_b", 2.0)),
        compact_c=float(cohesive.get("compact_c", 0.0)),
        compact_variant=str(cohesive.get("compact_variant", "beta_polynomial")),
        compact_rise_fraction=float(cohesive.get("compact_rise_fraction", .1)),
        compact_fall_fraction=float(cohesive.get("compact_fall_fraction", .85)),
        compact_plateau_drop=float(cohesive.get("compact_plateau_drop", .05)),
        compact_drop_fraction=float(cohesive.get("compact_drop_fraction", .2)),
        compact_tail_level=float(cohesive.get("compact_tail_level", .05)),
    ).validate_range((300, 450, 600, 750, 900, 1050, 1200))
    opening = ExponentialCohesiveBarrier(
        thermal=thermal,
        activation_area_m2=float(cohesive["activation_area_m2"]),
        attempt_frequency_s=float(cohesive["attempt_frequency_s"]),
        execution_identity=execution_identity,
    )
    manifest = replace(
        _simple_manifest(parent), name=str(candidate_id),
        candidate_id=str(candidate_id), cleavage=opening,
    )
    binding = _binding(
        row=parent, manifest=manifest, family=FAMILY, source=source,
        source_hash=_sha256(source), row_hash=record["complete_row_sha256"],
        hits=float(record["cleavage_hits"]),
        tau_s=float(record["cleavage_tau_s"]), candidate_id=candidate_id,
    )
    provenance = dict(binding.provenance)
    provenance.update({
        "family": FAMILY, "execution_identity": execution_identity,
        "cohesive_parameter_sha256": record["cohesive_parameter_sha256"],
        "parent_candidate_id": record["parent_candidate_id"],
        "parent_complete_row_sha256": record["parent_complete_row_sha256"],
        "opening_EXP_floor_fields_inactive": True,
        "barrier_changed": True,
        "mechanics_changed": execution_identity == COHESIVE_KINETIC_CZM,
        "event_law_changed": False,
    })
    return ArchivedMaterialBinding(
        binding.optimized_row, binding.parent_row, binding.manifest,
        binding.reduced_config, binding.persistent_sites, provenance,
    )


def build_cohesive_current_engine(binding: ArchivedMaterialBinding, *,
                                  kernel_family_path, local_strength_Pa=30.0e9):
    identity = str(binding.provenance.get("execution_identity", ""))
    if identity not in {COHESIVE_BARRIER_ONLY, COHESIVE_KINETIC_CZM}:
        raise RuntimeError("binding lacks a supported cohesive execution identity")
    engine = build_archived_current_engine(
        binding, kernel_family_path=kernel_family_path,
        local_strength_Pa=float(local_strength_Pa),
    )
    if not isinstance(engine.manifest.cleavage, ExponentialCohesiveBarrier):
        raise RuntimeError("final engine did not consume the cohesive barrier")
    engine._r10_opening_execution_identity = identity
    engine._r10_material_family = FAMILY
    engine._r10_final_bound_implementation_identity = (
        f"{FAMILY}:{identity}:EXPONENTIAL_COHESIVE_POTENTIAL_V1"
    )
    return engine


__all__ = ["build_cohesive_current_engine", "load_cohesive_binding"]
