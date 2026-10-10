"""Small shared loaders for the sealed V3.1 continuation row."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from arrhenius_fracture.canonical_v2_registry_v10230 import ThermodynamicBarrierAdapter
from scripts import run_v3_stage_d_reduced_developed_fatigue_v10230 as stage_d
from scripts.run_v3_1_coupled_sensitivity_v10230 import deserialize_thermodynamic
from scripts.fatigue_informed_joint_barrier_v3_v10230 import deserialize_surface

ROOT=Path(__file__).resolve().parents[1]
V3_REGISTRY=ROOT/'analysis_outputs/v3_fatigue_informed_joint_barrier_search/v3_complete_candidate_registry.json'


def load_v3_1_candidate(
    fatigue, optimized_row, *, registry_path=V3_REGISTRY,
    canonical_registry=None, canonical_manifest=None,
):
    registry_path = Path(registry_path).resolve()
    parents={x['candidate_id']:x for x in json.loads(registry_path.read_text())['rows']}
    parent=parents[optimized_row['parent_candidate_id']]
    base,physics,source=stage_d.load_candidate(
        fatigue,parent,canonical_registry=canonical_registry,
        canonical_manifest=canonical_manifest,
    )
    opening=deserialize_surface(optimized_row['opening_surface'])
    emission=deserialize_thermodynamic(optimized_row['emission_surface'])
    candidate=replace(base,candidate_id=optimized_row['candidate_id'],
        cleavage=ThermodynamicBarrierAdapter(opening,base.cleavage.parent),
        emission=ThermodynamicBarrierAdapter(emission,base.emission.parent))
    if 'hits' in optimized_row['opening_surface']:
        physics=replace(physics,cleavage_hits=float(optimized_row['opening_surface']['hits']))
    if 'tau_s' in optimized_row['opening_surface']:
        physics=replace(physics,cleavage_correlation_time_s=float(optimized_row['opening_surface']['tau_s']))
    return candidate,physics,source,parent


__all__=['load_v3_1_candidate']
