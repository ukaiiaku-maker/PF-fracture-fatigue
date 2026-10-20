import json
from pathlib import Path

import numpy as np

from scripts.fatigue_informed_joint_barrier_v3_v10230 import deserialize_surface
from scripts.run_r10_alternative_gstar_joint_search import make_variant,physical


ROOT=Path(__file__).resolve().parents[1]
BASE=Path("/Volumes/Data/Data/Nanopillar_calculation/PF-fracture-fatigue_worktrees/v3-fatigue-informed-joint-barrier-search/analysis_outputs/v3_1_coupled_slope_continuation/v3_1_optimized_candidate_registry.json")


def base_surface():
    row=next(r for r in json.loads(BASE.read_text())["rows"] if r["candidate_id"]=="V3_ceramic_V2_P25_007__V3_1_TRUST1")
    return deserialize_surface(row["opening_surface"])


def test_variant_preserves_declared_300K_barrier_anchors():
    base=base_surface();variant=make_variant(base,np.full(6,.5))
    from scripts.run_r10_alternative_gstar_joint_search import stress_from_K
    anchors=stress_from_K(np.array([0.,15.,75.]))
    assert np.allclose(base.G_eV(anchors,300.),variant.G_eV(anchors,300.),rtol=0,atol=2e-14)
    assert physical(variant)


def test_current_fracture_driver_exposes_alternative_binding():
    text=(ROOT/"scripts/run_r10_current_fracture_panel.py").read_text()
    assert '"alternative"' in text
    assert "load_archived_material_binding" in text
