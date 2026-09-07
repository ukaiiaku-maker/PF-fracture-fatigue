import numpy as np

from arrhenius_fracture.closure_mechanics_evidence import REGISTRY, GROUPS, MESHES, measurements
from arrhenius_fracture.closure_static_evidence import configuration, validate_solver_capture
from arrhenius_fracture.crack_void_mechanics_v5 import solve_crack_void_case
from arrhenius_fracture.finalization_v3_schema import canonical_hash


def test_radius_ligament_matrix_has_eighteen_unique_physical_configurations():
    keys = [GROUPS[f"matrix:{radius}:{ratio}:{n}:{r}"] for n,r in MESHES
            for radius in (4e-5,5e-5,6e-5) for ratio in (1.,2.,3.)]
    assert len(set(keys)) == 18
    for cfg in REGISTRY.values():
        assert canonical_hash(cfg) in REGISTRY
    for n,r in MESHES:
        assert GROUPS[f"centered:{n}:{r}"] == GROUPS[f"matrix:{5e-5}:3.0:{n}:{r}"]


def test_controls_are_exactly_matched_except_for_cavity_removal():
    for label,key in GROUPS.items():
        if not label.endswith(":matched_crack_only"): continue
        parent = dict(REGISTRY[GROUPS[label.removesuffix(":matched_crack_only")]])
        parent["cavity_enabled"] = False
        assert parent == REGISTRY[key]


def test_fixed_crack_and_fixed_cavity_derivatives_do_not_alias():
    for n,r in MESHES:
        cfg = REGISTRY[GROUPS[f"centered:{n}:{r}"]]
        for eps in (2.5e-6,1.25e-6):
            radius = REGISTRY[GROUPS[f"radius_perturb:{eps}:1:{n}:{r}"]]
            crack = REGISTRY[GROUPS[f"crack_perturb:{eps}:1:{n}:{r}"]]
            assert radius["crack_path_m"] == cfg["crack_path_m"]
            assert crack["cavity_radius_m"] == cfg["cavity_radius_m"]
            assert crack["cavity_center_m"] == cfg["cavity_center_m"]


def test_matched_crack_only_capture_reassembles_and_has_no_invented_cavity_metrics():
    cfg = dict(configuration(True,32,12), cavity_enabled=False)
    raw = solve_crack_void_case(**cfg)["source_capture"]
    validate_solver_capture(raw,cfg)
    m = measurements(raw,cfg)
    assert m["cavity_fields"] is None
    assert m["free_residual_relative"] < 1e-8
    assert m["energy_identity"] < .01
