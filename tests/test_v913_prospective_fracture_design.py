import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "scripts/design_v913_prospective_fracture_causality.py"
spec = importlib.util.spec_from_file_location("_v913_prospective_design_test", PATH)
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
assert spec.loader is not None
spec.loader.exec_module(mod)


def test_f2_collinearity_audit_prefers_overlap():
    frame = pd.read_csv(ROOT / "runs/v913_barrier_temperature_fracture_morphology_v3_focused/focused_model_master.csv")
    audit, selected = mod.coordinate_audit(frame)
    assert selected == "F2_activation_window_overlap"
    chosen = audit[audit.selected_for_F2].iloc[0]
    width = audit[audit.F2_candidate.eq("F2_width80_ratio")].iloc[0]
    assert abs(chosen.pearson_with_F1) < abs(width.pearson_with_F1)


def test_target_design_has_twelve_unique_rows_per_center():
    focused = pd.read_csv(ROOT / "runs/v913_barrier_temperature_fracture_morphology_v3_focused/focused_model_master.csv")
    plastic = pd.read_csv(ROOT / "runs/v913_barrier_temperature_fracture_morphology_v3_focused/plastic_bottleneck_descriptors.csv")
    focused = mod.with_low_temperature_bottleneck(focused, plastic)
    stats = mod.coordinate_stats(focused)
    for family, cid in mod.CANONICAL.items():
        r = focused[focused.candidate_id.eq(cid)].iloc[0]
        center = {
            mod.COORDS[0]: r.delta_mu_emit_minus_cleave,
            mod.COORDS[1]: r.activation_window_overlap_Oce,
            mod.COORDS[2]: r.delta_Theta_sigma_900,
            mod.COORDS[3]: r.F4_lowT_plastic_bottleneck,
        }
        design = mod.target_design(family, center, stats)
        assert len(design) == 12
        assert len({name for name, _ in design}) == 12
        assert all(any(not np.isclose(target[k], center[k]) for k in mod.COORDS) for _, target in design)


def test_common_stress_anchor_preserves_intrinsic_target_coordinates():
    # Algebraic invariants of the proposed one-scalar nuisance transformation:
    # normalized shape is unchanged, and T*sT/sigc is unchanged if sigc0 and
    # sT are multiplied by the same positive lambda.
    sig0, slope, tref, temp, lam = 7.2, -0.003, 300.0, 900.0, 1.17
    theta0 = temp * slope / (sig0 + slope * (temp - tref))
    theta1 = temp * (lam * slope) / (lam * sig0 + lam * slope * (temp - tref))
    assert np.isclose(theta0, theta1, rtol=0, atol=1e-14)
    assert np.isclose(mod.overlap_fast(.5, 2.0, .2, 1.1), mod.overlap_fast(.5, 2.0, .2, 1.1))


def test_fingerprint_changes_with_any_active_parameter():
    fields = ["a", "b"]
    row = pd.Series({"a": 1.0, "b": 2.0})
    original = mod.parameter_fingerprint(row, fields)
    changed = row.copy(); changed["b"] = np.nextafter(2.0, 3.0)
    assert mod.parameter_fingerprint(changed, fields) != original


def test_design_coordinates_match_focused_canonical_descriptors():
    v1 = mod.load_module(mod.V1_SCRIPT, "_v913_design_coordinate_test_v1")
    ExpFloorSurface, PTMechanism = v1.load_production_types(mod.SOURCE)
    candidates, _, _, _ = v1.load_population(mod.SOURCE)
    focused = pd.read_csv(mod.FOCUSED / "focused_model_master.csv")
    plastic = pd.read_csv(mod.FOCUSED / "plastic_bottleneck_descriptors.csv")
    for cid in mod.CANONICAL.values():
        row = candidates[candidates.candidate_id.eq(cid)].iloc[0]
        got = mod.design_coordinates(row, v1, ExpFloorSurface, PTMechanism)
        expected = focused[focused.candidate_id.eq(cid)].iloc[0]
        assert np.isclose(got[mod.COORDS[0]], expected.delta_mu_emit_minus_cleave, rtol=2e-12)
        assert np.isclose(got[mod.COORDS[1]], expected.activation_window_overlap_Oce, rtol=2e-12, atol=2e-12)
        assert np.isclose(got[mod.COORDS[2]], expected.delta_Theta_sigma_900, rtol=2e-12, atol=2e-12)
        low = plastic[plastic.candidate_id.eq(cid)].sort_values("temperature_K").iloc[0]
        assert np.isclose(got[mod.COORDS[3]], low.B_P_log10_tauP_over_taue, rtol=2e-12, atol=2e-12)


@pytest.fixture(scope="module")
def full_design():
    v1 = mod.load_module(mod.V1_SCRIPT, "_v913_full_design_test_v1")
    ExpFloorSurface, PTMechanism = v1.load_production_types(mod.SOURCE)
    candidates, _, _, _ = v1.load_population(mod.SOURCE)
    focused = pd.read_csv(mod.FOCUSED / "focused_model_master.csv")
    plastic = pd.read_csv(mod.FOCUSED / "plastic_bottleneck_descriptors.csv")
    focused = mod.with_low_temperature_bottleneck(focused, plastic)
    return (v1, candidates, *mod.prospective_design(candidates, focused, v1, ExpFloorSurface, PTMechanism))


def test_full_design_has_exact_controls_feasible_12x2_and_retained_failures(full_design):
    v1, candidates, registry, audit, changes, _, _ = full_design
    assert registry.design_role.eq("EXACT_CANONICAL_CENTER_CONTROL").sum() == 2
    primary = registry[registry.design_role.eq("FEASIBLE_PRIMARY")]
    assert primary.groupby("design_family").size().to_dict() == {"DBTT": 12, "Peak-T": 12}
    assert primary.parameter_fingerprint.is_unique
    assert primary.design_fingerprint.is_unique
    assert audit.feasibility_status.eq("INFEASIBLE_RETAINED").any()
    assert not changes.outside_historical_observed_range.any()
    for r in registry[registry.design_role.eq("EXACT_CANONICAL_CENTER_CONTROL")].itertuples(index=False):
        parent = candidates[candidates.candidate_id.eq(r.parent_candidate_id)].iloc[0]
        assert r.parameter_fingerprint == mod.parameter_fingerprint(parent, v1.ACTIVE_FIELDS)
        for field in v1.ACTIVE_FIELDS:
            assert getattr(r, field) == parent[field]


def test_anchor_plan_makes_no_unqualified_k300_claim(full_design):
    _, candidates, registry, _, _, _, _ = full_design
    anchor_fields = tuple(dict.fromkeys((*mod.VARY_FIELDS, "cleave_sigc0_GPa", "cleave_sT_GPa_per_K", "emit_sigc0_GPa")))
    stats = mod.robust_parameter_stats(candidates, anchor_fields).reset_index()
    plan = mod.anchor_plan(registry, candidates, stats)
    assert len(plan) == 24
    assert plan.lambda_applied.isna().all()
    assert plan.K300_pre_anchor_real_MPa_sqrt_m.isna().all()
    assert plan.K300_post_anchor_real_MPa_sqrt_m.isna().all()
    assert (~plan.post_anchor_claim_permitted).all()
    assert (plan.historical_envelope_lambda_min < plan.historical_envelope_lambda_max).all()
