"""Per-temperature joint fit: isothermal materials and censoring-aware scoring."""
import os
from pathlib import Path

import numpy as np
import pytest

from experimental_fits.ast2018.per_temperature_joint_fit import (
    CENSOR_PENALTY, isothermal_material, score_case, seed_coordinates)

NATIVE = Path(os.environ.get("AST2018_NATIVE_SOURCE", "/nonexistent"))


def _target():
    x = np.array([.1, .2, .4, .6])
    return {"J_da_um": x, "J": np.array([100., 200., 300., 400.]),
            "J_w": np.full(4, .25), "J_scale": 20.}


def test_censored_curve_never_beats_an_equal_supported_curve():
    target = _target()
    full = {"status": "complete", "J_da_um": [0, .1, .2, .4, .6], "J": [0, 100., 200., 300., 400.]}
    short = {"status": "model_limit_nominal_K_window", "J_da_um": [0, .1, .2],
             "J": [0, 100., 200.]}
    assert score_case(target, full)["J"] < score_case(target, short)["J"]
    # Unsupported points carry at least the declared penalty each.
    assert score_case(target, short)["J"] >= .5 * CENSOR_PENALTY ** 2


def test_failed_run_is_penalized_not_dropped():
    target = _target()
    failed = {"status": "numerical_failure"}
    assert score_case(target, failed)["J"] > score_case(
        target, {"status": "complete", "J_da_um": [0, .6], "J": [0, 0.]})["J"] * .99


@pytest.mark.skipif(not NATIVE.exists(), reason="pinned native source checkout required")
def test_isothermal_material_hits_targets_with_zero_temperature_slopes():
    from experimental_fits.ast2018.multi_temperature_bending_fit import base_material
    from experimental_fits.ast2018.section_plasticity_replay import selected_material
    _, _, base = base_material()
    base = selected_material(base)
    T = 373.15
    x0 = seed_coordinates(base, T)
    material = isothermal_material(base, T, x0)
    # Seed coordinates reproduce the selected material's barriers at T.
    for stress in (0., 1e9, 3e9):
        assert np.isclose(material.cleavage.barrier_eV(stress, T),
                          base.cleavage.barrier_eV(stress, T), rtol=1e-12)
        assert np.isclose(material.emission.barrier_eV(stress, T),
                          base.emission.barrier_eV(stress, T), rtol=1e-12)
        assert np.isclose(material.peierls.surface(material.emission).barrier_eV(stress / 3, T),
                          base.peierls.surface(base.emission).barrier_eV(stress / 3, T),
                          rtol=1e-9, atol=1e-6)
        assert np.isclose(material.taylor.surface(material.emission).sigc0_Pa,
                          base.taylor.surface(base.emission).sigc0_Pa
                          + base.taylor.surface(base.emission).sT_Pa_per_K
                          * (T - base.emission.Tref_K), rtol=1e-12)
    # No temperature dependence is imposed on the fitted barriers.
    for stress in (0., 2e9):
        assert np.isclose(material.cleavage.barrier_eV(stress, 294.15),
                          material.cleavage.barrier_eV(stress, 523.15))
        assert np.isclose(material.emission.barrier_eV(stress, 294.15),
                          material.emission.barrier_eV(stress, 523.15))


def test_bending_probe_is_constructible_at_the_fit_extension():
    # Regression: a default-target probe (2.3 um) raised in every forward run.
    from experimental_fits.ast2018.config import AstAssumptions
    from experimental_fits.ast2018.per_temperature_joint_fit import MAX_EXTENSION_UM, SHAPE
    with pytest.raises(ValueError):
        AstAssumptions(loading_mode="nominal_bending_shared_temperature",
                       bending_opening_shape_path=str(SHAPE))
    AstAssumptions(target_extension_m=MAX_EXTENSION_UM * 1e-6,
                   loading_mode="nominal_bending_shared_temperature",
                   bending_opening_shape_path=str(SHAPE))


def test_code_errors_abort_instead_of_being_scored(tmp_path, monkeypatch):
    # Regression: 644 evaluation errors were silently scored as penalties.
    from experimental_fits.ast2018 import per_temperature_joint_fit as fit

    from concurrent.futures import ThreadPoolExecutor

    def fake(job):
        return {"curve_id": job[2], "status": "evaluation_error",
                "numerical_failure": "ValueError: boom"}

    monkeypatch.setattr(fit, "run_case", fake)

    class Pool(ThreadPoolExecutor):
        pass

    tf = object.__new__(fit.TemperatureFit)
    tf.C, tf.T_K, tf.pool, tf.output, tf.cache = 21, 294.15, Pool(), tmp_path, {}
    tf.family, tf.tip_interaction = "A", "native"
    tf.cases = [{"curve_id": "c", "condition_id": "k"}]
    tf.lo, tf.hi = np.zeros(7), np.ones(7)
    tf.log = tmp_path / "log.jsonl"
    with pytest.raises(RuntimeError):
        tf.evaluate_many([np.full(7, .5)], "test")
    assert not tf.log.exists() and (tmp_path / "T21_ABORTED.json").is_file()


def test_search_box_is_the_same_at_every_temperature():
    # Regression: boxes centred on an inherited material's value at T imposed
    # its temperature dependence and kept 250 C emission away from easy values.
    from experimental_fits.ast2018.per_temperature_joint_fit import bounds_around
    for family in ("A", "B"):
        lo21, hi21 = bounds_around(np.array([4.5, .45, 1.6, 1.2, 2.1, .8, 2.1]), family)
        lo250, hi250 = bounds_around(np.array([4.9, .75, 1.6, 2.1, 2.0, .9, 2.0]), family)
        assert np.array_equal(lo21, lo250) and np.array_equal(hi21, hi250)
        assert lo21[-4] <= 1.2 and hi21[-4] >= 2.1


@pytest.mark.skipif(not NATIVE.exists(), reason="pinned native source checkout required")
def test_model_b_material_is_the_three_halves_envelope_at_T():
    from experimental_fits.ast2018.multi_temperature_bending_fit import base_material
    from experimental_fits.ast2018.section_plasticity_replay import selected_material
    _, _, base = base_material()
    base = selected_material(base)
    T = 523.15
    F, B, sB, sc = .6, 2.0, 3.0, 12.0
    x = np.array([F, B, np.log10(sB), np.log10(sc), 1.5, 2.0, 1.0, 1.8])
    material = isothermal_material(base, T, x, "B")
    for stress in (0., 2e9, 6e9, 11.9e9):
        q = 1 - stress / (sc * 1e9)
        expected = q ** 1.5 * (F + B * np.exp(-stress / (sB * 1e9)))
        assert np.isclose(material.cleavage.barrier_eV(stress, T), expected, rtol=1e-12)
    assert material.cleavage.barrier_eV(13e9, T) == 0.0
    # The emission and glide coordinates are shared with family A.
    xa = np.array([4.5, .5, 1.6, 1.5, 2.0, 1.0, 1.8])
    ma = isothermal_material(base, T, xa, "A")
    assert material.emission == ma.emission and material.peierls == ma.peierls
    with pytest.raises(ValueError):
        isothermal_material(base, T, np.array([-.1, 2., .5, 1., 1.5, 2., 1., 1.8]), "B")
    with pytest.raises(RuntimeError):
        isothermal_material(base, T, xa, "B")


def test_points_are_logged_as_they_complete_and_resume(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from experimental_fits.ast2018 import per_temperature_joint_fit as fit
    calls = []

    def fake(job):
        calls.append(job)
        if job[1][0] > .7:  # the slow-failing point never blocks finished ones
            return {"curve_id": job[2], "status": "evaluation_error",
                    "numerical_failure": "ValueError: boom"}
        return {"curve_id": job[2], "status": "numerical_failure", "numerical_failure": "x"}

    monkeypatch.setattr(fit, "run_case", fake)
    target = _target()
    tf = object.__new__(fit.TemperatureFit)
    tf.C, tf.T_K, tf.output, tf.cache, tf.family = 21, 294.15, tmp_path, {}, "A"
    tf.tip_interaction = "native"
    tf.cases = [{**target, "curve_id": "c", "condition_id": "k"}]
    tf.lo, tf.hi = np.zeros(7), np.ones(7)
    tf.log = tmp_path / "log.jsonl"
    with ThreadPoolExecutor(1) as tf.pool:
        tf.evaluate_many([np.full(7, .2), np.full(7, .3)], "test")
        assert len(tf.log.read_text().splitlines()) == 2
        with pytest.raises(RuntimeError):
            tf.evaluate_many([np.full(7, .4), np.full(7, .9)], "test")
    assert len(tf.log.read_text().splitlines()) == 3
    n = len(calls)
    # Cached points are not recomputed.
    with ThreadPoolExecutor(1) as tf.pool:
        tf.evaluate_many([np.full(7, .2), np.full(7, .4)], "test")
    assert len(calls) == n


@pytest.mark.skipif(not NATIVE.exists(), reason="pinned native source checkout required")
def test_glide_strength_does_not_follow_tip_emission():
    # Regression: Peierls/Taylor sc were multiples of the emission sc, so easier
    # tip emission softened the whole beam and the search could only reach fits
    # with the emission/opening competition switched off.
    from experimental_fits.ast2018.multi_temperature_bending_fit import base_material
    from experimental_fits.ast2018.section_plasticity_replay import selected_material
    _, _, base = base_material()
    base = selected_material(base)
    T = 294.15
    hard = isothermal_material(base, T, np.array([4.5, .45, 1.5, 1.5, 2.0, .8, 1.7]))
    easy = isothermal_material(base, T, np.array([4.5, .45, 1.5, 1.5, .7, .8, 1.7]))
    assert easy.emission.barrier_eV(2e9, T) < hard.emission.barrier_eV(2e9, T) - .1
    for m in (hard, easy):
        assert np.isclose(m.peierls.surface(m.emission).sigc0_Pa, 1.7e9, rtol=1e-12)
    ratio = base.taylor.stress_scale / base.peierls.stress_scale
    for name in ("peierls", "taylor"):
        a = getattr(hard, name).surface(hard.emission)
        b = getattr(easy, name).surface(easy.emission)
        assert np.isclose(a.sigc0_Pa, b.sigc0_Pa, rtol=1e-12)
        assert a.barrier_eV(5e8, T) == pytest.approx(b.barrier_eV(5e8, T), rel=1e-12)
    assert np.isclose(easy.taylor.surface(easy.emission).sigc0_Pa, ratio * 1.7e9, rtol=1e-12)


def test_load_is_not_scored_after_the_measured_crack_leaves_the_table():
    # Regression: the elastic table stops at 0.98 um, but load/stiffness were
    # scored to 3 um where the measured crack had reached ~1.5 um.
    u = np.linspace(0, 3, 31)
    stiff = np.where(u < .6, 1., 1. - .2 * (u - .6))  # passes 0.65 at u = 2.35
    target = {**_target(), "load_u_um": u, "load_K": np.full(31, 2.),
              "stiff_u_um": u, "stiff": stiff}
    model = {"status": "complete", "J_da_um": [0, .1, .2, .4, .6], "J": [0, 100., 200., 300., 400.],
             "u_um": u, "K_nominal": np.where(u <= 2.45, 2., 5.), "stiffness": stiff}
    base = score_case(target, model)
    windowed = score_case(target, {**model, "stiffness_at_extension_limit": .65})
    assert base["load"] > 1 and windowed["load"] == 0.0
    from experimental_fits.ast2018.per_temperature_joint_fit import measured_extension_limit_u
    assert measured_extension_limit_u(target, {"stiffness_at_extension_limit": .65}) == pytest.approx(2.4)
    assert measured_extension_limit_u(target, {}) == np.inf


def test_emission_active_box_only_caps_emission_and_is_temperature_independent():
    from experimental_fits.ast2018.per_temperature_joint_fit import FAMILIES, bounds_around
    for family in ("A", "B"):
        lo, hi = bounds_around(None, family)
        alo, ahi = bounds_around(None, family, emission_active=True)
        changed = {FAMILIES[family][i] for i in np.nonzero(ahi != hi)[0]}
        assert changed == {"G0_emit_eV", "sigc_emit_GPa"} and np.array_equal(lo, alo)
        assert np.all(ahi <= hi) and np.all(ahi > alo)


def test_plot_handles_a_temperature_without_a_load_trace(tmp_path, monkeypatch):
    # Regression: 500 C has only a J curve; the extension-limit lookup raised KeyError.
    import json
    from experimental_fits.ast2018 import plot_per_temperature_joint_fit as plot
    monkeypatch.setattr(plot, "targets", lambda: {500: [{**_target(), "curve_id": "c"}]})
    result = {"curve_id": "c", "status": "complete", "J_da_um": [0, .6], "J": [0, 400.],
              "u_um": [0, 1.], "K_nominal": [0, 2.], "stiffness": [1, .8], "bulk_upl_um": .1,
              "stiffness_at_extension_limit": .65}
    (tmp_path / "model_A").mkdir()
    (tmp_path / "model_A" / "T500_evaluations.jsonl").write_text(
        json.dumps({"x": [0.], "objective": -1., "stage": "s", "results": [result]}) + "\n")
    # The stored (possibly stale) objective is replaced by the current one.
    assert plot.main(500, tmp_path)["A"]["objective"] == pytest.approx(
        plot.objective([{**_target(), "curve_id": "c"}], [result])[0])
    assert (tmp_path / "T500_best_A_vs_B.png").is_file()


def test_plot_includes_the_signed_pileup_variant_and_a_run_without_a_crack_event(tmp_path, monkeypatch):
    # Regression: a best point with no crack event has no bulk/tip state
    # (bulk_upl_um None), which crashed the label formatting.
    import json
    from experimental_fits.ast2018 import plot_per_temperature_joint_fit as plot
    monkeypatch.setattr(plot, "targets", lambda: {500: [{**_target(), "curve_id": "c"}]})
    result = {"curve_id": "c", "status": "model_limit_nominal_K_window", "events": 0,
              "J_da_um": [0.], "J": [0.], "u_um": [0, 1.], "K_nominal": [0, 2.],
              "stiffness": [1, 1.], "bulk_upl_um": None, "tip_activations": None,
              "stiffness_at_extension_limit": .65}
    (tmp_path / "model_A_signed_pileup").mkdir()
    (tmp_path / "model_A_signed_pileup" / "T500_evaluations.jsonl").write_text(
        json.dumps({"x": [0.], "objective": 0., "stage": "s", "results": [result]}) + "\n")
    assert plot.main(500, tmp_path)["A_signed_pileup"] is not None
    assert (tmp_path / "T500_best_A_vs_B.png").is_file()


def test_stopping_the_crack_early_never_beats_growing_it_on_a_rising_curve():
    # Regression: unsupported points cost only the penalty when the measured J
    # rose above the model's last J, so fits whose crack stopped at 0.14 um beat
    # fits that grew the crack to 1 um at 100 and 250 C.
    x = np.linspace(.05, .95, 19)
    y = 100. + 900. * x
    target = {"J_da_um": x, "J": y, "J_w": np.full(19, 1 / 19), "J_scale": 45.}
    grown = {"status": "complete", "J_da_um": np.r_[0, x, .98],
             "J": np.r_[0, y + 2 * 45., y[-1] + 90.]}
    stopped = {"status": "model_limit_nominal_K_window", "J_da_um": [0, .05, .1, .14],
               "J": [0, 145., 190., 226.]}
    assert score_case(target, grown)["J"] < score_case(target, stopped)["J"]
    # A curve that falls below the model's last J is still penalized as before.
    low = {**target, "J": np.full(19, 100.)}
    assert score_case(low, stopped)["J"] >= (126. / 45. + CENSOR_PENALTY) ** 2 * 16 / 19


def test_resumed_points_are_rescored_with_the_current_objective(tmp_path):
    import json
    from experimental_fits.ast2018 import per_temperature_joint_fit as fit
    tf = object.__new__(fit.TemperatureFit)
    tf.cases = [{**_target(), "curve_id": "c", "condition_id": "k"}]
    tf.log = tmp_path / "log.jsonl"
    result = {"curve_id": "c", "status": "complete", "J_da_um": [0, .6], "J": [0, 400.]}
    tf.log.write_text(json.dumps({"x": [.1, .2], "objective": -1., "stage": "s",
                                  "detail": [], "results": [result]}) + "\n")
    tf.load_cache()
    row = tf.cache[(.1, .2)]
    assert row["objective"] > 0
    assert row["objective"] == pytest.approx(fit.objective(tf.cases, [result])[0])


def test_previous_points_with_the_current_coordinates_are_carried_unchanged(tmp_path):
    import json
    from types import SimpleNamespace
    from experimental_fits.ast2018.per_temperature_joint_fit import previous_best
    (tmp_path / "model_A").mkdir()
    x = [4.4, .45, 1.45, 1.56, 1.3, .81, 2.07]
    (tmp_path / "model_A" / "T100_evaluations.jsonl").write_text(json.dumps(
        {"x": x, "objective": 1., "results": [{"status": "complete"}]}) + "\n")
    base = SimpleNamespace(peierls=SimpleNamespace(stress_scale=1.))
    assert [p.tolist() for p in previous_best(100, previous=tmp_path, base=base)] == [x]
