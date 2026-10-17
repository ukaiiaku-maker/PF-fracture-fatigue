from __future__ import annotations

import math
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

from arrhenius_fracture.cohesive import (
    CohesiveElement, CohesiveNetwork, cohesive_contribution,
    cohesive_element_diagnostics,
)
from arrhenius_fracture.cohesive_arrhenius import (
    COHESIVE_BARRIER_ONLY, COHESIVE_KINETIC_CZM, FAMILY,
    CohesiveThermalLaw, ExponentialCohesiveBarrier, RateIndependentCohesiveControl,
    bilinear_barrier_J, finite_compliance_landscape, morse_barrier_J,
    potential_J_m2, stationary_points,
    tangent_Pa_per_m, traction_Pa, traction_barrier,
)
from arrhenius_fracture.current_coupled_execution_r4 import R4_KERNEL_FAMILY
from arrhenius_fracture.reduced_persistent_reference_r2 import runtime_receipt
from arrhenius_fracture.retained_material_cohesive_r10 import (
    build_cohesive_current_engine, load_cohesive_binding,
)


def law(entropy=0.0):
    return CohesiveThermalLaw(2.5, 20.0e9, entropy)


def element(**kwargs):
    values = dict(
        plus_nodes=(0, 1), minus_nodes=(2, 3),
        normal=np.array([0.0, 1.0]), tangent=np.array([1.0, 0.0]),
        length=2.0e-6, thermal=law(),
    )
    values.update(kwargs)
    return CohesiveElement(**values)


def opening_u(delta):
    u = np.zeros(8)
    u[1] = u[3] = 0.5 * delta
    u[5] = u[7] = -0.5 * delta
    return u


def test_potential_traction_tangent_and_envelope_scales():
    thermal = law()
    d0 = thermal.delta0_m
    assert traction_Pa(d0, thermal, 300.0) == pytest.approx(20.0e9)
    assert tangent_Pa_per_m(0.0, thermal, 300.0) == pytest.approx(
        thermal.initial_stiffness(300.0)
    )
    for x in (0.01, 0.4, 1.0, 3.0):
        d = x * d0; h = 1.0e-6 * d0
        derivative = (potential_J_m2(d + h, thermal, 300.0)
                      - potential_J_m2(d - h, thermal, 300.0)) / (2 * h)
        assert derivative == pytest.approx(traction_Pa(d, thermal, 300.0), rel=2e-8)


def test_stationary_roots_barrier_quadrature_and_activation_volume():
    thermal = law(); area = 2.0e-19; s = 0.37
    stress = s * thermal.t_hat(300.0)
    roots = stationary_points(stress, thermal, 300.0)
    assert roots["state"] == "METASTABLE"
    assert roots["x_min"] < 1.0 < roots["x_saddle"]
    result = traction_barrier(stress, thermal, 300.0, area)
    direct = area * quad(
        lambda delta: float(traction_Pa(delta, thermal, 300.0)) - stress,
        roots["x_min"] * thermal.delta0_m,
        roots["x_saddle"] * thermal.delta0_m,
        epsabs=1e-32,
    )[0]
    assert result["barrier_J"] == pytest.approx(direct, rel=2e-10)
    h = 1.0e-5 * stress
    derivative = (traction_barrier(stress + h, thermal, 300.0, area)["barrier_J"]
                  - traction_barrier(stress - h, thermal, 300.0, area)["barrier_J"]) / (2*h)
    assert -derivative == pytest.approx(result["activation_volume_m3"], rel=2e-8)


def test_endpoint_states_asymptotic_and_contact_rate():
    thermal = law(); area = 2.0e-19
    zero = traction_barrier(0.0, thermal, 300.0, area)
    assert zero["barrier_J"] == pytest.approx(area * thermal.gamma(300.0))
    assert math.isinf(zero["x_saddle"])
    assert math.isinf(zero["activation_volume_m3"])
    assert stationary_points(-1.0, thermal, 300.0)["state"] == "INADMISSIBLE_COMPRESSION"
    assert stationary_points(thermal.t_hat(300.0), thermal, 300.0)["state"] == "BARRIERLESS_SPINODAL"
    eps = 1.0e-9
    result = traction_barrier((1-eps)*thermal.t_hat(300.0), thermal, 300.0, area)
    expected = 4*math.sqrt(2)/(3*math.e)*area*thermal.gamma(300.0)*eps**1.5
    assert result["barrier_J"] == pytest.approx(expected, rel=1e-12)
    barrier = ExponentialCohesiveBarrier(thermal, area)
    assert barrier.rate(np.array([-1.0, 0.0]), 300.0).tolist() == [0.0, 0.0]
    control = RateIndependentCohesiveControl(thermal)
    assert not control.should_rupture(0.99*thermal.delta0_m, 300.0)
    assert control.should_rupture(thermal.delta0_m, 300.0)
    with pytest.raises(RuntimeError, match="no Arrhenius rate"):
        control.rate(1.0, 300.0)


def test_thermal_derivative_and_shape_controls():
    thermal = law(entropy=8.0e-4); area = 3.0e-19
    barrier = ExponentialCohesiveBarrier(thermal, area)
    stress = 0.3 * thermal.t_hat(300.0)
    entropy = barrier.entropy_J_K(stress, 650.0)
    assert math.isfinite(entropy)
    assert entropy > 0.0
    assert bilinear_barrier_J([0, 1], 2.5, area).tolist() == pytest.approx([2.5*area, 0])
    assert morse_barrier_J([0, 1], 2.5, area).tolist() == pytest.approx([2.5*area, 0])


def test_finite_compliance_distinguishes_no_saddle_from_spinodal():
    thermal = law(); d0 = thermal.delta0_m
    stiff = finite_compliance_landscape(
        2*d0, 2.0*thermal.initial_stiffness(300.0), thermal, 300.0,
    )
    assert stiff["state"] == "NO_ESCAPE_SADDLE"
    assert stiff["barrier_J_m2"] is None
    compliant = finite_compliance_landscape(
        3*d0, 0.05*thermal.initial_stiffness(300.0), thermal, 300.0,
    )
    assert compliant["state"] in {"METASTABLE", "NO_ESCAPE_SADDLE"}
    if compliant["state"] == "METASTABLE":
        assert compliant["barrier_J_m2"] > 0.0


def test_nonlinear_assembly_tangent_irreversibility_and_contact():
    elem = element(); network = CohesiveNetwork([elem], contact_penalty_Pa_per_m=3e22)
    delta = 0.35 * elem.thermal.delta0_m
    u = opening_u(delta)
    K, R = cohesive_contribution(network, u, 8)
    h = 1e-6 * elem.thermal.delta0_m
    direction = opening_u(1.0)
    _, Rp = cohesive_contribution(network, u + h*direction, 8)
    _, Rm = cohesive_contribution(network, u - h*direction, 8)
    numeric = (Rp-Rm)/(2*h)
    assert np.allclose(numeric, K @ direction, rtol=2e-7, atol=1e-5)
    assert np.linalg.norm(R) > 0.0
    elem.accept_rupture(7)
    _, broken_R = cohesive_contribution(network, u, 8)
    assert np.linalg.norm(broken_R) == pytest.approx(0.0)
    elem.accept_rupture(8)
    assert elem.accepted_event_index == 7
    compression = opening_u(-delta)
    _, contact_R = cohesive_contribution(network, compression, 8)
    assert np.linalg.norm(contact_R) > 0.0
    diag = cohesive_element_diagnostics(elem, compression, network)
    assert diag["contact_active"] is True
    assert diag["execution_identity"] == COHESIVE_KINETIC_CZM
    control_elem = element()
    assert not control_elem.accept_rate_independent_spinodal(0.9*control_elem.thermal.delta0_m)
    assert control_elem.accept_rate_independent_spinodal(control_elem.thermal.delta0_m)
    assert control_elem.metadata["rupture_owner"] == "RATE_INDEPENDENT_ENVELOPE_SPINODAL"


def test_binding_is_fail_closed_and_emits_final_runtime_identity(tmp_path):
    source = (Path(__file__).resolve().parents[1] / "arrhenius_fracture/data/materials"
              / "v10_2_27_v913_four_class_paper_registry.csv")
    with source.open(newline="") as stream:
        parent = dict(next(csv.DictReader(stream)))
    cohesive = {
        "Gamma_ref_J_m2": 2.5, "t_hat_ref_Pa": 20e9,
        "S_Gamma_J_m2_K": 0.0, "T_ref_K": 300.0,
        "activation_area_m2": 8e-20, "attempt_frequency_s": 1e12,
    }
    cohesive_hash = hashlib.sha256(json.dumps(
        cohesive, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    registry = tmp_path / "registry.csv"
    fields = ["candidate_id", "family", "complete_row_sha256",
              "parent_candidate_id", "parent_complete_row_sha256",
              "cleavage_hits", "cleavage_tau_s", "cohesive_parameter_sha256",
              "cohesive_parameter_vector", "parent_complete_parameter_vector"]
    with registry.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerow({
            "candidate_id": "fixture", "family": FAMILY,
            "complete_row_sha256": "complete", "parent_candidate_id": parent["candidate_id"],
            "parent_complete_row_sha256": "parent", "cleavage_hits": 3,
            "cleavage_tau_s": 1e-6, "cohesive_parameter_sha256": cohesive_hash,
            "cohesive_parameter_vector": json.dumps(cohesive),
            "parent_complete_parameter_vector": json.dumps(parent),
        })
    binding = load_cohesive_binding(registry, "fixture")
    assert binding.provenance["execution_identity"] == COHESIVE_BARRIER_ONLY
    engine = build_cohesive_current_engine(binding, kernel_family_path=R4_KERNEL_FAMILY)
    receipt = runtime_receipt(engine, stage="fixture")
    assert receipt["material_family"] == FAMILY
    assert receipt["selected_identity"] == COHESIVE_BARRIER_ONLY
    assert "EXPONENTIAL_COHESIVE_POTENTIAL" in receipt["final_bound_implementation_identity"]
    with pytest.raises(KeyError):
        load_cohesive_binding(registry, "missing")
