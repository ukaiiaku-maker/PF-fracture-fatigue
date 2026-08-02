from __future__ import annotations

from pathlib import Path

from arrhenius_fracture.plastic_flow_fixed_point_converged_v1043 import (
    transform_source,
)

ROOT = Path(__file__).resolve().parents[1]


def test_fixed_point_overlay_compiles_and_requires_convergence():
    source = (ROOT / "arrhenius_fracture" / "sharp_front.py").read_text()
    transformed = transform_source(source)
    compile(transformed, "sharp_front.py[v10.4.3-fixed-point-test]", "exec")

    required = {
        "maximum iterations": "Maximum relaxed mechanics/plasticity fixed-point iterations",
        "relaxation option": "--stagger-relaxation",
        "relative tolerance": "--stagger-rtol",
        "plastic strain tolerance": "--stagger-ep-atol",
        "density tolerance": "--stagger-rho-atol-m2",
        "state rebase": "ep_gp_step0_v1043, rho_gp_step0_v1043",
        "relaxed iterate": "ep_gp_iter_v1043 + _v1043_stagger_alpha",
        "convergence flag": "stagger_converged_v1043 = True",
        "strict rejection": "mechanics/plasticity fixed point did not converge",
        "final mechanics closure": "Close the staggered step with a",
        "positive directional J": "J_positive = max(J_signed, 0.0)",
    }
    missing = [label for label, token in required.items() if token not in transformed]
    assert not missing, missing

    assert "plastic_work_accepted_gp_v1042 +=" not in transformed
    assert "J_positive = max(sign_ref * J_signed, 0.0)" not in transformed


def test_unconverged_state_cannot_reach_final_mechanics_closure():
    source = (ROOT / "arrhenius_fracture" / "sharp_front.py").read_text()
    transformed = transform_source(source)

    gate = transformed.index("if not stagger_converged_v1043:")
    failure = transformed.index("mechanics/plasticity fixed point did not converge")
    final_equilibrium = transformed.index("The converged constitutive update changes")

    assert gate < failure < final_equilibrium
