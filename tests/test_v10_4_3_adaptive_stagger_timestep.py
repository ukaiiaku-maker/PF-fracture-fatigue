from __future__ import annotations

from pathlib import Path

from arrhenius_fracture.plastic_flow_adaptive_timestep_v1043 import (
    transform_source,
)

ROOT = Path(__file__).resolve().parents[1]


def _transformed() -> str:
    source = (ROOT / "arrhenius_fracture" / "sharp_front.py").read_text()
    transformed = transform_source(source)
    compile(transformed, "sharp_front.py[v10.4.3-adaptive-dt-test]", "exec")
    return transformed


def test_adaptive_stagger_retry_is_present_and_state_conservative():
    transformed = _transformed()
    required = {
        "dt shrink CLI": "--stagger-dt-shrink",
        "minimum fraction CLI": "--stagger-min-dt-fraction",
        "maximum retries CLI": "--stagger-max-dt-retries",
        "preserve accepted substep": "trial_frac = min(1.0, carry_frac * adaptive_grow)",
        "displacement rollback": "u = u_saved",
        "plastic strain rollback": "ep_gp = ep_saved",
        "density rollback": "rho_gp = rho_saved",
        "load rollback": "Uapp = Uapp_saved",
        "fraction shrink": "trial_frac * _v1043_dt_shrink",
        "retry continuation": "trial_frac = _v1043_next_trial_frac\n                        continue",
        "hard failure": "did not converge after adaptive timestep subdivision",
        "fixed rate statement": "retry with dt and dU reduced together at fixed rate",
    }
    missing = [label for label, token in required.items() if token not in transformed]
    assert not missing, missing


def test_retry_precedes_hard_failure_and_final_equilibrium():
    transformed = _transformed()
    retry = transformed.index("if _v1043_can_retry:")
    rollback = transformed.index("u = u_saved", retry)
    retry_continue = transformed.index("continue", rollback)
    hard_failure = transformed.index(
        "did not converge after adaptive timestep subdivision", retry_continue
    )
    final_equilibrium = transformed.index(
        "The converged constitutive update changes", hard_failure
    )
    assert retry < rollback < retry_continue < hard_failure < final_equilibrium


def test_rejected_trial_does_not_enter_accepted_work_or_hazard_path():
    transformed = _transformed()
    retry = transformed.index("if _v1043_can_retry:")
    retry_continue = transformed.index("continue", retry)
    accepted_work = transformed.index("W_p_acc += dWp", retry_continue)
    hazard_commit = transformed.index("eng.step(KJ, T, dt_cur)", retry_continue)
    assert retry_continue < accepted_work
    assert retry_continue < hazard_commit
