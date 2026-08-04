"""v10.4.8 campaign entry with complete numerical-failure arbitration.

The constitutive equations, FEM equilibrium, directional configurational J,
Arrhenius first-passage law, fracture event-energy gate, v10.4.6 physical
plastic terminal, and v10.4.7 accepted-substep stagnation terminal are
unchanged.

This entry adds a distinct unsuccessful outcome when the mechanics/plasticity
fixed point exhausts the configured minimum adaptive trial fraction before a
bounded accepted-substep terminal window can be formed.  That outcome is not
plasticity dominated and is not numerical stagnation: it is an explicit
nonlinear-solver failure after partial or zero crack growth.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys

from . import sharp_front_v10_4_4_plasticity_dominated_audited as _v1044
from . import sharp_front_v10_4_7_numerical_stagnation_audited as _v1047

MODEL_ID = "v10.4.8_complete_numerical_failure_arbitration"
FIXED_POINT_FAILURE_PREFIX = (
    "v10.4.3 mechanics/plasticity fixed point did not converge after adaptive "
    "timestep subdivision:"
)
FIXED_POINT_FAILURE_BASIS = (
    "minimum_adaptive_trial_fraction_exhausted_without_mechanics_plasticity_"
    "fixed_point_convergence_v1048"
)
FIXED_POINT_FAILURE_EXIT_CODE = 5


class FixedPointFailure(RuntimeError):
    """Structured form of the production fixed-point exhaustion exception."""

    def __init__(self, message: str, diagnostics: dict):
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def _coerce_value(value: str):
    text = value.strip()
    if text.endswith(" K"):
        text = text[:-2].strip()
    try:
        numeric = float(text)
    except ValueError:
        return value.strip()
    if numeric.is_integer() and all(token not in text.lower() for token in (".", "e")):
        return int(numeric)
    return numeric


def _parse_fixed_point_failure(exc: BaseException) -> FixedPointFailure | None:
    message = str(exc)
    if not message.startswith(FIXED_POINT_FAILURE_PREFIX):
        return None

    diagnostics = {}
    body = message[len(FIXED_POINT_FAILURE_PREFIX) :].strip()
    for key, raw_value in re.findall(r"([A-Za-z_]+)=([^,]+)", body):
        diagnostics[key] = _coerce_value(raw_value)

    aliases = {
        "T": "temperature_K",
        "dt_cur": "dt_cur_s",
    }
    for source, target in aliases.items():
        if source in diagnostics:
            diagnostics[target] = diagnostics[source]

    return FixedPointFailure(message, diagnostics)


def _fixed_point_failure_payload(failure: FixedPointFailure) -> dict:
    diagnostics = dict(failure.diagnostics)
    payload = {
        "schema": "v10.4.8_numerical_fixed_point_failure_audit_v1",
        "campaign_model_id": MODEL_ID,
        "classification": "numerical_fixed_point_failure",
        "failure_basis": FIXED_POINT_FAILURE_BASIS,
        "exit_code": FIXED_POINT_FAILURE_EXIT_CODE,
        "complete": False,
        "fracture_target_reached": False,
        "plasticity_dominated": False,
        "physical_plasticity_terminal_accepted": False,
        "numerical_stagnation_terminal_reached": False,
        "retry_same_numerics_recommended": False,
        "exception_type": "RuntimeError",
        "exception_message": str(failure),
        "diagnostics": diagnostics,
        "fracture_hazard_unchanged": True,
        "fracture_event_energy_gate_unchanged": True,
        "bulk_plastic_work_enters_fracture_hazard": False,
        "bulk_plastic_work_enters_fracture_energy_gate": False,
        "production_physics_modified": False,
    }
    payload.update(diagnostics)
    return payload


def _write_fixed_point_failure(root: Path, failure: FixedPointFailure) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = _fixed_point_failure_payload(failure)

    audit_tmp = root / "numerical_fixed_point_failure_audit.json.tmp"
    audit_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    audit_tmp.replace(root / "numerical_fixed_point_failure_audit.json")

    marker_tmp = root / "NUMERICAL_FIXED_POINT_FAILURE.tmp"
    marker_tmp.write_text(FIXED_POINT_FAILURE_BASIS + "\n")
    marker_tmp.replace(root / "NUMERICAL_FIXED_POINT_FAILURE")

    # Fail closed: neither numerical outcome may masquerade as completion.
    for name in (
        "COMPLETE",
        "PLASTIC_FLOW",
        "PLASTICITY_DOMINATED",
        "NUMERICAL_STAGNATION",
        "numerical_stagnation_audit.json",
        "stage3_case_status.json",
    ):
        (root / name).unlink(missing_ok=True)

    model_path = root / "v10_4_bulk_coupled_model_audit.json"
    model_payload = (
        json.loads(model_path.read_text()) if model_path.is_file() else {}
    )
    model_payload.update(
        {
            "schema": MODEL_ID,
            "numerical_fixed_point_failure_model": MODEL_ID,
            "numerical_fixed_point_failure_exit_code": (
                FIXED_POINT_FAILURE_EXIT_CODE
            ),
            "numerical_fixed_point_failure_is_successful_terminal": False,
            "fixed_point_failure_basis": FIXED_POINT_FAILURE_BASIS,
            "fracture_hazard_unchanged": True,
            "fracture_event_energy_gate_unchanged": True,
            "production_physics_modified": False,
        }
    )
    model_path.write_text(
        json.dumps(model_payload, indent=2, sort_keys=True) + "\n"
    )


def _rewrite_v1048_success_audit(root: Path) -> None:
    model_path = root / "v10_4_bulk_coupled_model_audit.json"
    payload = json.loads(model_path.read_text()) if model_path.is_file() else {}
    payload.update(
        {
            "schema": MODEL_ID,
            "numerical_fixed_point_failure_model": MODEL_ID,
            "numerical_fixed_point_failure_exit_code": (
                FIXED_POINT_FAILURE_EXIT_CODE
            ),
            "numerical_fixed_point_failure_is_successful_terminal": False,
            "fixed_point_failure_basis": FIXED_POINT_FAILURE_BASIS,
            "fracture_hazard_unchanged": True,
            "fracture_event_energy_gate_unchanged": True,
            "production_physics_modified": False,
        }
    )
    model_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    print(
        "  v10.4.8 numerical-failure arbitration: unchanged fracture and "
        "plasticity physics; accepted severe-substep plateaus retain the v10.4.7 "
        "exit-4 arbitration, while exhaustion of the minimum adaptive trial "
        "fraction is audited separately as NUMERICAL_FIXED_POINT_FAILURE "
        "with exit 5"
    )

    try:
        result = _v1047.main(args)
    except RuntimeError as exc:
        failure = _parse_fixed_point_failure(exc)
        if failure is None:
            raise
        out = _v1044._option_value(args, "--out")
        if out:
            _write_fixed_point_failure(Path(out), failure)
        print(
            "  NUMERICAL_FIXED_POINT_FAILURE: mechanics/plasticity fixed point "
            "did not converge before the minimum adaptive trial fraction was "
            "exhausted",
            file=sys.stderr,
        )
        raise SystemExit(FIXED_POINT_FAILURE_EXIT_CODE) from exc

    out = _v1044._option_value(args, "--out")
    if out:
        _rewrite_v1048_success_audit(Path(out))
    return result


if __name__ == "__main__":
    main()
