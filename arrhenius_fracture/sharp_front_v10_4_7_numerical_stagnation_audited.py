"""v10.4.7 campaign entry with fail-fast numerical-stagnation handling.

The constitutive model, FEM equilibrium, directional configurational J,
Arrhenius first-passage law, fracture event-energy gate, and v10.4.6 physical
plasticity terminal are unchanged.

After the bounded severe-substep window, a case has exactly one of two outcomes:

* if the v10.4.6 cumulative plastic-work fraction gate is satisfied, the case
  may terminate successfully as physically plasticity dominated; or
* if all severe-substep/no-growth/plateau gates are satisfied but cumulative
  plastic work is below the configured threshold, the case terminates
  unsuccessfully as numerical stagnation.

Numerical stagnation writes an explicit audit and marker, exits nonzero, and is
never converted into COMPLETE, PLASTIC_FLOW, or PLASTICITY_DOMINATED.
"""
from __future__ import annotations

from functools import wraps
import json
from pathlib import Path
import sys

from . import sharp_front_v10_4_4_plasticity_dominated_audited as _v1044
from . import sharp_front_v10_4_6_plasticity_dominance_audited as _v1046

MODEL_ID = "v10.4.7_physical_plastic_terminal_with_fail_fast_numerical_stagnation"
NUMERICAL_STAGNATION_BASIS = (
    "bounded_severe_substep_stagnation_without_cumulative_plastic_dominance_v1047"
)
NUMERICAL_STAGNATION_EXIT_CODE = 4


class NumericalStagnationError(RuntimeError):
    """Raised when bounded severe substepping is physical-terminal ineligible."""

    def __init__(self, metrics: dict):
        super().__init__(
            "bounded severe adaptive-substep stagnation without cumulative "
            "bulk-plastic-work dominance"
        )
        self.metrics = dict(metrics)


def _is_numerical_stagnation(metrics: dict | None) -> bool:
    if not isinstance(metrics, dict):
        return False
    criteria = dict(metrics.get("criteria", {}))
    return (
        bool(criteria.get("no_crack_event_in_window", False))
        and bool(criteria.get("negligible_crack_extension", False))
        and bool(criteria.get("load_carrying_response_plateau", False))
        and bool(criteria.get("adaptive_substep_stagnation", False))
        and not bool(
            criteria.get("cumulative_bulk_plastic_work_dominant", False)
        )
    )


def _numerical_stagnation_payload(metrics: dict) -> dict:
    payload = dict(metrics)
    payload.update(
        {
            "schema": "v10.4.7_numerical_stagnation_audit_v1",
            "campaign_model_id": MODEL_ID,
            "classification": "numerical_stagnation_not_plasticity_dominated",
            "failure_basis": NUMERICAL_STAGNATION_BASIS,
            "exit_code": NUMERICAL_STAGNATION_EXIT_CODE,
            "complete": False,
            "fracture_target_reached": False,
            "plasticity_dominated": False,
            "physical_plasticity_terminal_accepted": False,
            "retry_same_numerics_recommended": False,
            "fracture_hazard_unchanged": True,
            "fracture_event_energy_gate_unchanged": True,
            "bulk_plastic_work_enters_fracture_hazard": False,
            "bulk_plastic_work_enters_fracture_energy_gate": False,
        }
    )
    return payload


def _write_numerical_stagnation(root: Path, metrics: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = _numerical_stagnation_payload(metrics)

    audit_tmp = root / "numerical_stagnation_audit.json.tmp"
    audit_tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    audit_tmp.replace(root / "numerical_stagnation_audit.json")

    marker_tmp = root / "NUMERICAL_STAGNATION.tmp"
    marker_tmp.write_text(NUMERICAL_STAGNATION_BASIS + "\n")
    marker_tmp.replace(root / "NUMERICAL_STAGNATION")

    # Fail closed: this outcome is never a successful campaign terminal.
    for name in ("COMPLETE", "PLASTIC_FLOW", "PLASTICITY_DOMINATED"):
        (root / name).unlink(missing_ok=True)


def _load_transformed_sharp_front_v1047():
    """Add fail-fast arbitration after the v10.4.6 physical terminal."""
    module = _v1046._load_transformed_sharp_front_v1046()
    if getattr(module, "_v1047_numerical_stagnation_wrapped", False):
        return module

    physical_terminal_metrics = module._v1042_terminal_metrics

    @wraps(physical_terminal_metrics)
    def wrapped(window, args, **kwargs):
        result = physical_terminal_metrics(window, args, **kwargs)
        if result is not None and bool(result.get("criteria_pass", False)):
            return result

        fallback = _v1046._dominance_substep_metrics(window, args, **kwargs)
        if _is_numerical_stagnation(fallback):
            raise NumericalStagnationError(fallback)

        return result

    module._v1042_terminal_metrics = wrapped
    module._v1047_numerical_stagnation_wrapped = True
    module._v1047_numerical_stagnation_model_id = MODEL_ID
    return module


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)

    old_loader = _v1044.load_transformed_sharp_front
    _v1044.load_transformed_sharp_front = _load_transformed_sharp_front_v1047
    try:
        print(
            "  v10.4.7 numerical-stagnation arbitration: unchanged fracture and "
            "plasticity physics; after the bounded 128-substep severe-subdivision "
            "window, a physical plastic terminal still requires the configured "
            "cumulative plastic fraction; otherwise a no-growth J/force plateau "
            "fails fast as NUMERICAL_STAGNATION with a nonzero exit"
        )
        try:
            result = _v1044.main(args)
        except NumericalStagnationError as exc:
            out = _v1044._option_value(args, "--out")
            if out:
                _write_numerical_stagnation(Path(out), exc.metrics)
            print(
                "  NUMERICAL_STAGNATION: severe adaptive subdivision and a "
                "no-growth J/force plateau were reached without cumulative "
                "bulk-plastic-work dominance",
                file=sys.stderr,
            )
            raise SystemExit(NUMERICAL_STAGNATION_EXIT_CODE) from exc
    finally:
        _v1044.load_transformed_sharp_front = old_loader

    out = _v1044._option_value(args, "--out")
    if out:
        root = Path(out)
        _v1046._rewrite_v1046_audits(root)
        model_path = root / "v10_4_bulk_coupled_model_audit.json"
        payload = json.loads(model_path.read_text()) if model_path.is_file() else {}
        payload.update(
            {
                "schema": MODEL_ID,
                "numerical_stagnation_model": MODEL_ID,
                "numerical_stagnation_exit_code": NUMERICAL_STAGNATION_EXIT_CODE,
                "numerical_stagnation_is_successful_terminal": False,
                "numerical_stagnation_fail_fast_window_steps": 128,
                "fracture_hazard_unchanged": True,
                "fracture_event_energy_gate_unchanged": True,
            }
        )
        model_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return result


if __name__ == "__main__":
    main()
