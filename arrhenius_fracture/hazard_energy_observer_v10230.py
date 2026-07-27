"""Single-front mechanics observer for the v10.2.30 hazard-energy gate.

The low-level sharp-front driver imports its cleavage selectors and J-integral
routine inside ``run_2d``.  This module wraps those defining functions before the
run begins, records the selected relative plane factor, and mirrors the driver's
root-signed positive-J convention.  It does not alter the returned selector or
mechanics values.

The production v10.2.28/v10.2.29 campaigns are single-front.  v10.2.30 therefore
fails closed if more than one independently keyed front is requested; a future
branching implementation should replace this observer with a front-ID map.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import functools
import math
from typing import Any, Callable

from . import crystal
from . import j_integral

MODEL_ID = "v10.2.30_single_front_hazard_energy_observer"


@dataclass
class MechanicsObservation:
    gamma_rel: float = 1.0
    J_probe_J_per_m2: float = 0.0
    J_signed_J_per_m2: float = 0.0
    K_probe_Pa_sqrt_m: float = 0.0
    J_sign_reference: float = 0.0
    direction_angle_deg: float = 0.0
    selector_serial: int = 0
    mechanics_serial: int = 0
    source: str = "uninitialized"


_STATE = MechanicsObservation()
_ORIGINALS: dict[str, Callable[..., Any]] = {}
_INSTALLED = False


def reset_observer() -> None:
    global _STATE
    _STATE = MechanicsObservation()


def _candidate_rows(result: Any) -> list[dict[str, Any]]:
    selected = result[0] if isinstance(result, tuple) else result
    if selected is None:
        return []
    if isinstance(selected, list):
        return selected
    return list(selected)


def _gamma(candidate: dict[str, Any] | None) -> float:
    if not candidate:
        return 1.0
    raw = candidate.get("gamma_rel", candidate.get("gamma", 1.0))
    value = float(raw)
    if not math.isfinite(value) or value <= 0.0:
        raise RuntimeError("selected cleavage direction has invalid gamma_rel")
    candidate["gamma_rel"] = value
    return value


def _record_selection(result: Any, source: str) -> None:
    rows = _candidate_rows(result)
    if not rows:
        return
    winner = rows[0]
    gamma = _gamma(winner)
    direction = winner.get("t", [1.0, 0.0])
    try:
        angle = math.degrees(math.atan2(float(direction[1]), float(direction[0])))
    except Exception:
        angle = 0.0
    _STATE.gamma_rel = gamma
    _STATE.direction_angle_deg = float(angle)
    _STATE.selector_serial += 1
    _STATE.source = str(source)


def _selector_wrapper(original: Callable[..., Any], source: str):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        _record_selection(result, source)
        return result

    wrapped._v10230_hazard_energy_observer = True
    return wrapped


def _j_wrapper(original: Callable[..., Any]):
    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        result = original(*args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 3:
            raise RuntimeError("directional J routine returned an unexpected contract")
        _J_legacy, K_raw, info = result
        if not isinstance(info, dict):
            raise RuntimeError("directional J routine did not return an info dictionary")
        J_signed = float(info.get("J_signed", info.get("J", 0.0)) or 0.0)
        if not math.isfinite(J_signed):
            raise RuntimeError("directional J probe returned a nonfinite signed value")
        if _STATE.J_sign_reference == 0.0 and abs(J_signed) > 1.0e-30:
            _STATE.J_sign_reference = 1.0 if J_signed > 0.0 else -1.0
        J_effective = max(_STATE.J_sign_reference * J_signed, 0.0)
        K_value = max(float(K_raw), 0.0) if J_effective > 0.0 else 0.0
        if not math.isfinite(K_value):
            raise RuntimeError("directional J probe returned a nonfinite K")

        _STATE.J_signed_J_per_m2 = J_signed
        _STATE.J_probe_J_per_m2 = J_effective
        _STATE.K_probe_Pa_sqrt_m = K_value
        _STATE.mechanics_serial += 1
        info["v10230_gamma_rel"] = float(_STATE.gamma_rel)
        info["v10230_J_effective_J_per_m2"] = float(J_effective)
        info["v10230_K_probe_Pa_sqrt_m"] = float(K_value)
        info["v10230_J_sign_reference"] = float(_STATE.J_sign_reference)
        return result

    wrapped._v10230_hazard_energy_observer = True
    return wrapped


def install_observer() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    reset_observer()
    for name in ("cleave_direction_competition", "cleavage_branch_candidates"):
        original = getattr(crystal, name)
        _ORIGINALS[f"crystal.{name}"] = original
        setattr(crystal, name, _selector_wrapper(original, name))
    _ORIGINALS["j_integral.compute_J_integral"] = j_integral.compute_J_integral
    j_integral.compute_J_integral = _j_wrapper(j_integral.compute_J_integral)
    _INSTALLED = True


def restore_observer() -> None:
    global _INSTALLED
    if not _INSTALLED:
        return
    for name in ("cleave_direction_competition", "cleavage_branch_candidates"):
        original = _ORIGINALS.get(f"crystal.{name}")
        if original is not None:
            setattr(crystal, name, original)
    original_j = _ORIGINALS.get("j_integral.compute_J_integral")
    if original_j is not None:
        j_integral.compute_J_integral = original_j
    _ORIGINALS.clear()
    _INSTALLED = False


def current_observation() -> MechanicsObservation:
    if _STATE.mechanics_serial < 1:
        raise RuntimeError(
            "v10.2.30 hazard-energy gate requires a completed directional-J probe"
        )
    return MechanicsObservation(**asdict(_STATE))


def audit_payload() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "installed": bool(_INSTALLED),
        "single_front_only": True,
        "root_signed_positive_J": True,
        "continuous_and_discrete_gamma_contracts_normalized": True,
        "state": asdict(_STATE),
    }


__all__ = [
    "MODEL_ID",
    "MechanicsObservation",
    "audit_payload",
    "current_observation",
    "install_observer",
    "reset_observer",
    "restore_observer",
]
