"""Prescribed fixed-DeltaK waveform control for v10.2.1 fatigue runs.

The current v10 fatigue validation stage uses tip-only bulk mechanics and the
scalar K-waveform surrogate. In that setting the physically controlled fatigue
quantity is best prescribed directly: the FEM supplies crack geometry,
directional J information, and normalized tensor shape, while the front kinetics
receive an exactly fixed local DeltaK waveform after every stochastic geometry
event.

This is deliberately distinct from a full cyclic-FEM displacement controller.
When calibrated cyclic bulk plasticity is enabled later, the same target DeltaK
should be enforced by feedback on the displacement amplitude instead.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
import inspect
from typing import Any, Callable, Iterator


MODEL_ID = "v10.2.1_prescribed_fixed_deltaK"


@dataclass
class FixedDeltaKConfig:
    target_deltaK_MPa_sqrt_m: float

    def validate(self) -> "FixedDeltaKConfig":
        self.target_deltaK_MPa_sqrt_m = float(self.target_deltaK_MPa_sqrt_m)
        if not self.target_deltaK_MPa_sqrt_m > 0.0:
            raise ValueError("target DeltaK must be positive")
        return self

    @property
    def target_deltaK_Pa_sqrt_m(self) -> float:
        return self.target_deltaK_MPa_sqrt_m * 1.0e6

    def target_Kmax_Pa_sqrt_m(self, R: float) -> float:
        R = float(R)
        if not 0.0 <= R < 1.0:
            raise ValueError(
                "v10.2.1 fixed-DeltaK control currently requires 0 <= R < 1"
            )
        return self.target_deltaK_Pa_sqrt_m / max(1.0 - R, 1.0e-300)


_AUDIT: dict[str, Any] = {}


def reset_fixed_deltaK_audit(config: FixedDeltaKConfig) -> None:
    global _AUDIT
    _AUDIT = {
        "schema": MODEL_ID,
        "config": asdict(config.validate()),
        "waveforms_created": 0,
        "minimum_actual_deltaK_Pa_sqrt_m": None,
        "maximum_actual_deltaK_Pa_sqrt_m": None,
        "maximum_abs_target_error_Pa_sqrt_m": 0.0,
        "incoming_Kmax_min_Pa_sqrt_m": None,
        "incoming_Kmax_max_Pa_sqrt_m": None,
        "R_values": [],
        "control_mode": "prescribed_local_K_waveform",
        "fem_role": "geometry_directional_J_and_normalized_tensor_shape",
        "full_cyclic_displacement_feedback": False,
    }


def fixed_deltaK_audit_payload() -> dict[str, Any]:
    return dict(_AUDIT)


def _update_minmax(key_min: str, key_max: str, value: float) -> None:
    lo = _AUDIT.get(key_min)
    hi = _AUDIT.get(key_max)
    _AUDIT[key_min] = float(value) if lo is None else min(float(lo), float(value))
    _AUDIT[key_max] = float(value) if hi is None else max(float(hi), float(value))


def make_fixed_deltaK_waveform_factory(
    original: Callable[..., Any],
    config: FixedDeltaKConfig,
) -> Callable[..., Any]:
    """Return a constructor that replaces incoming Kmax by the target DeltaK."""
    cfg = FixedDeltaKConfig(config.target_deltaK_MPa_sqrt_m).validate()
    signature = inspect.signature(original)

    def factory(*args, **kwargs):
        bound = signature.bind_partial(*args, **kwargs)
        values = dict(bound.arguments)
        R = float(values.get("R", kwargs.get("R", 0.1)))
        incoming_Kmax = float(values.get("Kmax", kwargs.get("Kmax", 0.0)))
        target_Kmax = cfg.target_Kmax_Pa_sqrt_m(R)
        values["Kmax"] = target_Kmax
        waveform = original(**values)

        actual_deltaK = float(getattr(waveform, "DeltaK"))
        target_deltaK = cfg.target_deltaK_Pa_sqrt_m
        _AUDIT["waveforms_created"] = int(_AUDIT.get("waveforms_created", 0)) + 1
        _update_minmax(
            "minimum_actual_deltaK_Pa_sqrt_m",
            "maximum_actual_deltaK_Pa_sqrt_m",
            actual_deltaK,
        )
        _update_minmax(
            "incoming_Kmax_min_Pa_sqrt_m",
            "incoming_Kmax_max_Pa_sqrt_m",
            incoming_Kmax,
        )
        _AUDIT["maximum_abs_target_error_Pa_sqrt_m"] = max(
            float(_AUDIT.get("maximum_abs_target_error_Pa_sqrt_m", 0.0)),
            abs(actual_deltaK - target_deltaK),
        )
        r_values = list(_AUDIT.get("R_values", []))
        if not any(abs(float(old) - R) <= 1.0e-14 for old in r_values):
            r_values.append(R)
        _AUDIT["R_values"] = sorted(r_values)
        return waveform

    factory.__name__ = "FixedDeltaKFatigueWaveform"
    factory.__doc__ = (
        "Construct the inherited waveform with Kmax adjusted so DeltaK equals "
        f"{cfg.target_deltaK_MPa_sqrt_m:g} MPa*sqrt(m)."
    )
    return factory


@contextmanager
def install_fixed_deltaK_waveform(
    target_deltaK_MPa_sqrt_m: float,
) -> Iterator[FixedDeltaKConfig]:
    """Temporarily prescribe DeltaK for every fatigue waveform in the 2-D driver.

    ``sharp_front.run_2d`` imports ``FatigueWaveform`` from ``fatigue_v1`` when
    fatigue setup is entered. Patching that defining module before dispatch is
    therefore sufficient; ``sharp_front`` does not expose a module-level
    ``FatigueWaveform`` symbol.
    """
    from . import fatigue_v1

    cfg = FixedDeltaKConfig(target_deltaK_MPa_sqrt_m).validate()
    reset_fixed_deltaK_audit(cfg)

    original_module = fatigue_v1.FatigueWaveform
    factory = make_fixed_deltaK_waveform_factory(original_module, cfg)
    fatigue_v1.FatigueWaveform = factory
    try:
        yield cfg
    finally:
        fatigue_v1.FatigueWaveform = original_module


__all__ = [
    "FixedDeltaKConfig",
    "MODEL_ID",
    "fixed_deltaK_audit_payload",
    "install_fixed_deltaK_waveform",
    "make_fixed_deltaK_waveform_factory",
    "reset_fixed_deltaK_audit",
]
