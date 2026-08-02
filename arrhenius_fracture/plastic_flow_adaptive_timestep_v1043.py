"""Adaptive timestep retry for converged v10.4.3 bulk plasticity.

The strict fixed-point layer correctly refuses an unconverged mechanics/plasticity
state.  This outer layer makes that rejection recoverable: before any physical
step is accepted, it restores the beginning-of-trial displacement, plastic
strain, dislocation density, and applied opening, reduces the trial time and
opening increments by the same factor, and retries at the unchanged loading
rate.  A hard failure remains in force if the declared retry count or minimum
trial fraction is reached.

No fracture law, hazard, material parameter, event-energy criterion, or accepted
physical trajectory is modified by an unconverged trial because rejected trials
are fully rolled back before retry.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_fixed_point_converged_v1043 import (
    transform_source as _fixed_point_transform,
)

MODEL_ID = "v10.4.3_adaptive_dt_converged_stagger_fixed_point"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_adaptive_dt_converged"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _fixed_point_transform(source)

    text = _replace_once(
        text,
        """    p.add_argument('--stagger-rho-atol-m2', type=float, default=1.0e3,
                   dest='stagger_rho_atol_m2',
                   help='Absolute dislocation-density convergence tolerance [m^-2].')
""",
        """    p.add_argument('--stagger-rho-atol-m2', type=float, default=1.0e3,
                   dest='stagger_rho_atol_m2',
                   help='Absolute dislocation-density convergence tolerance [m^-2].')
    p.add_argument('--stagger-dt-shrink', type=float, default=0.25,
                   dest='stagger_dt_shrink',
                   help='Factor applied to dt and dU after an unconverged stagger trial.')
    p.add_argument('--stagger-min-dt-fraction', type=float, default=1.0e-8,
                   dest='stagger_min_dt_fraction',
                   help='Smallest accepted fraction of the nominal dt/dU increment.')
    p.add_argument('--stagger-max-dt-retries', type=int, default=16,
                   dest='stagger_max_dt_retries',
                   help='Maximum rejected-trial timestep subdivisions per accepted step.')
""",
        "v10.4.3 adaptive stagger timestep CLI",
    )

    text = _replace_once(
        text,
        """            trial_frac = min(1.0, carry_frac * adaptive_grow) if adaptive_events else 1.0
            while True:
""",
        """            # Preserve a successful constitutive substep size even when
            # hazard-based adaptive events are disabled.  It may regrow through
            # the existing adaptive_grow control on later accepted steps.
            trial_frac = min(1.0, carry_frac * adaptive_grow)
            stagger_dt_retries_v1043 = 0
            stagger_last_rejected_residual_v1043 = None
            stagger_last_rejected_ep_residual_v1043 = None
            stagger_last_rejected_rho_residual_v1043 = None
            stagger_last_rejected_dt_s_v1043 = None
            while True:
""",
        "v10.4.3 adaptive stagger trial initialization",
    )

    text = _replace_once(
        text,
        """                if not stagger_converged_v1043:
                    raise RuntimeError(
                        'v10.4.3 mechanics/plasticity fixed point did not converge: '
                        f'T={T:.17g} K, step={step_trial}, dt_cur={dt_cur:.17g} s, '
                        f'max_iterations={int(args.n_stagger)}, '
                        f'residual={stagger_residual_v1043:.17e}, '
                        f'ep_residual={stagger_ep_residual_v1043:.17e}, '
                        f'rho_residual={stagger_rho_residual_v1043:.17e}, '
                        f'relaxation={_v1043_stagger_alpha:.17g}'
                    )

""",
        """                if not stagger_converged_v1043:
                    _v1043_dt_shrink = float(args.stagger_dt_shrink)
                    _v1043_min_trial_frac = float(args.stagger_min_dt_fraction)
                    _v1043_max_dt_retries = int(args.stagger_max_dt_retries)
                    if not (0.0 < _v1043_dt_shrink < 1.0):
                        raise RuntimeError(
                            'v10.4.3 stagger dt shrink must satisfy 0 < factor < 1: '
                            f'{_v1043_dt_shrink}'
                        )
                    if not (0.0 < _v1043_min_trial_frac <= 1.0):
                        raise RuntimeError(
                            'v10.4.3 minimum stagger dt fraction must satisfy '
                            f'0 < fraction <= 1: {_v1043_min_trial_frac}'
                        )
                    if _v1043_max_dt_retries < 0:
                        raise RuntimeError(
                            'v10.4.3 stagger maximum dt retries must be non-negative'
                        )

                    stagger_last_rejected_residual_v1043 = float(
                        stagger_residual_v1043
                    )
                    stagger_last_rejected_ep_residual_v1043 = float(
                        stagger_ep_residual_v1043
                    )
                    stagger_last_rejected_rho_residual_v1043 = float(
                        stagger_rho_residual_v1043
                    )
                    stagger_last_rejected_dt_s_v1043 = float(dt_cur)
                    _v1043_next_trial_frac = max(
                        _v1043_min_trial_frac,
                        trial_frac * _v1043_dt_shrink,
                    )
                    _v1043_can_retry = (
                        stagger_dt_retries_v1043 < _v1043_max_dt_retries
                        and _v1043_next_trial_frac
                        < trial_frac * (1.0 - 32.0 * np.finfo(float).eps)
                    )
                    if _v1043_can_retry:
                        # The constitutive trial has not entered any accepted
                        # history, hazard clock, crack geometry, or work ledger.
                        # Restore the exact beginning-of-trial physical state and
                        # retry with dt and dU reduced together at fixed rate.
                        u = u_saved
                        ep_gp = ep_saved
                        rho_gp = rho_saved
                        Uapp = Uapp_saved
                        stagger_dt_retries_v1043 += 1
                        print(
                            '  v10.4.3 stagger retry: '
                            f'T={T:.6g}K step={step_trial} '
                            f'retry={stagger_dt_retries_v1043} '
                            f'dt={dt_cur:.6g}s -> '
                            f'{cfg.loading.dt * _v1043_next_trial_frac:.6g}s '
                            f'residual={stagger_residual_v1043:.6g} '
                            f'ep={stagger_ep_residual_v1043:.6g} '
                            f'rho={stagger_rho_residual_v1043:.6g}'
                        )
                        trial_frac = _v1043_next_trial_frac
                        continue

                    raise RuntimeError(
                        'v10.4.3 mechanics/plasticity fixed point did not converge '
                        'after adaptive timestep subdivision: '
                        f'T={T:.17g} K, step={step_trial}, dt_cur={dt_cur:.17g} s, '
                        f'trial_fraction={trial_frac:.17e}, '
                        f'min_trial_fraction={_v1043_min_trial_frac:.17e}, '
                        f'dt_retries={stagger_dt_retries_v1043}, '
                        f'max_dt_retries={_v1043_max_dt_retries}, '
                        f'max_iterations={int(args.n_stagger)}, '
                        f'residual={stagger_residual_v1043:.17e}, '
                        f'ep_residual={stagger_ep_residual_v1043:.17e}, '
                        f'rho_residual={stagger_rho_residual_v1043:.17e}, '
                        f'relaxation={_v1043_stagger_alpha:.17g}'
                    )

""",
        "v10.4.3 adaptive stagger rejection and retry",
    )

    text = _replace_once(
        text,
        """                        'mechanics_plasticity_stagger_rho_atol_m2': float(
                            _v1043_stagger_rho_atol
                        ),
                        'physical_time_advance_per_accepted_step': 'dt_cur_not_n_stagger_times_dt_cur',
""",
        """                        'mechanics_plasticity_stagger_rho_atol_m2': float(
                            _v1043_stagger_rho_atol
                        ),
                        'mechanics_plasticity_stagger_adaptive_dt': True,
                        'mechanics_plasticity_stagger_dt_shrink': float(
                            args.stagger_dt_shrink
                        ),
                        'mechanics_plasticity_stagger_min_dt_fraction': float(
                            args.stagger_min_dt_fraction
                        ),
                        'mechanics_plasticity_stagger_max_dt_retries': int(
                            args.stagger_max_dt_retries
                        ),
                        'mechanics_plasticity_stagger_dt_retries_used': int(
                            stagger_dt_retries_v1043
                        ),
                        'mechanics_plasticity_stagger_accepted_trial_fraction': float(
                            trial_frac
                        ),
                        'mechanics_plasticity_stagger_accepted_dt_s': float(dt_cur),
                        'mechanics_plasticity_stagger_last_rejected_residual': (
                            stagger_last_rejected_residual_v1043
                        ),
                        'mechanics_plasticity_stagger_last_rejected_ep_residual': (
                            stagger_last_rejected_ep_residual_v1043
                        ),
                        'mechanics_plasticity_stagger_last_rejected_rho_residual': (
                            stagger_last_rejected_rho_residual_v1043
                        ),
                        'mechanics_plasticity_stagger_last_rejected_dt_s': (
                            stagger_last_rejected_dt_s_v1043
                        ),
                        'physical_time_advance_per_accepted_step': 'dt_cur_not_n_stagger_times_dt_cur',
""",
        "v10.4.3 adaptive stagger audit provenance",
    )

    return text


def load_transformed_sharp_front() -> ModuleType:
    existing = sys.modules.get(MODULE_NAME)
    if existing is not None:
        return existing

    source_path = Path(__file__).with_name("sharp_front.py")
    transformed = transform_source(source_path.read_text())
    spec = importlib.util.spec_from_loader(MODULE_NAME, loader=None)
    if spec is None:
        raise RuntimeError("could not allocate v10.4.3 adaptive-dt module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(
                transformed,
                str(source_path) + "[v10.4.3-adaptive-dt-converged]",
                "exec",
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = ["MODEL_ID", "load_transformed_sharp_front", "transform_source"]
