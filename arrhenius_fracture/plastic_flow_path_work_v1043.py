"""Endpoint-path plastic-work accounting for the v10.4.3 fixed point.

The rebased fixed-point solve evaluates every constitutive candidate from the
beginning-of-step state, but the stress supplied to the converged candidate is
already close to the final relaxed stress.  The constitutive helper's local
``dWp_accepted_gp`` formula was written for a sequential return update where the
supplied stress is the pre-return stress.  Reusing it as the primary ledger in a
rebased fixed point therefore undercounts the work of the full accepted plastic
increment once relaxation becomes appreciable.

This overlay leaves the accepted plastic strain, dislocation density, fracture
hazard, event gate, and timestep controller unchanged.  For accepted monotonic
steps before any crack event it computes the diagnostic bulk-plastic work from

    0.5 * (sigma_begin + sigma_end) : (ep_end - ep_begin)

using the equilibrated accepted stresses at the two ends of the physical step
and the actual accepted plastic-strain increment.  The original constitutive
estimate is retained in a separate cumulative history for comparison.  Event
steps retain the constitutive ledger because the crack-topology change occurs
between the constitutive solve and final output and would otherwise mix two
geometries in one endpoint contraction.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_adaptive_timestep_v1043 import (
    transform_source as _adaptive_transform,
)

MODEL_ID = "v10.4.3_equilibrated_endpoint_path_plastic_work"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_endpoint_path_work"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _adaptive_transform(source)

    text = _replace_once(
        text,
        """        hist = {k: [] for k in ('Uapp', 'Ftop', 'KJ', 'W_ext', 'U_el', 'W_p',
                                'W_emit', 'rho_mean', 'rho_p95', 'rho_p99',
""",
        """        hist = {k: [] for k in ('Uapp', 'Ftop', 'KJ', 'W_ext', 'U_el', 'W_p',
                                'W_p_constitutive',
                                'W_emit', 'rho_mean', 'rho_p95', 'rho_p99',
""",
        "v10.4.3 constitutive comparison history",
    )

    text = _replace_once(
        text,
        """        W_ext_acc = 0.0; W_p_acc = 0.0; Ftop_prev = 0.0; Uapp_prev = 0.0
""",
        """        W_ext_acc = 0.0; W_p_acc = 0.0; Ftop_prev = 0.0; Uapp_prev = 0.0
        W_p_constitutive_acc_v1043 = 0.0
""",
        "v10.4.3 constitutive comparison accumulator",
    )

    text = _replace_once(
        text,
        """            # Preserve a successful constitutive substep size even when
            # hazard-based adaptive events are disabled.  It may regrow through
            # the existing adaptive_grow control on later accepted steps.
            trial_frac = min(1.0, carry_frac * adaptive_grow)
""",
        """            # Endpoint states for the accepted-step path-work diagnostic.
            # These snapshots are outside the rejected-trial loop, so every retry
            # is compared with the same beginning-of-accepted-step state.
            sigma_gp_step0_path_v1043 = np.asarray(
                sigma_gp, dtype=float
            ).copy()
            ep_gp_step0_path_v1043 = np.asarray(ep_gp, dtype=float).copy()

            # Preserve a successful constitutive substep size even when
            # hazard-based adaptive events are disabled.  It may regrow through
            # the existing adaptive_grow control on later accepted steps.
            trial_frac = min(1.0, carry_frac * adaptive_grow)
""",
        "v10.4.3 endpoint path snapshots",
    )

    old_work = """            if (
                plastic_work_accepted_gp_v1042 is not None
                and np.asarray(plastic_work_accepted_gp_v1042).size == mesh.ne
                and isinstance(plastic_work_info_v1042, dict)
            ):
                _v1043_dWp_gp = np.asarray(
                    plastic_work_accepted_gp_v1042, dtype=float
                ).reshape(-1)
                dWp = float(np.sum(_v1043_dWp_gp * mesh.area_e))
                _v1043_dWp_scale = float(
                    np.sum(np.abs(_v1043_dWp_gp) * mesh.area_e)
                )
                plastic_work_ledger_source_v1042 = (
                    'constitutive_dWp_accepted_gp_converged_stagger_rebased_state'
                )
            else:
                dWp = float(
                    np.sum(np.sum(sigma_gp * dot_ep, axis=0) * mesh.area_e)
                ) * dt_cur
                _v1043_dWp_scale = abs(dWp)
                plastic_work_ledger_source_v1042 = (
                    'post_update_sigma_dot_ep_fallback'
                )
            _v1043_negative_work_tol = max(
                1.0e-14 * max(_v1043_dWp_scale, 1.0),
                64.0 * np.finfo(float).eps * max(_v1043_dWp_scale, 1.0),
            )
            if dWp < -_v1043_negative_work_tol:
                raise RuntimeError(
                    'v10.4.3 accepted bulk plastic work is materially negative: '
                    f'dWp={dWp:.17e}, tolerance={_v1043_negative_work_tol:.17e}'
                )
            if dWp < 0.0:
                dWp = 0.0
            W_p_acc += dWp
"""

    new_work = """            if (
                plastic_work_accepted_gp_v1042 is not None
                and np.asarray(plastic_work_accepted_gp_v1042).size == mesh.ne
                and isinstance(plastic_work_info_v1042, dict)
            ):
                _v1043_dWp_constitutive_gp = np.asarray(
                    plastic_work_accepted_gp_v1042, dtype=float
                ).reshape(-1)
                dWp_constitutive_v1043 = float(
                    np.sum(_v1043_dWp_constitutive_gp * mesh.area_e)
                )
            else:
                dWp_constitutive_v1043 = float(
                    np.sum(np.sum(sigma_gp * dot_ep, axis=0) * mesh.area_e)
                ) * dt_cur

            _v1043_path_shapes_match = (
                np.asarray(sigma_gp_step0_path_v1043).shape
                == np.asarray(sigma_gp).shape
                and np.asarray(ep_gp_step0_path_v1043).shape
                == np.asarray(ep_gp).shape
                and np.asarray(sigma_gp).shape
                == np.asarray(ep_gp).shape
                and np.asarray(sigma_gp).shape[1] == mesh.ne
            )
            _v1043_prefracture_path_step = Kc_first is None
            if _v1043_path_shapes_match and _v1043_prefracture_path_step:
                _v1043_dep_path_gp = (
                    np.asarray(ep_gp, dtype=float)
                    - np.asarray(ep_gp_step0_path_v1043, dtype=float)
                )
                _v1043_sigma_path_avg_gp = 0.5 * (
                    np.asarray(sigma_gp_step0_path_v1043, dtype=float)
                    + np.asarray(sigma_gp, dtype=float)
                )
                _v1043_dWp_path_gp = np.sum(
                    _v1043_sigma_path_avg_gp * _v1043_dep_path_gp,
                    axis=0,
                )
                dWp = float(np.sum(_v1043_dWp_path_gp * mesh.area_e))
                _v1043_dWp_scale = float(
                    np.sum(np.abs(_v1043_dWp_path_gp) * mesh.area_e)
                )
                plastic_work_ledger_source_v1042 = (
                    'equilibrated_endpoint_trapezoid_sigma_colon_delta_ep'
                )
            else:
                dWp = dWp_constitutive_v1043
                _v1043_dWp_scale = abs(dWp)
                plastic_work_ledger_source_v1042 = (
                    'constitutive_event_step_or_shape_mismatch_fallback'
                )

            _v1043_negative_work_tol = max(
                1.0e-14 * max(_v1043_dWp_scale, 1.0),
                64.0 * np.finfo(float).eps * max(_v1043_dWp_scale, 1.0),
            )
            if dWp < -_v1043_negative_work_tol:
                raise RuntimeError(
                    'v10.4.3 accepted endpoint-path bulk plastic work is '
                    'materially negative: '
                    f'dWp={dWp:.17e}, tolerance={_v1043_negative_work_tol:.17e}'
                )
            if dWp < 0.0:
                dWp = 0.0
            if dWp_constitutive_v1043 < 0.0 and abs(dWp_constitutive_v1043) \
                    <= _v1043_negative_work_tol:
                dWp_constitutive_v1043 = 0.0
            W_p_acc += dWp
            W_p_constitutive_acc_v1043 += max(
                dWp_constitutive_v1043, 0.0
            )
"""

    text = _replace_once(
        text,
        old_work,
        new_work,
        "v10.4.3 equilibrated endpoint path-work integration",
    )

    text = _replace_once(
        text,
        """            hist['U_el'].append(U_el); hist['W_p'].append(W_p_acc)
            hist['W_emit'].append(W_emit_tot)
""",
        """            hist['U_el'].append(U_el); hist['W_p'].append(W_p_acc)
            hist['W_p_constitutive'].append(
                W_p_constitutive_acc_v1043
            )
            hist['W_emit'].append(W_emit_tot)
""",
        "v10.4.3 constitutive comparison history append",
    )

    text = text.replace(
        "constitutive_dWp_accepted_gp_converged_stagger_rebased_state",
        "equilibrated_endpoint_trapezoid_sigma_colon_delta_ep",
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
        raise RuntimeError("could not allocate v10.4.3 endpoint-path module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(
                transformed,
                str(source_path) + "[v10.4.3-endpoint-path-work]",
                "exec",
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = ["MODEL_ID", "load_transformed_sharp_front", "transform_source"]
