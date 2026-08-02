"""v10.4.3 stagger-consistent bulk-plasticity overlay.

The mechanics/plasticity stagger loop is a fixed-point iteration for one accepted
physical time increment.  Every constitutive trial must therefore start from the
same beginning-of-step plastic strain and dislocation-density state.  Only the
converged (last) stagger iterate is committed, and only its accepted constitutive
plastic-work increment is entered in the cumulative ledger.

This overlay is deliberately outside the shared ``sharp_front.py`` source so the
v10.4.0/v10.4.1 historical paths remain reproducible.  It is applied after the
v10.4.2 positive-directional-J transform and does not modify the fracture
hazard, first-passage clock, event-energy gate, or contour-shielding role.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .directional_j_positive_v1042 import transform_source as _positive_j_transform

MODEL_ID = "v10.4.3_stagger_consistent_bulk_plasticity"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_stagger_consistent"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _positive_j_transform(source)

    text = _replace_once(
        text,
        """                plastic_work_info_v1042 = None
                plastic_work_accepted_gp_v1042 = np.zeros(mesh.ne, dtype=float)
                for it in range(args.n_stagger):
""",
        """                plastic_work_info_v1042 = None
                plastic_work_accepted_gp_v1042 = None
                ep_gp_step0_v1043 = ep_gp.copy()
                rho_gp_step0_v1043 = rho_gp.copy()
                for it in range(args.n_stagger):
""",
        "v10.4.3 beginning-of-step constitutive snapshots",
    )

    text = _replace_once(
        text,
        """                        ep_gp, rho_gp, dot_ep, plastic_work_info_v1042 = update_plasticity(
                            ep_gp, rho_gp, sigma_gp, mat, T, dt_cur,
                            plast_model, cfg.dislocations, return_info=True)
                        _v1042_dWp_stagger_gp = np.asarray(
                            plastic_work_info_v1042.get('dWp_accepted_gp', []),
                            dtype=float,
                        ).reshape(-1)
                        if _v1042_dWp_stagger_gp.size != mesh.ne:
                            raise RuntimeError(
                                'v10.4.2 accepted plastic-work ledger size mismatch: '
                                f'{_v1042_dWp_stagger_gp.size} != {mesh.ne}'
                            )
                        plastic_work_accepted_gp_v1042 += np.maximum(
                            _v1042_dWp_stagger_gp, 0.0
                        )
""",
        """                        ep_gp, rho_gp, dot_ep, plastic_work_info_v1042 = update_plasticity(
                            ep_gp_step0_v1043, rho_gp_step0_v1043,
                            sigma_gp, mat, T, dt_cur,
                            plast_model, cfg.dislocations, return_info=True)
                        _v1043_dWp_converged_gp = np.asarray(
                            plastic_work_info_v1042.get('dWp_accepted_gp', []),
                            dtype=float,
                        ).reshape(-1)
                        if _v1043_dWp_converged_gp.size != mesh.ne:
                            raise RuntimeError(
                                'v10.4.3 accepted plastic-work ledger size mismatch: '
                                f'{_v1043_dWp_converged_gp.size} != {mesh.ne}'
                            )
                        if not np.all(np.isfinite(_v1043_dWp_converged_gp)):
                            raise RuntimeError(
                                'v10.4.3 accepted plastic-work ledger contains non-finite values'
                            )
                        # Assignment is intentional: each stagger is a trial from
                        # the same beginning-of-step state.  Only the final,
                        # converged trial belongs to the accepted physical step.
                        plastic_work_accepted_gp_v1042 = _v1043_dWp_converged_gp.copy()
""",
        "v10.4.3 re-based constitutive update and converged work ledger",
    )

    text = _replace_once(
        text,
        """            if (
                plastic_work_accepted_gp_v1042 is not None
                and np.asarray(plastic_work_accepted_gp_v1042).size == mesh.ne
                and isinstance(plastic_work_info_v1042, dict)
            ):
                dWp = float(
                    np.sum(
                        np.maximum(plastic_work_accepted_gp_v1042, 0.0)
                        * mesh.area_e
                    )
                )
                plastic_work_ledger_source_v1042 = (
                    'constitutive_dWp_accepted_gp_all_staggers'
                )
            else:
                dWp = float(
                    np.sum(np.sum(sigma_gp * dot_ep, axis=0) * mesh.area_e)
                ) * dt_cur
                plastic_work_ledger_source_v1042 = (
                    'post_update_sigma_dot_ep_fallback'
                )
            W_p_acc += max(dWp, 0.0)
""",
        """            if (
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
""",
        "v10.4.3 converged accepted-work integration",
    )

    text = _replace_once(
        text,
        """                        'W_bulk_plastic_primary_is_constitutive_accepted_work': (
                            plastic_work_ledger_source_v1042
                            == 'constitutive_dWp_accepted_gp_all_staggers'
                        ),
                        'W_bulk_plastic_stagger_iterations_accumulated': int(args.n_stagger),
""",
        """                        'W_bulk_plastic_primary_is_constitutive_accepted_work': (
                            plastic_work_ledger_source_v1042
                            == 'constitutive_dWp_accepted_gp_converged_stagger_rebased_state'
                        ),
                        'plastic_state_rebased_each_stagger': True,
                        'accepted_work_from_converged_stagger_only': True,
                        'mechanics_plasticity_stagger_iterations': int(args.n_stagger),
                        'physical_time_advance_per_accepted_step': 'dt_cur_not_n_stagger_times_dt_cur',
""",
        "v10.4.3 accepted-work audit provenance",
    )

    text = _replace_once(
        text,
        """    records = []
    sign_ref = float(sign_reference)
    if sign_ref == 0.0:
        sign_ref = 1.0
    for multiplier in _v1042_float_list(multipliers):
""",
        """    records = []
    sign_ref = float(sign_reference)
    if not np.isclose(sign_ref, 1.0, rtol=0.0, atol=0.0):
        raise RuntimeError(
            'v10.4.3 contour diagnostic received a non-production directional-J '
            f'sign reference: {sign_ref}'
        )
    for multiplier in _v1042_float_list(multipliers):
""",
        "v10.4.3 contour sign-reference assertion",
    )
    text = _replace_once(
        text,
        "        J_positive = max(sign_ref * J_signed, 0.0)\n",
        "        J_positive = max(J_signed, 0.0)\n",
        "v10.4.3 raw-positive contour directional J",
    )

    stale_label = "constitutive_dWp_accepted_gp_all_staggers"
    if stale_label in text:
        text = text.replace(
            stale_label,
            "constitutive_dWp_accepted_gp_converged_stagger_rebased_state",
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
        raise RuntimeError("could not allocate v10.4.3 stagger-consistent module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(transformed, str(source_path) + "[v10.4.3-stagger-consistent]", "exec"),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = ["MODEL_ID", "load_transformed_sharp_front", "transform_source"]
