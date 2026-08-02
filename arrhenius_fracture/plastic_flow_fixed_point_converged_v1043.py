"""Strict relaxed fixed-point closure for the v10.4.3 bulk-plasticity step.

The v10.4.3 state-rebase repair prevents ``n_stagger`` from multiplying the
physical constitutive time increment.  A fixed iteration count alone is not a
convergence criterion, however.  This outer overlay treats ``n_stagger`` as a
maximum iteration count, under-relaxes the mechanics/constitutive fixed-point
map, and refuses to accept a physical step unless the plastic-strain and
dislocation-density states satisfy explicit mixed absolute/relative tolerances.

No fracture, hazard, event-energy, or material parameter is changed.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from .plastic_flow_stagger_consistent_v1043 import (
    transform_source as _stagger_consistent_transform,
)

MODEL_ID = "v10.4.3_relaxed_converged_stagger_fixed_point"
MODULE_NAME = "arrhenius_fracture._sharp_front_v1043_fixed_point_converged"


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label} changed: expected 1 occurrence, found {count}")
    return text.replace(old, new, 1)


def transform_source(source: str) -> str:
    text = _stagger_consistent_transform(source)

    text = _replace_once(
        text,
        """    p.add_argument('--n-stagger', type=int, default=2, dest='n_stagger',
                   help='Mech<->plastic stagger iterations per step. The coupling '
                        'is well-converged at 2 here (Kc unchanged at 10).')
""",
        """    p.add_argument('--n-stagger', type=int, default=40, dest='n_stagger',
                   help='Maximum relaxed mechanics/plasticity fixed-point iterations per step.')
    p.add_argument('--stagger-relaxation', type=float, default=0.25,
                   dest='stagger_relaxation',
                   help='Under-relaxation factor for the stagger fixed-point map; 0 < alpha <= 1.')
    p.add_argument('--stagger-rtol', type=float, default=1.0e-6,
                   dest='stagger_rtol',
                   help='Relative convergence tolerance for plastic strain and density.')
    p.add_argument('--stagger-ep-atol', type=float, default=1.0e-12,
                   dest='stagger_ep_atol',
                   help='Absolute plastic-strain convergence tolerance.')
    p.add_argument('--stagger-rho-atol-m2', type=float, default=1.0e3,
                   dest='stagger_rho_atol_m2',
                   help='Absolute dislocation-density convergence tolerance [m^-2].')
""",
        "v10.4.3 strict fixed-point CLI",
    )

    text = _replace_once(
        text,
        """                plastic_work_info_v1042 = None
                plastic_work_accepted_gp_v1042 = None
                ep_gp_step0_v1043 = ep_gp.copy()
                rho_gp_step0_v1043 = rho_gp.copy()
                for it in range(args.n_stagger):
""",
        """                plastic_work_info_v1042 = None
                plastic_work_accepted_gp_v1042 = None
                ep_gp_step0_v1043 = ep_gp.copy()
                rho_gp_step0_v1043 = rho_gp.copy()
                ep_gp_iter_v1043 = ep_gp_step0_v1043.copy()
                rho_gp_iter_v1043 = rho_gp_step0_v1043.copy()
                stagger_converged_v1043 = False
                stagger_iterations_used_v1043 = 0
                stagger_residual_v1043 = float('inf')
                stagger_ep_residual_v1043 = float('inf')
                stagger_rho_residual_v1043 = float('inf')
                _v1043_stagger_alpha = float(args.stagger_relaxation)
                _v1043_stagger_rtol = float(args.stagger_rtol)
                _v1043_stagger_ep_atol = float(args.stagger_ep_atol)
                _v1043_stagger_rho_atol = float(args.stagger_rho_atol_m2)
                if int(args.n_stagger) < 1:
                    raise RuntimeError('v10.4.3 n_stagger must be at least one')
                if not (0.0 < _v1043_stagger_alpha <= 1.0):
                    raise RuntimeError(
                        'v10.4.3 stagger relaxation must satisfy 0 < alpha <= 1: '
                        f'{_v1043_stagger_alpha}'
                    )
                if _v1043_stagger_rtol < 0.0 or _v1043_stagger_ep_atol < 0.0 \
                        or _v1043_stagger_rho_atol < 0.0:
                    raise RuntimeError('v10.4.3 stagger tolerances must be non-negative')
                for it in range(args.n_stagger):
                    ep_gp = ep_gp_iter_v1043
                    rho_gp = rho_gp_iter_v1043
""",
        "v10.4.3 fixed-point initialization",
    )

    text = _replace_once(
        text,
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
        """                        (
                            ep_gp_candidate_v1043,
                            rho_gp_candidate_v1043,
                            dot_ep_candidate_v1043,
                            plastic_work_info_candidate_v1043,
                        ) = update_plasticity(
                            ep_gp_step0_v1043, rho_gp_step0_v1043,
                            sigma_gp, mat, T, dt_cur,
                            plast_model, cfg.dislocations, return_info=True)
                        ep_gp_candidate_v1043 = np.asarray(
                            ep_gp_candidate_v1043, dtype=float
                        )
                        rho_gp_candidate_v1043 = np.asarray(
                            rho_gp_candidate_v1043, dtype=float
                        )
                        if ep_gp_candidate_v1043.shape != ep_gp_iter_v1043.shape:
                            raise RuntimeError(
                                'v10.4.3 candidate plastic-strain shape mismatch: '
                                f'{ep_gp_candidate_v1043.shape} != {ep_gp_iter_v1043.shape}'
                            )
                        if rho_gp_candidate_v1043.shape != rho_gp_iter_v1043.shape:
                            raise RuntimeError(
                                'v10.4.3 candidate density shape mismatch: '
                                f'{rho_gp_candidate_v1043.shape} != {rho_gp_iter_v1043.shape}'
                            )
                        if not np.all(np.isfinite(ep_gp_candidate_v1043)) \
                                or not np.all(np.isfinite(rho_gp_candidate_v1043)):
                            raise RuntimeError(
                                'v10.4.3 fixed-point candidate contains non-finite state'
                            )
                        if np.any(rho_gp_candidate_v1043 < 0.0):
                            raise RuntimeError(
                                'v10.4.3 fixed-point candidate contains negative density'
                            )

                        _v1043_ep_delta = ep_gp_candidate_v1043 - ep_gp_iter_v1043
                        _v1043_rho_delta = rho_gp_candidate_v1043 - rho_gp_iter_v1043
                        _v1043_ep_scale = max(
                            float(np.max(np.abs(ep_gp_candidate_v1043))),
                            float(np.max(np.abs(ep_gp_iter_v1043))),
                            1.0,
                        )
                        _v1043_rho_scale = max(
                            float(np.max(np.abs(rho_gp_candidate_v1043))),
                            float(np.max(np.abs(rho_gp_iter_v1043))),
                            1.0,
                        )
                        _v1043_ep_den = (
                            _v1043_stagger_ep_atol
                            + _v1043_stagger_rtol * _v1043_ep_scale
                        )
                        _v1043_rho_den = (
                            _v1043_stagger_rho_atol
                            + _v1043_stagger_rtol * _v1043_rho_scale
                        )
                        stagger_ep_residual_v1043 = float(
                            np.max(np.abs(_v1043_ep_delta))
                            / max(_v1043_ep_den, np.finfo(float).tiny)
                        )
                        stagger_rho_residual_v1043 = float(
                            np.max(np.abs(_v1043_rho_delta))
                            / max(_v1043_rho_den, np.finfo(float).tiny)
                        )
                        stagger_residual_v1043 = max(
                            stagger_ep_residual_v1043,
                            stagger_rho_residual_v1043,
                        )
                        stagger_iterations_used_v1043 = it + 1

                        _v1043_dWp_candidate_gp = np.asarray(
                            plastic_work_info_candidate_v1043.get(
                                'dWp_accepted_gp', []
                            ),
                            dtype=float,
                        ).reshape(-1)
                        if _v1043_dWp_candidate_gp.size != mesh.ne:
                            raise RuntimeError(
                                'v10.4.3 accepted plastic-work ledger size mismatch: '
                                f'{_v1043_dWp_candidate_gp.size} != {mesh.ne}'
                            )
                        if not np.all(np.isfinite(_v1043_dWp_candidate_gp)):
                            raise RuntimeError(
                                'v10.4.3 accepted plastic-work ledger contains non-finite values'
                            )

                        if stagger_residual_v1043 <= 1.0:
                            ep_gp = ep_gp_candidate_v1043.copy()
                            rho_gp = rho_gp_candidate_v1043.copy()
                            dot_ep = dot_ep_candidate_v1043
                            plastic_work_info_v1042 = plastic_work_info_candidate_v1043
                            plastic_work_accepted_gp_v1042 = (
                                _v1043_dWp_candidate_gp.copy()
                            )
                            stagger_converged_v1043 = True
                            break

                        ep_gp_iter_v1043 = (
                            ep_gp_iter_v1043
                            + _v1043_stagger_alpha * _v1043_ep_delta
                        )
                        rho_gp_iter_v1043 = (
                            rho_gp_iter_v1043
                            + _v1043_stagger_alpha * _v1043_rho_delta
                        )
                        if not np.all(np.isfinite(ep_gp_iter_v1043)) \
                                or not np.all(np.isfinite(rho_gp_iter_v1043)) \
                                or np.any(rho_gp_iter_v1043 < 0.0):
                            raise RuntimeError(
                                'v10.4.3 relaxed fixed-point iterate became invalid'
                            )
""",
        "v10.4.3 relaxed convergence iteration",
    )

    text = _replace_once(
        text,
        """                # The last constitutive update changes ep_gp/rho_gp after the
                # last mechanics solve.  Close the staggered step with a
""",
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

                # The converged constitutive update changes ep_gp/rho_gp after the
                # last mechanics solve.  Close the staggered step with a
""",
        "v10.4.3 strict convergence gate",
    )

    text = _replace_once(
        text,
        """                        'mechanics_plasticity_stagger_iterations': int(args.n_stagger),
                        'physical_time_advance_per_accepted_step': 'dt_cur_not_n_stagger_times_dt_cur',
""",
        """                        'mechanics_plasticity_stagger_max_iterations': int(args.n_stagger),
                        'mechanics_plasticity_stagger_iterations_used': int(
                            stagger_iterations_used_v1043
                        ),
                        'mechanics_plasticity_stagger_converged': bool(
                            stagger_converged_v1043
                        ),
                        'mechanics_plasticity_stagger_scaled_residual': float(
                            stagger_residual_v1043
                        ),
                        'mechanics_plasticity_stagger_ep_scaled_residual': float(
                            stagger_ep_residual_v1043
                        ),
                        'mechanics_plasticity_stagger_rho_scaled_residual': float(
                            stagger_rho_residual_v1043
                        ),
                        'mechanics_plasticity_stagger_relaxation': float(
                            _v1043_stagger_alpha
                        ),
                        'mechanics_plasticity_stagger_rtol': float(
                            _v1043_stagger_rtol
                        ),
                        'mechanics_plasticity_stagger_ep_atol': float(
                            _v1043_stagger_ep_atol
                        ),
                        'mechanics_plasticity_stagger_rho_atol_m2': float(
                            _v1043_stagger_rho_atol
                        ),
                        'physical_time_advance_per_accepted_step': 'dt_cur_not_n_stagger_times_dt_cur',
""",
        "v10.4.3 fixed-point audit provenance",
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
        raise RuntimeError("could not allocate v10.4.3 fixed-point module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = str(source_path)
    module.__package__ = "arrhenius_fracture"
    sys.modules[MODULE_NAME] = module
    try:
        exec(
            compile(
                transformed,
                str(source_path) + "[v10.4.3-fixed-point-converged]",
                "exec",
            ),
            module.__dict__,
        )
    except Exception:
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


__all__ = ["MODEL_ID", "load_transformed_sharp_front", "transform_source"]
